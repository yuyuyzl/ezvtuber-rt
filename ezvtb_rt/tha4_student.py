"""THA4 Student Model (Mode 14) TensorRT Implementation
Two-stage inference architecture:
- face_morpher: SIREN network for face morphing (only needs pose)
- body_morpher: Multi-scale SIREN for full-body transformation
"""
import os
import numpy as np
import pycuda.driver as cuda
from collections import OrderedDict

from ezvtb_rt.trt_utils import *
from ezvtb_rt.trt_engine import HostDeviceMem, TRTEngine


class VRAMMem(object):
    """Manages allocation and lifecycle of CUDA device memory"""
    def __init__(self, nbytes: int):
        self.device = cuda.mem_alloc(nbytes)

    def __str__(self):
        return "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()
    
    def __del__(self):
        self.device.free()


class DirectVRAMCacher(object):
    """Implements LRU cache strategy for GPU memory management"""
    """Do not use nvcomp because face morpher output is small and compression overhead is high"""
    def __init__(self, nbytes: int, max_size: float):
        assert nbytes > 0
        self.pool = []
        if max_size > 0:
            while len(self.pool) * nbytes < max_size * 1024 * 1024 * 1024:
                self.pool.append(VRAMMem(nbytes))
        self.nbytes = nbytes
        self.cache = OrderedDict()
        self.hits = 0
        self.miss = 0
        self.max_size = max_size
    
    def query(self, hs: int) -> bool:
        cached = self.cache.get(hs)
        if cached is not None:
            return True
        else:
            return False
    
    def read_mem_set(self, hs: int) -> VRAMMem:
        cached = self.cache.get(hs)
        if cached is not None:
            self.hits += 1
            self.cache.move_to_end(hs)
            return cached
        else:
            self.miss += 1
            return None
    
    def write_mem_set(self, hs: int) -> VRAMMem:
        if len(self.pool) != 0:
            mem_set = self.pool.pop()
        else:
            mem_set = self.cache.popitem(last=False)[1]
        self.cache[hs] = mem_set
        return mem_set

class THA4StudentEngines():
    """THA4 Student Model (Mode 14) TensorRT implementation
    
    Two-stage SIREN-based inference:
    - Stage 1: Face Morpher - generates 128x128 face from pose only
    - Stage 2: Body Morpher - transforms full 512x512 image using face output
    
    Attributes:
        face_morpher_cacher (VRAMCacher): Cache for face morphing results
        streams (cuda.Stream): Dedicated CUDA streams for operations
        events (cuda.Event): Synchronization events
    """
    def __init__(self, model_dir, vram_cache_size: float = 1.0):
        """Initialize THA4 Student with model directory and caching
        
        Args:
            model_dir: Directory containing TensorRT engine files
            vram_cache_size: Total GPU memory for caching (MB)
        """
        face_morpher_trt_path = join(model_dir, 'face_morpher.trt')
        body_morpher_trt_path = join(model_dir, 'body_morpher.trt') 
        if not os.path.isfile(face_morpher_trt_path) or not os.path.isfile(body_morpher_trt_path):
            face_morpher_onnx_path = join(model_dir, 'face_morpher.onnx')
            body_morpher_onnx_path = join(model_dir, 'body_morpher.onnx')
            if not os.path.isfile(face_morpher_onnx_path) or \
               not os.path.isfile(body_morpher_onnx_path):
                raise FileNotFoundError('Required model files not found in directory')
            save_engine(build_engine(face_morpher_onnx_path, precision='fp16'), face_morpher_trt_path)
            save_engine(build_engine(body_morpher_onnx_path, precision='fp16'), body_morpher_trt_path)
        TRT_LOGGER.log(TRT_LOGGER.INFO, 'Creating Engines')
        self.face_morpher = TRTEngine(join(model_dir, 'face_morpher.trt'), 1)
        self.face_morpher.configure_in_out_tensors()
        self.body_morpher = TRTEngine(join(model_dir, 'body_morpher.trt'), 3)
        self.body_morpher.configure_in_out_tensors()

        # Create CUDA streams
        self.stream = cuda.Stream()  # Main stream for setImage and inference
        self.cachestream = cuda.Stream()  # Async cache writes
        
        # Setup single VRAMCacher with compressed storage (size in GB)
        self.cacher = DirectVRAMCacher(
            nbytes=self.face_morpher.outputs[0].host.nbytes,
            max_size=vram_cache_size
        ) if vram_cache_size > 0.0 else None

        # Create CUDA events for synchronization
        self.finishedFaceMorpher = cuda.Event()
        self.finishedFetchRes = cuda.Event()
    
    def setImage(self, img:np.ndarray, sync:bool):
        """Set input image and prepare for processing
        Args:
            img (np.ndarray): Input image array (512x512x4 RGBA format)
        """
        assert(len(img.shape) == 3 and 
               img.shape[0] == 512 and 
               img.shape[1] == 512 and 
               img.shape[2] == 4 and
               img.dtype == np.uint8)
        
        np.copyto(self.body_morpher.inputs[0].host, img)
        self.body_morpher.inputs[0].htod(self.stream)
        if sync:
            self.stream.synchronize()

    def syncSetImage(self, img:np.ndarray):
        self.setImage(img, sync=True)
    
    def asyncSetImage(self, img:np.ndarray):
        self.setImage(img, sync=False)
    
    def asyncInfer(self, pose:np.ndarray, stream=None):
        stream = stream if stream is not None else self.stream
        face_pose = pose[:, :39]
        
        # Copy pose to GPU
        np.copyto(self.face_morpher.inputs[0].host, face_pose)
        self.face_morpher.inputs[0].htod(stream)
        np.copyto(self.body_morpher.inputs[2].host, pose)
        self.body_morpher.inputs[2].htod(stream)
        
        # Stage 1: Face Morpher (pose only, generates 128x128 face)
        face_pose_hash = hash(str(face_pose))
        face_cached = None if self.cacher is None else \
            self.cacher.read_mem_set(face_pose_hash)
        
        self.cachestream.synchronize()

        if face_cached is not None:
            # Use cached result - copy from cache to input buffer (D2D)
            cuda.memcpy_dtod_async(
                self.body_morpher.inputs[1].device,
                face_cached.device,
                self.body_morpher.inputs[1].host.nbytes,
                stream)
        else:
            # Run inference and cache result
            self.face_morpher.asyncKickoff(stream)
            self.finishedFaceMorpher.record(stream)
            
            if self.cacher is not None:
                self.cachestream.wait_for_event(self.finishedFaceMorpher)
                face_cached = self.cacher.write_mem_set(face_pose_hash)
                cuda.memcpy_dtod_async(
                    face_cached.device,
                    self.face_morpher.outputs[0].device,
                    self.face_morpher.outputs[0].host.nbytes,
                    self.cachestream)
        
            self.body_morpher.inputs[1].bridgeFrom(self.face_morpher.outputs[0], stream)

        self.body_morpher.asyncKickoff(stream)
        self.body_morpher.outputs[1].dtoh(stream)
        self.finishedFetchRes.record(stream)

    def getOutputMem(self) -> HostDeviceMem:
        """Get output HostDeviceMem for further processing
        Returns:
            HostDeviceMem: Output memory buffer on GPU
        """
        return self.body_morpher.outputs[1]

    def syncAndGetOutput(self) -> np.ndarray:
        """Retrieve processed results from GPU
        Returns:
            np.ndarray: Output image array
        """
        self.finishedFetchRes.synchronize()
        return self.body_morpher.outputs[1].host