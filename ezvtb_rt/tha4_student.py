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
from ezvtb_rt.engine import Engine, createMemory


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


class VRAMCacher(object):
    """Implements LRU cache strategy for GPU memory management"""
    def __init__(self, nbytes1: int, nbytes2: int, max_size: float):
        sum_nkbytes = (nbytes1 + nbytes2) / 1024
        self.pool = []
        if max_size > 0:
            while len(self.pool) * sum_nkbytes < max_size * 1024 * 1024:
                self.pool.append((VRAMMem(nbytes1), VRAMMem(nbytes2)))
        self.nbytes1 = nbytes1
        self.nbytes2 = nbytes2
        self.cache = OrderedDict()
        self.hits = 0
        self.miss = 0
        if max_size <= 0:
            self.single_mem = (VRAMMem(nbytes1), VRAMMem(nbytes2))
        self.max_size = max_size
    
    def query(self, hs: int) -> bool:
        cached = self.cache.get(hs)
        if cached is not None:
            return True
        else:
            return False
    
    def read_mem_set(self, hs: int) -> tuple:
        cached = self.cache.get(hs)
        if cached is not None:
            self.hits += 1
            self.cache.move_to_end(hs)
            return cached
        else:
            self.miss += 1
            return None
    
    def write_mem_set(self, hs: int) -> tuple:
        if self.max_size <= 0:
            return self.single_mem
        if len(self.pool) != 0:
            mem_set = self.pool.pop()
        else:
            mem_set = self.cache.popitem(last=False)[1]
        self.cache[hs] = mem_set
        return mem_set


class THA4Student():
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
        self.prepareEngines(model_dir)
        self.prepareMemories()
        self.setMemsToEngines()
        self.prepare_cache(vram_cache_size)
        self.prepareStreams()
        print('THA4 Student Model initialized successfully')
    
    def prepareEngines(self, model_dir):
        """Load TensorRT engines for both models"""
        self.face_morpher = Engine(
            os.path.join(model_dir, 'face_morpher.trt'), 1)
        self.body_morpher = Engine(
            os.path.join(model_dir, 'body_morpher.trt'), 3)
    
    def prepareMemories(self):
        """Initialize GPU memory for inputs and outputs"""
        self.memories = {}
        
        # Face Morpher: pose input, face_morphed output
        self.memories['face_pose'] = createMemory(
            self.face_morpher.inputs[0])
        self.memories['face_morphed'] = createMemory(
            self.face_morpher.outputs[0])
        
        # Body Morpher: 3 inputs, 2 outputs
        self.memories['input_image'] = createMemory(
            self.body_morpher.inputs[0])
        self.memories['face_morphed_input'] = createMemory(
            self.body_morpher.inputs[1])
        self.memories['body_pose'] = createMemory(
            self.body_morpher.inputs[2])
        self.memories['result'] = createMemory(
            self.body_morpher.outputs[0])
        self.memories['cv_result'] = createMemory(
            self.body_morpher.outputs[1])
    
    def setMemsToEngines(self):
        """Bind memory to TensorRT engines"""
        # Face Morpher: 1 input, 1 output
        self.face_morpher.setInputMems([
            self.memories['face_pose']
        ])
        self.face_morpher.setOutputMems([
            self.memories['face_morphed']
        ])
        
        # Body Morpher: 3 inputs, 2 outputs
        self.body_morpher.setInputMems([
            self.memories['input_image'],
            self.memories['face_morphed_input'],
            self.memories['body_pose']
        ])
        self.body_morpher.setOutputMems([
            self.memories['result'],
            self.memories['cv_result']
        ])
    
    def prepare_cache(self, vram_cache_size: float):
        """Initialize GPU memory cache for face morphing results"""
        # Cache face_morphed outputs (1x4x128x128 = 262144 floats)
        # Store hash -> (face_morphed, padding)
        self.face_morpher_cacher = VRAMCacher(
            262144 * 4,  # face_morphed: 128*128*4*4 bytes
            4096,        # padding for alignment
            vram_cache_size
        )
    
    def prepareStreams(self):
        """Initialize CUDA streams for pipelined execution"""
        self.updatestream = cuda.Stream()
        self.instream = cuda.Stream()
        self.outstream = cuda.Stream()
        self.finishedFetchRes = cuda.Event()
        self.finishedExec = cuda.Event()
    
    def setImage(self, img: np.ndarray):
        """Set input image for processing
        
        Args:
            img: Input image [512, 512, 4] RGBA format
        """
        assert len(img.shape) == 3 and img.shape[0] == 512 and \
               img.shape[1] == 512 and img.shape[2] == 4, \
               "Image must be 512x512 RGBA"
        
        # Copy image to GPU memory
        np.copyto(self.memories['input_image'].host, img)
        self.memories['input_image'].htod(self.updatestream)
        self.updatestream.synchronize()
    
    def update_image(self, img: np.ndarray):
        """Alias for setImage for interface compatibility"""
        self.setImage(img)
    
    def inference(self, pose: np.ndarray) -> list:
        """Run inference with pose data
        
        Two-stage pipeline:
        1. Face Morpher: generates face from pose
        2. Body Morpher: transforms full body using face result
        
        Args:
            pose: Pose parameters [1, 45] (12 eyebrow + 27 face + 6 rotation)
        
        Returns:
            Output image [512, 512, 4] RGBA format
        """
        # Fetch result from previous inference
        self.outstream.wait_for_event(self.finishedExec)
        self.memories['cv_result'].dtoh(self.outstream)
        self.finishedFetchRes.record(self.outstream)
        
        # Copy pose to GPU
        np.copyto(self.memories['face_pose'].host, pose)
        self.memories['face_pose'].htod(self.instream)
        
        # Stage 1: Face Morpher (pose only, generates 128x128 face)
        pose_hash = hash(pose.tobytes())
        morpher_cached = self.face_morpher_cacher.read_mem_set(pose_hash)
        
        if morpher_cached is not None:
            # Use cached result - copy from cache to input buffer (D2D)
            cuda.memcpy_dtod_async(
                self.memories['face_morphed_input'].device,
                morpher_cached[0].device,
                self.memories['face_morphed_input'].host.nbytes,
                self.instream)
        else:
            # Run inference and cache result
            self.face_morpher.exec(self.instream)
            
            # Cache the face_morphed output
            mem_set = self.face_morpher_cacher.write_mem_set(pose_hash)
            cuda.memcpy_dtod_async(
                mem_set[0].device,
                self.memories['face_morphed'].device,
                self.memories['face_morphed'].host.nbytes,
                self.instream)
            
            # Copy to body_morpher input as well
            cuda.memcpy_dtod_async(
                self.memories['face_morphed_input'].device,
                self.memories['face_morphed'].device,
                self.memories['face_morphed_input'].host.nbytes,
                self.instream)
        
        # Copy full pose to body_morpher
        np.copyto(self.memories['body_pose'].host, pose)
        self.memories['body_pose'].htod(self.instream)
        
        # Wait for image upload to complete
        self.instream.wait_for_event(self.finishedFetchRes)
        
        # Stage 2: Body Morpher (full transformation)
        self.body_morpher.exec(self.instream)
        
        # Record completion event
        self.finishedExec.record(self.instream)
        
        # Synchronize and return result
        self.finishedFetchRes.synchronize()
        return [self.memories['cv_result'].host.copy()]
    
    def fetchRes(self):
        """Fetch results from GPU (for framework compatibility)
        
        Returns:
            List[np.ndarray]: List containing output image
        """
        self.finishedFetchRes.synchronize()
        return [self.memories['cv_result'].host.copy()]
