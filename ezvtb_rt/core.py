from ezvtb_rt.trt_utils import *
from ezvtb_rt.trt_engine import TRTEngine, HostDeviceMem
from ezvtb_rt.tha3 import THA3Engines
from ezvtb_rt.tha4 import THA4Engines
from ezvtb_rt.tha4_student import THA4Student
from ezvtb_rt.cache import Cacher
from ezvtb_rt.common import Core
import ezvtb_rt
import numpy as np
import os
from typing import List

def has_none_object_none_pattern(lst):
    if len(lst) < 3:
        return False
    for i in range(len(lst) - 2):
        if lst[i] is None and lst[i+1] is not None and lst[i+2] is None:
            return True
    return False

def find_none_block_indices(lst):
    if not lst:
        return None, None  # Empty list: no indices
    
    none_indices = [i for i, x in enumerate(lst) if x is None]
    if not none_indices:
        return None, None  # No Nones: no block
    
    first_none = min(none_indices)
    last_none = max(none_indices)
    
    # Optional: Verify contiguous block (all between first and last are None)
    if any(lst[i] is not None for i in range(first_none, last_none + 1)):
        raise ValueError("Nones are not contiguous")
    
    i1 = first_none - 1 if first_none > 0 else None  # None if block starts at 0
    i2 = last_none + 1 if last_none < len(lst) - 1 else None  # None if block ends at last index
    
    return i1, i2

class CoreTRT(Core):
    """Main inference pipeline combining THA face model with optional components:
    - RIFE for frame interpolation
    - SR for super resolution
    - Cacher for output caching
    
    Args:
        tha_dir: Path to THA model directory
        vram_cache_size: VRAM allocated for model caching (GB)
        use_eyebrow: Enable eyebrow motion processing
        rife_dir: Path to RIFE model directory (None to disable)
        sr_dir: Path to SR model directory (None to disable) 
        cache_max_giga: Max disk cache size (GB)
    """
    def __init__(self, 
                 tha_model_version:str = 'v3',
                 tha_model_seperable:bool = True,
                 tha_model_fp16:bool = False,
                 tha_model_name:str = None,
                 rife_model_enable:bool = False,
                 rife_model_scale:int = 2,
                 rife_model_fp16:bool = False,
                 sr_model_enable:bool = False,
                 sr_model_scale:int = 2,
                 sr_model_fp16:bool = False,
                 vram_cache_size:float = 1.0, 
                 cache_max_giga:float = 2.0, 
                 use_eyebrow:bool = False):
        if tha_model_version == 'v3':
            tha_path = os.path.join(ezvtb_rt.EZVTB_DATA, 'tha3',
                                    'seperable' if tha_model_seperable else 'standard', 
                                    'fp16' if tha_model_fp16 else 'fp32')
            self.v3 = True
        elif tha_model_version == 'v4':
            tha_path = os.path.join(ezvtb_rt.EZVTB_DATA, 'tha4', 
                                    'fp16' if tha_model_fp16 else 'fp32')
            self.v3 = False
        elif tha_model_version == 'v4_student':
            # Support custom student models in data/models/custom_tha4_models
            if tha_model_name:
                # Build path relative to project root (parent of ezvtuber-rt)
                project_root = os.path.dirname(
                    os.path.dirname(os.path.dirname(__file__))
                )
                tha_path = os.path.normpath(os.path.join(
                    project_root, 'data', 'models',
                    'custom_tha4_models', tha_model_name
                ))
            else:
                tha_path = os.path.join(
                    ezvtb_rt.EZVTB_DATA, 'tha4_student'
                )
            self.v3 = False
        else:
            raise ValueError('Unsupported THA model version')
        rife_path = None
        if rife_model_enable:
            rife_path = os.path.join(ezvtb_rt.EZVTB_DATA, 'rife', 
                                     f'rife_x{rife_model_scale}_{"fp16" if rife_model_fp16 else "fp32"}.trt')
            
        sr_path = None
        if sr_model_enable:
            if sr_model_scale == 4:
                if sr_model_fp16:
                    sr_path = os.path.join(ezvtb_rt.EZVTB_DATA, 'Real-ESRGAN', 'exported_256_fp16.trt')
                else:
                    sr_path = os.path.join(ezvtb_rt.EZVTB_DATA, 'Real-ESRGAN', 'exported_256_fp32.trt')
            else: #x2
                if sr_model_fp16:
                    sr_path = os.path.join(ezvtb_rt.EZVTB_DATA, 'waifu2x', 'noise0_scale2x_fp16.trt')
                else:
                    sr_path = os.path.join(ezvtb_rt.EZVTB_DATA, 'waifu2x', 'noise0_scale2x_fp32.trt')

        # Initialize core THA face model
        if self.v3:
            self.tha = THA3Engines(tha_path, vram_cache_size, use_eyebrow)
        elif tha_model_version == 'v4_student':
            self.tha = THA4Student(tha_path)
        elif tha_model_version == 'v4':
            self.tha = THA4Engines(tha_path, vram_cache_size, use_eyebrow)
        else:
            raise ValueError('Unsupported THA model version')

        self.tha_model_fp16: bool = tha_model_fp16

        # Initialize optional components
        self.rife: TRTEngine = None  # Frame interpolation module (default)
        self.rifes: List[TRTEngine] = []  # Additional RIFE engines for different scales
        self.sr: TRTEngine = None    # Super resolution module
        self.cacher: Cacher = None# Output caching system
        self.sr_cacher: Cacher = None # SR output caching
        self.last_tha_output: np.ndarray | None = None

        # Initialize RIFE if model path provided
        if rife_path is not None:
            self.rife = TRTEngine(rife_path, 2)
            self.rife.configure_in_out_tensors()
            # Load additional RIFE engines for x2, x3, x4 similar to ORT
            for i in [2, 3, 4]:
                path = os.path.join(ezvtb_rt.EZVTB_DATA, 'rife', f'rife_x{i}_{"fp16" if rife_model_fp16 else "fp32"}.trt')
                if os.path.exists(path):
                    engine = TRTEngine(path, 2)
                    engine.configure_in_out_tensors()
                    self.rifes.append(engine)
        # Initialize SR if model path provided
        if sr_path is not None:
            self.sr = TRTEngine(sr_path, 1)
            self.sr.configure_in_out_tensors(rife_model_scale if rife_model_enable else 1)
            if cache_max_giga > 0.0:
                # SR outputs are upscaled (expected 1024x1024 RGBA)
                self.sr_cacher = Cacher(cache_max_giga, width=1024, height=1024)

        # Initialize cache if enabled
        if cache_max_giga > 0.0:
            self.cacher = Cacher(cache_max_giga)

        self.main_stream: cuda.Stream = cuda.Stream()
        self.cache_stream: cuda.Stream = cuda.Stream()

    def setImage(self, img:np.ndarray):
        """Set input image for processing pipeline
        Args:
            img: Input image in BGR format (HWC, uint8)
        """
        self.tha.syncSetImage(img)
        self.last_tha_output = img

    def inference(self, poses: List[np.ndarray]) -> np.ndarray:
        """Run full inference pipeline
        Args:
            poses: One or more facial pose arrays. If multiple are provided,
                   RIFE interpolation and per-pose caching are enabled similar to ORT.
        Returns:
            Batched output images from the final stage in the pipeline.
        """
        assert isinstance(poses, list) and all(isinstance(p, np.ndarray) for p in poses), "poses must be a list of numpy arrays"

        if len(poses) == 0:
            raise ValueError('poses must not be empty')

        # Normalize dtype for all poses
        for i in range(len(poses)):
            poses[i] = poses[i].astype(np.float32)
            if self.tha_model_fp16 and not self.v3:
                poses[i] = poses[i].astype(np.float16)

        tha_pose = poses[-1]

        tha_mem_res: HostDeviceMem = self.tha.getOutputMem()

        cached_output = None
        # THA cache lookup for the last pose only (matches ORT semantics)
        if self.cacher is not None:
            self.cache_stream.synchronize()
            cached_output = self.cacher.read(hash(str(tha_pose)))
            if cached_output is not None:
                np.copyto(tha_mem_res.host, cached_output)
                tha_mem_res.htod(self.main_stream)
        # Run THA when not cached
        if cached_output is None:
            self.tha.asyncInfer(tha_pose, self.main_stream)
            # Need host data for caching or SR-only path
            tha_mem_res.dtoh(self.main_stream)
            self.main_stream.synchronize()
            if self.cacher is not None:
                self.cacher.write(hash(str(tha_pose)), tha_mem_res.host)

        # If no RIFE and no SR, just return THA result
        if self.rife is None and self.sr is None:
            return np.expand_dims(cached_output if cached_output is not None else np.copy(tha_mem_res.host), axis=0)
        
        # RIFE interpolation stage
        rife_mem_res : HostDeviceMem = None
        if self.rife is not None:
            # Select appropriate RIFE engine based on number of interpolated frames (x2/x3/x4)
            rife_engine = self.rife
            if len(poses) > 1 and len(poses) - 2 < len(self.rifes):
                rife_engine = self.rifes[len(poses) - 2]

            # Configure batch for dynamic outputs
            rife_engine.configure_in_out_tensors()
            rife_mem_res = rife_engine.outputs[0]

            # Fast path: all poses except the last are cached
            all_cached = len(poses) > 1 and self.cacher is not None and all(self.cacher.query(hash(str(p))) for p in poses[:-1])
            if all_cached:
                cached_frames = [self.cacher.get(hash(str(p))) for p in poses[:-1]]
                for i in range(len(poses) - 1):
                    np.copyto(rife_mem_res.host[i], cached_frames[i])
                np.copyto(rife_mem_res.host[-1], tha_mem_res.host)
                rife_mem_res.htod(self.main_stream)
            else: # Cases: one pose given but use interpolation, multiple poses with some missing cache, or no cache at all
                # Prepare previous frame
                np.copyto(rife_engine.inputs[0].host, self.last_tha_output)
                rife_engine.inputs[0].htod(self.main_stream)

                # Current frame
                rife_engine.inputs[1].bridgeFrom(tha_mem_res, self.main_stream)

                rife_engine.asyncKickoff(self.main_stream)
                rife_engine.outputs[0].dtoh(self.main_stream)
                self.main_stream.synchronize()

                # Cache interpolated frames when they align with provided poses
                if len(poses) > 1 and self.cacher is not None and len(poses) == rife_mem_res.host.shape[0]:
                    for i in range(len(poses) - 1):
                        self.cacher.write(hash(str(poses[i])), rife_mem_res.host[i])
                elif self.cacher is not None and len(poses) == 1:
                    # Single pose with RIFE (no interpolation), cache THA output
                    self.cacher.write(hash(str(poses[0])), tha_mem_res.host)

            # Track last THA output for future interpolation
            self.last_tha_output = np.copy(tha_mem_res.host)
        else:
            # No RIFE, SR-only uses THA output as a single-frame batch
            rife_mem_res = tha_mem_res
        
        if self.sr is None:
            return np.copy(rife_mem_res.host)

        # Special handling when only one pose was provided
        if len(poses) == 1:
            hs = hash(str(poses[0]))
            cached_sr = None if self.sr_cacher is None else self.sr_cacher.get(hs)
            if cached_sr is not None: # SR cache hit
                # Run SR on remaining frames only
                if len(rife_mem_res.host.shape) == 4 and rife_mem_res.host.shape[0] > 1:
                    res_host = rife_mem_res.host[:-1]
                    sr_batch = res_host.shape[0]
                    self.sr.configure_in_out_tensors(sr_batch)
                    cuda.memcpy_dtod_async(self.sr.inputs[0].device, rife_mem_res.device, res_host.nbytes, self.main_stream)
                    self.sr.asyncKickoff(self.main_stream)
                    self.sr.outputs[0].dtoh(self.main_stream)
                    self.main_stream.synchronize()
                    return np.concatenate((self.sr.outputs[0].host, np.expand_dims(cached_sr, axis=0)), axis=0)
                else: # there is only one frame to SR, which is cached
                    return np.expand_dims(cached_sr, axis=0)
            else: # No SR cache hit, dealing with single pose with RIFE interpolation followed by SR, or SR for a single tha result
                sr_batch = rife_mem_res.host.shape[0] if len(rife_mem_res.host.shape) == 4 else 1
                self.sr.configure_in_out_tensors(sr_batch)
                self.sr.inputs[0].bridgeFrom(rife_mem_res, self.main_stream)
                self.sr.asyncKickoff(self.main_stream)
                self.sr.outputs[0].dtoh(self.main_stream)
                self.main_stream.synchronize()
                if self.sr_cacher is not None:
                    self.sr_cacher.write(hs, self.sr.outputs[0].host[-1])
                return np.copy(self.sr.outputs[0].host)
        else: # Multiple poses with RIFE followed by SR
            assert len(rife_mem_res.host.shape) == 4 and rife_mem_res.host.shape[0] == len(poses)
            sr_results = [None] * len(poses)
            to_sr_images = []
            for i in range(len(poses)):
                hs = hash(str(poses[i]))
                sr_results[i] = None if self.sr_cacher is None else self.sr_cacher.get(hs)
                if sr_results[i] is None:
                    to_sr_images.append(rife_mem_res.host[i])
            if all(x is not None for x in sr_results): # All SR cached
                return np.stack(sr_results, axis=0)
            else: # Some SR missing, run SR on missing ones
                sr_batch = len(to_sr_images)
                self.sr.configure_in_out_tensors(sr_batch)
                for i, to_sr_image in enumerate(to_sr_images):
                    np.copyto(self.sr.inputs[0].host[i], to_sr_image)
                self.sr.inputs[0].htod(self.main_stream)
                self.sr.asyncKickoff(self.main_stream)
                self.sr.outputs[0].dtoh(self.main_stream)
                self.main_stream.synchronize()
                # Fill in SR results and update cache
                sr_output_idx = 0
                for i in range(len(poses)):
                    if sr_results[i] is None:
                        sr_results[i] = np.copy(self.sr.outputs[0].host[sr_output_idx])
                        if self.sr_cacher is not None:
                            self.sr_cacher.write(hash(str(poses[i])), sr_results[i])
                        sr_output_idx += 1
                return np.stack(sr_results, axis=0)     
