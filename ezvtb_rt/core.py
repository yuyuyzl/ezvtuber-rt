from ezvtb_rt.trt_utils import *
from ezvtb_rt.trt_engine import TRTEngine, HostDeviceMem
from ezvtb_rt.tha3 import THA3Engines
# from ezvtb_rt.tha4 import THA4
from ezvtb_rt.cache import Cacher
from ezvtb_rt.common import Core
import ezvtb_rt

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
        else:
            # self.tha = THA4(tha_path, vram_cache_size, use_eyebrow)
            pass

        self.tha_model_fp16: bool = tha_model_fp16

        # Initialize optional components
        self.rife: TRTEngine = None  # Frame interpolation module
        self.sr: TRTEngine = None    # Super resolution module
        self.cacher_512: Cacher = None# Output caching system

        # Initialize RIFE if model path provided
        if rife_path is not None:
            self.rife = TRTEngine(rife_path, 2)
            self.rife.configure_in_out_tensors()
        # Initialize SR if model path provided
        if sr_path is not None:
            self.sr = TRTEngine(sr_path, 2)
            self.sr.configure_in_out_tensors(rife_model_scale if rife_model_enable else 1)

        # Initialize cache if enabled
        if cache_max_giga > 0.0:
            self.cacher_512 = Cacher(cache_max_giga)

        self.main_stream: cuda.Stream = cuda.Stream()
        self.tha_finished_event: cuda.Event = cuda.Event()
        self.cache_stream: cuda.Stream = cuda.Stream()

    def setImage(self, img:np.ndarray):
        """Set input image for processing pipeline
        Args:
            img: Input image in BGR format (HWC, uint8)
        """
        self.tha.syncSetImage(img)

    def inference(self, pose:np.ndarray) -> np.ndarray:
        """Run full inference pipeline
        Args:
            pose: Facial pose parameters (45 floats)
            
        Returns:
            List of output images from final stage in pipeline.
            Note: Numpy arrays should be copied/used before next inference
        """
        # Convert pose to required precision
        pose = pose.astype(np.float32)
        if self.tha_model_fp16 and not self.v3: #For THA4 with FP16 model poses are fp16 inputs
            pose = pose.astype(np.float16)

        tha_mem_res: HostDeviceMem = self.tha.getOutputMem()

        # Cache bypass path
        if self.cacher_512 is None:
            # Directly run THA inference
            self.tha.asyncInfer(pose, self.main_stream)
            if self.rife is not None:
                cuda.memcpy_dtod_async(self.rife.inputs[0].device, self.rife.inputs[1].device, self.rife.inputs[0].host.nbytes, self.main_stream)
                cuda.memcpy_dtod_async(self.rife.inputs[1].device, tha_mem_res.device, tha_mem_res.host.nbytes, self.main_stream)
                self.rife.asyncKickoff(self.main_stream)
                if self.sr is not None:
                    cuda.memcpy_dtod_async(self.sr.inputs[0].device, self.rife.outputs[0].device, self.sr.inputs[0].host.nbytes, self.main_stream)
                    self.sr.asyncKickoff(self.main_stream)
            elif self.sr is not None: # Directly SR after THA
                cuda.memcpy_dtod_async(self.sr.inputs[0].device, tha_mem_res.device, tha_mem_res.host.nbytes, self.main_stream)
                self.sr.asyncKickoff(self.main_stream)
        else: # With caching
            self.cache_stream.synchronize()
            hs = hash(str(pose))
            cached_output = self.cacher_512.read(hs)
            if cached_output is not None: # Cache hit
                if self.rife is not None:
                    np.copyto(self.rife.inputs[1].host, cached_output)
                    cuda.memcpy_dtod_async(self.rife.inputs[0].device, self.rife.inputs[1].device, self.rife.inputs[0].host.nbytes, self.main_stream)
                    self.rife.inputs[1].htod(self.main_stream)
                    self.rife.asyncKickoff(self.main_stream)
                    if self.sr is not None:
                        cuda.memcpy_dtod_async(self.sr.inputs[0].device, self.rife.outputs[0].device, self.sr.inputs[0].host.nbytes, self.main_stream)
                        self.sr.asyncKickoff(self.main_stream)
                elif self.sr is not None: # Directly SR after cache
                    np.copyto(self.sr.inputs[0].host, cached_output)
                    self.sr.inputs[0].htod(self.main_stream)
                    self.sr.asyncKickoff(self.main_stream)
            else: # Cache miss
                # Run THA inference
                self.tha.asyncInfer(pose, self.main_stream)
                self.tha_finished_event.record(self.main_stream)
                if self.rife is not None:
                    cuda.memcpy_dtod_async(self.rife.inputs[0].device, self.rife.inputs[1].device, self.rife.inputs[0].host.nbytes, self.main_stream)
                    cuda.memcpy_dtod_async(self.rife.inputs[1].device, tha_mem_res.device, tha_mem_res.host.nbytes, self.main_stream)
                    self.rife.asyncKickoff(self.main_stream)
                    if self.sr is not None:
                        cuda.memcpy_dtod_async(self.sr.inputs[0].device, self.rife.outputs[0].device, self.sr.inputs[0].host.nbytes, self.main_stream)
                        self.sr.asyncKickoff(self.main_stream)
                elif self.sr is not None: # Directly SR after THA
                    cuda.memcpy_dtod_async(self.sr.inputs[0].device, tha_mem_res.device, tha_mem_res.host.nbytes, self.main_stream)
                    self.sr.asyncKickoff(self.main_stream)
                # Write to cache
                self.cacher_512.write(hs, tha_mem_res.host)

        if self.sr is not None:
            self.sr.outputs[0].dtoh(self.main_stream)
        elif self.rife is not None:
            self.rife.outputs[0].dtoh(self.main_stream)
        self.main_stream.synchronize()

        if self.sr is not None:
            return self.sr.outputs[0].host
        elif self.rife is not None:
            return self.rife.outputs[0].host
        elif self.cacher_512 is not None and cached_output is not None:
            return np.expand_dims(cached_output, axis=0)
        else:
            return np.expand_dims(tha_mem_res.host, axis=0)
