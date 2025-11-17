from ezvtb_rt.trt_utils import *
from ezvtb_rt.rife import RIFE
from ezvtb_rt.tha import THA
from ezvtb_rt.tha4 import THA4
from ezvtb_rt.cache import Cacher
from ezvtb_rt.sr import SR
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
                 sr_model_noise:int = 1,
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
            tha_path = os.path.join(ezvtb_rt.EZVTB_DATA, 'tha4')
            self.v3 = False
        else:
            raise ValueError('Unsupported THA model version')
        rife_path = None
        if rife_model_enable:
            rife_path = os.path.join(ezvtb_rt.EZVTB_DATA, 'rife_512', 
                                     f'x{rife_model_scale}', 
                                     'fp16' if rife_model_fp16 else 'fp32')
            
        sr_path = None
        if sr_model_enable:
            if sr_model_scale == 4:
                if sr_model_fp16:
                    sr_path = os.path.join(ezvtb_rt.EZVTB_DATA, 'Real-ESRGAN', 'exported_256_fp16')
                else:
                    sr_path = os.path.join(ezvtb_rt.EZVTB_DATA, 'Real-ESRGAN', 'exported_256')
            else: #x2
                if sr_model_fp16:
                    sr_path = os.path.join(ezvtb_rt.EZVTB_DATA, 'waifu2x_upconv', 'fp16', 'upconv_7', 'art', f'noise{sr_model_noise}_scale2x')
                else:
                    sr_path = os.path.join(ezvtb_rt.EZVTB_DATA, 'waifu2x_upconv', 'fp32', 'upconv_7', 'art', f'noise{sr_model_noise}_scale2x')

        # Initialize core THA face model
        if self.v3:
            self.tha = THA(tha_path, vram_cache_size, use_eyebrow)
        else:
            self.tha = THA4(tha_path, vram_cache_size, use_eyebrow)

        # Initialize optional components
        self.rife = None  # Frame interpolation module
        self.sr = None    # Super resolution module
        self.cacher = None# Output caching system
        self.scale = 1    # Output scaling factor

        # Initialize RIFE if model path provided
        if rife_path is not None:
            self.rife = RIFE(rife_path, self.tha.instream, 
                             self.tha.memories['output_cv_img' if self.v3 else 'cv_result'])
            self.scale = self.rife.scale
        # Initialize SR if model path provided
        if sr_path is not None:
            instream = None  # Will be set based on RIFE/THA
            mems = []        # Memory buffers from previous stage
            if self.rife is not None:
                instream = self.rife.instream
                for i in range(self.rife.scale):
                    mems.append(self.rife.memories['framegen_'+str(i)])
            else:
                instream = self.tha.instream
                mems.append(self.tha.memories['output_cv_img' if self.v3 else 'cv_result'])
            self.sr = SR(sr_path, instream, mems)

        # Initialize cache if enabled
        if cache_max_giga > 0.0:
            self.cacher = Cacher(cache_max_giga)

    def setImage(self, img:np.ndarray):
        """Set input image for processing pipeline
        Args:
            img: Input image in BGR format (HWC, uint8)
        """
        self.tha.setImage(img)

    def inference(self, pose:np.ndarray) -> List[np.ndarray]:
        """Run full inference pipeline
        Args:
            pose: Facial pose parameters (45 floats)
            
        Returns:
            List of output images from final stage in pipeline.
            Note: Numpy arrays should be copied/used before next inference
        """
        # Convert pose to required precision
        pose = pose.astype(np.float32)

        # Cache management variables
        need_cache_write = 0  # Hash value if cache needs updating
        res_carrier = None    # Current result container

        # Cache bypass path
        if self.cacher is None:
            # Directly run THA inference
            self.tha.inference(pose)
            res_carrier = self.tha
        else:  # Cache enabled path
            hs = hash(str(pose))  # Create pose hash key
            cached = self.cacher.read(hs)

            if cached is not None:  # Cache hit
                # Copy cached data to GPU memory
                np.copyto(self.tha.memories['output_cv_img' if self.v3 else 'cv_result'].host, cached)
                self.tha.memories['output_cv_img' if self.v3 else 'cv_result'].htod(self.tha.instream)
                res_carrier = [cached]
            else:  # Cache miss
                # Run THA inference and flag for cache storage
                self.tha.inference(pose)
                need_cache_write = hs
                res_carrier = self.tha

        # Run frame interpolation if enabled
        if self.rife is not None:
            self.rife.inference()
            res_carrier = self.rife
        # Run super resolution if enabled
        if self.sr is not None:
            self.sr.inference()
            res_carrier = self.sr

        # Update cache if we had a miss
        if need_cache_write != 0:
            self.cacher.write(need_cache_write, self.tha.fetchRes()[0])

        if type(res_carrier) is not list:
            res_carrier = res_carrier.fetchRes()
        return res_carrier
