"""THA4 TensorRT Core wrapper similar to CoreTRT"""
from ezvtb_rt.trt_utils import *
from ezvtb_rt.rife import RIFE
from ezvtb_rt.tha4 import THA4
from ezvtb_rt.cache import Cacher
from ezvtb_rt.sr import SR
from ezvtb_rt.common import Core


class CoreTHA4TRT(Core):
    """Main inference pipeline for THA4 with TensorRT acceleration
    
    Combines THA4 face model with optional components:
    - RIFE for frame interpolation
    - SR for super resolution  
    - Cacher for output caching
    
    Args:
        tha4_dir: Path to THA4 model directory (onnx_model_tha4/fp16 or fp32)
        vram_cache_size: VRAM allocated for model caching (GB)
        use_eyebrow: Enable eyebrow motion processing
        rife_dir: Path to RIFE model directory (None to disable)
        sr_dir: Path to SR model directory (None to disable)
        cache_max_volume: Max disk cache size (GB)
        cache_quality: Cache compression quality (1-3)
    """
    def __init__(self, tha4_dir: str, vram_cache_size: float,
                 use_eyebrow: bool, rife_dir: str, sr_dir: str,
                 cache_max_volume: float, cache_quality: int = 2):
        # Initialize core THA4 face model
        self.tha = THA4(tha4_dir, vram_cache_size, use_eyebrow)

        # Initialize optional components
        self.rife = None
        self.sr = None
        self.cacher = None
        self.scale = 1

        # Initialize RIFE if model path provided
        if rife_dir is not None:
            self.rife = RIFE(rife_dir, self.tha.instream,
                           self.tha.memories['cv_result'])
            self.scale = self.rife.scale
        
        # Initialize SR if model path provided
        if sr_dir is not None:
            instream = None
            mems = []
            if self.rife is not None:
                instream = self.rife.instream
                for i in range(self.rife.scale):
                    mems.append(self.rife.memories['framegen_'+str(i)])
            else:
                instream = self.tha.instream
                mems.append(self.tha.memories['cv_result'])
            self.sr = SR(sr_dir, instream, mems)

        # Initialize cache if enabled
        if cache_max_volume > 0.0:
            self.cacher = Cacher(cache_max_volume, cache_quality)

    def setImage(self, img: np.ndarray):
        """Set input image for processing pipeline
        
        Args:
            img: Input image in BGRA format (HWC, uint8)
        """
        self.tha.setImage(img)

    def inference(self, pose: np.ndarray) -> List[np.ndarray]:
        """Run full inference pipeline
        
        Args:
            pose: Facial pose parameters (45 floats)
            
        Returns:
            List of output images from final stage in pipeline
        """
        pose = pose.astype(np.float32)

        need_cache_write = 0
        res_carrier = None

        if self.cacher is None:
            self.tha.inference(pose)
            res_carrier = self.tha
        else:
            hs = hash(str(pose))
            cached = self.cacher.read(hs)

            if cached is not None:
                np.copyto(self.tha.memories['cv_result'].host, cached)
                self.tha.memories['cv_result'].htod(self.tha.instream)
                res_carrier = [cached]
            else:
                self.tha.inference(pose)
                need_cache_write = hs
                res_carrier = self.tha

        if self.rife is not None:
            self.rife.inference()
            res_carrier = self.rife
        
        if self.sr is not None:
            self.sr.inference()
            res_carrier = self.sr

        if need_cache_write != 0:
            self.cacher.write(need_cache_write, self.tha.fetchRes()[0])

        if type(res_carrier) is not list:
            res_carrier = res_carrier.fetchRes()
        return res_carrier
