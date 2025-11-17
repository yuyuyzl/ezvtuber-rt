"""THA4 ONNX Runtime Core wrapper similar to CoreORT"""
from typing import List, Optional
import numpy as np
from ezvtb_rt.rife_ort import RIFEORT
from ezvtb_rt.tha4_ort import THA4ORT, THA4ORTNonDefault
from ezvtb_rt.cache import Cacher
from ezvtb_rt.sr_ort import SRORT
from ezvtb_rt.common import Core


class CoreTHA4ORT(Core):
    """THA4 ONNX Runtime inference pipeline
    
    Args:
        tha4_path: Path to THA4 ONNX models directory
        rife_path: Path to RIFE ONNX models (None to disable)
        sr_path: Path to SR ONNX models (None to disable)
        device_id: GPU device ID
        cache_max_volume: Max cache volume (GB)
        cache_quality: Cache quality (0-100)
        use_eyebrow: Enable eyebrow processing
    """
    def __init__(self, tha4_path: Optional[str] = None,
                 rife_path: Optional[str] = None,
                 sr_path: Optional[str] = None,
                 device_id: int = 0,
                 cache_max_volume: float = 2.0,
                 cache_quality: int = 90,
                 use_eyebrow: bool = True):
        if device_id == 0:
            self.tha = THA4ORT(tha4_path, use_eyebrow)
        else:
            self.tha = THA4ORTNonDefault(tha4_path, device_id, use_eyebrow)

        self.rife = None
        self.sr = None
        self.cacher = None

        if rife_path is not None:
            self.rife = RIFEORT(rife_path, device_id)
        if sr_path is not None:
            self.sr = SRORT(sr_path, device_id)
        if cache_max_volume > 0.0:
            self.cacher = Cacher(cache_max_volume, width=512, height=512)

    def setImage(self, img: np.ndarray):
        """Set input character image
        
        Args:
            img: Input image in BGRA format (512x512x4)
        """
        self.tha.update_image(img)

    def inference(self, pose: np.ndarray) -> List[np.ndarray]:
        """Run inference
        
        Args:
            pose: Pose parameters (1, 45)
            
        Returns:
            List of output images
        """
        pose = pose.astype(np.float32)

        if self.cacher is None:
            res = self.tha.inference(pose)
        else:
            hs = hash(str(pose))
            cached = self.cacher.read(hs)

            if cached is not None:
                res = [cached]
            else:
                res = self.tha.inference(pose)
                self.cacher.write(hs, res[0])

        if self.rife is not None:
            res = self.rife.inference(res)
        if self.sr is not None:
            res = self.sr.inference(res)
        return res
