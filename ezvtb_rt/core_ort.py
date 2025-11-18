from typing import List, Optional
import numpy as np
import os
from ezvtb_rt.rife_ort import RIFEORT
from ezvtb_rt.tha_ort import THAORT, THAORTNonDefault
from ezvtb_rt.cache import Cacher
from ezvtb_rt.sr_ort import SRORT
from ezvtb_rt.tha4_ort import THA4ORT, THA4ORTNonDefault
from ezvtb_rt.common import Core
import ezvtb_rt

class CoreORT(Core):
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
                 vram_cache_size:float = 1.0,  #For compatibility, not used
                 cache_max_giga:float = 2.0, 
                 use_eyebrow:bool = False):
        if tha_model_version == 'v3':
            tha_path = os.path.join(ezvtb_rt.EZVTB_DATA, 'tha3',
                                    'seperable' if tha_model_seperable else 'standard', 
                                    'fp16' if tha_model_fp16 else 'fp32')
            
        elif tha_model_version == 'v4':
            tha_path = os.path.join(ezvtb_rt.EZVTB_DATA, 'tha4', 
                                    'fp16' if tha_model_fp16 else 'fp32')
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
        
        device_id = int(os.environ.get('EZVTB_DEVICE_ID', '0'))
        if tha_model_version == 'v3':
            if device_id == 0:
                self.tha = THAORT(tha_path, use_eyebrow)
            else:
                self.tha = THAORTNonDefault(tha_path, device_id, use_eyebrow)
        else:
            if device_id == 0:
                self.tha = THA4ORT(tha_path, use_eyebrow)
            else:
                self.tha = THA4ORTNonDefault(tha_path, device_id, use_eyebrow)
        self.rife = None
        self.sr = None
        self.cacher = None

        if rife_path is not None:
            self.rife = RIFEORT(rife_path, device_id)
        if sr_path is not None:
            self.sr = SRORT(sr_path, device_id)
        if cache_max_giga > 0.0:
            self.cacher = Cacher(cache_max_giga)
    def setImage(self, img:np.ndarray):
        self.tha.update_image(img)
    def inference(self, pose:np.ndarray) -> List[np.ndarray]:
        pose = pose.astype(np.float32)

        if self.cacher is None:# Do not use cacher
            res = self.tha.inference(pose)
        else:
            #use cacher 
            hs = hash(str(pose))
            cached = self.cacher.read(hs)

            if cached is not None:# Cache hits
                res = [cached]
            else: #cache missed
                res = self.tha.inference(pose)
                self.cacher.write(hs, res[0])

        if self.rife is not None:
            res = self.rife.inference(res)
        if self.sr is not None:
            res = self.sr.inference(res)
        return res

