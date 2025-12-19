from typing import List, Optional
import numpy as np
import os
from ezvtb_rt.tha3_ort import THA3ORTSessions, THA3ORTNonDefaultSessions
from ezvtb_rt.cache import Cacher
from ezvtb_rt.tha4_ort import THA4ORTSessions, THA4ORTNonDefaultSessions
from ezvtb_rt.tha4_student_ort import THA4StudentORTSessions
from ezvtb_rt.common import Core
import ezvtb_rt
from ezvtb_rt.ort_utils import createORTSession

class CoreORT(Core):
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
        elif tha_model_version == 'v4':
            tha_path = os.path.join(ezvtb_rt.EZVTB_DATA, 'tha4', 
                                    'fp16' if tha_model_fp16 else 'fp32')
        elif tha_model_version == 'v4_student':
            # Support custom student models in data/models/custom_tha4_models
            if tha_model_name:
                tha_path = os.path.normpath(os.path.join(
                    ezvtb_rt.EZVTB_DATA,
                    'custom_tha4_models', tha_model_name
                ))
            else:
                tha_path = os.path.join(
                    ezvtb_rt.EZVTB_DATA, 'tha4_student'
                )
        else:
            raise ValueError('Unsupported THA model version')
        rife_path = None
        if rife_model_enable:
            rife_path = os.path.join(ezvtb_rt.EZVTB_DATA, 'rife', 
                                     f'rife_x{rife_model_scale}_{"fp16" if rife_model_fp16 else "fp32"}.onnx')
            
        sr_path = None
        if sr_model_enable:
            if sr_model_scale == 4:
                sr_path = os.path.join(ezvtb_rt.EZVTB_DATA, 'Real-ESRGAN', f'exported_256_{ "fp16" if sr_model_fp16 else "fp32"}.onnx')
            else: #x2
                sr_path = os.path.join(ezvtb_rt.EZVTB_DATA, 'waifu2x', f"noise0_scale2x_{ 'fp16' if sr_model_fp16 else 'fp32'}.onnx")
        
        device_id = int(os.environ.get('EZVTB_DEVICE_ID', '0'))
        if tha_model_version == 'v3':
            if device_id == 0:
                self.tha = THA3ORTSessions(tha_path, use_eyebrow)
            else:
                self.tha = THA3ORTNonDefaultSessions(tha_path, device_id, use_eyebrow)
        elif tha_model_version == 'v4_student':
            self.tha = THA4StudentORTSessions(tha_path, device_id)
        elif tha_model_version == 'v4':
            if device_id == 0:
                self.tha = THA4ORTSessions(tha_path, use_eyebrow)
            else:
                self.tha = THA4ORTNonDefaultSessions(tha_path, device_id, use_eyebrow)
        else:
            raise ValueError('Unsupported THA model version')
        self.tha_model_fp16: bool = tha_model_fp16
        self.v3: bool = (tha_model_version == 'v3')
        self.rife: Optional[ort.InferenceSession] = None
        self.rifes: List[ort.InferenceSession] = []
        self.sr: Optional[ort.InferenceSession] = None
        self.sr_cacher: Optional[Cacher] = None
        self.cacher: Optional[Cacher] = None
        self.last_tha_output: Optional[np.ndarray] = None

        if rife_path is not None:
            self.rife = createORTSession(rife_path, device_id)
        if sr_path is not None:
            self.sr = createORTSession(sr_path, device_id)
            if cache_max_giga > 0.0:
                self.sr_cacher = Cacher(cache_max_giga, width=1024, height=1024)
        if cache_max_giga > 0.0:
            self.cacher = Cacher(cache_max_giga)
    def setImage(self, img:np.ndarray):
        self.tha.update_image(img)
        self.last_tha_output = img
    def inference(self, poses:List[np.ndarray]) -> np.ndarray:
        for i in range(len(poses)):
            poses[i] = poses[i].astype(np.float32)
            if self.tha_model_fp16 and not self.v3: #For THA4 with FP16 model poses are fp16 inputs
                poses[i] = poses[i].astype(np.float16)

        tha_result: np.ndarray = self.cacher.read(hash(str(poses[-1]))) if self.cacher is not None else None
        if tha_result is None:# Do not use cacher or cache missed
            tha_result = self.tha.inference(poses[-1])
            if self.cacher is not None:
                self.cacher.write(hash(str(poses[-1])), tha_result)

        if not self.rife and not self.sr: # Only THA
            return np.expand_dims(tha_result, axis=0)

        rife_result: np.ndarray = None
        if self.rife is not None: # RIFE
            self.rife = self.rifes[len(poses)-2] if len(poses) > 1 and len(poses)-2 < len(self.rifes) else self.rife
            all_cached: bool = len(poses) > 1 and self.cacher is not None and all(self.cacher.query(hash(str(pose))) for pose in poses[:-1])
            if all_cached:
                all_cached_images = [self.cacher.get(hash(str(pose))) for pose in poses[:-1]] + [tha_result]
                rife_result = np.stack(all_cached_images, axis=0)
            else:
                rife_result = self.rife.run(None, {'tha_img_0': self.last_tha_output, 'tha_img_1': tha_result})[0]
                if len(poses) > 1 and self.cacher is not None and len(poses) == len(rife_result):
                    for i in range(len(poses)-1):
                        self.cacher.write(hash(str(poses[i])), rife_result[i])
            self.last_tha_output = tha_result
        else:
            rife_result = np.expand_dims(tha_result, axis=0)

        if not self.sr:  # Only RIFE
            return rife_result
        
        # SR
        if len(poses) == 1: # Only one pose provided, 
            print("Single pose SR processing")
            hs = hash(str(poses[-1]))
            cached_sr = self.sr_cacher.read(hs) if self.sr_cacher is not None else None
            sr_batch = rife_result.shape[0]  if cached_sr is None else rife_result.shape[0] - 1
            if sr_batch > 0: #Need to run SR on some frames
                sr_result = self.sr.run(None, {self.sr.get_inputs()[0].name: rife_result[:sr_batch]})[0]
                if self.sr_cacher is not None and cached_sr is None:
                    self.sr_cacher.write(hs, sr_result[-1])
                elif cached_sr is not None:
                    sr_result = np.concatenate([sr_result, np.expand_dims(cached_sr, axis=0)], axis=0)
            else: # The only frame is cached
                sr_result =  np.expand_dims(cached_sr, axis=0)
        else:
            assert len(poses) == rife_result.shape[0]
            all_cached: bool = self.sr_cacher is not None and all(self.sr_cacher.query(hash(str(pose))) for pose in poses)
            if all_cached:
                sr_result = np.stack([self.sr_cacher.get(hash(str(pose))) for pose in poses], axis=0)
            else:
                if self.sr_cacher is None:
                    sr_result = self.sr.run(None, {self.sr.get_inputs()[0].name: rife_result})[0]
                else:
                    sr_result = []
                    to_sr_images = []
                    for i in range(len(poses)):
                        hs = hash(str(poses[i]))
                        cached_sr = self.sr_cacher.read(hs)
                        if cached_sr is None:
                            to_sr_images.append(rife_result[i])
                            sr_result.append(None)  # Placeholder
                        else:
                            sr_result.append(cached_sr)
                    assert len(to_sr_images) > 0
                    to_sr_images_np = np.stack(to_sr_images, axis=0)
                    sr_outputs = self.sr.run(None, {self.sr.get_inputs()[0].name: to_sr_images_np})[0]
                    sr_idx = 0
                    for i in range(len(poses)):
                        if sr_result[i] is None:
                            sr_result[i] = sr_outputs[sr_idx]
                            hs = hash(str(poses[i]))
                            self.sr_cacher.write(hs, sr_outputs[sr_idx])
                            sr_idx += 1
                    sr_result = np.stack(sr_result, axis=0)

        return sr_result