from typing import List, Optional
import numpy as np
import os
from ezvtb_rt.tha3_ort import THA3ORTSessions, THA3ORTNonDefaultSessions
from ezvtb_rt.cache import Cacher
from ezvtb_rt.tha4_ort import THA4ORTSessions, THA4ORTNonDefaultSessions
from ezvtb_rt.common import Core
import ezvtb_rt
import onnxruntime as ort

def createORTSession(model_path:str, device_id:int = 0):
    provider = 'DmlExecutionProvider'
    providers = [ provider]
    options = ort.SessionOptions()
    options.enable_mem_pattern = True
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.enable_cpu_mem_arena = True
    provider_options = [{'device_id':device_id, "execution_mode": "parallel", "arena_extend_strategy": "kSameAsRequested"}]
    session = ort.InferenceSession(model_path, sess_options=options, providers=providers, provider_options=provider_options)
    return session

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
        else:
            if device_id == 0:
                self.tha = THA4ORTSessions(tha_path, use_eyebrow)
            else:
                self.tha = THA4ORTNonDefaultSessions(tha_path, device_id, use_eyebrow)
        self.tha_model_fp16: bool = tha_model_fp16
        self.v3: bool = (tha_model_version == 'v3')
        self.rife: Optional[ort.InferenceSession] = None
        self.sr: Optional[ort.InferenceSession] = None
        self.cacher: Optional[Cacher] = None
        self.last_tha_output: Optional[np.ndarray] = None

        if rife_path is not None:
            self.rife = createORTSession(rife_path, device_id)
        if sr_path is not None:
            self.sr = createORTSession(sr_path, device_id)
        if cache_max_giga > 0.0:
            self.cacher = Cacher(cache_max_giga)
    def setImage(self, img:np.ndarray):
        self.tha.update_image(img)
        self.last_tha_output = img
    def inference(self, pose:np.ndarray) -> np.ndarray:
        pose = pose.astype(np.float32)
        if self.tha_model_fp16 and not self.v3: #For THA4 with FP16 model poses are fp16 inputs
            pose = pose.astype(np.float16)

        if self.cacher is None:# Do not use cacher
            tha_result = self.tha.inference(pose)
        else:
            #use cacher 
            hs = hash(str(pose))
            cached = self.cacher.read(hs)

            if cached is not None:# Cache hits
                tha_result = cached
            else: #cache missed
                tha_result = self.tha.inference(pose)
                self.cacher.write(hs, tha_result)

        if not self.rife and not self.sr: # Only THA
            return np.expand_dims(tha_result, axis=0)

        if self.rife is not None: # RIFE + (SR)
            res = self.rife.run(None, {'tha_img_0': self.last_tha_output, 'tha_img_1': tha_result})[0]
            self.last_tha_output = tha_result
            if self.sr is not None:
                res = self.sr.run(None, {self.sr.get_inputs()[0].name: res})[0]
            return res
        
        # Only SR
        return self.sr.run(None, {self.sr.get_inputs()[0].name: np.expand_dims(tha_result, axis=0)})[0]