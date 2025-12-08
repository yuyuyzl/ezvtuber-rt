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
        self.rifes: List[ort.InferenceSession] = []
        self.sr: Optional[ort.InferenceSession] = None
        self.sr_cacher: Optional[Cacher] = None
        self.cacher: Optional[Cacher] = None
        self.last_tha_output: Optional[np.ndarray] = None

        if rife_path is not None:
            self.rife = createORTSession(rife_path, device_id)
            for i in [2, 3, 4]:
                path = os.path.join(ezvtb_rt.EZVTB_DATA, 'rife', f'rife_x{i}_{"fp16" if rife_model_fp16 else "fp32"}.onnx')
                self.rifes.append(createORTSession(path, device_id))
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

        if self.cacher is None:# Do not use cacher
            tha_result = self.tha.inference(poses[-1])
        else:
            #use cacher 
            hs = hash(str(poses[-1]))
            cached = self.cacher.read(hs)

            if cached is not None:# Cache hits
                tha_result = cached
            else: #cache missed
                tha_result = self.tha.inference(poses[-1])
                self.cacher.write(hs, tha_result)

        if not self.rife and not self.sr: # Only THA
            return np.expand_dims(tha_result, axis=0)

        if self.rife is not None: # RIFE + (SR)
            self.rife = self.rifes[len(poses)-2] if len(poses) > 1 and len(poses)-2 < len(self.rifes) else self.rife
            all_cached: bool = len(poses) > 1 and self.cacher is not None and all(self.cacher.query(hash(str(pose))) for pose in poses[:-1])
            if all_cached:
                results = [self.cacher.get(hash(str(pose))) for pose in poses[:-1]] + [tha_result]
                res = np.stack(results, axis=0)
            else:
                res = self.rife.run(None, {'tha_img_0': self.last_tha_output, 'tha_img_1': tha_result})[0]
                if len(poses) > 1 and self.cacher is not None and len(poses) == len(res):
                    for i in range(len(poses)-1):
                        self.cacher.write(hash(str(poses[i])), res[i])
            self.last_tha_output = tha_result
            if self.sr is not None: # SR after RIFE
                if self.sr_cacher is not None: # Use SR cacher
                    if len(poses) == 1: #single pose input
                        hs = hash(str(poses[0]))
                        cached_sr = self.sr_cacher.get(hs)
                        if cached_sr is not None: #SR cache hit
                            if res.shape[0] > 1: #Multiple frames rife output
                                additional_sr = self.sr.run(None, {self.sr.get_inputs()[0].name: res[:-1]})[0]
                                res = np.concatenate([np.expand_dims(cached_sr, axis=0), additional_sr], axis=0)
                            else: #Single frame rife output
                                res = np.expand_dims(cached_sr, axis=0)
                        else: #SR cache miss
                            res = self.sr.run(None, {self.sr.get_inputs()[0].name: res})[0]
                            self.sr_cacher.write(hs, res[0])
                    elif len(poses) == res.shape[0]: #interpolated poses are provided
                        sr_results = []
                        to_run_list = []
                        for i in range(len(poses)): #check cache for each rife frame
                            hs = hash(str(poses[i]))
                            sr_results.append(self.sr_cacher.get(hs))
                            if sr_results[-1] is None:
                                to_run_list.append(res[i])
                        # Run SR on frames that missed the cache
                        sr_inputs = np.stack(to_run_list, axis=0) if len(to_run_list) > 1 else np.expand_dims(to_run_list[0], axis=0)
                        sr_outputs = self.sr.run(None, {self.sr.get_inputs()[0].name: sr_inputs})[0]
                        idx = 0
                        for i in range(len(poses)):
                            if sr_results[i] is None:
                                sr_results[i] = sr_outputs[idx]
                                self.sr_cacher.write(hash(str(poses[i])), sr_outputs[idx])
                                idx += 1
                        res = np.stack(sr_results, axis=0)
                    else:
                        raise ValueError('RIFE output length does not match input poses length')
                else:
                    res = self.sr.run(None, {self.sr.get_inputs()[0].name: res})[0]
            return res
        
        # Only SR
        hs = hash(str(poses[-1]))
        cached_sr = self.sr_cacher.read(hs) if self.sr_cacher is not None else None
        if cached_sr is not None:
            return np.expand_dims(cached_sr, axis=0)
        else:
            sr_result = self.sr.run(None, {self.sr.get_inputs()[0].name: np.expand_dims(tha_result, axis=0)})[0]
            if self.sr_cacher is not None:
                self.sr_cacher.write(hs, sr_result[0])
            return sr_result