from pathlib import Path
import os
import numpy as np
import tensorrt as trt
from typing import List, Dict, Tuple
import pycuda.driver as cuda
from os.path import join
import numpy
from tqdm import tqdm
from ezvtb_rt.init_utils import check_exist_all_models

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

# Solution from https://github.com/NVIDIA/TensorRT/issues/1050#issuecomment-775019583
def cudaSetDevice(device_idx):
    from ctypes import cdll, c_char_p
    libcudart = cdll.LoadLibrary('cudart64_12.dll')
    libcudart.cudaGetErrorString.restype = c_char_p
    ret = libcudart.cudaSetDevice(device_idx)
    if ret != 0:
        error_string = libcudart.cudaGetErrorString(ret)
        raise RuntimeError("cudaSetDevice: " + str(error_string))

def check_build_all_models():
    all_models_list = check_exist_all_models()
    #Check TRT support
    print('Start testing if TensorRT works on this machine')
    for fullpath in tqdm(all_models_list):
        dir, filename = os.path.split(fullpath)
        trt_filename = filename.split('.')[0] + '.trt'
        trt_fullpath = os.path.join(dir, trt_filename)
        if os.path.isfile(trt_fullpath):
            try:
                if load_engine(trt_fullpath) is not None:
                    continue
                else:
                    print(f'Can not successfully load {trt_fullpath}, build again')
            except:
                print(f'Can not successfully load {trt_fullpath}, build again')
        dtype = 'fp16' if 'fp16' in fullpath else 'fp32'
        engine_seri = build_engine(fullpath, dtype)
        if engine_seri is None:
            print('Error while building engine')
            return False
        save_engine(engine_seri, trt_fullpath)


def build_engine(onnx_file_path:str, precision:str = 'fp16') -> bytes:
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network()
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)
    # Parse model file
    TRT_LOGGER.log(TRT_LOGGER.INFO, f'Loading ONNX file from path {onnx_file_path}...')
    with open(onnx_file_path, 'rb') as model:
        TRT_LOGGER.log(TRT_LOGGER.INFO, 'Beginning ONNX file parsing')
        parse_res = parser.parse(model.read())
        if not parse_res:
            for error in range(parser.num_errors):
                TRT_LOGGER.log(TRT_LOGGER.ERROR, parser.get_error(error))
            raise ValueError('Failed to parse the ONNX file.')
    TRT_LOGGER.log(TRT_LOGGER.INFO, 'Completed parsing of ONNX file')
    TRT_LOGGER.log(TRT_LOGGER.INFO, f'Input number: {network.num_inputs}')
    TRT_LOGGER.log(TRT_LOGGER.INFO, f'Output number: {network.num_outputs}')
    def GiB(val):
        return val * 1 << 30
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, GiB(4)) # 4G
    config.tiling_optimization_level = trt.TilingOptimizationLevel.FULL
    
    if precision == 'fp32':
        config.set_flag(trt.BuilderFlag.TF32)
    elif precision == 'fp16':
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == 'int8':
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.INT8)
    else:
        raise ValueError('precision must be one of fp32 or fp16')
    
    def is_dynamic_shape()->bool:
        for i in range(network.num_inputs):
            input_name = network.get_input(i).name
            dims = network.get_input(i).shape
            if dims[0] == -1:
                return True
        return False

    if is_dynamic_shape():
        profile = builder.create_optimization_profile()
        for i in range(network.num_inputs):
            input_name = network.get_input(i).name
            print('Setting dynamic shape for input:', input_name)
            dims = network.get_input(i).shape
            min_shape = trt.Dims(dims)
            opt_shape = trt.Dims(dims)
            max_shape = trt.Dims(dims)
            min_shape[0] = 1
            opt_shape[0] = 4
            max_shape[0] = 4
            profile.set_shape(input_name, min_shape, opt_shape, max_shape)
            TRT_LOGGER.log(TRT_LOGGER.INFO, f'Setting dynamic shape for input {input_name}: min={min_shape}, opt={opt_shape}, max={max_shape}')
        config.add_optimization_profile(profile)
        config.builder_optimization_level = 5
    # Build engine.
    TRT_LOGGER.log(TRT_LOGGER.INFO, f'Building an engine from file {onnx_file_path}; this may take a while...')
    serialized_engine = builder.build_serialized_network(network, config)
    TRT_LOGGER.log(TRT_LOGGER.INFO, 'Completed creating Engine')
    return serialized_engine

def save_engine(engine, path):
    TRT_LOGGER.log(TRT_LOGGER.INFO, f'Saving engine to file {path}')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        f.write(engine)
    TRT_LOGGER.log(TRT_LOGGER.INFO, 'Completed saving engine')

def load_engine(path):
    TRT_LOGGER.log(TRT_LOGGER.WARNING, f'Loading engine from file {path}')
    runtime = trt.Runtime(TRT_LOGGER)
    with open(path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    TRT_LOGGER.log(TRT_LOGGER.INFO, 'Completed loading engine')
    return engine
