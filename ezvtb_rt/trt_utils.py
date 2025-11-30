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


def build_engine(onnx_file_path:str, precision:str = 'fp16', dynamic_batch:bool = False) -> bytes:
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
    
    if dynamic_batch:
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

#memory management
class HostDeviceMem(object):
    def __init__(self, host_mem:numpy.ndarray, device_mem: cuda.DeviceAllocation):
        self.host: numpy.ndarray = host_mem
        self.device: cuda.DeviceAllocation = device_mem

    @classmethod
    def create(cls, shape, dtype):
        host_mem = cuda.pagelocked_empty(shape, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        return cls(host_mem, device_mem)

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()
    def __del__(self):
        self.device.free()

    def dtoh(self, stream:cuda.Stream):
        cuda.memcpy_dtoh_async(self.host, self.device, stream) 
    def htod(self, stream:cuda.Stream):
        cuda.memcpy_htod_async(self.device, self.host, stream)

    def bridgeFrom(self, other: 'HostDeviceMem', stream:cuda.Stream):
        assert self.host.nbytes == other.host.nbytes, "Memory sizes must match for bridging"
        cuda.memcpy_dtod_async(self.device, other.device, self.host.nbytes, stream)



class TRTEngine:
    def __init__(self, engine: trt.ICudaEngine | str, n_input:int):
        if isinstance(engine, str):
            engine = load_engine(engine)
            assert engine is not None, f'Failed to load engine from path {engine}'
        self.engine: trt.ICudaEngine = engine
        TRT_LOGGER.log(TRT_LOGGER.INFO, 'Creating inference context')
        # create execution context
        self.context: trt.IExecutionContext = engine.create_execution_context()
        self.n_batch: int = -1
        self.in_out_tensors: dict = {}
        self.inputs: List[HostDeviceMem] = []
        self.outputs: List[HostDeviceMem] = []
        
        # get input and output tensor names
        self.input_tensor_names: List[str] = [engine.get_tensor_name(i) for i in range(n_input)]
        self.output_tensor_names: List[str] = [engine.get_tensor_name(i) for i in range(n_input, self.engine.num_io_tensors)]
        TRT_LOGGER.log(TRT_LOGGER.INFO, 'Input nodes: '+ str(self.input_tensor_names))
        TRT_LOGGER.log(TRT_LOGGER.INFO, 'Output nodes: '+ str(self.output_tensor_names))

        self.input_tensors_original_shapes: Dict[str, List[int]] = {}
        self.output_tensors_original_shapes: Dict[str, List[int]] = {}
        for input_tensor_name in self.input_tensor_names:
            shape = [dim for dim in self.context.get_tensor_shape(input_tensor_name)]
            self.input_tensors_original_shapes[input_tensor_name] = shape
        for output_tensor_name in self.output_tensor_names:
            shape = [dim for dim in self.context.get_tensor_shape(output_tensor_name)]
            self.output_tensors_original_shapes[output_tensor_name] = shape

        # create stream
        self.stream: cuda.Stream = cuda.Stream()
        # For measuring inference time
        self.start_event: cuda.Event = cuda.Event()
        self.end_event: cuda.Event = cuda.Event()
            
    def get_last_inference_time(self):
        return self.start_event.time_till(self.end_event)

    def syncPutInputs(self, np_inputs: List[np.ndarray | HostDeviceMem], n_batch:int = 1):
        # set input shapes, the output shapes are inferred automatically
        input_tensors, output_tensors = self.configure_in_out_tensors(n_batch)
        for inp, inp_mem, inp_name in zip(np_inputs, input_tensors, self.input_tensor_names):
            if inp.dtype != inp_mem.host.dtype or inp.shape != inp_mem.host.shape:
                print('Given:', inp.dtype, inp.shape)
                print('Expected:',inp_mem.host.dtype, inp_mem.host.shape)
                raise ValueError(f'Input shape or type does not match for input tensor {inp_name}')
            if isinstance(inp, HostDeviceMem):
                cuda.memcpy_dtod_async(inp_mem.device, inp.device, 
                                   inp_mem.host.nbytes, self.stream)
            else:
                np.copyto(inp_mem.host, inp)
                inp_mem.htod(self.stream)
        self.stream.synchronize()

    def kickoff(self, stream: cuda.Stream = None, sync:bool = False):
        if stream is None:
            stream = self.stream
        # Record the start event
        self.start_event.record(stream)
        # Run inference.
        self.context.execute_async_v3(stream.handle)
        # Record the end event
        self.end_event.record(stream)
        # Synchronize the stream if requested
        if sync:
            stream.synchronize()

    def asyncKickoff(self, stream: cuda.Stream = None):
        self.kickoff(stream, sync=False)

    def syncKickoff(self, stream: cuda.Stream = None):
        self.kickoff(stream, sync=True)

    def syncGetOutputs(self, copy:bool = True) -> List[np.ndarray]:
        for out_mem in self.outputs: out_mem.dtoh(self.stream)
        # Synchronize the stream
        self.stream.synchronize()
        if copy:
            return [np.copy(outp.host) for outp in self.outputs]
        else:
            return [outp.host for outp in self.outputs]
        
    def syncInfer(self, np_inputs: List[np.ndarray], n_batch:int = 1) -> List[np.ndarray]:
        self.syncPutInputs(np_inputs, n_batch)
        self.asyncKickoff()
        return self.syncGetOutputs(True)
    
    def configure_in_out_tensors(self, n_batch:int = 1) -> Tuple[List[HostDeviceMem], List[HostDeviceMem]]:
        inputs: List[HostDeviceMem] = []
        outputs: List[HostDeviceMem] = []
        if  n_batch in self.in_out_tensors:
            inputs, outputs = self.in_out_tensors[n_batch]
            if n_batch != self.n_batch:
                TRT_LOGGER.log(TRT_LOGGER.INFO, f'Reconfiguring tensor addresses for batch size {n_batch}')
                for input_tensor_name, input_mem in zip(self.input_tensor_names, inputs):
                    self.context.set_tensor_address(input_tensor_name, int(input_mem.device))
                    self.context.set_input_shape(input_tensor_name, trt.Dims(input_mem.host.shape))
                for output_tensor_name, output_mem in zip(self.output_tensor_names, outputs):
                    self.context.set_tensor_address(output_tensor_name, int(output_mem.device))
        else:
            for input_tensor_name in self.input_tensor_names:
                shape = self.input_tensors_original_shapes[input_tensor_name].copy()
                for i in range(len(shape)):
                    if shape[i] == -1:
                        shape[i] = n_batch
                dtype = trt.nptype(self.engine.get_tensor_dtype(input_tensor_name))
                TRT_LOGGER.log(TRT_LOGGER.INFO, f'Allocating input tensor: {input_tensor_name} with shape {shape} and dtype {dtype}')
                mem = HostDeviceMem.create(shape, dtype)
                self.context.set_tensor_address(input_tensor_name, int(mem.device)) # Use this setup without binding for v3
                self.context.set_input_shape(input_tensor_name, trt.Dims(shape))
                inputs.append(mem)
            for output_tensor_name in self.output_tensor_names:
                shape = self.output_tensors_original_shapes[output_tensor_name].copy()
                for i in range(len(shape)):
                    if shape[i] == -1:
                        shape[i] = n_batch
                dtype = trt.nptype(self.engine.get_tensor_dtype(output_tensor_name))
                mem = HostDeviceMem.create(shape, dtype)
                self.context.set_tensor_address(output_tensor_name, int(mem.device)) # Use this setup without binding for v3
                outputs.append(mem)
            self.in_out_tensors[n_batch] = (inputs, outputs)
        self.n_batch = n_batch
        self.inputs = inputs
        self.outputs = outputs
        return inputs, outputs