from ezvtb_rt.trt_utils import *
from ezvtb_rt.engine import Engine, createMemory, HostDeviceMem
import numpy as np


class SREngine(TRTEngine):
    """Single-stream super-resolution processor using TensorRT engine."""
    
    def __init__(self, model_path:str):
        super().__init__(model_path, 1)

    def syncInfer(self, np_input:np.ndarray)->np.ndarray:
        return super().syncInfer([np_input], 1)[0]
    
class SR():
    """Multi-stream parallel super-resolution processor."""
    
    def __init__(self, model_dir:str, instream = None, in_mems:List[HostDeviceMem] = None):
        """Initialize parallel processing infrastructure.
        
        Args:
            model_dir: Path to TensorRT engine file (without .trt extension)
            instream: Optional shared CUDA stream for execution
            in_mems: Optional pre-allocated input memory buffers
        """
        self.instream = instream  # Shared CUDA stream
        self.scale = 1 if in_mems is None else len(in_mems)  # Number of parallel streams
        self.fetchstream = cuda.Stream()  # Dedicated stream for data transfers
        self.finishedExec = [cuda.Event() for _ in range(self.scale)]  # Sync events
        self.engines = []  # TensorRT engine instances
        self.memories = {}  # Memory buffers
        
        # Initialize each processing stream
        for i in range(self.scale):
            # Load TensorRT engine
            engine = Engine(model_dir + '.trt', 1)
            
            # Configure memory buffers (reuse existing or create new)
            self.memories[f'framegen_{i}'] = in_mems[i] if in_mems else createMemory(engine.inputs[0])
            self.memories[f'output_{i}'] = createMemory(engine.outputs[0])
            
            # Bind engine memory
            engine.setInputMems([self.memories[f'framegen_{i}']])
            engine.setOutputMems([self.memories[f'output_{i}']])
            
            self.engines.append(engine)

    def inference(self):
        """Execute parallel inference across all engines."""
        for i in range(len(self.engines)):
            # Execute engine and record completion event
            self.engines[i].exec(self.instream)
            self.finishedExec[i].record(self.instream)
        
    def fetchRes(self)->List[np.ndarray]:
        """Retrieve processed results from all streams.
        
        Returns:
            List of super-resolved output images
        """
        for i in range(len(self.engines)):
            # Wait for engine completion before transferring
            self.fetchstream.wait_for_event(self.finishedExec[i])
            # Copy output from device to host memory
            self.memories[f'output_{i}'].dtoh(self.fetchstream)
            
        # Ensure all transfers complete
        self.fetchstream.synchronize()
        
        return [self.memories[f'output_{i}'].host for i in range(self.scale)]
    
    def viewRes(self)->List[np.ndarray]:
        """Get current output buffers without synchronization.
        
        Note: May return stale data if inference not complete
        
        Returns:
            List of output buffers in current state
        """
        return [self.memories[f'output_{i}'].host for i in range(self.scale)]
