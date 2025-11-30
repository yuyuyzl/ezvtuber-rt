from ezvtb_rt.trt_utils import *
from nvidia import nvcomp
import pycuda.driver as cuda

class VRAMCodeC:
    def __init__(self, stream:cuda.Stream = None):
        self.stream = stream if stream is not None else cuda.Stream()
        self.nvcomp_stream = nvcomp.CudaStream.borrow(self.stream.handle)
        self.codec = nvcomp.Codec(algorithm="Bitcomp", cuda_stream = self.nvcomp_stream.device)
    def encode(self, data_in:HostDeviceMem) -> nvcomp.Array:
        nvarr:nvcomp.Array = nvcomp.as_array(data_in.device.as_buffer(data_in.host.nbytes))
        return self.codec.encode(nvarr).cpu()
    def decode(self, data_in:nvcomp.Array, data_out:HostDeviceMem):
        nvarr_dec:nvcomp.Array = self.codec.decode(data_in.cuda())
        src = nvarr_dec.__cuda_array_interface__['data'][0]
        dst = data_out.device
        cuda.memcpy_dtod_async(dst, src, data_out.host.nbytes, self.stream)