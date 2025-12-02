from ezvtb_rt.trt_utils import *
from nvidia import nvcomp
import pycuda.driver as cuda
from collections import OrderedDict
from typing import List, Optional
from ezvtb_rt.trt_engine import HostDeviceMem

class NVCompDeviceBuffer:
    def __init__(self, array:nvcomp.Array):
        self.array = array
        self.device = array.__cuda_array_interface__['data'][0]
        self.nbytes = array.item_size * array.size


class VRAMCodeC:
    def __init__(self, stream:cuda.Stream = None):
        self.stream = stream if stream is not None else cuda.Stream()
        self.nvcomp_stream = nvcomp.CudaStream.borrow(self.stream.handle)
        self.codec = nvcomp.Codec(algorithm="Bitcomp", cuda_stream = self.nvcomp_stream.device)
    def encode(self, data_in:HostDeviceMem) -> NVCompDeviceBuffer:
        nvarr:nvcomp.Array = nvcomp.as_array(data_in.device.as_buffer(data_in.host.nbytes))
        return NVCompDeviceBuffer(self.codec.encode(nvarr).cpu())
    def decode(self, encoded_data:NVCompDeviceBuffer) -> NVCompDeviceBuffer:
        nvarr_dec:nvcomp.Array = self.codec.decode(encoded_data.array.cuda())
        return NVCompDeviceBuffer(nvarr_dec)


class VRAMCacher:
    """LRU cache for compressed VRAM buffers with configurable memory limit.
    
    Stores HostDeviceMem data in compressed form using VRAMCodeC to reduce VRAM usage.
    """
    
    def __init__(self, max_size_gb: float, stream: cuda.Stream = None):
        """
        Initialize the VRAM cacher.
        
        Args:
            max_size_gb: Maximum cache size in gigabytes.
            stream: Optional CUDA stream for operations.
        """
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        self.current_size_bytes = 0
        stream = stream if stream is not None else cuda.Stream()
        self.codec = VRAMCodeC(stream=stream)
        self._cache: OrderedDict[int, List[NVCompDeviceBuffer]] = OrderedDict()
        self.hits = 0
        self.miss = 0
    
    def _calculate_entry_size(self, buffers: List[NVCompDeviceBuffer]) -> int:
        """Calculate total size of a cache entry in bytes."""
        return sum(buf.nbytes for buf in buffers)
    
    def _evict_until_fit(self, required_bytes: int) -> None:
        """Evict LRU entries until there's enough space for required_bytes."""
        while self._cache and (self.current_size_bytes + required_bytes > self.max_size_bytes):
            oldest_key, oldest_buffers = self._cache.popitem(last=False)
            evicted_size = self._calculate_entry_size(oldest_buffers)
            self.current_size_bytes -= evicted_size
    
    def put(self, key: int, buffers: List[HostDeviceMem]) -> None:
        """
        Store a list of HostDeviceMem buffers in the cache (compressed).
        
        Args:
            key: Integer key for the cache entry.
            buffers: List of HostDeviceMem objects to compress and cache.
        """
        # Encode all buffers using VRAMCodeC
        encoded_buffers = [self.codec.encode(buf) for buf in buffers]
        entry_size = self._calculate_entry_size(encoded_buffers)
        
        # If key already exists, remove old entry first
        if key in self._cache:
            old_buffers = self._cache.pop(key)
            self.current_size_bytes -= self._calculate_entry_size(old_buffers)
        
        # Evict entries if needed to make room
        self._evict_until_fit(entry_size)
        
        # Add new entry at the end (most recently used)
        self._cache[key] = encoded_buffers
        self.current_size_bytes += entry_size
    
    def get(self, key: int) -> Optional[List[NVCompDeviceBuffer]]:
        """
        Retrieve and decode a cached entry by key.
        
        Args:
            key: Integer key to look up.
            
        Returns:
            List of decoded NVCompDeviceBuffer if found, None otherwise.
            Use the .device pointer to access decompressed data on GPU.
        """
        if key not in self._cache:
            self.miss += 1
            return None
        
        self.hits += 1
        # Move to end (most recently used)
        self._cache.move_to_end(key)
        
        # Decode all buffers
        encoded_buffers = self._cache[key]
        decoded_buffers = [self.codec.decode(buf) for buf in encoded_buffers]
        return decoded_buffers
    
    def __contains__(self, key: int) -> bool:
        """Check if a key exists in the cache."""
        return key in self._cache
    
    def __len__(self) -> int:
        """Return the number of entries in the cache."""
        return len(self._cache)
    
    def clear(self) -> None:
        """Clear all entries from the cache."""
        self._cache.clear()
        self.current_size_bytes = 0
        self.hits = 0
        self.miss = 0
    
    @property
    def size_gb(self) -> float:
        """Return current cache size in gigabytes."""
        return self.current_size_bytes / (1024 * 1024 * 1024)
    
    @property
    def hit_rate(self) -> float:
        """Return cache hit rate as a fraction."""
        total = self.hits + self.miss
        return self.hits / total if total > 0 else 0.0


if __name__ == "__main__":
    import cv2
    stream = cuda.Stream()
    data = cv2.imread("test/data/base.png", cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
    hdm_in = HostDeviceMem.create(data.shape, data.dtype)
    np.copyto(hdm_in.host, data)
    hdm_in.htod(stream)
    codec = VRAMCodeC(stream=stream)
    encoded = codec.encode(hdm_in)
    print(f"Original size: {hdm_in.host.nbytes}, Compressed size: {encoded.nbytes}")
    decoded_buf = codec.decode(encoded)
    hdm_out = HostDeviceMem.create(data.shape, data.dtype)
    cuda.memcpy_dtod_async(hdm_out.device, decoded_buf.device, decoded_buf.nbytes, stream)
    hdm_out.dtoh(stream)
    stream.synchronize()
    print("Compression and decompression successful!")
    print("Max absolute difference:", np.max(np.abs(hdm_in.host - hdm_out.host)))