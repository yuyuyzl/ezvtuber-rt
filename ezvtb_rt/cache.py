import numpy as np
from collections import OrderedDict
import os
import glob

def find_ffmpeg_from_path():
    """Scan PATH for FFmpeg bin dir containing key DLLs."""
    # Key FFmpeg DLLs to check for (adjust if your package uses others)
    required_dlls = ['avcodec-*.dll', 'avformat-*.dll', 'avutil-*.dll']
    
    path_dirs = os.environ.get('PATH', '').split(os.pathsep)
    for dir_path in path_dirs:
        if not os.path.isdir(dir_path):
            continue
        # Check if this dir has the required DLLs
        has_all_dlls = all(any(os.path.isfile(os.path.join(dir_path, dll))
                               for dll in glob.glob(os.path.join(dir_path, pattern)))
                           for pattern in required_dlls)
        if has_all_dlls:
            return dir_path
    return None

# Find and add the dir (if Python 3.8+)
ffmpeg_dir = find_ffmpeg_from_path()
if ffmpeg_dir:
    with os.add_dll_directory(ffmpeg_dir):
        import ezvtb_rt.ffmpeg_codec as ffmpeg_codec
    print(f"Added FFmpeg DLL path from PATH: {ffmpeg_dir}")
else:
    raise ImportError("FFmpeg DLLs not found in PATH. Ensure FFmpeg bin is added to system PATH.")


"""
Cache system with compression and LRU eviction.
Handles image data with synchronous compression/decompression.
Uses HuffYUV for efficient compression/decompression.
"""


class Cacher:
    """Main cache interface with synchronous compression.
    
    Attributes:
        cache: OrderedDict storing compressed entries
        encoder: HuffYUV encoder for compression
        decoder: HuffYUV decoder for decompression
        max_kbytes: Maximum cache size in kilobytes
        cached_kbytes: Current cache size in kilobytes
        hits: Total cache hits
        miss: Total cache misses
        continues_hits: Counter for sequential hits (anti-thrashing)
        last_hs: Last accessed hash key
    """
    
    def __init__(self, max_volume_giga:float = 2.0, width:int = 512, height:int = 512):
        """Initialize cache with specified size.
        
        Args:
            max_volume_giga: Maximum cache size in gigabytes (default 2.0)
        """
        self.cache = OrderedDict()  # LRU cache storage
        self.encoder = ffmpeg_codec.HuffYUVEncoderBGRA(width, height)
        self.decoder = ffmpeg_codec.HuffYUVDecoderBGRA(width, height)
        
        # Cache size management
        self.max_kbytes = max_volume_giga * 1024 * 1024  # Convert GB to KB
        self.cached_kbytes = 0  # Tracks total cached data size

        # Performance tracking
        self.hits = 0  # Total successful cache retrievals
        self.miss = 0  # Total cache misses
        
        # Cache state tracking
        self.continues_hits = 0  # Sequential hit counter
        self.last_hs = -1  # Last accessed hash key
    def read(self, hs:int) -> np.ndarray:
        """Retrieve cached data by hash key.
        
        Args:
            hs: Hash key of requested data
            
        Returns:
            np.ndarray: Decompressed image data or None if miss
        """
        # Check cache existence
        cached = self.cache.get(hs)
        
        # Anti-thrashing: Force miss after 5 sequential hits
        if self.continues_hits > 5:
            cached = None
            
        if cached is not None:
            # Update hit tracking
            if self.last_hs != hs:
                self.continues_hits += 1  # Increment sequential counter
            else:
                self.continues_hits = 0  # Reset if same key
            self.last_hs = hs
            self.hits += 1
            
            # Promote to MRU position
            self.cache.move_to_end(hs)
            
            result_img = self.decoder.decode(cached)
            return result_img
        else:
            # Update miss tracking
            self.miss += 1
            self.continues_hits = 0
            self.last_hs = hs
            return None
    def write(self, hs:int, data:np.ndarray):
        """Write data to cache with compression.
        
        Args:
            hs: Hash key for the data
            data: Raw image data to cache
        """
        # Skip if already cached
        if hs in self.cache:
            return
            
        # Compress data
        compressed = self.encoder.encode(data)
        
        # Add to cache and update size tracking
        self.cache[hs] = compressed
        self.cached_kbytes += len(compressed) / 1024  # Track size in KB
        
        # LRU eviction when over capacity
        while self.cached_kbytes > self.max_kbytes:
            poped = self.cache.popitem(last=False)  # Remove oldest entry
            self.cached_kbytes -= len(poped[1]) / 1024
            poped = None  # Allow GC
