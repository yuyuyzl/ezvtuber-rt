"""ezvtuber-rt: Real-time virtual tuber animation library.

A high-performance library for real-time virtual tuber animation powered by
TensorRT and ONNX. Supports THA3, RIFE frame interpolation, and super-resolution.

Main Components:
    - CoreTRT: TensorRT-accelerated inference pipeline (NVIDIA GPUs)
    - CoreORT: ONNX Runtime inference pipeline (AMD/Intel GPUs, CPU)
    - THA: Talking Head Anime model wrapper
    - RIFE: Real-time frame interpolation
    - SR: Super-resolution model wrapper
    - Cacher: Memory and disk caching system

Example:
    >>> from ezvtb_rt import CoreTRT
    >>> core = CoreTRT(
    ...     tha_dir="data/tha3",
    ...     vram_cache_size=2.0,
    ...     use_eyebrow=True,
    ...     rife_dir="data/rife",
    ...     sr_dir="data/sr",
    ...     cache_max_volume=5.0
    ... )
"""

__version__ = "0.1.0"
__author__ = "zpeng11"
__license__ = "MIT"

# Import main classes for easy access
try:
    from ezvtb_rt.core import CoreTRT
    __all__ = ["CoreTRT"]
except ImportError:
    # TensorRT not available
    __all__ = []

try:
    from ezvtb_rt.core_ort import CoreORT
    __all__.append("CoreORT")
except ImportError:
    # ONNX Runtime not available
    pass

# Import common base class
try:
    from ezvtb_rt.common import Core
    __all__.append("Core")
except ImportError:
    pass

# Import utility modules
try:
    from ezvtb_rt.cache import Cacher
    __all__.append("Cacher")
except ImportError:
    pass

try:
    import ezvtb_rt.rgba_utils
    __all__.append("rgba_utils")
except ImportError:
    pass

try:
    import os
    with os.add_dll_directory(os.path.join(os.environ['FFMPEG_DIR'], 'bin')):
        import ezvtb_rt.ffmpeg_codec
        __all__.append("ffmpeg_codec")
except ImportError:
    pass