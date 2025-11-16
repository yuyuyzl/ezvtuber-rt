# Installation Guide

This guide explains how to install `ezvtuber-rt` as a Python library.

## Download Model Files and FFmpeg

Before using the library, you need to download the ONNX models:

1. Download the model package from: [Release Page](https://github.com/zpeng11/ezvtuber-rt/releases/download/0.0.1/20241220.zip)
2. Extract the contents to the `data/` folder in your project directory or provide `EZVTB_DATA` environmental variable points to location.
3. Download the FFmpeg LGPL release package from [Release Page](https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-lgpl-shared.zip)
4. Extract FFmpeg and provide location to package in `FFMPEG_DIR` environmental variable. This is necessary for both build time and runtime.

## Install VS2022+ from Microsoft

You should be able to find other resource on how to install with VC++ build support.

## Installation Methods

Make sure you got FFmpeg and VC++ ready in last section. Clone the repository and install in VS2022+ dev prompt:

```bash
git clone https://github.com/zpeng11/ezvtuber-rt.git && cd ezvtuber-rt
conda install -c nvidia/label/cuda-12.9.1 cuda-toolkit cudnn # You may want to chose a proper version
pip install tensorrt_cu12_libs==10.11.0.33 tensorrt_cu12_bindings==10.11.0.33 tensorrt==10.11.0.33 --extra-index-url https://pypi.nvidia.com #need to fit with cuda-toolkit version
pip install . #This will require a VS2022+ dev environment to build pycuda and cpp packages
```

You can also do development install, requirements are the same
```bash
pip install -e .
```



## Basic Usage

### Using TensorRT (NVIDIA GPUs)

```python
from ezvtb_rt import CoreTRT

# Initialize the core with TensorRT backend
core = CoreTRT(
    tha_dir="data/tha3",           # Path to THA3 models
    vram_cache_size=2.0,            # VRAM cache size in GB
    use_eyebrow=True,               # Enable eyebrow animation
    rife_dir="data/rife",           # Path to RIFE models (frame interpolation)
    sr_dir="data/sr",               # Path to SR models (super resolution)
    cache_max_volume=5.0,           # Max cache volume in GB
    cache_quality=2                 # Cache quality (1-3)
)

# Use the core for inference
# ... your inference code here ...
```

### Using ONNX Runtime (AMD/Intel GPUs)

```python
from ezvtb_rt import CoreORT

# Initialize the core with ONNX Runtime backend
core = CoreORT(
    tha_dir="data/tha3",
    vram_cache_size=2.0,
    use_eyebrow=True,
    rife_dir="data/rife",
    sr_dir="data/sr",
    cache_max_volume=5.0,
    cache_quality=2
)

# Use the core for inference
# ... your inference code here ...
```

## Verifying Installation

Test your installation:

```python
import ezvtb_rt
print(f"ezvtuber-rt version: {ezvtb_rt.__version__}")
print(f"Available modules: {ezvtb_rt.__all__}")
```

