# Installation Guide

This guide explains how to install `ezvtuber-rt` as a Python library.

## Download Model Files and FFmpeg

Before using the library, you need to download the ONNX models:

1. Download the model package from: [Release Page](https://github.com/zpeng11/ezvtuber-rt/releases/download/0.0.1/20241220.zip)
2. Extract the contents to the `data/` folder in your project directory or provide `EZVTB_DATA` environmental variable points to location.
3. Download the FFmpeg LGPL shared release package from [Release Page](https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-lgpl-shared.zip)
4. Extract FFmpeg and provide location to package in `FFMPEG_DIR` environmental variable. This is necessary for both build time and runtime.

## Install VS2022+ from Microsoft

You should be able to find other resource on how to install with VC++ build support.

## Installation Methods

Make sure you got FFmpeg and VC++ ready in last section. Clone the repository and install in VS2022+ dev prompt:

```bash
set FFMPEG_DIR=C:\Dir\To\FFmpeg
set EZVTB_DEVICE_ID=0
set EZVTB_DATA=C:\Dir\To\Model
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
### Environment Variables necessary at runtime
* `set FFMPEG_DIR=C:\Dir\To\FFmpeg` neccesarry
* `set EZVTB_DEVICE_ID=0` optional to indicate if want to run on non-default GPU
* `set EZVTB_DATA=C:\Dir\To\Model` optional if want to save model in other location
### Using TensorRT (NVIDIA GPUs)

```python
from ezvtb_rt import CoreTRT

# Initialize the core with TensorRT backend
core = CoreTRT()

#setup input image
core.setImage(cv2_bgra_image)
# Use the core for inference
while True:
    pose = np.array([0.0, ... 0.0]).astype(np.float32).reshape((1,45)) #Get pose from sensor
    results = core.inference(pose)
    for result in results:
        #This result is a (N, N, 4) BGRA image, display or port to streaming
```

### Using ONNX Runtime (AMD/Intel GPUs)

```python
from ezvtb_rt import CoreORT

# Initialize the core with TensorRT backend
core = CoreTRT()

#setup input image
core.setImage(cv2_bgra_image)
# Use the core for inference
while True:
    pose = np.array([0.0, ... 0.0]).astype(np.float32).reshape((1,45)) #Get pose from sensor
    results = core.inference(pose)
    for result in results:
        #This result is a (N, N, 4) BGRA image, display or port to streaming

```
