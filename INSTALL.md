# Installation Guide

This guide explains how to install `ezvtuber-rt` as a Python library.

## Download FFmpeg

1. Download the FFmpeg LGPL shared release package from [Release Page](https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-lgpl-shared.zip)
2. Extract FFmpeg and add the `bin` folder to `PATH` environmental variable
environmental variable. This is necessary for both building and running.

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
### Download Model Data
Download model data from [release]() and extract it to `data` folder for default, or you will need to explicitly provide the path when using
### Environment Variables
* FFmpeg package available in `PATH`
* `set EZVTB_DEVICE_ID=0` optional to indicate if want to run on non-default GPU

### Using TensorRT (NVIDIA GPUs)

```python
from ezvtb_rt import CoreTRT
from ezvtb_rt import init_model_path

# Initialize the model path
init_model_path('C:/Path/To/Model/Folder')

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
from ezvtb_rt import init_model_path

# Initialize the model path
init_model_path('C:/Path/To/Model/Folder')

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
