# Installation Guide

This guide explains how to install `ezvtuber-rt` as a Python library.

## Install VS2022+ from Microsoft or install pycuda prebuilt

You should be able to find other resource on how to install with VC++ build support. This is necessary for building and installing `pycuda` package. To avoid doing that, you can install conda prebuilt like this in your env prior to other step:  
```bash
conda install conda-forge::pycuda
```
## Installation Methods

```bash
git clone https://github.com/zpeng11/ezvtuber-rt.git && cd ezvtuber-rt
conda install -c nvidia/label/cuda-12.9.1 cuda-toolkit cudnn # You may want to chose a proper version
pip install tensorrt_cu12_libs==10.11.0.33 tensorrt_cu12_bindings==10.11.0.33 tensorrt==10.11.0.33 --extra-index-url https://pypi.nvidia.com #need to fit with cuda-toolkit version
pip install . #This will require a VS2022+ dev environment to build pycuda
```

You can also do development install, requirements are the same
```bash
pip install -e .
```



## Basic Usage
### Download Model Data
Download model data from [release]() and extract it to `data` folder for default, or you will need to explicitly provide the path when using
### Environment Variables
* `set EZVTB_DEVICE_ID=0`(other than 0 for other GPU) optional to indicate if want to run on non-default GPU

### Using ONNX Runtime DirectML (ALL GPUs)

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
    pose: np.ndarray = np.array(get_from_current_sensor_collect()).astype(np.float32).reshape((1,45)) #Get pose from sensor
    results: np.ndarray = core.inference([pose])
    #This results is a (N, H, W, 4) BGRA image, display or port to streaming

```

### Using TensorRT (NVIDIA GPUs)


```python
from ezvtb_rt import CoreTRT
from ezvtb_rt import init_model_path
from ezvtb_rt.trt_utils import check_build_all_models()

# Initialize the model path
init_model_path('C:/Path/To/Model/Folder')

check_build_all_models() #Necesarry for each device/GPU, only run once per-install

# Initialize the core with TensorRT backend
core = CoreTRT()

#setup input image
core.setImage(cv2_bgra_image)
# Use the core for inference
while True:
    pose: np.ndarray = np.array(get_from_current_sensor_collect()).astype(np.float32).reshape((1,45)) #Get pose from sensor
    results: np.ndarray = core.inference(pose)
    #This results is a (N, H, W, 4) BGRA image, display or port to streaming
```




### Using interpolation
```python
from ezvtb_rt import CoreORT
from ezvtb_rt import init_model_path

# Initialize the model path
init_model_path('C:/Path/To/Model/Folder')

# Initialize the core with TensorRT backend
core = CoreTRT(rife_model_enable = True, rife_model_scale = 3)

#setup input image
core.setImage(cv2_bgra_image)
# Use the core for inference
while True:
    pose0: np.ndarray = get_from_last_sensor_collect() #design you own way to get this from last run
    pose1: np.ndarray = np.array(get_from_current_sensor_collect()).astype(np.float32).reshape((1,45)) #Get current pose from sensor
    pose_0_33: np.ndarray = pose0 + (pose0 + pose1)/3  # Pose interpolation on 1/3
    pose_0_66: np.ndarray = pose0 + 2*(pose0 + pose1)/3 # Pose interpolation on 2/3
    #You may need to qunatize poses for simplification in order to hit catch of poses.
    results: np.ndarray = core.inference([pose_0_33, pose_0_66, pose1])
    #This results is a (3, 512, 512, 4) BGRA image, display or port to streaming
```