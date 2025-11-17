from ezvtb_rt.trt_utils import check_build_all_models
import numpy as np
import pycuda.driver as cuda
from ezvtb_rt import CoreTRT
from tqdm import tqdm
import json
import cv2
from typing import List, Tuple
from ezvtb_rt import init_model_path
import ezvtb_rt

init_model_path('D:\\EasyVtuber\\data\\models')
print(ezvtb_rt.EZVTB_DATA)

# Function to generate video
def generate_video(imgs:List[np.ndarray], video_path:str, framerate:float): #Images should be prepared to be opencv image layout

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(video_path, fourcc, framerate, (imgs[0].shape[0], imgs[0].shape[1]))
    if not video.isOpened():
        raise ValueError("CV2 video encoder Not supported")

    # Appending images to video
    for i in range(len(imgs)):
        video.write(imgs[i])

    # Release the video file
    video.release()
    cv2.destroyAllWindows()
    print("Video generated successfully!")

def CorePerf():
    core = CoreTRT(tha_model_version='v4')
    core.setImage( cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED))
    cuda.start_profiler()
    with open('./test/data/pose_20fps.json', 'r') as file:
        pose_data = json.load(file)
    for i in tqdm(range(len(pose_data[800:1000]))):
        ret = core.inference(np.array(pose_data[i]).reshape(1,45))
        for item in ret:
            item.copy()
    for i in tqdm(range(len(pose_data[800:1000]))):
        ret = core.inference(np.array(pose_data[i]).reshape(1,45))
        for item in ret:
            item.copy()
    cuda.stop_profiler()

def CoreShow():
    core = CoreTRT(tha_model_version='v4', tha_model_seperable=True, tha_model_fp16=True, sr_model_enable=True, sr_model_scale=4, sr_model_fp16=True, rife_model_enable=True, rife_model_scale=4, rife_model_fp16=True)
    core.setImage( cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED))

    with open('./test/data/pose_20fps.json', 'r') as file:
        pose_data = json.load(file)

    def createInterpolatedVideo(poses, core):
        new_vid = []
        for i in tqdm(range(len(poses))):
            outputs = core.inference(np.array(poses[i]).reshape(1,45))
            for output in outputs:
                new_vid.append(output[:,:,:3].copy())
        for i in tqdm(range(len(poses))):
            outputs = core.inference(np.array(poses[i]).reshape(1,45))
            for output in outputs:
                new_vid.append(output[:,:,:3].copy())
        return new_vid
    
    vid = createInterpolatedVideo(pose_data[800:1000], core)
    generate_video(vid, './test/data/test.mp4', 80)
    if core.cacher is not None:
        print(core.cacher.hits, core.cacher.miss)


if __name__ == "__main__":
    # check_build_all_models()
    CoreShow()
    CorePerf()