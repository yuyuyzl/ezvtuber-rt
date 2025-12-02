import sys
import os
from ezvtb_rt.init_utils import check_exist_all_models
from ezvtb_rt.core_ort import CoreORT
from ezvtb_rt.tha_ort import THAORT
import numpy as np
from typing import List, Tuple
from tqdm import tqdm
import json
import cv2

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
  
def CoreORTPerf():
    core = CoreORT(tha_model_version='v3', tha_model_fp16=False, tha_model_seperable=False, use_eyebrow=False)
    core.setImage( cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED))
    with open('./test/data/pose_20fps.json', 'r') as file:
        pose_data = json.load(file)
    for pose in tqdm(pose_data[800:1000]):
        ret = core.inference(np.array(pose).reshape(1,45))
        for item in ret:
            item.copy()
    for pose in tqdm(pose_data[800:1000]):
        ret = core.inference(np.array(pose).reshape(1,45))
        for item in ret:
            item.copy()

def CoreORTTestShow():
    core = CoreORT(tha_model_version='v3', tha_model_fp16=True, use_eyebrow=False, tha_model_seperable=False)
    core.setImage( cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED))
    with open('./test/data/pose_20fps.json', 'r') as file:
        pose_data = json.load(file)

    def createInterpolatedVideo(poses, core):
        new_vid = []
        for i in tqdm(range(len(poses))):
            outputs = core.inference(np.array(poses[i]).reshape(1,45))
            for output in outputs:
                new_vid.append(output[:,:,:3])
        for i in tqdm(range(len(poses))):
            outputs = core.inference(np.array(poses[i]).reshape(1,45))
            for output in outputs:
                new_vid.append(output[:,:,:3])
        return new_vid
    
    vid = createInterpolatedVideo(pose_data[800:1000], core)
    generate_video(vid, './test/data/test.mp4', 20)
    if core.cacher is not None:
        print(core.cacher.hits, core.cacher.miss)



if __name__ == "__main__":
    # check_exist_all_models()
    # CoreORTPerf()
    CoreORTTestShow()