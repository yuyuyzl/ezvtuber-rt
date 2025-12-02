import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from ezvtb_rt.tha3 import THA3Engines, THA3EnginesSimple
from tqdm import tqdm
import json
import cv2
from typing import List


# Function to generate video
def generate_video(imgs: List[np.ndarray], video_path: str, framerate: float):
    """Generate video from a list of images
    Args:
        imgs: List of images in opencv format (BGR)
        video_path: Output video path
        framerate: Video framerate
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(video_path, fourcc, framerate, (imgs[0].shape[1], imgs[0].shape[0]))
    if not video.isOpened():
        raise ValueError("CV2 video encoder Not supported")

    for img in imgs:
        video.write(img)

    video.release()
    cv2.destroyAllWindows()
    print("Video generated successfully!")


def THA3EnginesPerf():
    """Performance test for THA3Engines with VRAM caching"""
    model_dir = './data/tha3/seperable/fp16'
    engine = THA3Engines(model_dir, vram_cache_size=0.0, use_eyebrow=True)
    
    # Load test image (512x512 RGBA)
    img = cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED)
    engine.syncSetImage(img)
    
    # Load pose data
    with open('./test/data/pose_20fps.json', 'r') as file:
        pose_data = json.load(file)
    
    cuda.start_profiler()
    
    # First pass - mostly cache misses
    print("First pass (cache cold):")
    for i in tqdm(range(len(pose_data[800:1000]))):
        engine.asyncInfer(np.array(pose_data[i]).reshape(1, 45))
        output = engine.syncAndGetOutput()
        output.copy()  # Force copy to measure actual throughput
    
    # Second pass - should have cache hits
    print("Second pass (cache warm):")
    for i in tqdm(range(len(pose_data[800:1000]))):
        engine.asyncInfer(np.array(pose_data[i]).reshape(1, 45))
        output = engine.syncAndGetOutput()
        output.copy()
    
    cuda.stop_profiler()
    
    # Print cache statistics
    if engine.cacher is not None:
        print(f"\nMorpher cache - Hits: {engine.cacher.hits}, Misses: {engine.cacher.miss}")


def THA3EnginesShow():
    """Generate a test video using THA3Engines"""
    model_dir = './data/tha3/seperable/fp16'
    engine = THA3Engines(model_dir, vram_cache_size=1.0, use_eyebrow=True)
    
    # Load test image (512x512 RGBA)
    img = cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED)
    engine.syncSetImage(img)
    
    # Load pose data
    with open('./test/data/pose_20fps.json', 'r') as file:
        pose_data = json.load(file)
    
    def create_video_frames(poses, engine):
        frames = []
        # First pass
        print("Generating frames (pass 1):")
        for i in tqdm(range(len(poses))):
            engine.asyncInfer(np.array(poses[i]).reshape(1, 45))
            output = engine.syncAndGetOutput()
            # Convert RGBA to BGR for opencv
            frames.append(output[:, :, :3].copy())
        
        # Second pass to show cache effectiveness
        print("Generating frames (pass 2):")
        for i in tqdm(range(len(poses))):
            engine.asyncInfer(np.array(poses[i]).reshape(1, 45))
            output = engine.syncAndGetOutput()
            frames.append(output[:, :, :3].copy())
        
        return frames
    
    # Use subset of pose data
    frames = create_video_frames(pose_data[800:1200], engine)
    generate_video(frames, './test/data/tha_engines_test.mp4', 20)
    
    # Print cache statistics
    if engine.cacher is not None:
        print(f"\nMorpher cache - Hits: {engine.cacher.hits}, Misses: {engine.cacher.miss}")


def THA3EnginesimpleShow():
    """Generate a test video using THA3Enginesimple (no caching)"""
    model_dir = './data/tha3/seperable/fp16'
    engine = THA3EnginesSimple(model_dir)
    
    # Load test image (512x512 RGBA)
    img = cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED)
    engine.setImage(img)
    
    # Load pose data
    with open('./test/data/pose_20fps.json', 'r') as file:
        pose_data = json.load(file)
    
    def create_video_frames(poses, engine):
        frames = []
        print("Generating frames:")
        for i in tqdm(range(len(poses))):
            output = engine.inference(np.array(poses[i]).reshape(1, 45))
            # Convert RGBA to BGR for opencv
            frames.append(output[:, :, :3].copy())
        return frames
    
    # Use subset of pose data
    frames = create_video_frames(pose_data[800:1200], engine)
    generate_video(frames, './test/data/tha_simple_test.mp4', 20)




if __name__ == "__main__":
    # Run different tests
    # THA3EnginesPerf()      # Performance profiling
    THA3EnginesShow()        # Generate video with cached engine
    # THA3EnginesimpleShow() # Generate video with simple engine
