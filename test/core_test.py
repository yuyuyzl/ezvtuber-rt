import unittest
import json
import os
import sys
import time
from typing import List

import cv2
import numpy as np
import pycuda.autoinit  # noqa: F401 - ensure CUDA context is ready
from tqdm import tqdm

from ezvtb_rt.core import CoreTRT


def generate_video(imgs: List[np.ndarray], video_path: str, framerate: float):
    """Write frames to an mp4 for manual inspection."""
    if len(imgs) == 0:
        raise ValueError("No images to generate video")

    h, w = imgs[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(video_path, fourcc, framerate, (w, h))
    if not video.isOpened():
        raise ValueError("CV2 video encoder not supported")

    for img in imgs:
        video.write(img)

    video.release()
    cv2.destroyAllWindows()
    print(f"Video generated successfully at {video_path}!")


class TestCoreTRTBase(unittest.TestCase):
    """Common test utilities for CoreTRT option matrix."""

    TEST_IMAGE_PATH = './test/data/base.png'
    POSE_DATA_PATH = './test/data/pose_20fps.json'

    @classmethod
    def setUpClass(cls):
        cls.test_image = cv2.imread(cls.TEST_IMAGE_PATH, cv2.IMREAD_UNCHANGED)
        if cls.test_image is None:
            raise unittest.SkipTest(f"Test image missing at {cls.TEST_IMAGE_PATH}")

        if not os.path.isfile(cls.POSE_DATA_PATH):
            raise unittest.SkipTest(f"Pose data missing at {cls.POSE_DATA_PATH}")
        with open(cls.POSE_DATA_PATH, 'r') as f:
            cls.pose_data = json.load(f)

    def get_pose(self, index: int) -> np.ndarray:
        return np.array(self.pose_data[index]).reshape(1, 45).astype(np.float32)

    def make_core(self, *, version: str = 'v3', **overrides) -> CoreTRT:
        """Create a CoreTRT instance with defaults and skip when unavailable."""
        try:
            return CoreTRT(
                tha_model_version=version,
                tha_model_seperable=True,
                tha_model_fp16=True,
                use_eyebrow=False,
                cache_max_giga=0.0,
                **overrides,
            )
        except Exception as exc:  # pragma: no cover - environment dependent
            self.skipTest(f"CoreTRT init failed: {exc}")


class TestCoreTRTCache(TestCoreTRTBase):
    """Validate caching behaviour (hits, misses, anti-thrashing)."""

    def test_cache_disabled(self):
        core = self.make_core(cache_max_giga=0.0)
        self.assertIsNone(core.cacher_512)

    def test_cache_enabled(self):
        core = self.make_core(cache_max_giga=1.0)
        self.assertIsNotNone(core.cacher_512)

    def test_cache_hits_same_pose(self):
        core = self.make_core(cache_max_giga=1.0)
        core.setImage(self.test_image)

        pose = self.get_pose(800)
        out1 = core.inference(pose)
        misses = core.cacher_512.miss
        hits = core.cacher_512.hits

        out2 = core.inference(pose)

        self.assertEqual(core.cacher_512.hits, hits + 1)
        self.assertEqual(core.cacher_512.miss, misses)
        self.assertEqual(out1.shape, (1, 512, 512, 4))
        np.testing.assert_array_equal(out1, out2)

    def test_cache_miss_for_different_poses(self):
        core = self.make_core(cache_max_giga=1.0)
        core.setImage(self.test_image)

        for i in range(3):
            core.inference(self.get_pose(800 + i))

        self.assertEqual(core.cacher_512.miss, 3)
        self.assertEqual(core.cacher_512.hits, 0)

    def test_cache_anti_thrashing_for_different_keys(self):
        core = self.make_core(cache_max_giga=1.0)
        core.setImage(self.test_image)

        poses: List[np.ndarray] = [self.get_pose(800 + i) for i in range(8)]
        for pose in poses:
            core.inference(pose)

        initial_misses = core.cacher_512.miss

        for pose in poses:
            core.inference(pose)

        self.assertGreater(core.cacher_512.hits, 0)
        self.assertGreater(core.cacher_512.miss, initial_misses)

    def test_cache_same_key_resets_counter(self):
        core = self.make_core(cache_max_giga=1.0)
        core.setImage(self.test_image)

        pose = self.get_pose(800)
        core.inference(pose)

        for _ in range(5):
            core.inference(pose)

        self.assertEqual(core.cacher_512.miss, 1)
        self.assertEqual(core.cacher_512.hits, 5)


class TestCoreTRTBasicTHA4(TestCoreTRTBase):
    """Basic THA4 coverage mirrors THA3 paths."""

    def test_tha_v4_fp16_no_cache(self):
        core = self.make_core(version='v4', tha_model_fp16=True, cache_max_giga=0.0)
        core.setImage(self.test_image)
        out = core.inference(self.get_pose(800))
        self.assertEqual(out.shape, (1, 512, 512, 4))

    def test_tha_v4_fp32_no_cache(self):
        core = self.make_core(version='v4', tha_model_fp16=False, cache_max_giga=0.0)
        core.setImage(self.test_image)
        out = core.inference(self.get_pose(800))
        self.assertEqual(out.shape, (1, 512, 512, 4))

    def test_tha_v4_with_cache_hits(self):
        core = self.make_core(version='v4', tha_model_fp16=True, cache_max_giga=1.0)
        core.setImage(self.test_image)

        pose = self.get_pose(800)
        out1 = core.inference(pose)
        hits = core.cacher_512.hits
        misses = core.cacher_512.miss

        out2 = core.inference(pose)

        self.assertEqual(out1.shape, out2.shape)
        self.assertEqual(core.cacher_512.hits, hits + 1)
        self.assertEqual(core.cacher_512.miss, misses)


class TestCoreTRTRife(TestCoreTRTBase):
    """Check RIFE interpolation variants and cache interaction."""

    def test_rife_disabled(self):
        core = self.make_core(rife_model_enable=False)
        self.assertIsNone(core.rife)

    def test_rife_x2_fp16(self):
        core = self.make_core(rife_model_enable=True, rife_model_scale=2, rife_model_fp16=True)
        self.assertIsNotNone(core.rife)
        core.setImage(self.test_image)

        output = core.inference(self.get_pose(800))
        self.assertEqual(output.shape[0], 2)

    def test_rife_x3_fp16(self):
        core = self.make_core(rife_model_enable=True, rife_model_scale=3, rife_model_fp16=True)
        self.assertIsNotNone(core.rife)
        core.setImage(self.test_image)

        output = core.inference(self.get_pose(800))
        self.assertEqual(output.shape[0], 3)

    def test_rife_cache_hit_uses_cached_tha_output(self):
        core = self.make_core(
            rife_model_enable=True,
            rife_model_scale=2,
            rife_model_fp16=True,
            cache_max_giga=1.0,
        )
        self.assertIsNotNone(core.cacher_512)
        core.setImage(self.test_image)

        pose = self.get_pose(800)
        first = core.inference(pose)
        hits = core.cacher_512.hits
        misses = core.cacher_512.miss

        second = core.inference(pose)

        self.assertEqual(core.cacher_512.hits, hits + 1)
        self.assertEqual(core.cacher_512.miss, misses)
        self.assertEqual(first.shape, second.shape)


class TestCoreTRTSuperResolution(TestCoreTRTBase):
    """Validate SR scaling behaviour and cache reuse."""

    def test_sr_disabled(self):
        core = self.make_core(sr_model_enable=False)
        self.assertIsNone(core.sr)

    def test_sr_waifu2x_x2_fp16(self):
        core = self.make_core(sr_model_enable=True, sr_model_scale=2, sr_model_fp16=True)
        self.assertIsNotNone(core.sr)
        core.setImage(self.test_image)

        output = core.inference(self.get_pose(800))
        self.assertEqual(output.shape[1], 1024)
        self.assertEqual(output.shape[2], 1024)

    def test_sr_realesrgan_x4_fp16(self):
        core = self.make_core(sr_model_enable=True, sr_model_scale=4, sr_model_fp16=True)
        self.assertIsNotNone(core.sr)
        core.setImage(self.test_image)

        output = core.inference(self.get_pose(800))
        self.assertEqual(output.shape[1], 1024)
        self.assertEqual(output.shape[2], 1024)

    def test_sr_cache_hit_reuses_cached_base(self):
        core = self.make_core(sr_model_enable=True, sr_model_scale=2, sr_model_fp16=True, cache_max_giga=1.0)
        self.assertIsNotNone(core.cacher_512)
        core.setImage(self.test_image)

        pose = self.get_pose(800)
        out1 = core.inference(pose)
        hits = core.cacher_512.hits
        misses = core.cacher_512.miss

        out2 = core.inference(pose)

        self.assertEqual(core.cacher_512.hits, hits + 1)
        self.assertEqual(core.cacher_512.miss, misses)
        self.assertEqual(out1.shape, out2.shape)


class TestCoreTRTCombined(TestCoreTRTBase):
    """Ensure RIFE + SR pipelines work together."""

    def test_rife_x2_with_sr_x2(self):
        core = self.make_core(
            rife_model_enable=True,
            rife_model_scale=2,
            rife_model_fp16=True,
            sr_model_enable=True,
            sr_model_scale=2,
            sr_model_fp16=True,
            cache_max_giga=0.0,
        )
        self.assertIsNotNone(core.rife)
        self.assertIsNotNone(core.sr)
        core.setImage(self.test_image)

        output = core.inference(self.get_pose(800))
        self.assertEqual(output.shape[0], 2)
        self.assertEqual(output.shape[1], 1024)
        self.assertEqual(output.shape[2], 1024)

    def test_rife_x3_with_sr_x4_and_cache(self):
        core = self.make_core(
            rife_model_enable=True,
            rife_model_scale=3,
            rife_model_fp16=True,
            sr_model_enable=True,
            sr_model_scale=4,
            sr_model_fp16=True,
            cache_max_giga=1.0,
        )
        self.assertIsNotNone(core.rife)
        self.assertIsNotNone(core.sr)
        core.setImage(self.test_image)

        pose = self.get_pose(800)
        first = core.inference(pose)
        hits = core.cacher_512.hits

        second = core.inference(pose)

        self.assertEqual(first.shape[0], 3)
        self.assertEqual(first.shape[1], 1024)
        self.assertEqual(first.shape[2], 1024)
        self.assertEqual(core.cacher_512.hits, hits + 1)
        self.assertEqual(second.shape, first.shape)

    def test_v4_with_rife_and_sr(self):
        core = self.make_core(
            version='v4',
            tha_model_fp16=True,
            rife_model_enable=True,
            rife_model_scale=2,
            rife_model_fp16=True,
            sr_model_enable=True,
            sr_model_scale=2,
            sr_model_fp16=True,
            cache_max_giga=0.0,
        )
        self.assertIsNotNone(core.rife)
        self.assertIsNotNone(core.sr)
        core.setImage(self.test_image)

        output = core.inference(self.get_pose(800))
        self.assertEqual(output.shape[0], 2)
        self.assertEqual(output.shape[1], 1024)
        self.assertEqual(output.shape[2], 1024)


def CoreTRT_ShowVideo_THAOnly():
    print("=" * 60)
    print("Generating video: THA only (no RIFE, no SR)")
    print("=" * 60)

    core = CoreTRT(
        tha_model_version='v3',
        tha_model_seperable=True,
        tha_model_fp16=True,
        use_eyebrow=False,
        cache_max_giga=2.0,
    )

    img = cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED)
    core.setImage(img)

    with open('./test/data/pose_20fps.json', 'r') as f:
        pose_data = json.load(f)

    frames = []
    print("Pass 1 (cache cold):")
    for i in tqdm(range(200)):
        pose = np.array(pose_data[800 + i]).reshape(1, 45)
        output = core.inference(pose)
        frames.append(output[0, :, :, :3])

    print("Pass 2 (cache warm):")
    for i in tqdm(range(200)):
        pose = np.array(pose_data[800 + i]).reshape(1, 45)
        output = core.inference(pose)
        frames.append(output[0, :, :, :3])

    generate_video(frames, './test/data/core_trt_tha_only.mp4', 20)

    if core.cacher_512:
        print(f"Cache stats - Hits: {core.cacher_512.hits}, Misses: {core.cacher_512.miss}")


def CoreTRT_ShowVideo_THA_v4():
    print("=" * 60)
    print("Generating video: THA v4")
    print("=" * 60)

    try:
        core = CoreTRT(
            tha_model_version='v4',
            tha_model_fp16=True,
            use_eyebrow=False,
            cache_max_giga=2.0,
        )
    except Exception as exc:
        print(f"Skipping THA v4 video: {exc}")
        return

    img = cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED)
    core.setImage(img)

    with open('./test/data/pose_20fps.json', 'r') as f:
        pose_data = json.load(f)

    frames = []
    print("Generating THA v4 frames:")
    for i in tqdm(range(200)):
        pose = np.array(pose_data[800 + i]).reshape(1, 45)
        output = core.inference(pose)
        frames.append(output[0, :, :, :3])

    generate_video(frames, './test/data/core_trt_tha_v4.mp4', 20)

    if core.cacher_512:
        print(f"Cache stats - Hits: {core.cacher_512.hits}, Misses: {core.cacher_512.miss}")


def CoreTRT_ShowVideo_WithRIFE_x2():
    print("=" * 60)
    print("Generating video: THA + RIFE x2")
    print("=" * 60)

    core = CoreTRT(
        tha_model_version='v3',
        tha_model_seperable=True,
        tha_model_fp16=True,
        use_eyebrow=False,
        rife_model_enable=True,
        rife_model_scale=2,
        rife_model_fp16=True,
        cache_max_giga=2.0,
    )

    img = cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED)
    core.setImage(img)

    with open('./test/data/pose_20fps.json', 'r') as f:
        pose_data = json.load(f)

    frames = []
    print("Generating interpolated frames:")
    for i in tqdm(range(200)):
        pose = np.array(pose_data[800 + i]).reshape(1, 45)
        output = core.inference(pose)
        for j in range(output.shape[0]):
            frames.append(output[j, :, :, :3])

    generate_video(frames, './test/data/core_trt_rife_x2.mp4', 40)

    if core.cacher_512:
        print(f"Cache stats - Hits: {core.cacher_512.hits}, Misses: {core.cacher_512.miss}")


def CoreTRT_ShowVideo_WithSR_waifu2x():
    print("=" * 60)
    print("Generating video: THA + waifu2x x2")
    print("=" * 60)

    core = CoreTRT(
        tha_model_version='v3',
        tha_model_seperable=True,
        tha_model_fp16=True,
        use_eyebrow=False,
        sr_model_enable=True,
        sr_model_scale=2,
        sr_model_fp16=True,
        cache_max_giga=2.0,
    )

    img = cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED)
    core.setImage(img)

    with open('./test/data/pose_20fps.json', 'r') as f:
        pose_data = json.load(f)

    frames = []
    print("Generating SR frames (1024x1024):")
    for i in tqdm(range(100)):
        pose = np.array(pose_data[800 + i]).reshape(1, 45)
        output = core.inference(pose)
        frames.append(output[0, :, :, :3])

    generate_video(frames, './test/data/core_trt_sr_waifu2x.mp4', 20)

    if core.cacher_512:
        print(f"Cache stats - Hits: {core.cacher_512.hits}, Misses: {core.cacher_512.miss}")


def CoreTRT_ShowVideo_RIFE_x2_SR_x2():
    print("=" * 60)
    print("Generating video: THA + RIFE x2 + waifu2x x2")
    print("=" * 60)

    core = CoreTRT(
        tha_model_version='v3',
        tha_model_seperable=True,
        tha_model_fp16=True,
        use_eyebrow=False,
        rife_model_enable=True,
        rife_model_scale=2,
        rife_model_fp16=True,
        sr_model_enable=True,
        sr_model_scale=2,
        sr_model_fp16=True,
        cache_max_giga=2.0,
    )

    img = cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED)
    core.setImage(img)

    with open('./test/data/pose_20fps.json', 'r') as f:
        pose_data = json.load(f)

    frames = []
    print("Generating interpolated + SR frames:")
    for i in tqdm(range(100)):
        pose = np.array(pose_data[800 + i]).reshape(1, 45)
        output = core.inference(pose)
        for j in range(output.shape[0]):
            frames.append(output[j, :, :, :3])

    generate_video(frames, './test/data/core_trt_rife_x2_sr_x2.mp4', 40)

    if core.cacher_512:
        print(f"Cache stats - Hits: {core.cacher_512.hits}, Misses: {core.cacher_512.miss}")


def CoreTRT_ShowVideo_CachePerformance():
    print("=" * 60)
    print("Demonstrating cache performance (CoreTRT)")
    print("=" * 60)

    core = CoreTRT(
        tha_model_version='v3',
        tha_model_seperable=True,
        tha_model_fp16=True,
        use_eyebrow=False,
        cache_max_giga=2.0,
    )

    img = cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED)
    core.setImage(img)

    with open('./test/data/pose_20fps.json', 'r') as f:
        pose_data = json.load(f)

    frames_pass1 = []
    start_time = time.time()
    print("Pass 1 (cache cold):")
    for i in tqdm(range(200)):
        pose = np.array(pose_data[800 + i]).reshape(1, 45)
        output = core.inference(pose)
        frames_pass1.append(output[0, :, :, :3])
    pass1_time = time.time() - start_time
    print(f"Pass 1 time: {pass1_time:.2f}s")
    if core.cacher_512:
        print(f"Cache stats - Hits: {core.cacher_512.hits}, Misses: {core.cacher_512.miss}")

    frames_pass2 = []
    start_time = time.time()
    print("\nPass 2 (cache warm):")
    for i in tqdm(range(200)):
        pose = np.array(pose_data[800 + i]).reshape(1, 45)
        output = core.inference(pose)
        frames_pass2.append(output[0, :, :, :3])
    pass2_time = time.time() - start_time
    print(f"Pass 2 time: {pass2_time:.2f}s")
    if core.cacher_512:
        print(f"Cache stats - Hits: {core.cacher_512.hits}, Misses: {core.cacher_512.miss}")

    if pass2_time > 0:
        print(f"\nSpeedup from cache: {pass1_time / pass2_time:.2f}x")

    all_frames = frames_pass1 + frames_pass2
    generate_video(all_frames, './test/data/core_trt_cache_demo.mp4', 20)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == '--show-all':
            CoreTRT_ShowVideo_THAOnly()
            CoreTRT_ShowVideo_THA_v4()
            CoreTRT_ShowVideo_WithRIFE_x2()
            CoreTRT_ShowVideo_WithSR_waifu2x()
            CoreTRT_ShowVideo_RIFE_x2_SR_x2()
            CoreTRT_ShowVideo_CachePerformance()
        elif sys.argv[1] == '--show-tha':
            CoreTRT_ShowVideo_THAOnly()
            CoreTRT_ShowVideo_THA_v4()
        elif sys.argv[1] == '--show-rife':
            CoreTRT_ShowVideo_WithRIFE_x2()
        elif sys.argv[1] == '--show-sr':
            CoreTRT_ShowVideo_WithSR_waifu2x()
        elif sys.argv[1] == '--show-combined':
            CoreTRT_ShowVideo_RIFE_x2_SR_x2()
        elif sys.argv[1] == '--show-cache':
            CoreTRT_ShowVideo_CachePerformance()
        else:
            unittest.main(verbosity=2)
    else:
        unittest.main(verbosity=2)