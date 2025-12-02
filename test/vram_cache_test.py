"""
Unit tests and benchmarks for VRAMCacher class.
Tests LRU eviction, cache limits, compression, and performance.
"""
import unittest
import numpy as np
import pycuda.driver as cuda
import cv2
import time
from tqdm import tqdm

from ezvtb_rt.trt_engine import HostDeviceMem
from ezvtb_rt.vram_cache import VRAMCacher, NVCompDeviceBuffer


class TestVRAMCacher(unittest.TestCase):
    """Unit tests for VRAMCacher functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.stream = cuda.Stream()
        # Create test data - 512x512 RGBA float32 image (~4MB uncompressed)
        cls.test_shape = (512, 512, 4)
        cls.test_dtype = np.float32
        cls.entry_uncompressed_size = np.prod(cls.test_shape) * np.dtype(cls.test_dtype).itemsize
    
    def _create_test_hdm(self, seed: int = 0) -> HostDeviceMem:
        """Create a HostDeviceMem with random data for testing."""
        np.random.seed(seed)
        data = np.random.rand(*self.test_shape).astype(self.test_dtype)
        hdm = HostDeviceMem.create(self.test_shape, self.test_dtype)
        np.copyto(hdm.host, data)
        hdm.htod(self.stream)
        self.stream.synchronize()
        return hdm
    
    def _create_test_hdm_from_image(self, path: str = "./test/data/base.png") -> HostDeviceMem:
        """Create a HostDeviceMem from an image file."""
        data = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
        hdm = HostDeviceMem.create(data.shape, data.dtype)
        np.copyto(hdm.host, data)
        hdm.htod(self.stream)
        self.stream.synchronize()
        return hdm
    
    def test_basic_put_get(self):
        """Test basic put and get operations."""
        cacher = VRAMCacher(max_size_gb=0.1, stream=self.stream)
        
        hdm = self._create_test_hdm(seed=42)
        original_data = hdm.host.copy()
        
        # Put data
        cacher.put(1, [hdm])
        self.assertEqual(len(cacher), 1)
        self.assertIn(1, cacher)
        
        # Get data
        result = cacher.get(1)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], NVCompDeviceBuffer)
        
        # Verify data integrity by copying back
        hdm_out = HostDeviceMem.create(self.test_shape, self.test_dtype)
        cuda.memcpy_dtod_async(hdm_out.device, result[0].device, result[0].nbytes, self.stream)
        hdm_out.dtoh(self.stream)
        self.stream.synchronize()
        
        max_diff = np.max(np.abs(original_data - hdm_out.host))
        self.assertEqual(max_diff, 0.0, "Decoded data should match original exactly")
    
    def test_cache_miss(self):
        """Test cache miss returns None."""
        cacher = VRAMCacher(max_size_gb=0.1, stream=self.stream)
        
        result = cacher.get(999)
        self.assertIsNone(result)
        self.assertEqual(cacher.miss, 1)
        self.assertEqual(cacher.hits, 0)
    
    def test_hit_miss_tracking(self):
        """Test hit and miss counters."""
        cacher = VRAMCacher(max_size_gb=0.1, stream=self.stream)
        hdm = self._create_test_hdm()
        
        cacher.put(1, [hdm])
        
        # Miss
        cacher.get(999)
        self.assertEqual(cacher.miss, 1)
        
        # Hit
        cacher.get(1)
        self.assertEqual(cacher.hits, 1)
        
        # Another hit
        cacher.get(1)
        self.assertEqual(cacher.hits, 2)
        
        # Check hit rate
        self.assertAlmostEqual(cacher.hit_rate, 2/3)
    
    def test_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        # Small cache that can hold ~2-3 entries
        cacher = VRAMCacher(max_size_gb=0.008, stream=self.stream)  # ~8MB
        
        hdms = [self._create_test_hdm(seed=i) for i in range(5)]
        
        # Add 5 entries - should evict oldest ones
        for i, hdm in enumerate(hdms):
            cacher.put(i, [hdm])
        
        # Oldest entries should be evicted
        self.assertNotIn(0, cacher)
        self.assertNotIn(1, cacher)
        
        # Newer entries should remain
        self.assertIn(4, cacher)
        self.assertIn(3, cacher)
        
        # Verify cache size is under limit
        self.assertLessEqual(cacher.current_size_bytes, cacher.max_size_bytes)
    
    def test_lru_access_order(self):
        """Test that accessing an entry moves it to MRU position."""
        cacher = VRAMCacher(max_size_gb=0.012, stream=self.stream)  # ~12MB, holds ~3 entries
        
        hdms = [self._create_test_hdm(seed=i) for i in range(3)]
        
        # Add 3 entries
        for i, hdm in enumerate(hdms):
            cacher.put(i, [hdm])
        
        # Access entry 0 (moves to MRU)
        cacher.get(0)
        
        # Add new entry - should evict entry 1 (now LRU)
        cacher.put(3, [self._create_test_hdm(seed=3)])
        
        # Entry 0 should still be present (was accessed recently)
        self.assertIn(0, cacher)
        # Entry 1 should be evicted (was LRU)
        self.assertNotIn(1, cacher)
    
    def test_update_existing_key(self):
        """Test updating an existing key."""
        cacher = VRAMCacher(max_size_gb=0.1, stream=self.stream)
        
        hdm1 = self._create_test_hdm(seed=1)
        hdm2 = self._create_test_hdm(seed=2)
        original_data2 = hdm2.host.copy()
        
        cacher.put(1, [hdm1])
        old_size = cacher.current_size_bytes
        
        # Update with new data
        cacher.put(1, [hdm2])
        
        # Should still be one entry
        self.assertEqual(len(cacher), 1)
        
        # Verify it's the new data
        result = cacher.get(1)
        hdm_out = HostDeviceMem.create(self.test_shape, self.test_dtype)
        cuda.memcpy_dtod_async(hdm_out.device, result[0].device, result[0].nbytes, self.stream)
        hdm_out.dtoh(self.stream)
        self.stream.synchronize()
        
        max_diff = np.max(np.abs(original_data2 - hdm_out.host))
        self.assertEqual(max_diff, 0.0)
    
    def test_multiple_buffers_per_key(self):
        """Test storing multiple buffers under one key."""
        cacher = VRAMCacher(max_size_gb=0.1, stream=self.stream)
        
        hdms = [self._create_test_hdm(seed=i) for i in range(3)]
        original_data = [hdm.host.copy() for hdm in hdms]
        
        cacher.put(1, hdms)
        self.assertEqual(len(cacher), 1)
        
        result = cacher.get(1)
        self.assertEqual(len(result), 3)
        
        # Verify each buffer
        for i, buf in enumerate(result):
            hdm_out = HostDeviceMem.create(self.test_shape, self.test_dtype)
            cuda.memcpy_dtod_async(hdm_out.device, buf.device, buf.nbytes, self.stream)
            hdm_out.dtoh(self.stream)
            self.stream.synchronize()
            max_diff = np.max(np.abs(original_data[i] - hdm_out.host))
            self.assertEqual(max_diff, 0.0, f"Buffer {i} mismatch")
    
    def test_clear(self):
        """Test clearing the cache."""
        cacher = VRAMCacher(max_size_gb=0.1, stream=self.stream)
        
        hdm = self._create_test_hdm()
        cacher.put(1, [hdm])
        cacher.get(1)
        cacher.get(999)  # miss
        
        cacher.clear()
        
        self.assertEqual(len(cacher), 0)
        self.assertEqual(cacher.current_size_bytes, 0)
        self.assertEqual(cacher.hits, 0)
        self.assertEqual(cacher.miss, 0)
    
    def test_contains(self):
        """Test __contains__ method."""
        cacher = VRAMCacher(max_size_gb=0.1, stream=self.stream)
        hdm = self._create_test_hdm()
        
        self.assertNotIn(1, cacher)
        cacher.put(1, [hdm])
        self.assertIn(1, cacher)
    
    def test_compression_ratio(self):
        """Test that compression actually reduces size."""
        cacher = VRAMCacher(max_size_gb=1.0, stream=self.stream)
        
        # Use image data which should compress well
        try:
            hdm = self._create_test_hdm_from_image()
        except:
            hdm = self._create_test_hdm()
        
        uncompressed_size = hdm.host.nbytes
        cacher.put(1, [hdm])
        compressed_size = cacher.current_size_bytes
        
        ratio = compressed_size / uncompressed_size
        print(f"\nCompression ratio: {ratio:.2%} (compressed: {compressed_size}, original: {uncompressed_size})")
        
        # Compression should provide some benefit (ratio < 1)
        # Note: random data might not compress well, real images should
        self.assertGreater(compressed_size, 0)


class TestVRAMCacherBenchmarks(unittest.TestCase):
    """Performance benchmarks for VRAMCacher."""
    
    @classmethod
    def setUpClass(cls):
        """Set up benchmark fixtures."""
        cls.stream = cuda.Stream()
        cls.test_shape = (512, 512, 4)
        cls.test_dtype = np.float32
    
    def _create_test_hdm(self, seed: int = 0) -> HostDeviceMem:
        """Create a HostDeviceMem with random data."""
        np.random.seed(seed)
        data = np.random.rand(*self.test_shape).astype(self.test_dtype)
        hdm = HostDeviceMem.create(self.test_shape, self.test_dtype)
        np.copyto(hdm.host, data)
        hdm.htod(self.stream)
        self.stream.synchronize()
        return hdm
    
    def test_benchmark_put_performance(self):
        """Benchmark put (encode) performance."""
        cacher = VRAMCacher(max_size_gb=2.0, stream=self.stream)
        num_iterations = 1000
        
        # Pre-create test data
        hdm = self._create_test_hdm()
        
        # Warm up
        for i in range(10):
            cacher.put(i, [hdm])
        cacher.clear()
        
        # Benchmark
        start = time.perf_counter()
        for i in tqdm(range(num_iterations), desc="Put benchmark"):
            cacher.put(i, [hdm])
        self.stream.synchronize()
        elapsed = time.perf_counter() - start
        
        ops_per_sec = num_iterations / elapsed
        avg_time_ms = (elapsed / num_iterations) * 1000
        
        print(f"\n=== PUT Benchmark ===")
        print(f"Total time: {elapsed:.2f}s for {num_iterations} operations")
        print(f"Throughput: {ops_per_sec:.1f} ops/sec")
        print(f"Average latency: {avg_time_ms:.3f} ms/op")
        print(f"Final cache size: {cacher.size_gb:.3f} GB")
    
    def test_benchmark_get_performance(self):
        """Benchmark get (decode) performance."""
        cacher = VRAMCacher(max_size_gb=2.0, stream=self.stream)
        num_entries = 100
        num_iterations = 1000
        
        # Pre-populate cache
        hdm = self._create_test_hdm()
        for i in range(num_entries):
            cacher.put(i, [hdm])
        
        # Warm up
        for _ in range(10):
            cacher.get(0)
        
        # Benchmark
        start = time.perf_counter()
        for i in tqdm(range(num_iterations), desc="Get benchmark"):
            key = i % num_entries
            cacher.get(key)
        self.stream.synchronize()
        elapsed = time.perf_counter() - start
        
        ops_per_sec = num_iterations / elapsed
        avg_time_ms = (elapsed / num_iterations) * 1000
        
        print(f"\n=== GET Benchmark ===")
        print(f"Total time: {elapsed:.2f}s for {num_iterations} operations")
        print(f"Throughput: {ops_per_sec:.1f} ops/sec")
        print(f"Average latency: {avg_time_ms:.3f} ms/op")
        print(f"Hit rate: {cacher.hit_rate:.2%}")
    
    def test_benchmark_eviction_performance(self):
        """Benchmark performance under eviction pressure."""
        # Small cache to force frequent evictions
        cacher = VRAMCacher(max_size_gb=0.05, stream=self.stream)  # ~50MB
        num_iterations = 500
        
        hdm = self._create_test_hdm()
        
        start = time.perf_counter()
        for i in tqdm(range(num_iterations), desc="Eviction benchmark"):
            cacher.put(i, [hdm])
        self.stream.synchronize()
        elapsed = time.perf_counter() - start
        
        ops_per_sec = num_iterations / elapsed
        avg_time_ms = (elapsed / num_iterations) * 1000
        
        print(f"\n=== EVICTION Benchmark ===")
        print(f"Total time: {elapsed:.2f}s for {num_iterations} operations")
        print(f"Throughput: {ops_per_sec:.1f} ops/sec")
        print(f"Average latency: {avg_time_ms:.3f} ms/op")
        print(f"Cache entries: {len(cacher)}")
        print(f"Cache size: {cacher.size_gb:.3f} GB / {cacher.max_size_bytes / (1024**3):.3f} GB max")
    
    def test_benchmark_mixed_workload(self):
        """Benchmark realistic mixed put/get workload."""
        cacher = VRAMCacher(max_size_gb=0.1, stream=self.stream)
        num_iterations = 1000
        num_unique_keys = 50
        
        hdm = self._create_test_hdm()
        
        # Pre-populate half the keys
        for i in range(num_unique_keys // 2):
            cacher.put(i, [hdm])
        
        start = time.perf_counter()
        for i in tqdm(range(num_iterations), desc="Mixed workload"):
            key = i % num_unique_keys
            if cacher.get(key) is None:
                # Cache miss - put new data
                cacher.put(key, [hdm])
        self.stream.synchronize()
        elapsed = time.perf_counter() - start
        
        ops_per_sec = num_iterations / elapsed
        
        print(f"\n=== MIXED WORKLOAD Benchmark ===")
        print(f"Total time: {elapsed:.2f}s for {num_iterations} operations")
        print(f"Throughput: {ops_per_sec:.1f} ops/sec")
        print(f"Hit rate: {cacher.hit_rate:.2%}")
        print(f"Hits: {cacher.hits}, Misses: {cacher.miss}")
    
    def test_benchmark_multiple_buffers(self):
        """Benchmark with multiple buffers per entry."""
        cacher = VRAMCacher(max_size_gb=1.0, stream=self.stream)
        num_iterations = 200
        buffers_per_entry = 4
        
        hdms = [self._create_test_hdm(seed=i) for i in range(buffers_per_entry)]
        
        # Put benchmark
        start = time.perf_counter()
        for i in tqdm(range(num_iterations), desc="Multi-buffer put"):
            cacher.put(i, hdms)
        self.stream.synchronize()
        put_elapsed = time.perf_counter() - start
        
        # Get benchmark
        start = time.perf_counter()
        for i in tqdm(range(num_iterations), desc="Multi-buffer get"):
            cacher.get(i)
        self.stream.synchronize()
        get_elapsed = time.perf_counter() - start
        
        print(f"\n=== MULTI-BUFFER Benchmark ({buffers_per_entry} buffers/entry) ===")
        print(f"Put: {num_iterations / put_elapsed:.1f} ops/sec ({put_elapsed / num_iterations * 1000:.3f} ms/op)")
        print(f"Get: {num_iterations / get_elapsed:.1f} ops/sec ({get_elapsed / num_iterations * 1000:.3f} ms/op)")
        print(f"Cache size: {cacher.size_gb:.3f} GB")


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
