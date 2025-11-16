"""THA4 TensorRT Implementation
Similar to tha.py but adapted for THA4's 5-model architecture:
- decomposer, combiner, morpher (face_morpher), body_morpher, upscaler
"""
from ezvtb_rt.trt_utils import *
from ezvtb_rt.engine import Engine, createMemory
from collections import OrderedDict

# VRAMMem and VRAMCacher are same as THA3
class VRAMMem(object):
    """Manages allocation and lifecycle of CUDA device memory"""
    def __init__(self, nbytes:int):
        self.device = cuda.mem_alloc(nbytes)

    def __str__(self):
        return "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()
    
    def __del__(self):
        self.device.free()


class VRAMCacher(object):
    """Implements LRU cache strategy for GPU memory management
    Attributes:
        pool (list): Available memory sets
        cache (OrderedDict): LRU cache of memory sets
        hits/miss (int): Cache performance metrics
    """
    def __init__(self, nbytes1:int, nbytes2:int, max_size:float):
        sum_nkbytes = (nbytes1 + nbytes2)/1024
        self.pool = []
        while len(self.pool) * sum_nkbytes < max_size * 1024 * 1024:
            self.pool.append((VRAMMem(nbytes1), VRAMMem(nbytes2)))
        self.nbytes1 = nbytes1
        self.nbytes2 = nbytes2
        self.cache = OrderedDict()
        self.hits = 0
        self.miss = 0
        if max_size <= 0:
            self.single_mem = (VRAMMem(nbytes1), VRAMMem(nbytes2))
        self.max_size = max_size
    
    def query(self, hs:int)->bool:
        cached = self.cache.get(hs)
        if cached is not None:
            return True
        else:
            return False
    
    def read_mem_set(self, hs:int)->set[VRAMMem, VRAMMem]:
        cached = self.cache.get(hs)
        if cached is not None:
            self.hits += 1
            self.cache.move_to_end(hs)
            return cached
        else:
            self.miss += 1
            return None
    
    def write_mem_set(self, hs:int)->set[VRAMMem, VRAMMem]:
        if self.max_size <= 0:
            return self.single_mem
        if len(self.pool) != 0:
            mem_set = self.pool.pop()
        else:
            mem_set = self.cache.popitem(last=False)[1]
        self.cache[hs] = mem_set
        return mem_set


class THA4():
    """THA4 TensorRT implementation with GPU memory caching
    
    THA4 uses 5 models instead of THA3's 5:
    - decomposer (same)
    - combiner (same)
    - morpher (face_morpher) (same)
    - body_morpher (replaces rotator)
    - upscaler (replaces editor)
    
    Attributes:
        combiner_cacher (VRAMCacher): Cache for eyebrow combination results
        morpher_cacher (VRAMCacher): Cache for face morphing results
        streams (cuda.Stream): Dedicated CUDA streams for different operations
        events (cuda.Event): Synchronization events for pipeline stages
    """
    def __init__(self, model_dir, vram_cache_size:float = 1.0, use_eyebrow:bool = True):
        """Initialize THA4 with model directory and caching configuration
        
        Args:
            model_dir (str): Directory containing TensorRT engine files
            vram_cache_size (float): Total GPU memory allocated for caching (in MB)
            use_eyebrow (bool): Enable eyebrow pose processing
        """
        self.prepareEngines(model_dir)
        self.prepareMemories()
        self.setMemsToEngines()
        
        # THA4: Combiner cache only needs eyebrow_image (no decoded features)
        if use_eyebrow:
            # Use a dummy size for the second buffer since THA4 doesn't have it
            self.combiner_cacher = VRAMCacher(
                self.memories['eyebrow_image'].host.nbytes, 
                self.memories['eyebrow_image'].host.nbytes,  # Same size, won't be used
                0.1 * vram_cache_size)
        else:
            self.combiner_cacher = None
        
        self.morpher_cacher = VRAMCacher(
            self.memories['face_morphed_full'].host.nbytes,
            self.memories['face_morphed_half'].host.nbytes,
            (0.9 if use_eyebrow else 1.0) * vram_cache_size)
        
        self.use_eyebrow = use_eyebrow

        # Create CUDA streams
        self.updatestream = cuda.Stream()
        self.instream = cuda.Stream()
        self.cachestream = cuda.Stream()
        self.outstream = cuda.Stream()
        
        # Create CUDA events
        self.finishedMorpher = cuda.Event()
        self.finishedCombiner = cuda.Event()
        self.finishedFetch = cuda.Event()
        self.finishedExec = cuda.Event()
    
    def setImage(self, img:np.ndarray):
        """Prepare input image and run initial processing
        
        Args:
            img (np.ndarray): Input image array (512x512x4 BGRA format)
        """
        assert(len(img.shape) == 3 and 
               img.shape[0] == 512 and 
               img.shape[1] == 512 and 
               img.shape[2] == 4)
        np.copyto(self.memories['input_img'].host, img)
        self.memories['input_img'].htod(self.updatestream)
        self.decomposer.exec(self.updatestream)
        
        if not self.use_eyebrow:
            self.memories['eyebrow_pose'].host[:,:] = 0.0
            self.memories['eyebrow_pose'].htod(self.updatestream)
            self.combiner.exec(self.updatestream)
        
        self.updatestream.synchronize()

    def inference(self, pose:np.ndarray):
        """Execute full inference pipeline with pose data and caching
        
        Args:
            pose (np.ndarray): Combined pose parameters array containing:
                - eyebrow_pose: First 12 elements
                - face_pose: Next 27 elements 
                - rotation_pose: Remaining 6 elements (total 45)
        """
        eyebrow_pose = pose[:, :12]
        face_pose = pose[:, 12:12+27]
        rotation_pose = pose[:, 12+27:]

        np.copyto(self.memories['rotation_pose'].host, rotation_pose)
        self.memories['rotation_pose'].htod(self.instream)

        morpher_hash = hash(str(pose[0, :12+27]))
        morpher_cached = self.morpher_cacher.read_mem_set(morpher_hash)
        combiner_hash = hash(str(pose[0, :12]))
        
        if self.use_eyebrow:
            combiner_cached = self.combiner_cacher.read_mem_set(combiner_hash)
        else:
            combiner_cached = None

        self.cachestream.synchronize()
        self.outstream.synchronize()
        
        if morpher_cached is not None:
            # Fast path: morpher cached, only run body_morpher and upscaler
            cuda.memcpy_dtod_async(
                self.memories['face_morphed_full'].device, 
                morpher_cached[0].device, 
                self.memories['face_morphed_full'].host.nbytes, 
                self.instream)
            cuda.memcpy_dtod_async(
                self.memories['face_morphed_half'].device, 
                morpher_cached[1].device, 
                self.memories['face_morphed_half'].host.nbytes, 
                self.instream)
            self.body_morpher.exec(self.instream)
            self.upscaler.exec(self.instream)
            self.finishedExec.record(self.instream)
            
        elif combiner_cached is not None or not self.use_eyebrow:
            # Medium path: combiner cached, run morpher + body_morpher + upscaler
            if self.use_eyebrow:
                # THA4: Only cache eyebrow_image, no morpher_decoded
                cuda.memcpy_dtod_async(
                    self.memories['eyebrow_image'].device, 
                    combiner_cached[0].device, 
                    self.memories['eyebrow_image'].host.nbytes, 
                    self.instream)
            
            np.copyto(self.memories['face_pose'].host, face_pose)
            self.memories['face_pose'].htod(self.instream)
            
            # Execute morpher
            self.morpher.exec(self.instream)
            self.finishedMorpher.record(self.instream)
            
            # Execute body_morpher and upscaler
            self.body_morpher.exec(self.instream)
            self.upscaler.exec(self.instream)
            self.finishedExec.record(self.instream)

            # Cache morpher result
            self.cachestream.wait_for_event(self.finishedMorpher)
            morpher_cache_write = self.morpher_cacher.write_mem_set(morpher_hash)
            cuda.memcpy_dtod_async(
                morpher_cache_write[0].device, 
                self.memories['face_morphed_full'].device, 
                self.memories['face_morphed_full'].host.nbytes, 
                self.cachestream)
            cuda.memcpy_dtod_async(
                morpher_cache_write[1].device, 
                self.memories['face_morphed_half'].device, 
                self.memories['face_morphed_half'].host.nbytes, 
                self.cachestream)
                
        else:
            # Full path: nothing cached, run all stages
            np.copyto(self.memories['face_pose'].host, face_pose)
            self.memories['face_pose'].htod(self.instream)
            np.copyto(self.memories['eyebrow_pose'].host, eyebrow_pose)
            self.memories['eyebrow_pose'].htod(self.instream)

            # Execute combiner
            self.combiner.exec(self.instream)
            self.finishedCombiner.record(self.instream)
            
            # Execute morpher
            self.morpher.exec(self.instream)
            self.finishedMorpher.record(self.instream)

            # Execute body_morpher and upscaler
            self.body_morpher.exec(self.instream)
            self.upscaler.exec(self.instream)
            self.finishedExec.record(self.instream)

            # Cache combiner result - THA4: only eyebrow_image
            combiner_cache_write = self.combiner_cacher.write_mem_set(combiner_hash)
            self.cachestream.wait_for_event(self.finishedCombiner)
            cuda.memcpy_dtod_async(
                combiner_cache_write[0].device, 
                self.memories['eyebrow_image'].device, 
                self.memories['eyebrow_image'].host.nbytes, 
                self.cachestream)
            
            # Cache morpher result
            morpher_cache_write = self.morpher_cacher.write_mem_set(morpher_hash)
            self.cachestream.wait_for_event(self.finishedMorpher)
            cuda.memcpy_dtod_async(
                morpher_cache_write[0].device, 
                self.memories['face_morphed_full'].device, 
                self.memories['face_morphed_full'].host.nbytes, 
                self.cachestream)
            cuda.memcpy_dtod_async(
                morpher_cache_write[1].device, 
                self.memories['face_morphed_half'].device, 
                self.memories['face_morphed_half'].host.nbytes, 
                self.cachestream)
        
        self.outstream.wait_for_event(self.finishedExec)
        self.memories['cv_result'].dtoh(self.outstream)
        self.finishedFetch.record(self.outstream)

    def fetchRes(self)->List[np.ndarray]:
        """Retrieve processed results from GPU
        
        Returns:
            List[np.ndarray]: List containing output image array
        """
        self.finishedFetch.synchronize()
        return [self.memories['cv_result'].host]
    
    def viewRes(self)->List[np.ndarray]:
        """Get current output without synchronization
        
        Returns:
            List[np.ndarray]: List containing most recent output image
        """
        return [self.memories['cv_result'].host]

    def prepareEngines(self, model_dir):
        """Load and initialize TensorRT engines from model directory"""
        TRT_LOGGER.log(TRT_LOGGER.INFO, 'Creating THA4 Engines')
        self.decomposer = Engine(join(model_dir, 'decomposer.trt'), 1)
        self.combiner = Engine(join(model_dir, 'combiner.trt'), 4)
        self.morpher = Engine(join(model_dir, 'morpher.trt'), 3)
        self.body_morpher = Engine(join(model_dir, 'body_morpher.trt'), 2)
        self.upscaler = Engine(join(model_dir, 'upscaler.trt'), 4)

    def prepareMemories(self):
        """Allocate GPU memory for all engine inputs/outputs"""
        TRT_LOGGER.log(TRT_LOGGER.INFO, 'Creating memories on VRAM for THA4')
        self.memories = {}
        
        # Decomposer I/O
        self.memories['input_img'] = createMemory(self.decomposer.inputs[0])
        self.memories["background_layer"] = createMemory(self.decomposer.outputs[0])
        self.memories["eyebrow_layer"] = createMemory(self.decomposer.outputs[1])
        self.memories["image_prepared"] = createMemory(self.decomposer.outputs[2])

        # Combiner I/O
        self.memories['eyebrow_pose'] = createMemory(self.combiner.inputs[3])
        self.memories['eyebrow_image'] = createMemory(self.combiner.outputs[0])
        # THA4 combiner only outputs eyebrow_image, no decoded features
        self.memories['morpher_decoded'] = None

        # Morpher (face_morpher) I/O - THA4 has only 3 inputs
        self.memories['face_pose'] = createMemory(self.morpher.inputs[2])
        self.memories['face_morphed_full'] = createMemory(self.morpher.outputs[0])
        self.memories['face_morphed_half'] = createMemory(self.morpher.outputs[1])

        # Body morpher I/O
        self.memories['rotation_pose'] = createMemory(self.body_morpher.inputs[1])
        self.memories['half_res_posed_image'] = createMemory(self.body_morpher.outputs[0])
        self.memories['half_res_grid_change'] = createMemory(self.body_morpher.outputs[1])

        # Upscaler I/O
        self.memories['result'] = createMemory(self.upscaler.outputs[0])
        self.memories['cv_result'] = createMemory(self.upscaler.outputs[1])

    def setMemsToEngines(self):
        """Connect allocated memory buffers to engine I/O ports"""
        TRT_LOGGER.log(TRT_LOGGER.INFO, 'Linking memories on VRAM to THA4 engine graph nodes')
        
        # Decomposer
        decomposer_inputs = [self.memories['input_img']]
        self.decomposer.setInputMems(decomposer_inputs)
        decomposer_outputs = [
            self.memories["background_layer"], 
            self.memories["eyebrow_layer"], 
            self.memories["image_prepared"]
        ]
        self.decomposer.setOutputMems(decomposer_outputs)

        # Combiner - THA4 has 4 inputs, 1 output
        combiner_inputs = [
            self.memories['image_prepared'], 
            self.memories["background_layer"], 
            self.memories["eyebrow_layer"], 
            self.memories['eyebrow_pose']
        ]
        self.combiner.setInputMems(combiner_inputs)
        combiner_outputs = [self.memories['eyebrow_image']]
        self.combiner.setOutputMems(combiner_outputs)

        # Morpher (face_morpher) - THA4 has 3 inputs (no decoded features)
        morpher_inputs = [
            self.memories['image_prepared'], 
            self.memories['eyebrow_image'], 
            self.memories['face_pose']
        ]
        self.morpher.setInputMems(morpher_inputs)
        morpher_outputs = [
            self.memories['face_morphed_full'], 
            self.memories['face_morphed_half']
        ]
        self.morpher.setOutputMems(morpher_outputs)

        # Body morpher
        body_morpher_inputs = [
            self.memories['face_morphed_half'], 
            self.memories['rotation_pose']
        ]
        self.body_morpher.setInputMems(body_morpher_inputs)
        body_morpher_outputs = [
            self.memories['half_res_posed_image'], 
            self.memories['half_res_grid_change']
        ]
        self.body_morpher.setOutputMems(body_morpher_outputs)

        # Upscaler
        upscaler_inputs = [
            self.memories['face_morphed_full'], 
            self.memories['half_res_posed_image'], 
            self.memories['half_res_grid_change'], 
            self.memories['rotation_pose']
        ]
        self.upscaler.setInputMems(upscaler_inputs)
        upscaler_outputs = [
            self.memories['result'], 
            self.memories['cv_result']
        ]
        self.upscaler.setOutputMems(upscaler_outputs)
