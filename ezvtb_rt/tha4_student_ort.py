"""THA4 Student Model (Mode 14) ONNX Runtime Implementation
Two-stage inference architecture with DirectML GPU acceleration
"""
import os
import onnxruntime as ort
import numpy as np


class THA4StudentORT:
    """THA4 Student Model ONNX Runtime implementation with DirectML GPU support
    
    Two-stage SIREN-based inference:
    - Stage 1: Face Morpher - generates 128x128 face from pose only
    - Stage 2: Body Morpher - transforms full 512x512 image
    
    Attributes:
        face_morpher: ONNX Runtime session for face generation
        body_morpher: ONNX Runtime session for body transformation
        face_cache: Cache for face morphing results based on pose hash
    """
    
    def __init__(self, model_dir: str, device_id: int = 0):
        """Initialize THA4 Student ONNX Runtime sessions
        
        Args:
            model_dir: Directory containing ONNX model files
            device_id: GPU device ID (default: 0)
        
        Raises:
            ValueError: If DirectML GPU provider not available
        """
        self.model_dir = model_dir
        self.device_id = device_id
        self.dtype = np.float32  # Student model is FP32 only
        self.device = 'dml'
        
        # Check for DirectML support
        available_providers = ort.get_available_providers()
        if 'DmlExecutionProvider' not in available_providers:
            raise ValueError(
                'DirectML not available. Available providers: ' +
                str(available_providers)
            )
        
        print('THA4 Student Model using DmlExecutionProvider')
        
        # Configure session options
        sess_options = ort.SessionOptions()
        sess_options.enable_mem_pattern = False
        sess_options.graph_optimization_level = \
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        providers = [('DmlExecutionProvider', {'device_id': device_id})]
        
        # Load Face Morpher session (pose only -> 128x128 face)
        face_morpher_path = os.path.join(model_dir, 'face_morpher.onnx')
        self.face_morpher = ort.InferenceSession(
            face_morpher_path,
            sess_options=sess_options,
            providers=providers
        )
        print(f'Loaded face_morpher from {face_morpher_path}')
        
        # Load Body Morpher session (image + face + pose -> output)
        body_morpher_path = os.path.join(model_dir, 'body_morpher.onnx')
        self.body_morpher = ort.InferenceSession(
            body_morpher_path,
            sess_options=sess_options,
            providers=providers
        )
        print(f'Loaded body_morpher from {body_morpher_path}')
        
        # Create GPU memory buffers using OrtValues
        # Face Morpher outputs: [1, 4, 128, 128]
        self.face_morphed_buffer = ort.OrtValue.ortvalue_from_shape_and_type(
            (1, 4, 128, 128), self.dtype, self.device
        )
        
        # Body Morpher inputs/outputs
        self.input_image_buffer = ort.OrtValue.ortvalue_from_shape_and_type(
            (512, 512, 4), np.uint8, self.device
        )
        self.pose_buffer = ort.OrtValue.ortvalue_from_shape_and_type(
            (1, 45), self.dtype, self.device
        )
        self.result_buffer = ort.OrtValue.ortvalue_from_shape_and_type(
            (1, 4, 512, 512), self.dtype, self.device
        )
        self.cv_result_buffer = ort.OrtValue.ortvalue_from_shape_and_type(
            (512, 512, 4), np.uint8, self.device
        )
        
        # Setup IO binding for body_morpher for faster execution
        self.body_binding = self.body_morpher.io_binding()
        self.body_binding.bind_ortvalue_input(
            'input_image', self.input_image_buffer
        )
        self.body_binding.bind_ortvalue_input(
            'face_morphed', self.face_morphed_buffer
        )
        self.body_binding.bind_ortvalue_input(
            'pose', self.pose_buffer
        )
        self.body_binding.bind_ortvalue_output(
            'result', self.result_buffer
        )
        self.body_binding.bind_ortvalue_output(
            'cv_result', self.cv_result_buffer
        )
        
        # Cache for face morphing results (pose_hash -> face_morphed_data)
        self.face_cache = {}
        self.current_image = None
    
    def setImage(self, img: np.ndarray):
        """Set input image for processing
        
        Args:
            img: Input image [512, 512, 4] in RGBA format
        
        Raises:
            ValueError: If image is not valid format
        """
        if len(img.shape) != 3 or img.shape != (512, 512, 4):
            raise ValueError(
                f'Image must be (512, 512, 4) RGBA, got {img.shape}'
            )
        
        # Update image buffer on GPU
        self.input_image_buffer.update_inplace(img)
        self.current_image = img
    
    def update_image(self, img: np.ndarray):
        """Alias for setImage for interface compatibility"""
        self.setImage(img)
    
    def inference(self, pose: np.ndarray) -> list:
        """Run inference with pose data
        
        Two-stage pipeline:
        1. Face Morpher: generates 128x128 face from pose only
        2. Body Morpher: transforms full body using face result
        
        Args:
            pose: Pose parameters [1, 45]
                - 0-11: eyebrow parameters
                - 12-38: face expression parameters
                - 39-44: rotation parameters
        
        Returns:
            Output image [512, 512, 4] in RGBA uint8 format
        """
        # Convert pose to FP32 if needed
        pose_fp32 = pose.astype(np.float32)
        
        # Stage 1: Check cache first
        pose_hash = hash(pose_fp32.tobytes())
        
        if pose_hash not in self.face_cache:
            # Generate face using Face Morpher (pose only)
            face_morphed = self.face_morpher.run(
                None,
                {'pose': pose_fp32}
            )[0]
            
            # Cache the result
            self.face_cache[pose_hash] = face_morphed
            
            # Keep cache size reasonable (max 100 entries)
            if len(self.face_cache) > 100:
                # Remove oldest entry (FIFO)
                self.face_cache.pop(next(iter(self.face_cache)))
        else:
            face_morphed = self.face_cache[pose_hash]
        
        # Update face_morphed buffer
        self.face_morphed_buffer.update_inplace(face_morphed)
        
        # Update pose buffer
        self.pose_buffer.update_inplace(pose_fp32)
        
        # Stage 2: Run Body Morpher with IO binding
        self.body_morpher.run_with_iobinding(self.body_binding)
        
        # Return result (copy to CPU)
        cv_result = self.cv_result_buffer.numpy().copy()
        return [cv_result]
    
    def fetchRes(self):
        """Fetch results from GPU (for framework compatibility)
        
        Returns:
            List[np.ndarray]: List containing output image
        """
        cv_result = self.cv_result_buffer.numpy().copy()
        return [cv_result]
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics
        
        Returns:
            Dictionary with cache hit/miss statistics
        """
        return {
            'cache_size': len(self.face_cache),
            'max_cache_size': 100
        }
    
    def clear_cache(self):
        """Clear the face morphing cache"""
        self.face_cache.clear()
