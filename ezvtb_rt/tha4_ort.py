"""THA4 ONNX Runtime Implementation
Similar to tha_ort.py but adapted for THA4's 5-model architecture
"""
import os
import onnxruntime as ort
import onnx
import numpy as np
from typing import List



class THA4ORTSessions:
    """THA4 ONNX Runtime implementation for default device (device_id=0)
    
    Uses ONNX Runtime with GPU acceleration (CUDA or DirectML) for inference.
    Supports eyebrow processing and maintains intermediate results on GPU.
    """
    def __init__(self, tha4_dir: str, use_eyebrow: bool = True):
        """Initialize THA4 ORT inference session
        
        Args:
            tha4_dir: Directory containing THA4 ONNX models
            use_eyebrow: Enable eyebrow pose processing
        """
        self.tha4_dir = tha4_dir
        if 'fp16' in tha4_dir:
            self.dtype = np.float16
        else:
            self.dtype = np.float32
        
        # Check for available GPU providers
        available = ort.get_available_providers()
        if 'CUDAExecutionProvider' in available:
            self.provider = 'CUDAExecutionProvider'
            self.device = 'cuda'
        elif 'DmlExecutionProvider' in available:
            self.provider = 'DmlExecutionProvider'
            self.device = 'dml'
        else:
            raise ValueError('No GPU provider available for ONNX Runtime')
        
        print('Using THA4 ORT with EP:', self.provider)
        
        self.use_eyebrow = use_eyebrow
        
        # Session options
        providers = [self.provider]
        options = ort.SessionOptions()
        options.enable_mem_pattern = False
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        # Create OrtValues for GPU memory
        if use_eyebrow:
            self.decomposed_background_layer = ort.OrtValue.ortvalue_from_shape_and_type(
                (1, 4, 128, 128), self.dtype, self.device)
            self.decomposed_eyebrow_layer = ort.OrtValue.ortvalue_from_shape_and_type(
                (1, 4, 128, 128), self.dtype, self.device)
            self.eyebrow_pose = ort.OrtValue.ortvalue_from_shape_and_type(
                (1, 12), self.dtype, self.device)
        else:
            self.combiner_eyebrow_image = ort.OrtValue.ortvalue_from_shape_and_type(
                (1, 4, 192, 192), self.dtype, self.device)
        
        self.image_prepared = ort.OrtValue.ortvalue_from_shape_and_type(
            (1, 4, 512, 512), self.dtype, self.device)
        self.face_pose = ort.OrtValue.ortvalue_from_shape_and_type(
            (1, 27), self.dtype, self.device)
        self.rotation_pose = ort.OrtValue.ortvalue_from_shape_and_type(
            (1, 6), self.dtype, self.device)
        self.result_image = ort.OrtValue.ortvalue_from_shape_and_type(
            (512, 512, 4), np.uint8, self.device)
        
        # Load ONNX sessions
        self.decomposer = ort.InferenceSession(
            os.path.join(tha4_dir, 'decomposer.onnx'),
            sess_options=options,
            providers=providers)
        
        if not use_eyebrow:
            self.combiner = ort.InferenceSession(
                os.path.join(tha4_dir, 'combiner.onnx'),
                sess_options=options,
                providers=providers)
        
        merge_filename = 'merge.onnx' if use_eyebrow else 'merge_no_eyebrow.onnx'
        self.merged = ort.InferenceSession(
            os.path.join(tha4_dir, merge_filename),
            sess_options=options,
            providers=providers)
        
        # Setup IO binding for merged model
        self.binding = self.merged.io_binding()
        if use_eyebrow:
            self.binding.bind_ortvalue_input('combiner_eyebrow_background_layer', self.decomposed_background_layer)
            self.binding.bind_ortvalue_input('combiner_eyebrow_layer', self.decomposed_eyebrow_layer)
            self.binding.bind_ortvalue_input('combiner_eyebrow_pose', self.eyebrow_pose)
            self.binding.bind_ortvalue_input('combiner_image_prepared', self.image_prepared)
        else:
            self.binding.bind_ortvalue_input('morpher_im_morpher_crop', self.combiner_eyebrow_image)
        
        self.binding.bind_ortvalue_input('morpher_image_prepared', self.image_prepared)
        self.binding.bind_ortvalue_input('morpher_face_pose', self.face_pose)
        self.binding.bind_ortvalue_input('body_morpher_rotation_pose', self.rotation_pose)
        self.binding.bind_ortvalue_input('upscaler_rotation_pose', self.rotation_pose)
        self.binding.bind_ortvalue_output('upscaler_cv_result', self.result_image)

    def update_image(self, img: np.ndarray):
        """Process input image through decomposer and optionally combiner
        
        Args:
            img: Input image in BGRA format (512x512x4)
        """
        shapes = img.shape
        if len(shapes) != 3 or shapes[0] != 512 or shapes[1] != 512 or shapes[2] != 4:
            raise ValueError('Invalid update image shape')
        
        decomposed = self.decomposer.run(None, {'input_image': img})
        self.image_prepared.update_inplace(decomposed[2])
        
        if self.use_eyebrow:
            self.decomposed_background_layer.update_inplace(decomposed[0])
            self.decomposed_eyebrow_layer.update_inplace(decomposed[1])
        else:
            # Run combiner with zero eyebrow pose
            combined = self.combiner.run(None, {
                'image_prepared': decomposed[2],
                'eyebrow_background_layer': decomposed[0],
                'eyebrow_layer': decomposed[1],
                'eyebrow_pose': np.zeros((1, 12), dtype=self.dtype)
            })
            self.combiner_eyebrow_image.update_inplace(combined[0])

    def inference(self, poses: np.ndarray) -> np.ndarray:
        """Run inference with pose parameters
        
        Args:
            poses: Pose parameters array (1, 45) containing:
                   [0:12] - eyebrow pose
                   [12:39] - face pose  
                   [39:45] - rotation pose
                   
        Returns:
            Output image as numpy array
        """
        poses = poses.astype(self.dtype)
        if self.use_eyebrow:
            self.eyebrow_pose.update_inplace(poses[:, :12])
        self.face_pose.update_inplace(poses[:, 12:12+27])
        self.rotation_pose.update_inplace(poses[:, 12+27:])
        
        self.merged.run_with_iobinding(self.binding)
        
        return self.result_image.numpy()


class THA4ORTNonDefaultSessions:
    """THA4 ONNX Runtime implementation for non-default devices
    
    ORT has limitations with non-default devices and OrtValue, so this
    implementation uses a fully merged graph to minimize data transfer.
    """
    def __init__(self, tha4_dir: str, device_id: int, use_eyebrow=True):
        """Initialize THA4 ORT for non-default device
        
        Args:
            tha4_dir: Directory containing THA4 ONNX models
            device_id: GPU device ID (non-zero)
            use_eyebrow: Enable eyebrow pose processing
        """
        self.tha4_dir = tha4_dir
        if 'fp16' in tha4_dir:
            self.dtype = np.float16
        else:
            self.dtype = np.float32
        
        available = ort.get_available_providers()
        if 'CUDAExecutionProvider' in available:
            self.provider = ('CUDAExecutionProvider', {'device_id': device_id})
        elif 'DmlExecutionProvider' in available:
            self.provider = ('DmlExecutionProvider', {'device_id': device_id})
        else:
            raise ValueError('No GPU provider available')
        
        print(f'Using THA4 ORT Non-Default with device_id={device_id}')
        
        self.use_eyebrow = use_eyebrow
        
        # For non-default device, use fully merged model
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Load separate sessions
        self.decomposer = ort.InferenceSession(
            os.path.join(tha4_dir, 'decomposer.onnx'),
            sess_options=options,
            providers=[self.provider])
        
        if not use_eyebrow:
            self.combiner = ort.InferenceSession(
                os.path.join(tha4_dir, 'combiner.onnx'),
                sess_options=options,
                providers=[self.provider])
        
        merge_filename = 'merge.onnx' if use_eyebrow else 'merge_no_eyebrow.onnx'
        self.merged = ort.InferenceSession(
            os.path.join(tha4_dir, merge_filename),
            sess_options=options,
            providers=[self.provider])
        
        # Store intermediate results
        self.decomposed_results = None
        self.combiner_results = None

    def update_image(self, img: np.ndarray):
        """Process input image
        
        Args:
            img: Input image in BGRA format (512x512x4)
        """
        shapes = img.shape
        if len(shapes) != 3 or shapes[0] != 512 or shapes[1] != 512 or shapes[2] != 4:
            raise ValueError('Invalid update image shape')
        
        self.decomposed_results = self.decomposer.run(None, {'input_image': img})
        
        if not self.use_eyebrow:
            self.combiner_results = self.combiner.run(None, {
                'image_prepared': self.decomposed_results[2],
                'eyebrow_background_layer': self.decomposed_results[0],
                'eyebrow_layer': self.decomposed_results[1],
                'eyebrow_pose': np.zeros((1, 12), dtype=self.dtype)
            })

    def inference(self, poses: np.ndarray) -> np.ndarray:
        """Run inference with pose parameters
        
        Args:
            poses: Pose parameters (1, 45)
            
        Returns:
            Output image as numpy array
        """
        poses = poses.astype(self.dtype)
        eyebrow_pose = poses[:, :12]
        face_pose = poses[:, 12:12+27]
        rotation_pose = poses[:, 12+27:]
        
        if self.use_eyebrow:
            inputs = {
                'combiner_eyebrow_background_layer': self.decomposed_results[0],
                'combiner_eyebrow_layer': self.decomposed_results[1],
                'combiner_eyebrow_pose': eyebrow_pose,
                'combiner_image_prepared': self.decomposed_results[2],
                'morpher_image_prepared': self.decomposed_results[2],
                'morpher_face_pose': face_pose,
                'body_morpher_rotation_pose': rotation_pose,
                'upscaler_rotation_pose': rotation_pose
            }
        else:
            inputs = {
                'morpher_im_morpher_crop': self.combiner_results[0],
                'morpher_image_prepared': self.decomposed_results[2],
                'morpher_face_pose': face_pose,
                'body_morpher_rotation_pose': rotation_pose,
                'upscaler_rotation_pose': rotation_pose
            }
        
        result = self.merged.run(None, inputs)
        return result[0]
