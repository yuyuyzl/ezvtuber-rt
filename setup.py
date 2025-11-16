"""Setup script for ezvtuber-rt library.

A real-time virtual tuber animation library powered by TensorRT and ONNX.
Provides ONNX models and integrated code for THA3, RIFE, Waifu2x, and Real-ESRGAN.
"""

from setuptools import setup, find_packages, Extension
from pathlib import Path
import sys
import platform
import os

# Read the README file for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Core dependencies required for all installations
core_dependencies = [
    "numpy",
    "opencv-python",
    "tqdm",
    "onnx",
    "turbojpeg",
    "onnxruntime-directml",
    "pycuda",
    "pybind11>=2.6.0"
]

# Define the C++ extension module for RGBA_utils
def get_rgba_utils_extension():
    """Configure the RGBA_utils C++ extension with pybind11"""
    
    # Get pybind11 include path
    try:
        import pybind11
        pybind11_include = pybind11.get_include()
    except ImportError as e:
        raise RuntimeError(
            "pybind11 is required to build this package. "
            "It should be automatically installed via pyproject.toml. "
            "If you see this error, try: pip install pybind11>=2.6.0"
        ) from e
    
    # Platform-specific compiler flags
    extra_compile_args = []
    extra_link_args = []
    
    if platform.system() == "Windows":
        # MSVC compiler flags
        extra_compile_args = [
            "/O2",           # Optimization
            "/openmp",       # OpenMP support
            "/arch:SSE2",    # SSE2 instructions
            "/std:c++14",    # C++14 standard
        ]
    else:
        # GCC/Clang compiler flags
        extra_compile_args = [
            "-O3",           # Optimization
            "-fopenmp",      # OpenMP support
            "-msse2",        # SSE2 instructions
            "-std=c++14",    # C++14 standard
        ]
        extra_link_args = ["-fopenmp"]
    
    return Extension(
        "ezvtb_rt.rgba_utils",  # Module name
        sources=["ezvtb_rt/cpp/RGBA_utils.cpp"],
        include_dirs=[pybind11_include],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++"
    )

# Define the C++ extension module for ffmpeg_codec
def get_ffmpeg_codec_extension():
    """Configure the ffmpeg_codec C++ extension with FFmpeg libraries"""
    
    # Check for FFMPEG_DIR environment variable
    ffmpeg_dir = os.environ.get("FFMPEG_DIR")
    if not ffmpeg_dir:
        raise RuntimeError(
            "FFMPEG_DIR environment variable is not set. "
            "Please set FFMPEG_DIR to point to your FFmpeg installation directory. "
            "Example: set FFMPEG_DIR=C:\\path\\to\\ffmpeg"
        )
    
    ffmpeg_path = Path(ffmpeg_dir)
    if not ffmpeg_path.exists():
        raise RuntimeError(
            f"FFMPEG_DIR points to non-existent directory: {ffmpeg_dir}"
        )
    
    # Get pybind11 include path
    try:
        import pybind11
        pybind11_include = pybind11.get_include()
    except ImportError as e:
        raise RuntimeError(
            "pybind11 is required to build this package. "
            "It should be automatically installed via pyproject.toml. "
            "If you see this error, try: pip install pybind11>=2.6.0"
        ) from e
    
    # Set up include and library directories
    include_dirs = [
        pybind11_include,
        str(ffmpeg_path / "include")
    ]
    
    library_dirs = [
        str(ffmpeg_path / "lib")
    ]
    
    # FFmpeg libraries to link
    libraries = [
        "avcodec",
        "avutil",
        "swscale"
    ]
    
    # Platform-specific compiler flags
    extra_compile_args = []
    extra_link_args = []
    
    if platform.system() == "Windows":
        # MSVC compiler flags
        extra_compile_args = [
            "/O2",           # Optimization
            "/std:c++14",    # C++14 standard
        ]
    else:
        # GCC/Clang compiler flags
        extra_compile_args = [
            "-O3",           # Optimization
            "-std=c++14",    # C++14 standard
        ]
    
    return Extension(
        "ezvtb_rt.ffmpeg_codec",  # Module name
        sources=["ezvtb_rt/cpp/ffmpeg_codec.cpp"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++"
    )

setup(
    name="ezvtuber-rt",
    version="0.1.0",
    author="zpeng11",
    author_email="",
    description="Real-time virtual tuber animation library with TensorRT and ONNX support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zpeng11/ezvtuber-rt",
    project_urls={
        "Bug Tracker": "https://github.com/zpeng11/ezvtuber-rt/issues",
        "Source Code": "https://github.com/zpeng11/ezvtuber-rt",
    },
    packages=find_packages(exclude=["test", "test.*"]),
    ext_modules=[
        get_rgba_utils_extension(),
        get_ffmpeg_codec_extension()
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires=">=3.8",
    install_requires=core_dependencies,
    include_package_data=True,
    package_data={
        "ezvtb_rt": ["*.py"],
    },
    zip_safe=False,
    keywords=[
        "vtuber",
        "real-time",
        "animation",
        "tensorrt",
        "onnx",
        "deeplearning",
        "computer-vision",
        "gpu-acceleration",
        "frame-interpolation",
        "super-resolution",
    ],
    entry_points={
        # Add console scripts if needed in the future
        # "console_scripts": [
        #     "ezvtuber-rt=ezvtb_rt.cli:main",
        # ],
    },
)
