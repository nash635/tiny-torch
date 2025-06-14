#!/usr/bin/env python3
"""
Setup script for Tiny-Torch
参考 pytorch/setup.py 的实现
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
from setuptools import setup, Extension, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11

# Project metadata
PACKAGE_NAME = "tiny-torch"
VERSION = "0.1.0"
DESCRIPTION = "A PyTorch-inspired deep learning framework"
AUTHOR = "Tiny-Torch Contributors"
EMAIL = "tiny-torch@example.com"
URL = "https://github.com/your-username/tiny-torch"

# Build configuration
DEBUG = os.getenv("DEBUG", "0") == "1"
WITH_CUDA = os.getenv("WITH_CUDA", "1") == "1"
WITH_MKL = os.getenv("WITH_MKL", "0") == "1"
WITH_OPENMP = os.getenv("WITH_OPENMP", "1") == "1"
VERBOSE = os.getenv("VERBOSE", "0") == "1"

# Paths
ROOT_DIR = Path(__file__).parent.absolute()
CSRC_DIR = ROOT_DIR / "csrc"
BUILD_DIR = ROOT_DIR / "build"

def check_env():
    """检查构建环境"""
    print("=== Tiny-Torch Build Environment ===")
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.machine()}")
    print(f"WITH_CUDA: {WITH_CUDA}")
    print(f"WITH_MKL: {WITH_MKL}")
    print(f"WITH_OPENMP: {WITH_OPENMP}")
    print(f"DEBUG: {DEBUG}")
    print("=====================================")

def get_cuda_version():
    """获取CUDA版本"""
    try:
        result = subprocess.run(
            ["nvcc", "--version"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        for line in result.stdout.split('\n'):
            if 'release' in line:
                version = line.split('release')[1].split(',')[0].strip()
                return version
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return None

def get_extensions():
    """构建扩展模块"""
    extensions = []
    
    # 源文件
    sources = [
        # Python API bindings
        "csrc/api/src/python_bindings.cpp",
        "csrc/api/src/tensor_api.cpp",
        "csrc/api/src/autograd_api.cpp",
        
        # ATen core
        "csrc/aten/src/ATen/core/Tensor.cpp",
        "csrc/aten/src/ATen/core/TensorImpl.cpp",
        "csrc/aten/src/ATen/core/Storage.cpp",
        "csrc/aten/src/ATen/core/Allocator.cpp",
        
        # Native operations
        "csrc/aten/src/ATen/native/BinaryOps.cpp",
        "csrc/aten/src/ATen/native/UnaryOps.cpp",
        "csrc/aten/src/ATen/native/LinearAlgebra.cpp",
        "csrc/aten/src/ATen/native/Activation.cpp",
        "csrc/aten/src/ATen/native/Reduction.cpp",
        
        # TH layer
        "csrc/aten/src/TH/THTensor.cpp",
        "csrc/aten/src/TH/THStorage.cpp",
        "csrc/aten/src/TH/THAllocator.cpp",
        
        # Autograd
        "csrc/autograd/engine.cpp",
        "csrc/autograd/function.cpp",
        "csrc/autograd/variable.cpp",
        "csrc/autograd/functions/basic_ops.cpp",
    ]
    
    # CUDA源文件
    if WITH_CUDA:
        cuda_sources = [
            "csrc/aten/src/ATen/cuda/CUDAContext.cu",
            "csrc/aten/src/ATen/native/cuda/BinaryOps.cu",
            "csrc/aten/src/ATen/native/cuda/UnaryOps.cu",
            "csrc/aten/src/ATen/native/cuda/LinearAlgebra.cu",
            "csrc/aten/src/ATen/native/cuda/Activation.cu",
            "csrc/aten/src/ATen/native/cuda/Reduction.cu",
        ]
        sources.extend(cuda_sources)
    
    # 包含目录
    include_dirs = [
        str(CSRC_DIR),
        str(CSRC_DIR / "api" / "include"),
        str(CSRC_DIR / "aten" / "include"),
        pybind11.get_include(),
    ]
    
    # 编译器选项
    cxx_flags = ["-std=c++17", "-fPIC"]
    if DEBUG:
        cxx_flags.extend(["-g", "-O0", "-DDEBUG"])
    else:
        cxx_flags.extend(["-O3", "-DNDEBUG"])
    
    # 平台特定选项
    if platform.system() == "Darwin":  # macOS
        cxx_flags.extend(["-stdlib=libc++"])
    elif platform.system() == "Linux":
        cxx_flags.extend(["-Wall", "-Wextra"])
    
    # 链接库
    libraries = []
    library_dirs = []
    
    # CUDA支持
    if WITH_CUDA:
        cuda_version = get_cuda_version()
        if cuda_version:
            print(f"Found CUDA version: {cuda_version}")
            cxx_flags.append("-DWITH_CUDA")
            
            # CUDA路径
            cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
            include_dirs.append(f"{cuda_home}/include")
            library_dirs.append(f"{cuda_home}/lib64")
            
            # CUDA库
            libraries.extend(["cudart", "cublas", "curand"])
            
            # 尝试找到cuDNN
            cudnn_lib_dir = f"{cuda_home}/lib64"
            if os.path.exists(f"{cudnn_lib_dir}/libcudnn.so"):
                libraries.append("cudnn")
        else:
            print("CUDA requested but nvcc not found, disabling CUDA")
            WITH_CUDA = False
    
    # OpenMP支持
    if WITH_OPENMP:
        cxx_flags.append("-DWITH_OPENMP")
        if platform.system() == "Darwin":
            # macOS通常使用clang，可能需要单独安装libomp
            try:
                result = subprocess.run(
                    ["brew", "--prefix", "libomp"], 
                    capture_output=True, 
                    text=True
                )
                if result.returncode == 0:
                    omp_prefix = result.stdout.strip()
                    include_dirs.append(f"{omp_prefix}/include")
                    library_dirs.append(f"{omp_prefix}/lib")
                    libraries.append("omp")
                    cxx_flags.append("-Xpreprocessor")
                    cxx_flags.append("-fopenmp")
            except FileNotFoundError:
                print("Warning: libomp not found via brew")
        else:
            cxx_flags.append("-fopenmp")
            libraries.append("gomp")
    
    # MKL支持
    if WITH_MKL:
        mkl_root = os.environ.get("MKLROOT")
        if mkl_root:
            cxx_flags.append("-DWITH_MKL")
            include_dirs.append(f"{mkl_root}/include")
            library_dirs.append(f"{mkl_root}/lib/intel64")
            libraries.extend(["mkl_intel_lp64", "mkl_sequential", "mkl_core"])
        else:
            print("MKL requested but MKLROOT not set, falling back to standard BLAS")
            libraries.extend(["blas", "lapack"])
    else:
        # 标准BLAS/LAPACK
        libraries.extend(["blas", "lapack"])
    
    # 创建扩展
    ext = Pybind11Extension(
        "torch._C",
        sources=sources,
        include_dirs=include_dirs,
        libraries=libraries,
        library_dirs=library_dirs,
        cxx_std=17,
        extra_compile_args=cxx_flags,
    )
    
    extensions.append(ext)
    return extensions

def get_requirements():
    """读取依赖"""
    requirements = []
    req_file = ROOT_DIR / "requirements.txt"
    if req_file.exists():
        with open(req_file, 'r') as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return requirements

def get_long_description():
    """读取长描述"""
    readme_file = ROOT_DIR / "README.md"
    if readme_file.exists():
        with open(readme_file, 'r', encoding='utf-8') as f:
            return f.read()
    return DESCRIPTION

class CustomBuildExt(build_ext):
    """自定义构建类"""
    
    def build_extensions(self):
        # 设置编译器选项
        if self.compiler.compiler_type == 'unix':
            for ext in self.extensions:
                # 添加-fvisibility=hidden以减少符号暴露
                ext.extra_compile_args.append('-fvisibility=hidden')
        
        super().build_extensions()
    
    def run(self):
        # 确保构建目录存在
        BUILD_DIR.mkdir(exist_ok=True)
        super().run()

if __name__ == "__main__":
    check_env()
    
    setup(
        name=PACKAGE_NAME,
        version=VERSION,
        description=DESCRIPTION,
        long_description=get_long_description(),
        long_description_content_type="text/markdown",
        author=AUTHOR,
        author_email=EMAIL,
        url=URL,
        
        packages=find_packages(exclude=["test*", "benchmarks*"]),
        ext_modules=get_extensions(),
        cmdclass={"build_ext": CustomBuildExt},
        
        install_requires=get_requirements(),
        python_requires=">=3.8",
        
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: BSD License",
            "Operating System :: POSIX :: Linux",
            "Operating System :: MacOS :: MacOS X",
            "Programming Language :: C++",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
        
        zip_safe=False,
        include_package_data=True,
    )
