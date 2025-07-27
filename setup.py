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
import shutil

# Project metadata
PACKAGE_NAME = "tiny-torch"
VERSION = "0.1.0"
DESCRIPTION = "A PyTorch-inspired deep learning framework"
AUTHOR = "Tiny-Torch Contributors"
EMAIL = "tiny-torch@example.com"
URL = "https://github.com/your-username/tiny-torch"

# Build configuration
DEBUG = os.getenv("DEBUG", "0") == "1"
WITH_CUDA = os.getenv("WITH_CUDA", "1") == "1"  # Default to True, but will be auto-detected
WITH_MKL = os.getenv("WITH_MKL", "0") == "1"
WITH_OPENMP = os.getenv("WITH_OPENMP", "1") == "1"
VERBOSE = os.getenv("VERBOSE", "0") == "1"

# Paths
ROOT_DIR = Path(__file__).parent.absolute()
CSRC_DIR = ROOT_DIR / "csrc"
BUILD_DIR = ROOT_DIR / "build"

def test_ninja_compatibility():
    """测试Ninja是否真正可用"""
    if not shutil.which("ninja"):
        return False
        
    try:
        # 测试ninja版本命令
        result = subprocess.run(
            ["ninja", "--version"], 
            capture_output=True, 
            text=True, 
            check=True,
            timeout=5
        )
        version = result.stdout.strip()
        if VERBOSE:
            print(f"Ninja version: {version}")
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return False

def should_use_ninja():
    """智能判断是否应该使用Ninja"""
    # 检查环境变量设置
    if os.getenv("USE_NINJA", "1") == "0":
        return False
        
    # 检查Ninja兼容性
    if not test_ninja_compatibility():
        if VERBOSE:
            print("Ninja compatibility test failed, falling back to Make")
        return False
        
    return True

# 设置USE_NINJA变量
USE_NINJA = should_use_ninja()

def setup_color_output():
    """设置强制彩色输出的环境变量"""
    # 设置各种工具的彩色输出环境变量
    color_env = {
        'FORCE_COLOR': '1',
        'CMAKE_COLOR_MAKEFILE': 'ON',
        'CMAKE_COLOR_DIAGNOSTICS': 'ON', 
        'CMAKE_FORCE_COLORED_OUTPUT': 'ON',
        'CLICOLOR_FORCE': '1',
        'NINJA_STATUS': '[%f/%t] ',
        # 为编译器设置彩色输出
        'CFLAGS': os.environ.get('CFLAGS', '') + ' -fdiagnostics-color=always',
        'CXXFLAGS': os.environ.get('CXXFLAGS', '') + ' -fdiagnostics-color=always',
    }
    
    # 更新环境变量
    for key, value in color_env.items():
        os.environ[key] = value
    
    if VERBOSE:
        print("[INFO] Enabled forced color output for build tools")

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
    print(f"USE_NINJA: {USE_NINJA}")
    print("=====================================")

def get_sources():
    """获取所有C++源文件列表"""
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
    
    return sources

def use_cmake_build():
    """检查是否应该使用CMake构建 (当项目变复杂时)"""
    # 当源文件数量超过阈值时，使用CMake+ninja构建
    sources = get_sources()
    return len(sources) > 20 or WITH_CUDA

def cmake_build():
    """使用CMake+ninja构建"""
    print("Using CMake+ninja build system...")
    
    # 设置彩色输出
    setup_color_output()
    
    # 创建构建目录
    cmake_build_dir = BUILD_DIR / "cmake"
    cmake_build_dir.mkdir(parents=True, exist_ok=True)
    
    # CMake配置命令
    cmake_args = [
        f"-DCMAKE_BUILD_TYPE={'Debug' if DEBUG else 'Release'}",
        f"-DWITH_CUDA={'ON' if WITH_CUDA else 'OFF'}",
        f"-DWITH_MKL={'ON' if WITH_MKL else 'OFF'}",
        f"-DWITH_OPENMP={'ON' if WITH_OPENMP else 'OFF'}",
        f"-DCMAKE_INSTALL_PREFIX={ROOT_DIR}/tiny_torch",
    ]
    
    # 使用ninja如果可用
    if USE_NINJA:
        cmake_args.append("-GNinja")
        print("Using Ninja generator for faster builds")
    
    # 运行CMake配置
    print(f"Running CMake configure in {cmake_build_dir}")
    try:
        result = subprocess.run([
            "cmake", str(ROOT_DIR), *cmake_args
        ], cwd=cmake_build_dir, capture_output=True, text=True, check=True)
        print("[PASS] CMake configure successful")
        if VERBOSE:
            print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"[FAIL] CMake configure failed: {e}")
        if e.stderr:
            print("Error output:", e.stderr)
        if e.stdout:
            print("Output:", e.stdout)
        return False
    
    # 运行构建
    print("Running build...")
    try:
        if USE_NINJA:
            build_cmd = ["ninja"]
            if VERBOSE:
                build_cmd.append("-v")
        else:
            build_cmd = ["make", "-j"]
            if VERBOSE:
                build_cmd.append("VERBOSE=1")
                
        result = subprocess.run(build_cmd, cwd=cmake_build_dir, capture_output=True, text=True, check=True)
        print("[PASS] Build successful")
        if VERBOSE:
            print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"[FAIL] Build failed: {e}")
        if e.stderr:
            print("Error output:", e.stderr)
        if e.stdout:
            print("Output:", e.stdout)
        
        # 如果Ninja失败，尝试降级到Make
        if USE_NINJA:
            print("[RETRY] Ninja build failed, trying fallback to Make...")
            try:
                # 重新配置使用Make
                cmake_args_make = [arg for arg in cmake_args if not arg.startswith("-GNinja")]
                subprocess.run([
                    "cmake", str(ROOT_DIR), *cmake_args_make
                ], cwd=cmake_build_dir, check=True)
                
                # 用Make构建
                subprocess.run(["make", "-j"], cwd=cmake_build_dir, check=True)
                print("[PASS] Make fallback build successful")
            except subprocess.CalledProcessError as fallback_e:
                print(f"[FAIL] Make fallback also failed: {fallback_e}")
                return False
        else:
            return False
    
    print("CMake build completed successfully!")
    return True

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

def check_cuda_available():
    """检查CUDA是否可用"""
    try:
        # 检查nvcc是否存在
        result = subprocess.run(
            ["nvcc", "--version"], 
            capture_output=True, 
            text=True, 
            check=True,
            timeout=10
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False

def auto_detect_cuda():
    """自动检测是否应该启用CUDA"""
    global WITH_CUDA
    
    # 如果环境变量明确设置了WITH_CUDA=0，则禁用CUDA
    if os.getenv("WITH_CUDA", "").lower() in ["0", "false", "off", "no"]:
        WITH_CUDA = False
        print("CUDA disabled by environment variable")
        return
    
    # 如果环境变量明确设置了WITH_CUDA=1，则尝试启用CUDA
    if os.getenv("WITH_CUDA", "").lower() in ["1", "true", "on", "yes"]:
        if check_cuda_available():
            WITH_CUDA = True
            print("CUDA enabled by environment variable and is available")
        else:
            WITH_CUDA = False
            print("Warning: CUDA requested by environment variable but nvcc not found. Falling back to CPU-only build.")
        return
    
    # 如果没有明确设置，则自动检测
    if check_cuda_available():
        WITH_CUDA = True
        print("CUDA automatically detected and enabled")
    else:
        WITH_CUDA = False
        print("CUDA not available, using CPU-only build")

def use_cmake_build():
    """检查是否应该使用CMake构建 (当项目变复杂时)"""
    # 当源文件数量超过阈值时，使用CMake+ninja构建
    sources = get_sources()
    return len(sources) > 20 or WITH_CUDA

def cmake_build():
    """使用CMake+ninja构建"""
    print("Using CMake+ninja build system...")
    
    # 设置彩色输出
    setup_color_output()
    
    # 创建构建目录
    cmake_build_dir = BUILD_DIR / "cmake"
    cmake_build_dir.mkdir(parents=True, exist_ok=True)
    
    # CMake配置命令
    cmake_args = [
        f"-DCMAKE_BUILD_TYPE={'Debug' if DEBUG else 'Release'}",
        f"-DWITH_CUDA={'ON' if WITH_CUDA else 'OFF'}",
        f"-DWITH_MKL={'ON' if WITH_MKL else 'OFF'}",
        f"-DWITH_OPENMP={'ON' if WITH_OPENMP else 'OFF'}",
        f"-DCMAKE_INSTALL_PREFIX={ROOT_DIR}/tiny_torch",
    ]
    
    # 使用ninja如果可用
    if USE_NINJA:
        cmake_args.append("-GNinja")
        print("Using Ninja generator for faster builds")
    
    # 运行CMake配置
    subprocess.run([
        "cmake", str(ROOT_DIR), *cmake_args
    ], cwd=cmake_build_dir, check=True)
    
    # 运行构建
    if USE_NINJA:
        subprocess.run(["ninja"], cwd=cmake_build_dir, check=True)
    else:
        subprocess.run(["make", "-j"], cwd=cmake_build_dir, check=True)
    
    print("CMake build completed successfully!")

def get_extensions():
    """获取扩展模块配置"""
    global WITH_CUDA  # Declare as global to avoid UnboundLocalError
    extensions = []
    
    # 检查是否应该使用CMake构建
    if use_cmake_build():
        print("Complex project detected, using CMake+ninja build system")
        # 对于复杂项目，我们仍然返回setuptools扩展，但主要工作由CMake完成
        # 这是一个过渡方案，确保setuptools仍能正确处理生成的库
        pass
    
    # 获取源文件列表
    sources = get_sources()
    
    # CUDA源文件 - 暂时禁用以使用CMake构建CUDA代码
    # TODO: 在Phase 1.2中实现CUDA支持的Python绑定
    # if WITH_CUDA:
    #     cuda_sources = [
    #         "csrc/aten/src/ATen/cuda/CUDAContext.cu",
    #         "csrc/aten/src/ATen/native/cuda/BinaryOps.cu",
    #         "csrc/aten/src/ATen/native/cuda/UnaryOps.cu",
    #         "csrc/aten/src/ATen/native/cuda/LinearAlgebra.cu",
    #         "csrc/aten/src/ATen/native/cuda/Activation.cu",
    #         "csrc/aten/src/ATen/native/cuda/Reduction.cu",
    #     ]
    #     sources.extend(cuda_sources)
    
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
    
    # Linux 系统特定选项
    if platform.system() == "Linux":
        cxx_flags.extend(["-Wall", "-Wextra"])
    else:
        raise RuntimeError(f"Unsupported platform: {platform.system()}. Only Linux is supported.")
    
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
    
    # OpenMP支持 - 仅限 Linux
    if WITH_OPENMP:
        cxx_flags.append("-DWITH_OPENMP")
        cxx_flags.append("-fopenmp")
        libraries.append("gomp")
    
    # BLAS/LAPACK 支持 - 使用更灵活的方法
    if WITH_MKL:
        mkl_root = os.environ.get("MKLROOT")
        if mkl_root:
            cxx_flags.append("-DWITH_MKL")
            include_dirs.append(f"{mkl_root}/include")
            library_dirs.append(f"{mkl_root}/lib/intel64")
            libraries.extend(["mkl_intel_lp64", "mkl_sequential", "mkl_core"])
        else:
            print("MKL requested but MKLROOT not set, skipping BLAS/LAPACK")
    else:
        # 尝试找到BLAS/LAPACK但不强制要求
        print("Warning: Building without BLAS/LAPACK libraries, tensor operations will be CPU-only")
    
    # 创建扩展
    ext = Pybind11Extension(
        "tiny_torch._C",
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
        # 设置彩色输出
        setup_color_output()
        
        # 确保构建目录存在
        BUILD_DIR.mkdir(exist_ok=True)
        super().run()

if __name__ == "__main__":
    # Auto-detect CUDA availability before any other checks
    auto_detect_cuda()
    
    check_env()
    
    # 如果源文件较多或需要CUDA支持，使用CMake构建
    if use_cmake_build():
        cmake_build()

# Setup configuration (always available for pip)
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
    ext_modules=get_extensions() if not use_cmake_build() else [],
    cmdclass={"build_ext": CustomBuildExt},
    
    # Dependencies are now managed in pyproject.toml to avoid duplication
    # install_requires=get_requirements(),  # Commented out to avoid setuptools warning
    python_requires=">=3.8",
    
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: POSIX :: Linux",
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
