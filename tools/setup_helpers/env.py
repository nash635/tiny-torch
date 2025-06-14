"""
tools/setup_helpers/env.py
构建环境检查和配置工具 (参考 pytorch/tools/setup_helpers/env.py)
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    if version < (3, 8):
        raise RuntimeError(f"Python 3.8+ required, got {version.major}.{version.minor}")
    return f"{version.major}.{version.minor}.{version.micro}"

def check_cmake():
    """检查CMake版本"""
    try:
        result = subprocess.run(
            ["cmake", "--version"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        version_line = result.stdout.split('\n')[0]
        version = version_line.split()[2]
        
        # 检查版本是否满足要求
        major, minor = map(int, version.split('.')[:2])
        if (major, minor) < (3, 18):
            raise RuntimeError(f"CMake 3.18+ required, got {version}")
        return version
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError("CMake not found. Please install CMake 3.18+")

def check_cuda():
    """检查CUDA环境"""
    cuda_info = {
        'available': False,
        'version': None,
        'home': None,
        'nvcc_path': None
    }
    
    try:
        # 检查nvcc
        result = subprocess.run(
            ["nvcc", "--version"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        
        # 解析版本
        for line in result.stdout.split('\n'):
            if 'release' in line:
                version = line.split('release')[1].split(',')[0].strip()
                cuda_info['version'] = version
                break
        
        # 检查CUDA_HOME
        cuda_home = os.environ.get('CUDA_HOME')
        if not cuda_home:
            # 尝试常见路径
            common_paths = ['/usr/local/cuda', '/opt/cuda']
            for path in common_paths:
                if os.path.exists(path):
                    cuda_home = path
                    break
        
        if cuda_home and os.path.exists(cuda_home):
            cuda_info['home'] = cuda_home
            cuda_info['nvcc_path'] = os.path.join(cuda_home, 'bin', 'nvcc')
            cuda_info['available'] = True
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    return cuda_info

def check_compiler():
    """检查C++编译器"""
    compiler_info = {
        'type': None,
        'version': None,
        'path': None
    }
    
    # 检查GCC
    try:
        result = subprocess.run(
            ["gcc", "--version"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        version_line = result.stdout.split('\n')[0]
        if 'gcc' in version_line.lower():
            compiler_info['type'] = 'gcc'
            # 提取版本号
            parts = version_line.split()
            for part in parts:
                if part.replace('.', '').isdigit():
                    compiler_info['version'] = part
                    break
            compiler_info['path'] = subprocess.run(
                ["which", "gcc"], capture_output=True, text=True
            ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # 检查Clang (macOS默认)
    if not compiler_info['type']:
        try:
            result = subprocess.run(
                ["clang", "--version"], 
                capture_output=True, 
                text=True, 
                check=True
            )
            version_line = result.stdout.split('\n')[0]
            if 'clang' in version_line.lower():
                compiler_info['type'] = 'clang'
                # 提取版本号
                parts = version_line.split()
                for part in parts:
                    if 'version' in part:
                        idx = parts.index(part)
                        if idx + 1 < len(parts):
                            compiler_info['version'] = parts[idx + 1]
                        break
                compiler_info['path'] = subprocess.run(
                    ["which", "clang"], capture_output=True, text=True
                ).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
    
    return compiler_info

def check_dependencies():
    """检查依赖库"""
    deps = {}
    
    # 检查BLAS/LAPACK
    try:
        import numpy
        deps['numpy'] = numpy.__version__
    except ImportError:
        deps['numpy'] = None
    
    # 检查pybind11
    try:
        import pybind11
        deps['pybind11'] = pybind11.__version__
    except ImportError:
        deps['pybind11'] = None
    
    return deps

def get_build_env():
    """获取完整的构建环境信息"""
    env_info = {
        'platform': platform.platform(),
        'architecture': platform.machine(),
        'python_version': check_python_version(),
        'cmake_version': None,
        'cuda': check_cuda(),
        'compiler': check_compiler(),
        'dependencies': check_dependencies(),
        'build_flags': {
            'WITH_CUDA': os.environ.get('WITH_CUDA', '0') == '1',
            'WITH_MKL': os.environ.get('WITH_MKL', '0') == '1',
            'WITH_OPENMP': os.environ.get('WITH_OPENMP', '1') == '1',
            'DEBUG': os.environ.get('DEBUG', '0') == '1',
        }
    }
    
    try:
        env_info['cmake_version'] = check_cmake()
    except RuntimeError as e:
        env_info['cmake_error'] = str(e)
    
    return env_info

def print_env_info():
    """打印环境信息"""
    env = get_build_env()
    
    print("=== Tiny-Torch Build Environment ===")
    print(f"Platform: {env['platform']}")
    print(f"Architecture: {env['architecture']}")
    print(f"Python: {env['python_version']}")
    
    if 'cmake_version' in env:
        print(f"CMake: {env['cmake_version']}")
    else:
        print(f"CMake: ERROR - {env.get('cmake_error', 'Unknown error')}")
    
    # 编译器信息
    compiler = env['compiler']
    if compiler['type']:
        print(f"Compiler: {compiler['type']} {compiler['version']} ({compiler['path']})")
    else:
        print("Compiler: Not found")
    
    # CUDA信息
    cuda = env['cuda']
    if cuda['available']:
        print(f"CUDA: {cuda['version']} ({cuda['home']})")
    else:
        print("CUDA: Not available")
    
    # 依赖信息
    deps = env['dependencies']
    print(f"NumPy: {deps.get('numpy', 'Not found')}")
    print(f"pybind11: {deps.get('pybind11', 'Not found')}")
    
    # 构建标志
    flags = env['build_flags']
    print("\nBuild flags:")
    for flag, value in flags.items():
        print(f"  {flag}: {value}")
    
    print("====================================")
    
    return env

if __name__ == "__main__":
    print_env_info()
