#!/usr/bin/env python3
"""
test_build_system.py
构建系统的基础测试
"""

import os
import sys
import subprocess
from pathlib import Path

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

class BuildSystemTestFailure(Exception):
    """构建系统测试失败异常"""
    pass

def test_import_torch():
    """测试能否导入torch模块"""
    try:
        import torch
        if torch.__version__ != "0.1.0":
            raise BuildSystemTestFailure(f"Expected version 0.1.0, got {torch.__version__}")
        print(f"✓ Successfully imported torch v{torch.__version__}")
    except ImportError as e:
        raise BuildSystemTestFailure(f"Failed to import torch: {e}")

def test_submodule_imports():
    """测试子模块导入"""
    try:
        import torch.nn
        import torch.optim
        import torch.autograd
        print("✓ Successfully imported all submodules")
    except ImportError as e:
        raise BuildSystemTestFailure(f"Failed to import submodules: {e}")

def test_placeholder_functions():
    """测试占位符函数是否正确抛出NotImplementedError"""
    import torch
    
    try:
        torch.tensor([1, 2, 3])
        raise BuildSystemTestFailure("torch.tensor should raise NotImplementedError")
    except NotImplementedError:
        pass
    
    try:
        torch.add(1, 2)
        raise BuildSystemTestFailure("torch.add should raise NotImplementedError")
    except NotImplementedError:
        pass
    
    try:
        torch.nn.Module()
        raise BuildSystemTestFailure("torch.nn.Module should raise NotImplementedError")
    except NotImplementedError:
        pass
    
    try:
        torch.optim.SGD([])
        raise BuildSystemTestFailure("torch.optim.SGD should raise NotImplementedError")
    except NotImplementedError:
        pass
    
    print("✓ All placeholder functions correctly raise NotImplementedError")

def test_build_environment():
    """测试构建环境"""
    try:
        from tools.setup_helpers.env import get_build_env
        env = get_build_env()
        
        # 检查基本环境
        assert env['python_version']
        assert env['platform']
        assert env['architecture']
        
        print(f"✓ Build environment check passed")
        print(f"  Python: {env['python_version']}")
        print(f"  Platform: {env['platform']}")
        print(f"  Architecture: {env['architecture']}")
        
    except Exception as e:
        raise BuildSystemTestFailure(f"Build environment check failed: {e}")

def test_cmake_availability():
    """测试CMake是否可用"""
    try:
        result = subprocess.run(
            ["cmake", "--version"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        version_line = result.stdout.split('\n')[0]
        print(f"✓ CMake available: {version_line}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠ CMake not available (skipping)")

def test_project_structure():
    """测试项目结构"""
    required_dirs = [
        "csrc",
        "torch",
        "test", 
        "tools",
        "docs",
        "examples",
        "benchmarks"
    ]
    
    for dir_name in required_dirs:
        dir_path = PROJECT_ROOT / dir_name
        if not dir_path.exists():
            raise BuildSystemTestFailure(f"Required directory {dir_name} not found")
    
    required_files = [
        "CMakeLists.txt",
        "setup.py",
        "pyproject.toml",
        "requirements.txt",
        "Makefile",
        ".gitignore",
        ".clang-format",
        "LICENSE"
    ]
    
    for file_name in required_files:
        file_path = PROJECT_ROOT / file_name
        if not file_path.exists():
            raise BuildSystemTestFailure(f"Required file {file_name} not found")
    
    print("✓ Project structure is complete")

if __name__ == "__main__":
    # 直接运行测试
    print("Running build system tests...")
    test_import_torch()
    test_submodule_imports()
    test_placeholder_functions()
    test_build_environment()
    test_cmake_availability()
    test_project_structure()
    print("All tests passed! ✓")
