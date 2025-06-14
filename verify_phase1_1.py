#!/usr/bin/env python3
"""
验证Phase 1.1构建系统设置是否完成
"""

import os
import sys
import subprocess
from pathlib import Path

# 获取项目根目录
if '__file__' in globals():
    PROJECT_ROOT = Path(__file__).parent
else:
    PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

def check_files():
    """检查必需的文件是否存在"""
    print("🔍 Checking required files...")
    
    required_files = {
        # 核心构建文件
        "CMakeLists.txt": "CMake配置文件",
        "setup.py": "Python包设置文件", 
        "pyproject.toml": "现代Python项目配置",
        "requirements.txt": "Python依赖文件",
        "requirements-dev.txt": "开发依赖文件",
        "Makefile": "便捷构建命令",
        
        # 配置文件
        ".gitignore": "Git忽略文件",
        ".clang-format": "C++代码格式化配置",
        ".pre-commit-config.yaml": "预提交钩子配置", 
        ".editorconfig": "编辑器配置",
        "pytest.ini": "测试配置",
        
        # CI/CD
        ".github/workflows/ci.yml": "GitHub Actions CI配置",
        
        # 文档
        "LICENSE": "许可证文件",
        "README.md": "项目说明文档",
        "CONTRIBUTING.md": "贡献指南",
        "CHANGELOG.md": "更新日志",
        
        # 脚本和工具
        "scripts/build.sh": "构建脚本",
        "tools/setup_helpers/env.py": "环境检查工具",
        "tools/check_env.py": "独立环境检查脚本",
        
        # Python包结构
        "torch/__init__.py": "主模块初始化",
        "torch/nn/__init__.py": "神经网络模块",
        "torch/optim/__init__.py": "优化器模块", 
        "torch/autograd/__init__.py": "自动微分模块",
        "torch/py.typed": "类型提示标记文件",
        
        # 测试
        "test/__init__.py": "测试包初始化",
        "test/test_build_system.py": "构建系统测试",
    }
    
    missing_files = []
    for file_path, description in required_files.items():
        full_path = PROJECT_ROOT / file_path
        if full_path.exists():
            print(f"  ✅ {file_path} ({description})")
        else:
            print(f"  ❌ {file_path} ({description}) - MISSING")
            missing_files.append(file_path)
    
    return len(missing_files) == 0

def check_directories():
    """检查必需的目录结构"""
    print("\n🔍 Checking directory structure...")
    
    required_dirs = {
        "csrc": "C++源码根目录",
        "csrc/api/include": "Python API头文件",
        "csrc/api/src": "Python API源文件",
        "csrc/aten/src/ATen/core": "ATen核心", 
        "csrc/aten/src/ATen/native": "ATen CPU实现",
        "csrc/aten/src/ATen/cuda": "ATen CUDA实现",
        "csrc/autograd": "自动微分C++实现",
        "torch/nn/modules": "神经网络模块",
        "torch/_C": "C扩展绑定目录",
        "test/cpp": "C++测试目录",
        "benchmarks/cpp": "C++性能测试",
        "docs/design": "设计文档",
        "examples": "示例代码",
        "tools/build": "构建工具",
    }
    
    missing_dirs = []
    for dir_path, description in required_dirs.items():
        full_path = PROJECT_ROOT / dir_path
        if full_path.exists():
            print(f"  ✅ {dir_path}/ ({description})")
        else:
            print(f"  ❌ {dir_path}/ ({description}) - MISSING")
            missing_dirs.append(dir_path)
    
    return len(missing_dirs) == 0

def check_build_environment():
    """检查构建环境"""
    print("\n🔍 Checking build environment...")
    
    try:
        from tools.setup_helpers.env import get_build_env
        env = get_build_env()
        
        # 检查Python版本
        python_version = tuple(map(int, env['python_version'].split('.')))
        if python_version >= (3, 8):
            print(f"  ✅ Python {env['python_version']} (>= 3.8)")
        else:
            print(f"  ❌ Python {env['python_version']} (需要 >= 3.8)")
            return False
        
        # 检查CMake
        if env.get('cmake_version'):
            print(f"  ✅ CMake {env['cmake_version']}")
        else:
            print(f"  ❌ CMake not found")
            return False
        
        # 检查编译器
        compiler = env['compiler']
        if compiler['type']:
            print(f"  ✅ {compiler['type']} {compiler['version']}")
        else:
            print(f"  ❌ C++ compiler not found")
            return False
        
        # 检查依赖
        deps = env['dependencies']
        for dep_name, version in deps.items():
            if version:
                print(f"  ✅ {dep_name} {version}")
            else:
                print(f"  ❌ {dep_name} not found")
                return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Environment check failed: {e}")
        return False

def check_basic_functionality():
    """检查基本功能"""
    print("\n🔍 Checking basic functionality...")
    
    try:
        # 检查能否导入torch
        import torch
        print(f"  ✅ torch import successful (v{torch.__version__})")
        
        # 检查子模块导入
        import torch.nn
        import torch.optim  
        import torch.autograd
        print(f"  ✅ All submodules import successfully")
        
        # 检查占位符函数
        try:
            torch.tensor([1, 2, 3])
            print(f"  ❌ torch.tensor should raise NotImplementedError")
            return False
        except NotImplementedError:
            print(f"  ✅ torch.tensor correctly raises NotImplementedError")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Basic functionality check failed: {e}")
        return False

def main():
    """主检查函数"""
    print("=" * 50)
    print("🚀 Tiny-Torch Phase 1.1 验证")
    print("=" * 50)
    
    checks = [
        ("文件结构", check_files),
        ("目录结构", check_directories), 
        ("构建环境", check_build_environment),
        ("基本功能", check_basic_functionality),
    ]
    
    all_passed = True
    for check_name, check_func in checks:
        try:
            result = check_func()
            if result:
                print(f"\n✅ {check_name} 检查通过")
            else:
                print(f"\n❌ {check_name} 检查失败")
                all_passed = False
        except Exception as e:
            print(f"\n❌ {check_name} 检查出错: {e}")
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 Phase 1.1 构建系统设置完成！")
        print("\n📋 下一步:")
        print("  1. 可以运行 'make build' 尝试构建")
        print("  2. 运行 'make test' 执行测试")
        print("  3. 开始实施 Phase 1.2: 张量核心库(ATen)")
        print("=" * 50)
        return 0
    else:
        print("❌ Phase 1.1 还未完全完成，请检查上述问题")
        print("=" * 50)
        return 1

if __name__ == "__main__":
    sys.exit(main())
