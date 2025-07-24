"""
tools/check_env.py
检查构建环境的独立脚本
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def main():
    """检查并打印构建环境信息"""
    try:
        from tools.setup_helpers.env import print_env_info
        env = print_env_info()
        
        # 检查关键依赖
        missing_deps = []
        
        if not env.get('cmake_version'):
            missing_deps.append("CMake 3.18+")
        
        if not env['compiler']['type']:
            missing_deps.append("C++ compiler (GCC/Clang)")
        
        if not env['dependencies'].get('numpy'):
            missing_deps.append("NumPy")
        
        if not env['dependencies'].get('pybind11'):
            missing_deps.append("pybind11")
        
        if missing_deps:
            print(f"\n[FAIL] Missing dependencies:")
            for dep in missing_deps:
                print(f"   - {dep}")
            print("\nPlease install missing dependencies before building.")
            return 1
        else:
            print(f"\n[PASS] All required dependencies are available!")
            
            # 显示推荐的构建命令
            print(f"\nRecommended build commands:")
            if env['cuda']['available']:
                print(f"  WITH_CUDA=1 python setup.py build_ext --inplace")
            else:
                print(f"  python setup.py build_ext --inplace")
            print(f"  pip install -e .")
            
            return 0
            
    except Exception as e:
        print(f"[ERROR] Error checking environment: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
