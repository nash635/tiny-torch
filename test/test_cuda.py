#!/usr/bin/env python3
"""
test_cuda_comprehensive.py
Tiny-Torch CUDA功能综合测试套件

合并了原有的多个CUDA测试文件，提供统一的测试接口。
"""

import subprocess
import sys
import os
import time
import argparse
from pathlib import Path

# 动态获取项目根目录路径
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

def run_environment_tests():
    """运行环境测试"""
    print("[环境] 系统环境测试")
    print("-" * 30)
    
    results = {}
    
    # 测试CUDA驱动
    print("[检测] 检测NVIDIA GPU驱动...")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("[成功] NVIDIA GPU检测成功")
            lines = result.stdout.split('\n')[:3]
            for line in lines:
                if line.strip():
                    print(f"   {line}")
            results['CUDA驱动'] = True
        else:
            print("[失败] nvidia-smi命令失败")
            results['CUDA驱动'] = False
    except:
        print("[失败] nvidia-smi命令未找到")
        results['CUDA驱动'] = False
    
    # 测试CUDA编译器
    print("\n[检测] 检测CUDA编译器...")
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("[成功] CUDA编译器(nvcc)可用")
            lines = result.stdout.split('\n')
            for line in lines:
                if 'release' in line.lower():
                    print(f"   {line.strip()}")
            results['CUDA编译器'] = True
        else:
            print("[失败] nvcc编译器不可用")
            results['CUDA编译器'] = False
    except:
        print("[失败] nvcc编译器未找到")
        results['CUDA编译器'] = False
    
    # 测试GPU属性
    print("\n[检测] 查询GPU设备信息...")
    try:
        result = subprocess.run([
            'nvidia-smi', '--query-gpu=gpu_name,memory.total,compute_cap', 
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("[成功] GPU设备信息:")
            lines = result.stdout.strip().split('\n')
            for i, line in enumerate(lines):
                if line.strip():
                    parts = line.split(', ')
                    if len(parts) >= 3:
                        name, memory, compute_cap = parts[0], parts[1], parts[2]
                        print(f"   GPU {i}: {name}")
                        print(f"   内存: {memory}MB, 计算能力: {compute_cap}")
            results['GPU属性'] = True
        else:
            print("[失败] 无法获取GPU设备信息")
            results['GPU属性'] = False
    except:
        print("[失败] GPU信息查询失败")
        results['GPU属性'] = False
    
    return results

def run_functional_tests():
    """运行功能测试"""
    print("[功能] 基本功能测试")
    print("-" * 30)
    
    results = {}
    
    # 测试tiny_torch导入
    print("[检测] 测试tiny_torch模块导入...")
    try:
        import tiny_torch
        print(f"[成功] 导入tiny_torch成功，版本: {tiny_torch.__version__}")
        results['tiny_torch导入'] = True
    except ImportError as e:
        print(f"[失败] 无法导入tiny_torch: {e}")
        results['tiny_torch导入'] = False
        return results
    
    # 测试CUDA模块
    print("\n[检测] 测试tiny_torch.cuda模块...")
    try:
        if hasattr(tiny_torch, 'cuda'):
            print("[成功] tiny_torch.cuda模块存在")
            
            # 检查基本函数
            functions = ['is_available', 'device_count', 'current_device', 'get_device_name']
            all_exist = True
            for func in functions:
                if hasattr(tiny_torch.cuda, func):
                    print(f"   [OK] {func}")
                else:
                    print(f"   [缺失] {func} 缺失")
                    all_exist = False
            
            results['cuda模块'] = all_exist
        else:
            print("[失败] tiny_torch.cuda模块不存在")
            results['cuda模块'] = False
    except Exception as e:
        print(f"[失败] 测试tiny_torch.cuda模块失败: {e}")
        results['cuda模块'] = False
    
    # 测试CUDA功能
    print("\n[检测] 测试CUDA基本功能...")
    try:
        if tiny_torch.cuda.is_available():
            print(f"[成功] CUDA可用性: {tiny_torch.cuda.is_available()}")
            print(f"[成功] 设备数量: {tiny_torch.cuda.device_count()}")
            print(f"[成功] 当前设备: {tiny_torch.cuda.current_device()}")
            
            # 测试设备信息
            for i in range(tiny_torch.cuda.device_count()):
                name = tiny_torch.cuda.get_device_name(i)
                props = tiny_torch.cuda.get_device_properties(i)
                print(f"[成功] GPU {i}: {name}")
                if props:
                    total_mem = props.get('total_memory', 0)
                    compute_cap = props.get('compute_capability', 'Unknown')
                    print(f"   内存: {total_mem // (1024**3)} GB, 计算能力: {compute_cap}")
            
            results['CUDA功能'] = True
        else:
            print("[警告] CUDA不可用")
            results['CUDA功能'] = False
    except Exception as e:
        print(f"[失败] CUDA功能测试失败: {e}")
        results['CUDA功能'] = False
    
    return results

def run_build_tests():
    """运行构建测试"""
    print("[构建] 构建系统测试")
    print("-" * 30)
    
    results = {}
    
    # 检查CMake配置
    print("[检测] 检查CMake CUDA配置...")
    cmake_file = PROJECT_ROOT / "CMakeLists.txt"
    if cmake_file.exists():
        with open(cmake_file, 'r') as f:
            content = f.read()
        
        checks = [
            ("WITH_CUDA选项", "WITH_CUDA" in content),
            ("CUDA语言支持", "enable_language(CUDA)" in content),
            ("CUDAToolkit查找", "find_package(CUDAToolkit" in content),
        ]
        
        print("[成功] CMake CUDA配置检查:")
        all_passed = True
        for check_name, passed in checks:
            status = "[OK]" if passed else "[缺失]"
            print(f"   {status} {check_name}")
            if not passed:
                all_passed = False
        
        results['CMake配置'] = all_passed
    else:
        print("[失败] CMakeLists.txt文件不存在")
        results['CMake配置'] = False
    
    # 检查CUDA源文件
    print("\n[检测] 检查CUDA源文件...")
    cuda_files = [
        "csrc/aten/src/ATen/cuda/CUDAContext.cu",
        "csrc/aten/src/ATen/native/cuda/BinaryOps.cu",
        "csrc/aten/src/ATen/native/cuda/UnaryOps.cu",
    ]
    
    all_exist = True
    for cuda_file in cuda_files:
        full_path = PROJECT_ROOT / cuda_file
        if full_path.exists():
            print(f"   [OK] {cuda_file}")
        else:
            print(f"   [缺失] {cuda_file}")
            all_exist = False
    
    results['源文件'] = all_exist
    
    # 检查构建产物
    print("\n[检测] 检查构建产物...")
    # 查找可能的构建目录和静态库
    possible_lib_paths = [
        PROJECT_ROOT / "build" / "cmake" / "libtiny_torch_cpp.a",  # 标准构建目录
        PROJECT_ROOT / "build" / "libtiny_torch_cpp.a"            # 备用位置
    ]
    
    lib_found = False
    for lib_file in possible_lib_paths:
        if lib_file.exists():
            lib_size = lib_file.stat().st_size
            print(f"   [OK] 静态库: {lib_size // 1024} KB ({lib_file.relative_to(PROJECT_ROOT)})")
            results['构建产物'] = True
            lib_found = True
            break
    
    if not lib_found:
        print("   [缺失] 静态库文件不存在")
        results['构建产物'] = False
    
    return results

def run_demo():
    """运行演示"""
    print("[演示] 功能演示")
    print("-" * 30)
    
    try:
        import tiny_torch
        
        print(f"Tiny-Torch版本: {tiny_torch.__version__}")
        print(f"CUDA可用性: {tiny_torch.cuda.is_available()}")
        
        if tiny_torch.cuda.is_available():
            print(f"GPU设备数量: {tiny_torch.cuda.device_count()}")
            print(f"当前设备: {tiny_torch.cuda.current_device()}")
            print(f"CUDA版本: {tiny_torch.cuda.version()}")
            
            print("\nGPU设备详细信息:")
            for i in range(tiny_torch.cuda.device_count()):
                name = tiny_torch.cuda.get_device_name(i)
                props = tiny_torch.cuda.get_device_properties(i)
                print(f"   GPU {i}: {name}")
                if props:
                    print(f"     内存: {props['total_memory'] // (1024**3)} GB")
                    print(f"     计算能力: {props['compute_capability']}")
        else:
            print("警告: CUDA当前不可用")
            
    except Exception as e:
        print(f"演示失败: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Tiny-Torch CUDA综合测试套件')
    parser.add_argument('--basic', action='store_true', help='只运行基本功能测试')
    parser.add_argument('--demo', action='store_true', help='只运行功能演示')
    parser.add_argument('--build', action='store_true', help='只运行构建系统测试')
    parser.add_argument('--env', action='store_true', help='只运行环境测试')
    
    args = parser.parse_args()
    
    print("Tiny-Torch CUDA综合测试套件")
    print("=" * 60)
    print(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    all_results = {}
    
    # 根据参数选择要运行的测试
    run_all = not any([args.basic, args.demo, args.build, args.env])
    
    if args.env or run_all:
        env_results = run_environment_tests()
        all_results.update(env_results)
        print()
    
    if args.basic or run_all:
        func_results = run_functional_tests()
        all_results.update(func_results)
        print()
    
    if args.build or run_all:
        build_results = run_build_tests()
        all_results.update(build_results)
        print()
    
    if args.demo or run_all:
        run_demo()
        print()
    
    # 测试结果总结
    if all_results:
        print("=" * 60)
        print("测试结果总结")
        print("=" * 60)
        
        passed = 0
        for test_name, result in all_results.items():
            status = "通过" if result else "失败"
            print(f"   {test_name:12}: {status}")
            if result:
                passed += 1
        
        total = len(all_results)
        print(f"\n总体结果: {passed}/{total} 测试通过 ({passed/total*100:.1f}%)")
        
        if passed >= total * 0.8:
            print("\nCUDA支持良好！")
            print("   已为Phase 1.2做好准备")
            return True
        else:
            print("\n警告: CUDA支持需要改进")
            return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
