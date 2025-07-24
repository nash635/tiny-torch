#!/usr/bin/env python3
"""
Tiny-Torch Phase 1.1 Build System Status Report
验证构建系统集成并生成状态报告
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import platform

def print_header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def print_section(title):
    print(f"\n{title}")
    print("-" * 40)

def check_status(condition, message):
    status = "[PASS]" if condition else "[FAIL]"
    print(f"  {status} {message}")
    return condition

def check_tool(cmd, name):
    """检查工具是否可用"""
    try:
        result = subprocess.run([cmd, "--version"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            version = result.stdout.split('\n')[0]
            print(f"  [PASS] {name}: {version}")
            return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    print(f"  [FAIL] {name}: Not available")
    return False

def main():
    print_header("Tiny-Torch Phase 1.1 Build System Status Report")
    print(f"Generated: {platform.platform()}")
    print(f"Python: {sys.version}")
    
    # 1. 环境检查
    print_section("1. Build Environment")
    
    python_ok = check_status(sys.version_info >= (3, 7), f"Python >= 3.7 (current: {sys.version_info[:2]})")
    cmake_ok = check_tool("cmake", "CMake")
    make_ok = check_tool("make", "Make") 
    ninja_ok = check_tool("ninja", "Ninja")
    gcc_ok = check_tool("gcc", "GCC")
    
    # 2. 项目文件检查
    print_section("2. Project Structure")
    
    project_files = {
        "CMakeLists.txt": "CMake build configuration",
        "setup.py": "Python package setup",
        "Makefile": "Convenience build interface", 
        "pyproject.toml": "Modern Python project config"
    }
    
    all_files_ok = True
    for file, desc in project_files.items():
        exists = Path(file).exists()
        check_status(exists, f"{file} - {desc}")
        all_files_ok &= exists
    
    # 3. 源码结构
    print_section("3. Source Code Structure")
    
    source_dirs = {
        "csrc/": "C++/CUDA source code",
        "torch/": "Python frontend package",
        "test/": "Test suite",
        "docs/": "Documentation"
    }
    
    for dir_path, desc in source_dirs.items():
        exists = Path(dir_path).exists()
        if exists:
            if dir_path == "csrc/":
                # Count source files
                cpp_files = list(Path(dir_path).rglob("*.cpp"))
                cu_files = list(Path(dir_path).rglob("*.cu"))
                check_status(True, f"{dir_path} - {desc} ({len(cpp_files)} .cpp, {len(cu_files)} .cu files)")
            else:
                check_status(True, f"{dir_path} - {desc}")
        else:
            check_status(False, f"{dir_path} - {desc}")
    
    # 4. 构建系统分析
    print_section("4. Build System Analysis")
    
    # 检查 Ninja 集成
    ninja_integrated = False
    if Path("setup.py").exists():
        with open("setup.py", "r") as f:
            content = f.read()
            ninja_integrated = "ninja" in content.lower() and "USE_NINJA" in content
    
    check_status(ninja_integrated, "Ninja integration in setup.py")
    
    # 检查 CMake Ninja 支持
    cmake_ninja_support = False
    if Path("CMakeLists.txt").exists():
        with open("CMakeLists.txt", "r") as f:
            content = f.read()
            cmake_ninja_support = "Ninja" in content
    
    check_status(cmake_ninja_support, "CMake Ninja optimizations")
    
    # 5. 构建测试
    print_section("5. Build System Test")
    
    # 创建简单测试
    test_dir = Path("build/status_test")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True)
    
    # 测试 CMake 配置
    cmake_success = False
    ninja_build_files = False
    
    try:
        cmake_args = ["cmake", "../..", "-DCMAKE_BUILD_TYPE=Release", "-DWITH_CUDA=OFF", "-DBUILD_TESTS=OFF"]
        if ninja_ok:
            cmake_args.append("-GNinja")
        
        result = subprocess.run(cmake_args, cwd=test_dir, capture_output=True, text=True, timeout=30)
        cmake_success = result.returncode == 0
        
        if cmake_success and ninja_ok:
            ninja_build_files = (test_dir / "build.ninja").exists()
            
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pass
    
    check_status(cmake_success, "CMake configuration test")
    if ninja_ok:
        check_status(ninja_build_files, "Ninja build files generation")
    
    # 6. 性能预期
    print_section("6. Performance Expectations")
    
    if ninja_ok and ninja_build_files:
        print("  [HIGH-PERFORMANCE] High-Performance Build Mode Active")
        print("     • Expected 2-4x faster builds vs Make")
        print("     • Parallel compilation optimized")
        print("     • Incremental builds enhanced")
    elif make_ok:
        print("  [STANDARD] Standard Build Mode (Make fallback)")
        print("     • Reliable cross-platform builds")
        print("     • Consider installing Ninja for performance")
    else:
        print("  [WARNING] Limited build capabilities")
        print("     • Install CMake and Make for full functionality")
    
    # 7. 总结和建议
    print_section("7. Summary & Recommendations")
    
    # 计算整体就绪度
    core_tools = python_ok and cmake_ok and (make_ok or ninja_ok)
    project_ready = all_files_ok and ninja_integrated and cmake_ninja_support
    build_tested = cmake_success
    
    overall_score = sum([core_tools, project_ready, build_tested])
    
    if overall_score == 3:
        status_emoji = "[EXCELLENT]"
        status_text = "FULLY READY"
        color = "\033[92m"  # Green
    elif overall_score == 2:
        status_emoji = "[GOOD]"
        status_text = "MOSTLY READY"
        color = "\033[93m"  # Yellow
    else:
        status_emoji = "[WARNING]"
        status_text = "NEEDS ATTENTION"
        color = "\033[91m"  # Red
    
    reset = "\033[0m"
    
    print(f"\n{color}BUILD SYSTEM STATUS: {status_emoji} {status_text}{reset}")
    print(f"   Overall Score: {overall_score}/3")
    
    if ninja_ok and ninja_build_files:
        print(f"\nNINJA INTEGRATION: {color}[ACTIVE]{reset}")
        print("   • High-performance builds enabled")
        print("   • 2-4x speed improvement expected")
        print("   • Modern build pipeline operational")
    elif ninja_ok:
        print(f"\nNINJA INTEGRATION: {color}[PARTIAL]{reset}")
        print("   • Ninja available but build files generation failed")
        print("   • Check CMake configuration")
    else:
        print(f"\nNINJA INTEGRATION: [NOT AVAILABLE]")
        print("   • Install ninja: pip install ninja")
        print("   • Or: conda install ninja")
    
    # 下一步建议
    print(f"\nNEXT STEPS:")
    if not ninja_ok:
        print("   1. Install Ninja: pip install ninja")
    if not cmake_success:
        print("   2. Fix CMake configuration issues")
    if overall_score == 3:
        print("   [PASS] System ready for Phase 1.2 development!")
        print("   [PASS] Begin tensor implementation")
    
    # 快速命令参考
    print(f"\nQUICK COMMANDS:")
    print("   make build          # Standard build")
    print("   make clean          # Clean artifacts")
    if ninja_ok:
        print("   USE_NINJA=1 make build  # Force Ninja")
    print("   DEBUG=1 make build      # Debug build")
    
    print_header("Report Complete")

if __name__ == "__main__":
    main()
