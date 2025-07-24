#!/usr/bin/env python3
"""
Tiny-Torch Build Diagnostic Script
Helps diagnose and fix common build issues
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
import glob

def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def print_subsection(title):
    print(f"\n{title}")
    print("-" * 40)

def run_command(cmd, description, capture_output=True):
    """Run a command and return the result"""
    print(f"Running: {description}")
    try:
        if capture_output:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                print(f"[PASS] Success: {description}")
                return result.stdout.strip()
            else:
                print(f"[FAIL] Failed: {description}")
                print(f"Error: {result.stderr}")
                return None
        else:
            result = subprocess.run(cmd, shell=True, timeout=30)
            if result.returncode == 0:
                print(f"[PASS] Success: {description}")
                return True
            else:
                print(f"[FAIL] Failed: {description}")
                return False
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] Timeout: {description}")
        return None
    except Exception as e:
        print(f"[ERROR] Exception: {description} - {e}")
        return None

def main():
    print_section("Tiny-Torch Build Diagnostic")
    
    # 1. Environment check
    print_subsection("1. Environment Information")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Platform: {sys.platform}")
    
    # 2. Check for build artifacts
    print_subsection("2. Build Artifacts Analysis")
    
    # Check for .egg-info directories
    egg_info_dirs = glob.glob("*.egg-info")
    if egg_info_dirs:
        print(f"[FOUND] Found {len(egg_info_dirs)} .egg-info directories:")
        for dir_name in egg_info_dirs:
            print(f"   - {dir_name}")
        print("   This can cause the 'Multiple .egg-info directories found' error")
    else:
        print("[PASS] No .egg-info directories found")
    
    # Check for build directories
    build_dirs = ["build", "dist", "__pycache__", ".pytest_cache"]
    for dir_name in build_dirs:
        if Path(dir_name).exists():
            print(f"[FOUND] Found build directory: {dir_name}")
        else:
            print(f"[PASS] No {dir_name} directory")
    
    # 3. Package dependencies
    print_subsection("3. Package Dependencies")
    
    required_packages = {
        "setuptools": "setuptools",
        "wheel": "wheel", 
        "pybind11": "pybind11",
        "cmake": "cmake"
    }
    
    for package, import_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"[PASS] {package}: Available")
        except ImportError:
            print(f"[FAIL] {package}: Missing")
    
    # Check ninja separately since it's a command-line tool, not a Python package
    ninja_check = run_command("which ninja", "Check ninja availability", capture_output=True)
    if ninja_check:
        print(f"[PASS] ninja: Available")
    else:
        print(f"[FAIL] ninja: Missing")
    
    # 4. Tools availability
    print_subsection("4. Build Tools")
    
    tools = {
        "python3": "Python interpreter",
        "pip3": "Python package installer", 
        "cmake": "CMake build system",
        "ninja": "Ninja build tool",
        "make": "GNU Make",
        "gcc": "GCC compiler",
        "nvcc": "NVIDIA CUDA compiler"
    }
    
    for tool, description in tools.items():
        result = run_command(f"which {tool}", f"Check {description}")
        if result:
            if tool == "python3":
                version = run_command(f"{tool} --version", f"Get {tool} version")
            elif tool == "pip3":
                version = run_command(f"{tool} --version", f"Get {tool} version")
            elif tool == "ninja":
                version = run_command(f"{tool} --version 2>/dev/null", f"Get {tool} version")
            else:
                version = run_command(f"{tool} --version 2>/dev/null | head -1", f"Get {tool} version")
            if version:
                print(f"   Version: {version}")
    
    # 5. Clean build artifacts
    print_subsection("5. Cleanup Operations")
    
    cleanup_paths = [
        "build",
        "dist", 
        "*.egg-info",
        "__pycache__",
        ".pytest_cache",
        "torch/_C*.so"
    ]
    
    for pattern in cleanup_paths:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            print(f"[CLEANUP] Found {len(matches)} items matching '{pattern}':")
            for match in matches:
                print(f"   - {match}")
        else:
            print(f"[PASS] No items matching '{pattern}'")
    
    # 6. Suggested fixes
    print_subsection("6. Suggested Fixes")
    
    if egg_info_dirs:
        print("[FIX] Fix for multiple .egg-info directories:")
        print("   rm -rf *.egg-info")
        
    print("\n[FIX] Complete cleanup command:")
    print("   rm -rf build/ dist/ *.egg-info/ __pycache__/ .pytest_cache/")
    print("   find . -name '*.pyc' -delete")
    
    print("\n[RECOMMENDED] Recommended installation sequence:")
    print("   1. rm -rf build/ dist/ *.egg-info/")
    print("   2. pip3 install -r requirements.txt")
    print("   3. python3 setup.py build_ext --inplace")
    print("   4. pip3 install -e . --no-deps")
    
    # 7. Test simple operations
    print_subsection("7. Basic Functionality Test")
    
    # Test setup.py syntax
    syntax_ok = run_command("python3 -m py_compile setup.py", "Check setup.py syntax")
    
    # Test import
    import_ok = run_command("python3 -c 'import setuptools; print(setuptools.__version__)'", "Test setuptools import")
    
    print_section("Diagnostic Complete")
    
    if egg_info_dirs:
        print("[WARNING] Action required: Clean up .egg-info directories")
        return False
    else:
        print("[PASS] No major issues detected")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
