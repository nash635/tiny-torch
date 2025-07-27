#!/usr/bin/env python3
"""
Test script to verify CUDA auto-detection works correctly
"""

import os
import sys
import subprocess

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
    # 如果环境变量明确设置了WITH_CUDA=0，则禁用CUDA
    if os.getenv("WITH_CUDA", "").lower() in ["0", "false", "off", "no"]:
        print("CUDA disabled by environment variable")
        return False
    
    # 如果环境变量明确设置了WITH_CUDA=1，则尝试启用CUDA
    if os.getenv("WITH_CUDA", "").lower() in ["1", "true", "on", "yes"]:
        if check_cuda_available():
            print("CUDA enabled by environment variable and is available")
            return True
        else:
            print("Warning: CUDA requested by environment variable but nvcc not found. Falling back to CPU-only build.")
            return False
    
    # 如果没有明确设置，则自动检测
    if check_cuda_available():
        print("CUDA automatically detected and enabled")
        return True
    else:
        print("CUDA not available, using CPU-only build")
        return False

def test_cuda_detection():
    """Test the CUDA auto-detection logic"""
    print("=== Testing CUDA Auto-Detection ===")
    
    # Save original environment
    original_with_cuda = os.environ.get("WITH_CUDA")
    
    try:
        # Test 1: Auto-detection (no environment variable)
        print("\n1. Testing auto-detection (no WITH_CUDA env var)...")
        if "WITH_CUDA" in os.environ:
            del os.environ["WITH_CUDA"]
        
        cuda_available = check_cuda_available()
        print(f"   CUDA available: {cuda_available}")
        
        with_cuda_result = auto_detect_cuda()
        print(f"   Auto-detected WITH_CUDA: {with_cuda_result}")
        
        # Test 2: Force enable CUDA
        print("\n2. Testing WITH_CUDA=1...")
        os.environ["WITH_CUDA"] = "1"
        with_cuda_result = auto_detect_cuda()
        print(f"   WITH_CUDA=1 result: {with_cuda_result}")
        
        # Test 3: Force disable CUDA
        print("\n3. Testing WITH_CUDA=0...")
        os.environ["WITH_CUDA"] = "0"
        with_cuda_result = auto_detect_cuda()
        print(f"   WITH_CUDA=0 result: {with_cuda_result}")
        
        print("\n=== CUDA Auto-Detection Test Complete ===")
        
    finally:
        # Restore original environment
        if original_with_cuda is not None:
            os.environ["WITH_CUDA"] = original_with_cuda
        elif "WITH_CUDA" in os.environ:
            del os.environ["WITH_CUDA"]

if __name__ == "__main__":
    test_cuda_detection()
