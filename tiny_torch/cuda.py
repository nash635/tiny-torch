"""
tiny_torch.cuda
CUDA支持模块 - Tiny-Torch的CUDA功能接口
"""

import subprocess
import os
import sys

# 缓存CUDA可用性检查结果
_cuda_available = None
_device_count = None

def is_available():
    """检查CUDA是否可用"""
    global _cuda_available
    
    # 如果已经检查过，直接返回缓存结果
    if _cuda_available is not None:
        return _cuda_available
        
    try:
        # 检查nvidia-smi是否可用
        result = subprocess.run(['nvidia-smi'], capture_output=True)
        _cuda_available = result.returncode == 0
        return _cuda_available
    except (FileNotFoundError, subprocess.SubprocessError):
        _cuda_available = False
        return False

def device_count():
    """获取可用的CUDA设备数量"""
    global _device_count
    
    # 如果已经检查过，直接返回缓存结果
    if _device_count is not None:
        return _device_count
        
    if not is_available():
        _device_count = 0
        return 0
        
    try:
        result = subprocess.run([
            'nvidia-smi', '--query-gpu=count', '--format=csv,noheader'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            # nvidia-smi返回的是每个GPU的计数，我们需要计算行数
            lines = result.stdout.strip().split('\n')
            _device_count = len([line for line in lines if line.strip()])
            return _device_count
        else:
            _device_count = 0
            return 0
    except Exception:
        _device_count = 0
        return 0

def get_device_name(device=None):
    """获取CUDA设备名称"""
    if not is_available():
        return None
        
    device_id = device if device is not None else 0
    
    try:
        result = subprocess.run([
            'nvidia-smi', '--query-gpu=gpu_name', '--format=csv,noheader'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            names = result.stdout.strip().split('\n')
            if 0 <= device_id < len(names):
                return names[device_id].strip()
        return None
    except Exception:
        return None

def get_device_properties(device=None):
    """获取CUDA设备属性"""
    if not is_available():
        return None
        
    device_id = device if device is not None else 0
    
    try:
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=memory.total,compute_cap', 
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if 0 <= device_id < len(lines):
                memory, compute_cap = lines[device_id].split(', ')
                return {
                    'total_memory': int(memory) * 1024 * 1024,  # 转换为字节
                    'compute_capability': compute_cap
                }
        return None
    except Exception:
        return None

def current_device():
    """获取当前CUDA设备ID"""
    return 0

def version():
    """获取CUDA版本信息"""
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'release' in line.lower():
                    parts = line.split('release')[-1].split(',')[0].strip()
                    return parts
        return "Unknown"
    except Exception:
        return "Unknown"

# 导出的接口
__all__ = [
    'is_available',
    'device_count', 
    'get_device_name',
    'get_device_properties',
    'current_device',
    'version'
]
