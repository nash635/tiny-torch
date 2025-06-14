"""
torch/__init__.py
Tiny-Torch 主模块 (参考 pytorch/torch/__init__.py)
"""

# 版本信息
__version__ = "0.1.0"

# 导入核心模块（暂时为占位符，后续实现）
# from torch._C import *  # C扩展模块

# 占位符实现，确保包结构正确
def tensor(*args, **kwargs):
    """创建张量的占位符函数"""
    raise NotImplementedError("Tensor implementation coming in Phase 1.2")

def add(*args, **kwargs):
    """张量加法的占位符函数"""
    raise NotImplementedError("Add operation implementation coming in Phase 2")

# 子模块导入
from . import nn
from . import optim
from . import autograd

# 导出主要接口
__all__ = [
    'tensor',
    'add',
    'nn',
    'optim', 
    'autograd',
    '__version__'
]
