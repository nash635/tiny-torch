"""
tiny_torch/nn/__init__.py
神经网络模块 (参考 pytorch/torch/nn/__init__.py)
"""

# 占位符实现
class Module:
    """神经网络模块基类的占位符"""
    def __init__(self):
        raise NotImplementedError("Module implementation coming in Phase 5")

class Linear:
    """线性层的占位符"""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Linear layer implementation coming in Phase 5")

# 导出接口
__all__ = ['Module', 'Linear']
