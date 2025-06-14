"""
test/__init__.py
测试包初始化文件
"""

import os
import sys
from pathlib import Path

# 确保项目根目录在Python路径中
TEST_DIR = Path(__file__).parent
PROJECT_ROOT = TEST_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 测试配置
TEST_DEVICE = os.environ.get('TINY_TORCH_TEST_DEVICE', 'cpu')
ENABLE_SLOW_TESTS = os.environ.get('TINY_TORCH_TEST_SLOW', '0') == '1'

__all__ = ['TEST_DEVICE', 'ENABLE_SLOW_TESTS']
