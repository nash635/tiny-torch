"""
Tiny-Torch Test Suite

Comprehensive testing framework for tiny-torch with organized test categories:
- Unit tests: Component isolation testing
- Integration tests: Component interaction testing  
- System tests: End-to-end validation
- Performance tests: Speed and efficiency validation

Usage:
    python tests/run_tests.py                    # Run all tests
    python tests/run_tests.py --category unit    # Run unit tests only
    python tests/run_tests.py --category quick   # Quick smoke test
"""

import os
import sys
from pathlib import Path

# Ensure project root directory is in Python path
TEST_DIR = Path(__file__).parent
PROJECT_ROOT = TEST_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Test configuration
TEST_DEVICE = os.environ.get('TINY_TORCH_TEST_DEVICE', 'cpu')
ENABLE_SLOW_TESTS = os.environ.get('TINY_TORCH_TEST_SLOW', '0') == '1'

# Import test utilities for convenience
try:
    from .utils import (
        TestEnvironment,
        TestBase,
        UnitTestBase,
        IntegrationTestBase,
        SystemTestBase,
        CudaTestBase,
        BuildTestBase,
        requires_cuda,
        requires_build,
        requires_nvcc
    )
    
    __all__ = [
        'TEST_DEVICE',
        'ENABLE_SLOW_TESTS',
        'TestEnvironment',
        'TestBase',
        'UnitTestBase', 
        'IntegrationTestBase',
        'SystemTestBase',
        'CudaTestBase',
        'BuildTestBase',
        'requires_cuda',
        'requires_build',
        'requires_nvcc'
    ]
    
except ImportError:
    # Handle case where test utils are not available
    __all__ = ['TEST_DEVICE', 'ENABLE_SLOW_TESTS']

__version__ = "2.0.0"
