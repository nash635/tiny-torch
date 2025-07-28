"""
Common test utilities and base classes for tiny-torch testing.
Provides standardized testing infrastructure for all test modules.
"""

import os
import sys
import subprocess
import tempfile
import shutil
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Try to import pytest, but don't fail if not available
try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    # Define minimal pytest stubs for when pytest is not available
    class pytest:
        @staticmethod
        def skip(reason):
            class SkipException(Exception):
                pass
            raise SkipException(reason)
        
        class mark:
            @staticmethod
            def skipif(condition, reason):
                def decorator(func):
                    def wrapper(*args, **kwargs):
                        if condition:
                            pytest.skip(reason)
                        return func(*args, **kwargs)
                    return wrapper
                return decorator


class TestEnvironment:
    """Test environment management utilities."""
    
    @staticmethod
    def get_project_root() -> Path:
        """Get the project root directory."""
        return PROJECT_ROOT
    
    @staticmethod 
    def get_build_dir() -> Path:
        """Get the build directory."""
        return PROJECT_ROOT / "build"
    
    @staticmethod
    def get_cmake_dir() -> Path:
        """Get the cmake build directory."""
        return PROJECT_ROOT / "build" / "cmake"
    
    @staticmethod
    def is_cuda_available() -> bool:
        """Check if CUDA is available in the environment."""
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, timeout=10)
            return result.returncode == 0
        except (FileNotFoundError, subprocess.SubprocessError, subprocess.TimeoutExpired):
            return False
    
    @staticmethod
    def is_nvcc_available() -> bool:
        """Check if nvcc compiler is available."""
        # Use centralized CUDA detection
        from tests.utils.cuda import CudaTestUtils
        info = CudaTestUtils.get_cuda_info()
        return info['nvcc_available'] and info['nvcc_version'] is not None
    
    @staticmethod
    def is_build_available() -> bool:
        """Check if build artifacts are available."""
        build_dir = TestEnvironment.get_build_dir()
        cmake_dir = TestEnvironment.get_cmake_dir()
        return build_dir.exists() and cmake_dir.exists()
    
    @staticmethod
    def get_test_device() -> str:
        """Get the device to use for testing."""
        device = os.environ.get('TINY_TORCH_TEST_DEVICE', 'auto')
        if device == 'auto':
            return 'cuda' if TestEnvironment.is_cuda_available() else 'cpu'
        return device


class CommandRunner:
    """Utility for running shell commands in tests."""
    
    def __init__(self, default_timeout: int = 30):
        self.default_timeout = default_timeout
    
    def run(self, cmd: Union[str, List[str]], cwd: Optional[str] = None, 
            check: bool = True, timeout: Optional[int] = None,
            env: Optional[Dict[str, str]] = None) -> subprocess.CompletedProcess:
        """Run a command and return the result."""
        if isinstance(cmd, str):
            cmd = cmd.split()
        
        timeout = timeout or self.default_timeout
        
        return subprocess.run(
            cmd, 
            cwd=cwd,
            capture_output=True,
            text=True,
            check=check,
            timeout=timeout,
            env=env
        )
    
    def run_success(self, cmd: Union[str, List[str]], cwd: Optional[str] = None,
                   timeout: Optional[int] = None) -> subprocess.CompletedProcess:
        """Run a command and assert it succeeds."""
        result = self.run(cmd, cwd=cwd, check=False, timeout=timeout)
        if result.returncode != 0:
            cmd_str = ' '.join(cmd) if isinstance(cmd, list) else cmd
            raise AssertionError(
                f"Command failed: {cmd_str}\n"
                f"Return code: {result.returncode}\n"
                f"Stdout: {result.stdout}\n"
                f"Stderr: {result.stderr}"
            )
        return result


class TestBase:
    """Base class for all tiny-torch tests."""
    
    def setup_method(self):
        """Setup test environment before each test method."""
        self.env = TestEnvironment()
        self.runner = CommandRunner()
        self.original_cwd = os.getcwd()
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Cleanup after each test method."""
        os.chdir(self.original_cwd)
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def skip_if_no_cuda(self):
        """Skip test if CUDA is not available."""
        if not self.env.is_cuda_available():
            pytest.skip("CUDA not available")
    
    def skip_if_no_nvcc(self):
        """Skip test if nvcc is not available."""
        if not self.env.is_nvcc_available():
            pytest.skip("nvcc compiler not available")
    
    def skip_if_no_build(self):
        """Skip test if build artifacts are not available."""
        if not self.env.is_build_available():
            pytest.skip("Build artifacts not found. Run 'make build' first.")


class UnitTestBase(TestBase):
    """Base class for unit tests."""
    pass


class IntegrationTestBase(TestBase):
    """Base class for integration tests."""
    
    def setup_method(self):
        super().setup_method()
        self.skip_if_no_build()


class SystemTestBase(TestBase):
    """Base class for system tests."""
    
    def setup_method(self):
        super().setup_method()
        self.skip_if_no_build()


class CudaTestBase(TestBase):
    """Base class for CUDA-related tests."""
    
    def setup_method(self):
        super().setup_method()
        self.skip_if_no_cuda()


class BuildTestBase(TestBase):
    """Base class for build system tests."""
    
    def setup_method(self):
        super().setup_method()
        self.build_dir = self.env.get_build_dir()
        self.cmake_dir = self.env.get_cmake_dir()


# Pytest fixtures
if PYTEST_AVAILABLE:
    @pytest.fixture
    def test_env():
        """Provide test environment utilities."""
        return TestEnvironment()

    @pytest.fixture
    def command_runner():
        """Provide command runner utilities."""
        return CommandRunner()

    @pytest.fixture
    def temp_workspace():
        """Provide a temporary workspace for tests."""
        temp_dir = tempfile.mkdtemp()
        original_cwd = os.getcwd()
        
        yield temp_dir
        
        os.chdir(original_cwd)
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

# Pytest markers
def requires_cuda(func):
    """Mark test as requiring CUDA."""
    if PYTEST_AVAILABLE:
        return pytest.mark.skipif(
            not TestEnvironment.is_cuda_available(),
            reason="CUDA not available"
        )(func)
    else:
        def wrapper(*args, **kwargs):
            if not TestEnvironment.is_cuda_available():
                pytest.skip("CUDA not available")
            return func(*args, **kwargs)
        return wrapper

def requires_nvcc(func):
    """Mark test as requiring nvcc compiler."""
    if PYTEST_AVAILABLE:
        return pytest.mark.skipif(
            not TestEnvironment.is_nvcc_available(),
            reason="nvcc compiler not available"
        )(func)
    else:
        def wrapper(*args, **kwargs):
            if not TestEnvironment.is_nvcc_available():
                pytest.skip("nvcc compiler not available")
            return func(*args, **kwargs)
        return wrapper

def requires_build(func):
    """Mark test as requiring build artifacts."""
    if PYTEST_AVAILABLE:
        return pytest.mark.skipif(
            not TestEnvironment.is_build_available(),
            reason="Build artifacts not found"
        )(func)
    else:
        def wrapper(*args, **kwargs):
            if not TestEnvironment.is_build_available():
                pytest.skip("Build artifacts not found")
            return func(*args, **kwargs)
        return wrapper


# Test data utilities
class TestData:
    """Test data and fixtures management."""
    
    @staticmethod
    def get_fixtures_dir() -> Path:
        """Get the test fixtures directory."""
        return Path(__file__).parent.parent / "fixtures"
    
    @staticmethod
    def get_sample_data(name: str) -> Path:
        """Get path to sample data file."""
        fixtures_dir = TestData.get_fixtures_dir()
        data_path = fixtures_dir / name
        if not data_path.exists():
            raise FileNotFoundError(f"Test data not found: {name}")
        return data_path
