"""
CUDA-specific test utilities.
"""

import subprocess
import os
from typing import Dict, Optional, Tuple
from .common import TestEnvironment, CommandRunner


class CudaTestUtils:
    """CUDA testing utilities."""
    
    @staticmethod
    def get_cuda_info() -> Dict[str, any]:
        """Get CUDA environment information."""
        info = {
            'driver_available': False,
            'nvcc_available': False,
            'gpu_count': 0,
            'driver_version': None,
            'nvcc_version': None,
            'gpus': []
        }
        
        # Check CUDA driver
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                info['driver_available'] = True
                # Parse driver version from nvidia-smi output
                for line in result.stdout.split('\n'):
                    if 'Driver Version:' in line:
                        parts = line.split('Driver Version:')
                        if len(parts) > 1:
                            info['driver_version'] = parts[1].split()[0]
                        break
        except (subprocess.SubprocessError, subprocess.TimeoutExpired, FileNotFoundError, OSError):
            # nvidia-smi not available (common on CI servers without GPU)
            info['driver_available'] = False
        
        # Check nvcc compiler
        try:
            result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                info['nvcc_available'] = True
                # Parse nvcc version
                for line in result.stdout.split('\n'):
                    if 'release' in line.lower():
                        parts = line.split(',')
                        for part in parts:
                            if 'release' in part.lower():
                                info['nvcc_version'] = part.strip().split()[-1]
                        break
        except (subprocess.SubprocessError, subprocess.TimeoutExpired, FileNotFoundError, OSError):
            # nvcc not available (common on CI servers without CUDA toolkit)
            info['nvcc_available'] = False
        
        # Get GPU count and info
        if info['driver_available']:
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], 
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    for line in result.stdout.strip().split('\n'):
                        if line.strip():
                            parts = line.split(',')
                            if len(parts) >= 2:
                                info['gpus'].append({
                                    'name': parts[0].strip(),
                                    'memory': parts[1].strip()
                                })
                    info['gpu_count'] = len(info['gpus'])
            except (subprocess.SubprocessError, subprocess.TimeoutExpired, FileNotFoundError, OSError):
                # nvidia-smi not available or failed
                info['gpu_count'] = 0
                info['gpus'] = []
        
        return info
    
    @staticmethod
    def test_cuda_detection() -> Tuple[bool, str]:
        """Test CUDA auto-detection logic."""
        env_cuda = os.getenv("WITH_CUDA", "").lower()
        
        # If explicitly disabled
        if env_cuda in ["0", "false", "off", "no"]:
            return False, "CUDA disabled by environment variable"
        
        # If explicitly enabled
        if env_cuda in ["1", "true", "on", "yes"]:
            if TestEnvironment.is_nvcc_available():
                return True, "CUDA enabled by environment variable and is available"
            else:
                return False, "Warning: CUDA requested but nvcc not found. Falling back to CPU-only"
        
        # Auto-detection
        if TestEnvironment.is_nvcc_available():
            return True, "CUDA automatically detected and enabled"
        else:
            return False, "CUDA not available, using CPU-only build"
    
    @staticmethod
    def validate_cuda_build() -> Dict[str, bool]:
        """Validate CUDA build artifacts."""
        results = {}
        build_dir = TestEnvironment.get_cmake_dir()
        
        # Check for CUDA object files
        cuda_files = list(build_dir.glob("**/*.cu.o"))
        results['cuda_objects'] = len(cuda_files) > 0
        
        # Check for CUDA-related libraries
        cuda_libs = list(build_dir.glob("**/*cuda*"))
        results['cuda_libraries'] = len(cuda_libs) > 0
        
        return results


class CudaEnvironmentValidator:
    """Validates CUDA environment for testing."""
    
    def __init__(self):
        self.runner = CommandRunner()
        self.info = CudaTestUtils.get_cuda_info()
    
    def validate_environment(self) -> Dict[str, any]:
        """Run complete CUDA environment validation."""
        results = {
            'environment_valid': False,
            'driver_test': False,
            'compiler_test': False,
            'detection_test': False,
            'build_test': False,
            'details': {}
        }
        
        # Test 1: Driver availability
        results['driver_test'] = self.info['driver_available']
        results['details']['driver'] = {
            'available': self.info['driver_available'],
            'version': self.info['driver_version'],
            'gpu_count': self.info['gpu_count'],
            'gpus': self.info['gpus']
        }
        
        # Test 2: Compiler availability  
        results['compiler_test'] = self.info['nvcc_available']
        results['details']['compiler'] = {
            'available': self.info['nvcc_available'],
            'version': self.info['nvcc_version']
        }
        
        # Test 3: Detection logic
        cuda_enabled, message = CudaTestUtils.test_cuda_detection()
        results['detection_test'] = True  # Logic test always passes
        results['details']['detection'] = {
            'should_enable_cuda': cuda_enabled,
            'message': message
        }
        
        # Test 4: Build artifacts (if available)
        if TestEnvironment.is_build_available():
            build_results = CudaTestUtils.validate_cuda_build()
            results['build_test'] = any(build_results.values())
            results['details']['build'] = build_results
        else:
            results['build_test'] = None
            results['details']['build'] = {'message': 'Build artifacts not available'}
        
        # Overall validation
        required_tests = [results['driver_test'], results['compiler_test'], results['detection_test']]
        if results['build_test'] is not None:
            required_tests.append(results['build_test'])
        
        results['environment_valid'] = all(test for test in required_tests if test is not None)
        
        return results
