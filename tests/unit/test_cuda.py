"""
Unit tests for CUDA functionality.
"""

import pytest
from tests.utils import CudaTestBase, requires_cuda, requires_nvcc, CudaTestUtils


class TestCudaDetection(CudaTestBase):
    """Test CUDA detection logic."""
    
    @pytest.mark.cuda
    def test_cuda_environment_info(self):
        """Test CUDA environment information gathering."""
        info = CudaTestUtils.get_cuda_info()
        
        # Verify structure
        assert isinstance(info, dict)
        assert 'driver_available' in info
        assert 'nvcc_available' in info
        assert 'gpu_count' in info
        assert 'gpus' in info
        
        # Verify types
        assert isinstance(info['driver_available'], bool)
        assert isinstance(info['nvcc_available'], bool)
        assert isinstance(info['gpu_count'], int)
        assert isinstance(info['gpus'], list)
    
    @pytest.mark.cuda
    @requires_cuda
    def test_cuda_driver_detection(self):
        """Test CUDA driver detection."""
        info = CudaTestUtils.get_cuda_info()
        assert info['driver_available'] is True
        assert info['gpu_count'] >= 1
        assert len(info['gpus']) >= 1
    
    @pytest.mark.cuda
    @requires_nvcc
    def test_nvcc_detection(self):
        """Test nvcc compiler detection."""
        info = CudaTestUtils.get_cuda_info()
        assert info['nvcc_available'] is True
        assert info['nvcc_version'] is not None
    
    @pytest.mark.cuda
    def test_auto_detection_logic(self):
        """Test CUDA auto-detection logic."""
        enabled, message = CudaTestUtils.test_cuda_detection()
        
        assert isinstance(enabled, bool)
        assert isinstance(message, str)
        assert len(message) > 0
    
    @pytest.mark.cuda
    def test_module_interface(self):
        """Test tiny_torch.cuda module interface."""
        import tiny_torch.cuda
        
        # Test is_available() function
        result = tiny_torch.cuda.is_available()
        assert isinstance(result, bool)
        
        # Test device_count() function
        count = tiny_torch.cuda.device_count()
        assert isinstance(count, int)
        assert count >= 0
    
    @pytest.mark.cuda
    @requires_cuda
    def test_cuda_basic_functionality(self):
        """Test basic CUDA functionality."""
        import tiny_torch.cuda
        
        # Test basic interface
        assert callable(tiny_torch.cuda.is_available)
        assert callable(tiny_torch.cuda.device_count)
        
        # Test return types
        is_available = tiny_torch.cuda.is_available()
        device_count = tiny_torch.cuda.device_count()
        
        assert isinstance(is_available, bool)
        assert isinstance(device_count, int)
        assert device_count >= 0
        
        print(f"CUDA basic functionality - Available: {is_available}, Devices: {device_count}")
    
    @pytest.mark.cuda
    @requires_cuda
    def test_cuda_module_attributes(self):
        """Test CUDA module has expected attributes."""
        import tiny_torch.cuda
        
        # Expected attributes/functions
        expected_attrs = ['is_available', 'device_count']
        
        for attr in expected_attrs:
            assert hasattr(tiny_torch.cuda, attr), f"Missing attribute: {attr}"
            
        print("CUDA module attributes test passed")
    
    @pytest.mark.cuda
    @requires_cuda
    def test_cuda_error_handling(self):
        """Test CUDA error handling."""
        import tiny_torch.cuda
        
        # These calls should not raise exceptions
        try:
            is_available = tiny_torch.cuda.is_available()
            device_count = tiny_torch.cuda.device_count()
            
            # Basic sanity checks
            assert isinstance(is_available, bool)
            assert isinstance(device_count, int)
            assert device_count >= 0
            
            print(f"CUDA error handling test - Available: {is_available}, Devices: {device_count}")
            
        except Exception as e:
            pytest.fail(f"CUDA basic operations should not raise exceptions: {e}")


class TestCudaValidation(CudaTestBase):
    """Test CUDA environment validation."""
    
    @pytest.mark.cuda
    def test_cuda_validator(self):
        """Test CudaEnvironmentValidator."""
        from tests.utils import CudaEnvironmentValidator
        
        validator = CudaEnvironmentValidator()
        results = validator.validate_environment()
        
        # Check result structure
        assert isinstance(results, dict)
        assert 'environment_valid' in results
        assert 'details' in results
        
        # Check individual test results
        assert 'driver_test' in results
        assert 'compiler_test' in results
        assert 'detection_test' in results
        
        # Detection test should always pass (it's logic validation)
        assert results['detection_test'] is True
