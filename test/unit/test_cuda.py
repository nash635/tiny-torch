"""
Unit tests for CUDA functionality.
"""

from test.utils import CudaTestBase, requires_cuda, requires_nvcc, CudaTestUtils


class TestCudaDetection(CudaTestBase):
    """Test CUDA detection logic."""
    
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
    
    @requires_cuda
    def test_cuda_driver_detection(self):
        """Test CUDA driver detection."""
        info = CudaTestUtils.get_cuda_info()
        assert info['driver_available'] is True
        assert info['gpu_count'] >= 1
        assert len(info['gpus']) >= 1
    
    @requires_nvcc
    def test_nvcc_detection(self):
        """Test nvcc compiler detection."""
        info = CudaTestUtils.get_cuda_info()
        assert info['nvcc_available'] is True
        assert info['nvcc_version'] is not None
    
    def test_auto_detection_logic(self):
        """Test CUDA auto-detection logic."""
        enabled, message = CudaTestUtils.test_cuda_detection()
        
        assert isinstance(enabled, bool)
        assert isinstance(message, str)
        assert len(message) > 0
    
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


class TestCudaValidation(CudaTestBase):
    """Test CUDA environment validation."""
    
    def test_cuda_validator(self):
        """Test CudaEnvironmentValidator."""
        from test.utils import CudaEnvironmentValidator
        
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
