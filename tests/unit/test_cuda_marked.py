"""
Unit tests for CUDA functionality with pytest.mark.cuda markers.
"""

import pytest
from tests.utils import UnitTestBase, requires_cuda


class TestCudaUnit(UnitTestBase):
    """CUDA unit tests using pytest.mark.cuda markers."""
    
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
        
        print(f"CUDA unit test - Available: {is_available}, Devices: {device_count}")
    
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


class TestCudaDetectionMarked(UnitTestBase):
    """CUDA detection tests with proper markers."""
    
    @pytest.mark.cuda
    @requires_cuda
    def test_cuda_detection_logic(self):
        """Test CUDA detection logic with proper marker."""
        from tests.utils import CudaTestUtils
        
        # Get CUDA info
        info = CudaTestUtils.get_cuda_info()
        
        # Verify structure
        assert isinstance(info, dict)
        required_keys = ['driver_available', 'nvcc_available', 'gpu_count', 'gpus']
        
        for key in required_keys:
            assert key in info, f"Missing key in CUDA info: {key}"
        
        # Verify types
        assert isinstance(info['driver_available'], bool)
        assert isinstance(info['nvcc_available'], bool)
        assert isinstance(info['gpu_count'], int)
        assert isinstance(info['gpus'], list)
        
        print(f"CUDA detection test - Info: {info}")
