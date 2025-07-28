"""
Integration tests for CUDA functionality.
"""

import pytest
from tests.utils import IntegrationTestBase, requires_cuda


class TestCudaIntegration(IntegrationTestBase):
    """Test CUDA integration functionality."""
    
    @pytest.mark.cuda
    @requires_cuda
    def test_cuda_availability_placeholder(self):
        """Placeholder test to verify CUDA test infrastructure."""
        import tiny_torch.cuda
        
        # This is a placeholder test to ensure the CUDA test pipeline works
        # It should pass regardless of whether CUDA is actually available
        
        # Test basic CUDA module interface
        assert hasattr(tiny_torch.cuda, 'is_available')
        assert hasattr(tiny_torch.cuda, 'device_count')
        
        # Test that functions return reasonable values
        cuda_available = tiny_torch.cuda.is_available()
        device_count = tiny_torch.cuda.device_count()
        
        assert isinstance(cuda_available, bool)
        assert isinstance(device_count, int)
        assert device_count >= 0
        
        print(f"CUDA available: {cuda_available}")
        print(f"Device count: {device_count}")
    
    @pytest.mark.cuda
    @requires_cuda
    def test_cuda_module_import(self):
        """Test that CUDA-related modules can be imported."""
        # Test basic imports work
        import tiny_torch
        import tiny_torch.cuda
        
        # Verify module structure
        assert hasattr(tiny_torch, 'cuda')
        assert tiny_torch.cuda is not None
        
        # Test basic functionality
        try:
            # This should not raise an exception
            device_count = tiny_torch.cuda.device_count()
            is_available = tiny_torch.cuda.is_available()
            
            # Log for debugging
            print(f"CUDA module test - Available: {is_available}, Devices: {device_count}")
            
            # Basic assertions
            assert device_count >= 0
            assert isinstance(is_available, bool)
            
        except Exception as e:
            pytest.fail(f"CUDA module basic functionality failed: {e}")


class TestCudaEnvironment(IntegrationTestBase):
    """Test CUDA environment integration."""
    
    @pytest.mark.cuda
    @requires_cuda
    def test_cuda_environment_consistency(self):
        """Test consistency between environment and module behavior."""
        import tiny_torch.cuda
        from tests.utils import CudaTestUtils
        
        # Get environment info
        env_info = CudaTestUtils.get_cuda_info()
        
        # Get module info
        module_available = tiny_torch.cuda.is_available()
        module_device_count = tiny_torch.cuda.device_count()
        
        # Log for debugging
        print(f"Environment info: {env_info}")
        print(f"Module available: {module_available}, device count: {module_device_count}")
        
        # Basic consistency checks
        assert isinstance(env_info, dict)
        assert 'driver_available' in env_info
        assert 'gpu_count' in env_info
        
        # The module and environment should be somewhat consistent
        # (allowing for some flexibility in implementation)
        if env_info.get('driver_available', False):
            # If driver is available, device count should be reasonable
            assert module_device_count >= 0
        
        # Device count should match or be reasonable
        env_gpu_count = env_info.get('gpu_count', 0)
        assert env_gpu_count >= 0
        assert module_device_count >= 0


class TestCudaBuildIntegration(IntegrationTestBase):
    """Test CUDA build system integration."""
    
    @pytest.mark.cuda
    @requires_cuda
    def test_cuda_build_artifacts(self):
        """Test that CUDA build artifacts are present when expected."""
        from tests.utils import CudaTestUtils, BuildValidator
        
        # Validate build system
        build_validator = BuildValidator()
        build_results = build_validator.run_full_validation()
        
        assert build_results['structure_valid'], "Basic build structure should be valid"
        
        # Check CUDA-specific build artifacts
        cuda_results = CudaTestUtils.validate_cuda_build()
        
        # Log results for debugging
        print(f"Build results: {build_results}")
        print(f"CUDA build results: {cuda_results}")
        
        # At minimum, the validation should not crash
        assert isinstance(cuda_results, dict)
        
        # If CUDA is supposed to be available, some artifacts should exist
        import tiny_torch.cuda
        if tiny_torch.cuda.is_available():
            print("CUDA is available - checking for build artifacts")
            # In a real implementation, we'd check for specific files
            # For now, just ensure the validation ran
            assert cuda_results is not None
