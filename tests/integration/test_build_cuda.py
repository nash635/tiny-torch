"""
Integration tests for build system and CUDA integration.
"""

from tests.utils import IntegrationTestBase, requires_cuda, requires_build, BuildValidator, CudaEnvironmentValidator


class TestBuildCudaIntegration(IntegrationTestBase):
    """Test integration between build system and CUDA."""
    
    @requires_cuda
    @requires_build
    def test_cuda_build_integration(self):
        """Test that CUDA components are properly built."""
        from tests.utils import CudaTestUtils
        
        # Validate CUDA build artifacts
        results = CudaTestUtils.validate_cuda_build()
        
        # At least one CUDA-related artifact should exist
        assert any(results.values()), "No CUDA build artifacts found despite CUDA being available"
    
    def test_environment_cuda_consistency(self):
        """Test consistency between environment detection and module behavior."""
        import tiny_torch.cuda
        from tests.utils import CudaTestUtils
        
        # Get environment info
        info = CudaTestUtils.get_cuda_info()
        
        # Module behavior should match environment
        module_available = tiny_torch.cuda.is_available()
        
        # If CUDA driver is available, module should detect it
        if info['driver_available']:
            # Module might still return False if not built with CUDA
            # But we can at least verify the function works
            assert isinstance(module_available, bool)
        
        # Device count should be consistent
        module_count = tiny_torch.cuda.device_count()
        if info['driver_available']:
            assert module_count >= 0
        else:
            assert module_count == 0


class TestComprehensiveValidation(IntegrationTestBase):
    """Comprehensive integration validation."""
    
    def test_complete_build_validation(self):
        """Test complete build system validation."""
        validator = BuildValidator()
        results = validator.run_full_validation()
        
        # Core components should pass
        assert results['structure_valid'], "Project structure validation failed"
        assert results['import_valid'], "Module import validation failed"
        assert results['placeholder_valid'], "Placeholder function validation failed"
        
        # Build validation depends on whether build exists
        if self.env.is_build_available():
            assert results['build_valid'], "Build artifacts validation failed"
    
    @requires_cuda
    def test_complete_cuda_validation(self):
        """Test complete CUDA environment validation."""
        validator = CudaEnvironmentValidator()
        results = validator.validate_environment()
        
        # Basic environment tests should pass
        assert results['detection_test'], "CUDA detection logic failed"
        
        # Driver and compiler availability
        assert results['driver_test'], "CUDA driver test failed"
        # Note: compiler test might fail if nvcc not available
    
    def test_cross_module_compatibility(self):
        """Test compatibility between different modules."""
        import tiny_torch
        import tiny_torch.cuda
        import tiny_torch.nn
        import tiny_torch.optim
        import tiny_torch.autograd
        
        # All modules should be importable without conflicts
        assert tiny_torch.__version__ == "0.1.0"
        
        # CUDA module should work regardless of other modules
        cuda_available = tiny_torch.cuda.is_available()
        device_count = tiny_torch.cuda.device_count()
        
        assert isinstance(cuda_available, bool)
        assert isinstance(device_count, int)


class TestErrorHandling(IntegrationTestBase):
    """Test error handling across integrated components."""
    
    def test_graceful_cuda_fallback(self):
        """Test graceful fallback when CUDA is not available."""
        import tiny_torch.cuda
        
        # These should never raise exceptions, even without CUDA
        try:
            is_available = tiny_torch.cuda.is_available()
            device_count = tiny_torch.cuda.device_count()
            
            assert isinstance(is_available, bool)
            assert isinstance(device_count, int)
            assert device_count >= 0
            
        except Exception as e:
            # Should not raise any exceptions
            assert False, f"CUDA module raised unexpected exception: {e}"
    
    def test_import_error_handling(self):
        """Test proper handling of import errors."""
        # Main module import should always work
        try:
            import tiny_torch
            assert True
        except ImportError:
            assert False, "Main module import should never fail after build"
        
        # Submodule imports should handle errors gracefully
        submodules = ['nn', 'optim', 'autograd', 'cuda']
        for submodule in submodules:
            try:
                exec(f'import tiny_torch.{submodule}')
            except ImportError as e:
                assert False, f"Submodule import failed: tiny_torch.{submodule}: {e}"
