"""
System-level tests for the complete tiny-torch system.
"""

import os
import tempfile
import shutil
from pathlib import Path
from test.utils import SystemTestBase, BuildValidator, CudaEnvironmentValidator


class TestCompleteSystem(SystemTestBase):
    """Test the complete system end-to-end."""
    
    def test_phase_1_1_verification(self):
        """Test that Phase 1.1 requirements are met."""
        # This replicates the verify_phase1_1.py functionality
        
        # 1. Build system validation
        validator = BuildValidator()
        build_results = validator.run_full_validation()
        
        assert build_results['structure_valid'], "Project structure invalid"
        assert build_results['import_valid'], "Module imports invalid"  
        assert build_results['placeholder_valid'], "Placeholder functions invalid"
        
        # 2. CUDA validation (if available)
        if self.env.is_cuda_available():
            cuda_validator = CudaEnvironmentValidator()
            cuda_results = cuda_validator.validate_environment()
            
            # At minimum, detection should work
            assert cuda_results['detection_test'], "CUDA detection failed"
        
        # 3. Overall system health
        assert build_results['overall_valid'], "Overall system validation failed"
    
    def test_installation_verification(self):
        """Test installation verification workflow."""
        import tiny_torch
        
        # Verify version
        assert tiny_torch.__version__ == "0.1.0"
        
        # Verify CUDA module
        cuda_available = tiny_torch.cuda.is_available()
        device_count = tiny_torch.cuda.device_count()
        
        assert isinstance(cuda_available, bool)
        assert isinstance(device_count, int)
        assert device_count >= 0
        
        # Verify placeholder behavior
        try:
            tiny_torch.tensor([1, 2, 3])
            assert False, "Should raise NotImplementedError"
        except NotImplementedError:
            pass
        
        print(f"✓ Installation verification passed")
        print(f"  Version: {tiny_torch.__version__}")
        print(f"  CUDA Available: {cuda_available}")
        print(f"  GPU Count: {device_count}")
    
    def test_build_reproducibility(self):
        """Test that builds are reproducible."""
        validator = BuildValidator()
        
        # Get initial state
        initial_results = validator.run_full_validation()
        
        # Clean and rebuild (if possible)
        if hasattr(validator, 'rebuild'):
            rebuild_results = validator.rebuild()
            
            if rebuild_results['build_success']:
                # Verify results are consistent
                new_results = validator.run_full_validation()
                
                assert new_results['structure_valid'] == initial_results['structure_valid']
                assert new_results['import_valid'] == initial_results['import_valid']


class TestSystemRobustness(SystemTestBase):
    """Test system robustness and error handling."""
    
    def test_environment_variations(self):
        """Test behavior under different environment conditions."""
        import tiny_torch
        
        # Test with different CUDA environment variables
        original_env = os.environ.get('WITH_CUDA')
        
        try:
            # Test with CUDA explicitly disabled
            os.environ['WITH_CUDA'] = '0'
            # Module should still work
            assert tiny_torch.cuda.device_count() >= 0
            
            # Test with CUDA explicitly enabled  
            os.environ['WITH_CUDA'] = '1'
            # Should not crash even if CUDA not available
            assert tiny_torch.cuda.device_count() >= 0
            
        finally:
            # Restore original environment
            if original_env is not None:
                os.environ['WITH_CUDA'] = original_env
            elif 'WITH_CUDA' in os.environ:
                del os.environ['WITH_CUDA']
    
    def test_concurrent_usage(self):
        """Test concurrent usage scenarios."""
        import threading
        import tiny_torch
        
        results = []
        errors = []
        
        def worker():
            try:
                # Each thread should be able to use the module
                version = tiny_torch.__version__
                cuda_available = tiny_torch.cuda.is_available()
                device_count = tiny_torch.cuda.device_count()
                
                results.append({
                    'version': version,
                    'cuda_available': cuda_available,
                    'device_count': device_count
                })
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify no errors and consistent results
        assert len(errors) == 0, f"Concurrent usage errors: {errors}"
        assert len(results) == 5, "Not all threads completed"
        
        # All results should be identical
        first_result = results[0]
        for result in results[1:]:
            assert result == first_result, "Inconsistent results between threads"
    
    def test_memory_stability(self):
        """Test memory stability under repeated operations."""
        import gc
        import tiny_torch
        
        # Get baseline
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Perform repeated operations
        for _ in range(1000):
            version = tiny_torch.__version__
            cuda_available = tiny_torch.cuda.is_available()
            device_count = tiny_torch.cuda.device_count()
            
            # Try tensor creation (should fail)
            try:
                tiny_torch.tensor([1, 2, 3])
            except NotImplementedError:
                pass
        
        # Check for memory leaks
        gc.collect()
        final_objects = len(gc.get_objects())
        
        object_increase = final_objects - initial_objects
        # Allow some increase, but not excessive
        assert object_increase < 1000, f"Possible memory leak: {object_increase} new objects"


class TestSystemDocumentation(SystemTestBase):
    """Test system documentation and help features."""
    
    def test_module_help(self):
        """Test that modules provide useful help information."""
        import tiny_torch
        
        # Main module should have docstring
        assert tiny_torch.__doc__ is not None
        
        # Version should be accessible
        assert hasattr(tiny_torch, '__version__')
        
        # Help should work without errors
        try:
            help_text = help(tiny_torch)
            # help() returns None but shouldn't crash
        except Exception as e:
            assert False, f"help(tiny_torch) failed: {e}"
    
    def test_error_messages(self):
        """Test that error messages are informative."""
        import tiny_torch
        
        try:
            tiny_torch.tensor([1, 2, 3])
            assert False, "Should raise NotImplementedError"
        except NotImplementedError as e:
            # Error message should be informative
            error_msg = str(e)
            assert len(error_msg) > 0, "Empty error message"
    
    def test_module_introspection(self):
        """Test module introspection capabilities."""
        import tiny_torch
        
        # Should be able to inspect modules
        assert hasattr(tiny_torch, '__name__')
        assert hasattr(tiny_torch, '__file__')
        assert hasattr(tiny_torch, '__package__')
        
        # Submodules should be accessible
        submodules = ['nn', 'optim', 'autograd', 'cuda']
        for name in submodules:
            assert hasattr(tiny_torch, name)
            submodule = getattr(tiny_torch, name)
            assert hasattr(submodule, '__name__')
            assert submodule.__name__ == f'tiny_torch.{name}'
