"""
System-level tests for the complete tiny-torch system.
"""

import os
import tempfile
import shutil
import subprocess
import threading
import gc
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path
import pytest
from tests.utils import SystemTestBase, BuildValidator, CudaEnvironmentValidator


@pytest.mark.slow
class TestPhaseVerification(SystemTestBase):
    """Test phase verification - unified test cases for all project phases."""
    
    def test_phase_1_1_verification(self):
        """Test that Phase 1.1 (Build System Setup) requirements are met."""
        print("=== Phase 1.1 Verification: Build System Setup ===")
        
        # 1. Check required files
        required_files = {
            # Core build files
            "CMakeLists.txt": "CMake configuration file",
            "setup.py": "Python package setup file", 
            "pyproject.toml": "Modern Python project configuration",
            "requirements.txt": "Python dependencies file",
            "requirements-dev.txt": "Development dependencies file",
            "Makefile": "Convenient build commands",
            
            # Configuration files
            ".gitignore": "Git ignore file",
            
            # Documentation
            "LICENSE": "License file",
            "README.md": "Project documentation",
            
            # Scripts and tools
            "tools/build.sh": "Build script",
            "tools/check_env.py": "Independent environment check script",
            
            # Python package structure
            "tiny_torch/__init__.py": "Main module initialization",
            "tiny_torch/nn/__init__.py": "Neural network module",
            "tiny_torch/optim/__init__.py": "Optimizer module", 
            "tiny_torch/autograd/__init__.py": "Autograd module",
            "tiny_torch/py.typed": "Type hint marker file",
            
            # Tests
            "tests/__init__.py": "Test package initialization",
        }
        
        missing_files = []
        project_root = Path(__file__).parent.parent.parent
        
        for file_path, description in required_files.items():
            full_path = project_root / file_path
            if full_path.exists():
                print(f"  [PASS] {file_path} ({description})")
            else:
                print(f"  [FAIL] {file_path} ({description}) - MISSING")
                missing_files.append(file_path)
        
        # 2. Check required directories
        required_dirs = {
            "csrc": "C++ source root directory",
            "csrc/api": "Python API directory",
            "csrc/aten": "ATen implementation directory",
            "csrc/autograd": "Autograd C++ implementation",
            "tiny_torch/nn": "Neural network modules",
            "tests": "Test directory",
        }
        
        missing_dirs = []
        for dir_path, description in required_dirs.items():
            full_path = project_root / dir_path
            if full_path.exists():
                print(f"  [PASS] {dir_path}/ ({description})")
            else:
                print(f"  [FAIL] {dir_path}/ ({description}) - MISSING")
                missing_dirs.append(dir_path)
        
        # 3. Build system validation
        validator = BuildValidator()
        build_results = validator.run_full_validation()
        
        assert build_results['structure_valid'], "Project structure invalid"
        assert build_results['import_valid'], "Module imports invalid"  
        assert build_results['placeholder_valid'], "Placeholder functions invalid"
        
        # 4. Basic functionality check
        try:
            import tiny_torch
            print(f"  [PASS] tiny_torch import successful (v{tiny_torch.__version__})")
            
            # Check submodule imports
            import tiny_torch.nn
            import tiny_torch.optim  
            import tiny_torch.autograd
            print(f"  [PASS] All submodules import successfully")
            
            # Check placeholder functions
            try:
                tiny_torch.tensor([1, 2, 3])
                assert False, "tiny_torch.tensor should raise NotImplementedError"
            except NotImplementedError:
                print(f"  [PASS] tiny_torch.tensor correctly raises NotImplementedError")
        except Exception as e:
            assert False, f"Basic functionality check failed: {e}"
        
        # 5. CUDA validation (if available)
        if self.env.is_cuda_available():
            cuda_validator = CudaEnvironmentValidator()
            cuda_results = cuda_validator.validate_environment()
            
            # At minimum, detection should work
            assert cuda_results['detection_test'], "CUDA detection failed"
        
        # Assert all critical checks passed
        assert len(missing_files) == 0, f"Missing required files: {missing_files}"
        assert len(missing_dirs) == 0, f"Missing required directories: {missing_dirs}"
        
        print("✓ Phase 1.1 Build System Setup - PASSED")
    
    def test_phase_1_2_verification(self):
        """Test that Phase 1.2 (Tensor Core Library - ATen) requirements are met."""
        print("=== Phase 1.2 Verification: Tensor Core Library (ATen) ===")
        
        # This is a placeholder for future Phase 1.2 verification
        # When Phase 1.2 is implemented, add specific checks for:
        # - ATen core tensor operations
        # - Basic tensor creation and manipulation
        # - CPU tensor backend functionality
        
        pytest.skip("Phase 1.2 not yet implemented")
    
    def test_phase_1_3_verification(self):
        """Test that Phase 1.3 (CUDA Integration) requirements are met."""
        print("=== Phase 1.3 Verification: CUDA Integration ===")
        
        # This is a placeholder for future Phase 1.3 verification
        # When Phase 1.3 is implemented, add specific checks for:
        # - CUDA tensor operations
        # - GPU memory management
        # - CUDA kernel compilation and execution
        
        pytest.skip("Phase 1.3 not yet implemented")
    
    def test_phase_2_1_verification(self):
        """Test that Phase 2.1 (Autograd Engine) requirements are met."""
        print("=== Phase 2.1 Verification: Autograd Engine ===")
        
        # This is a placeholder for future Phase 2.1 verification
        # When Phase 2.1 is implemented, add specific checks for:
        # - Automatic differentiation functionality
        # - Gradient computation
        # - Backpropagation engine
        
        pytest.skip("Phase 2.1 not yet implemented")
    
    def test_phase_2_2_verification(self):
        """Test that Phase 2.2 (Neural Network Modules) requirements are met."""
        print("=== Phase 2.2 Verification: Neural Network Modules ===")
        
        # This is a placeholder for future Phase 2.2 verification
        # When Phase 2.2 is implemented, add specific checks for:
        # - Basic neural network layers (Linear, Conv2d, etc.)
        # - Activation functions
        # - Loss functions
        
        pytest.skip("Phase 2.2 not yet implemented")
    
    def test_phase_3_1_verification(self):
        """Test that Phase 3.1 (Optimizers) requirements are met."""
        print("=== Phase 3.1 Verification: Optimizers ===")
        
        # This is a placeholder for future Phase 3.1 verification
        # When Phase 3.1 is implemented, add specific checks for:
        # - SGD optimizer
        # - Adam optimizer
        # - Parameter updates
        
        pytest.skip("Phase 3.1 not yet implemented")

@pytest.mark.slow
class TestCompleteSystem(SystemTestBase):
    """Test the complete system end-to-end."""
    
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


@pytest.mark.slow
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
        
        # Perform repeated operations (reduced iterations for faster tests)
        for _ in range(100):  # Reduced from 1000 to 100
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
        # Allow some increase, but not excessive (adjusted for fewer iterations)
        assert object_increase < 500, f"Possible memory leak: {object_increase} new objects"


@pytest.mark.slow
class TestSystemDocumentation(SystemTestBase):
    """Test system documentation and help features."""
    
    def test_module_help(self):
        """Test that modules provide useful help information."""
        import tiny_torch
        import io
        import sys
        from contextlib import redirect_stdout
        
        # Main module should have docstring
        assert tiny_torch.__doc__ is not None
        
        # Version should be accessible
        assert hasattr(tiny_torch, '__version__')
        
        # Help should work without errors - capture output to avoid pager
        try:
            old_stdout = sys.stdout
            captured_output = io.StringIO()
            
            with redirect_stdout(captured_output):
                help(tiny_torch)
            
            help_text = captured_output.getvalue()
            # Should contain some documentation
            assert len(help_text) > 100, f"Help text too short: {len(help_text)} chars"
            assert "tiny_torch" in help_text.lower(), "Help should mention module name"
            
        except Exception as e:
            assert False, f"help(tiny_torch) failed: {e}"
        finally:
            sys.stdout = old_stdout
    
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
