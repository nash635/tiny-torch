"""
Integration tests for module interactions and API consistency.
"""

from test.utils import IntegrationTestBase


class TestModuleInteractions(IntegrationTestBase):
    """Test interactions between different modules."""
    
    def test_module_independence(self):
        """Test that modules can be imported independently."""
        # Import modules in different orders to test independence
        import tiny_torch.cuda
        import tiny_torch.nn
        import tiny_torch.optim
        import tiny_torch.autograd
        import tiny_torch
        
        # Verify they all imported correctly
        assert hasattr(tiny_torch, '__version__')
        assert hasattr(tiny_torch.cuda, 'is_available')
        assert hasattr(tiny_torch.nn, '__name__')
        assert hasattr(tiny_torch.optim, '__name__')
        assert hasattr(tiny_torch.autograd, '__name__')
    
    def test_api_consistency(self):
        """Test API consistency across modules."""
        import tiny_torch
        
        # Version should be accessible from main module
        version = tiny_torch.__version__
        assert isinstance(version, str)
        assert version == "0.1.0"
        
        # CUDA functions should be consistent
        cuda_available = tiny_torch.cuda.is_available()
        device_count = tiny_torch.cuda.device_count()
        
        # If CUDA is available, device count should be positive
        if cuda_available:
            assert device_count > 0
        else:
            assert device_count == 0
    
    def test_namespace_isolation(self):
        """Test that modules don't pollute each other's namespaces."""
        import tiny_torch.nn
        import tiny_torch.optim
        import tiny_torch.autograd
        import tiny_torch.cuda
        
        # Each module should have its own namespace
        modules = [tiny_torch.nn, tiny_torch.optim, tiny_torch.autograd, tiny_torch.cuda]
        names = [module.__name__ for module in modules]
        
        # All names should be unique
        assert len(names) == len(set(names))
        
        # Each should have proper module name
        expected_names = ['tiny_torch.nn', 'tiny_torch.optim', 'tiny_torch.autograd', 'tiny_torch.cuda']
        for expected in expected_names:
            assert expected in names


class TestAPIWorkflow(IntegrationTestBase):
    """Test complete API workflow scenarios."""
    
    def test_basic_usage_pattern(self):
        """Test basic expected usage pattern."""
        import tiny_torch
        
        # Step 1: Check version
        assert tiny_torch.__version__ == "0.1.0"
        
        # Step 2: Check CUDA availability
        cuda_available = tiny_torch.cuda.is_available()
        device_count = tiny_torch.cuda.device_count()
        
        assert isinstance(cuda_available, bool)
        assert isinstance(device_count, int)
        
        # Step 3: Try to create tensor (should fail gracefully)
        try:
            tensor = tiny_torch.tensor([1, 2, 3])
            assert False, "tensor() should raise NotImplementedError"
        except NotImplementedError:
            pass  # Expected behavior
    
    def test_error_propagation(self):
        """Test that errors propagate correctly through the API."""
        import tiny_torch
        
        # Test various ways to trigger NotImplementedError
        test_cases = [
            lambda: tiny_torch.tensor([1]),
            lambda: tiny_torch.tensor([[1, 2], [3, 4]]),
            lambda: tiny_torch.tensor([1.0, 2.0])
        ]
        
        for test_case in test_cases:
            try:
                test_case()
                assert False, f"Test case should raise NotImplementedError: {test_case}"
            except NotImplementedError:
                pass  # Expected
            except Exception as e:
                assert False, f"Unexpected exception type: {type(e).__name__}: {e}"


class TestPerformanceIntegration(IntegrationTestBase):
    """Test performance-related integration aspects."""
    
    def test_import_performance(self):
        """Test that imports complete in reasonable time."""
        import time
        
        # Test main module import time
        start_time = time.time()
        import tiny_torch
        import_time = time.time() - start_time
        
        # Should import quickly (less than 1 second)
        assert import_time < 1.0, f"Import took too long: {import_time:.2f}s"
    
    def test_function_call_performance(self):
        """Test that basic function calls perform reasonably."""
        import tiny_torch
        import time
        
        # CUDA availability check should be fast
        start_time = time.time()
        for _ in range(100):
            tiny_torch.cuda.is_available()
        call_time = time.time() - start_time
        
        # 100 calls should complete quickly
        assert call_time < 1.0, f"100 CUDA availability checks took too long: {call_time:.2f}s"
    
    def test_memory_usage(self):
        """Test that module loading doesn't consume excessive memory."""
        import gc
        import sys
        
        # Get baseline memory
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Import all modules
        import tiny_torch
        import tiny_torch.nn
        import tiny_torch.optim
        import tiny_torch.autograd
        import tiny_torch.cuda
        
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Should not create an excessive number of objects
        object_increase = final_objects - initial_objects
        assert object_increase < 10000, f"Too many objects created: {object_increase}"
