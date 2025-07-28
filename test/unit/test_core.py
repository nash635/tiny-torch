"""
Unit tests for core tiny-torch module structure and API.
"""

try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    class pytest:
        @staticmethod
        def raises(exception_type):
            class RaisesContext:
                def __enter__(self):
                    return self
                def __exit__(self, exc_type, exc_val, exc_tb):
                    if exc_type is None:
                        raise AssertionError(f"Expected {exception_type.__name__} but no exception was raised")
                    return issubclass(exc_type, exception_type)
            return RaisesContext()

from test.utils import UnitTestBase, TestEnvironment


class TestCoreModule(UnitTestBase):
    """Test core module structure and imports."""
    
    def test_main_module_import(self):
        """Test that main tiny_torch module can be imported."""
        import tiny_torch
        assert hasattr(tiny_torch, '__version__')
        assert tiny_torch.__version__ == "0.1.0"
    
    def test_submodule_imports(self):
        """Test that all expected submodules can be imported."""
        import tiny_torch.nn
        import tiny_torch.optim
        import tiny_torch.autograd
        import tiny_torch.cuda
        
        # Verify they are modules
        assert hasattr(tiny_torch.nn, '__name__')
        assert hasattr(tiny_torch.optim, '__name__')
        assert hasattr(tiny_torch.autograd, '__name__')
        assert hasattr(tiny_torch.cuda, '__name__')
    
    def test_placeholder_tensor_function(self):
        """Test that tensor function correctly raises NotImplementedError."""
        import tiny_torch
        
        with pytest.raises(NotImplementedError):
            tiny_torch.tensor([1, 2, 3])
    
    def test_cuda_module_interface(self):
        """Test CUDA module basic interface."""
        import tiny_torch.cuda
        
        # These should be callable without errors
        is_available = tiny_torch.cuda.is_available()
        assert isinstance(is_available, bool)
        
        device_count = tiny_torch.cuda.device_count()
        assert isinstance(device_count, int)
        assert device_count >= 0


class TestCoreAPI(UnitTestBase):
    """Test core API structure and behavior."""
    
    def test_module_attributes(self):
        """Test that modules have expected attributes."""
        import tiny_torch
        
        # Check version is properly set
        assert hasattr(tiny_torch, '__version__')
        assert isinstance(tiny_torch.__version__, str)
        
        # Check that tensor function exists
        assert hasattr(tiny_torch, 'tensor')
        assert callable(tiny_torch.tensor)
    
    def test_error_handling(self):
        """Test proper error handling for unimplemented features."""
        import tiny_torch
        
        # Test various parameter combinations for tensor
        with pytest.raises(NotImplementedError):
            tiny_torch.tensor([1, 2, 3, 4])
        
        with pytest.raises(NotImplementedError):
            tiny_torch.tensor([[1, 2], [3, 4]])
        
        with pytest.raises(NotImplementedError):
            tiny_torch.tensor([1.0, 2.0, 3.0])
    
    def test_module_documentation(self):
        """Test that modules have proper documentation."""
        import tiny_torch
        
        # Main module should have docstring
        assert tiny_torch.__doc__ is not None
        
        # Submodules should be importable
        submodules = ['nn', 'optim', 'autograd', 'cuda']
        for name in submodules:
            module = getattr(tiny_torch, name)
            assert hasattr(module, '__name__')
            assert module.__name__ == f'tiny_torch.{name}'
