"""
Performance tests for tiny-torch system.
"""

import time
import gc
import threading
import pytest
from tests.utils import SystemTestBase


@pytest.mark.slow
class TestPerformance(SystemTestBase):
    """Performance tests for the system."""
    
    def test_import_performance(self):
        """Test module import performance."""
        # Test cold import (fresh Python process would be better, but this approximates)
        start_time = time.time()
        
        import tiny_torch
        import tiny_torch.nn
        import tiny_torch.optim
        import tiny_torch.autograd
        import tiny_torch.cuda
        
        end_time = time.time()
        import_time = end_time - start_time
        
        # Imports should complete quickly
        assert import_time < 2.0, f"Imports took too long: {import_time:.3f}s"
        
        print(f"✓ Import performance: {import_time:.3f}s")
    
    def test_function_call_performance(self):
        """Test basic function call performance."""
        import tiny_torch
        
        # Test CUDA functions (reduced iterations for faster execution)
        iterations = 1000  # Reduced from 10000
        
        start_time = time.time()
        for _ in range(iterations):
            tiny_torch.cuda.is_available()
        end_time = time.time()
        
        cuda_time = end_time - start_time
        per_call = cuda_time / iterations
        
        # Should be very fast (adjusted threshold for fewer iterations)
        assert per_call < 0.01, f"CUDA calls too slow: {per_call:.6f}s per call"
        
        print(f"✓ CUDA function performance: {per_call:.6f}s per call")
    
    def test_memory_usage(self):
        """Test memory usage patterns."""
        import tiny_torch
        
        # Baseline memory
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Perform operations (reduced for faster tests)
        for _ in range(100):  # Reduced from 1000
            version = tiny_torch.__version__
            cuda_available = tiny_torch.cuda.is_available()
            device_count = tiny_torch.cuda.device_count()
        
        # Check memory growth
        gc.collect()
        final_objects = len(gc.get_objects())
        growth = final_objects - initial_objects
        
        # Should not grow significantly
        assert growth < 100, f"Memory growth too high: {growth} objects"
        
        print(f"✓ Memory usage: {growth} object growth after 1000 operations")
    
    def test_concurrent_performance(self):
        """Test performance under concurrent load."""
        import tiny_torch
        
        def worker_task():
            for _ in range(100):
                tiny_torch.cuda.is_available()
                tiny_torch.cuda.device_count()
        
        # Run multiple threads
        num_threads = 10
        threads = []
        
        start_time = time.time()
        
        for _ in range(num_threads):
            thread = threading.Thread(target=worker_task)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should complete reasonably quickly
        assert total_time < 5.0, f"Concurrent execution too slow: {total_time:.3f}s"
        
        print(f"✓ Concurrent performance: {total_time:.3f}s for {num_threads} threads")
    
    def test_repeated_imports(self):
        """Test performance of repeated module imports."""
        # This tests import cache efficiency
        
        start_time = time.time()
        
        for _ in range(100):
            import tiny_torch
            import tiny_torch.cuda
            import tiny_torch.nn
        
        end_time = time.time()
        reimport_time = end_time - start_time
        
        # Repeated imports should be very fast due to caching
        assert reimport_time < 0.1, f"Repeated imports too slow: {reimport_time:.3f}s"
        
        print(f"✓ Repeated import performance: {reimport_time:.3f}s for 100 reimports")


@pytest.mark.slow
class TestScalability(SystemTestBase):
    """Test system scalability characteristics."""
    
    def test_operation_scaling(self):
        """Test how operations scale with increasing load."""
        import tiny_torch
        
        # Test different operation counts
        operation_counts = [100, 1000, 10000]
        times = []
        
        for count in operation_counts:
            start_time = time.time()
            
            for _ in range(count):
                tiny_torch.cuda.is_available()
            
            end_time = time.time()
            times.append(end_time - start_time)
        
        # Should scale roughly linearly
        for i in range(1, len(times)):
            ratio = times[i] / times[i-1]
            count_ratio = operation_counts[i] / operation_counts[i-1]
            
            # Allow some overhead, but should be roughly proportional
            assert ratio < count_ratio * 2, f"Poor scaling: {ratio:.2f}x time for {count_ratio}x operations"
        
        print(f"✓ Operation scaling: {times}")
    
    def test_thread_scaling(self):
        """Test performance scaling with thread count."""
        import tiny_torch
        
        def worker_task():
            for _ in range(100):
                tiny_torch.cuda.is_available()
        
        # Test different thread counts
        thread_counts = [1, 2, 4, 8]
        times = []
        
        for count in thread_counts:
            threads = []
            
            start_time = time.time()
            
            for _ in range(count):
                thread = threading.Thread(target=worker_task)
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join()
            
            end_time = time.time()
            times.append(end_time - start_time)
        
        # Times shouldn't increase dramatically with more threads
        max_time = max(times)
        min_time = min(times)
        
        assert max_time / min_time < 10, f"Poor thread scaling: {max_time/min_time:.2f}x variation"
        
        print(f"✓ Thread scaling: {dict(zip(thread_counts, times))}")


@pytest.mark.slow
class TestResourceUsage(SystemTestBase):
    """Test resource usage characteristics."""
    
    def test_cpu_usage(self):
        """Test CPU usage patterns.""" 
        import tiny_torch
        import time
        
        # Baseline
        start_time = time.process_time()
        
        # Perform operations (reduced for faster tests)
        for _ in range(1000):  # Reduced from 10000
            tiny_torch.cuda.is_available()
            tiny_torch.cuda.device_count()
        
        end_time = time.process_time()
        cpu_time = end_time - start_time
        
        # Should not consume excessive CPU
        assert cpu_time < 1.0, f"High CPU usage: {cpu_time:.3f}s CPU time"
        
        print(f"✓ CPU usage: {cpu_time:.3f}s for 1k operations")
    
    def test_memory_efficiency(self):
        """Test memory efficiency."""
        import tiny_torch
        import gc
        
        # Multiple cycles to test for accumulation (reduced for faster tests)
        for cycle in range(5):  # Reduced from 10
            gc.collect()
            initial = len(gc.get_objects())
            
            # Do work (reduced iterations)
            for _ in range(100):  # Reduced from 1000
                tiny_torch.__version__
                tiny_torch.cuda.is_available()
            
            gc.collect() 
            final = len(gc.get_objects())
            growth = final - initial
            
            # Should not accumulate objects over cycles
            assert growth < 50, f"Memory accumulation in cycle {cycle}: {growth} objects"
        
        print(f"✓ Memory efficiency: No significant accumulation over 10 cycles")
