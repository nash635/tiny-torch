"""
Unified test runner for tiny-torch test suite.
Provides organized test execution with different test categories.
"""

import sys
import argparse
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from test.utils import TestEnvironment, CommandRunner


class TestRunner:
    """Unified test runner for the tiny-torch test suite."""
    
    def __init__(self):
        self.env = TestEnvironment()
        self.runner = CommandRunner()
        self.test_dir = Path(__file__).parent
        
    def run_unit_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run unit tests."""
        print("🧪 Running Unit Tests...")
        return self._run_pytest(
            self.test_dir / "unit",
            "Unit Tests",
            verbose=verbose
        )
    
    def run_integration_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run integration tests."""
        print("🔗 Running Integration Tests...")
        return self._run_pytest(
            self.test_dir / "integration", 
            "Integration Tests",
            verbose=verbose
        )
    
    def run_system_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run system tests."""
        print("🖥️  Running System Tests...")
        return self._run_pytest(
            self.test_dir / "system",
            "System Tests", 
            verbose=verbose
        )
    
    def run_cpp_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run C++ tests."""
        print("⚙️  Running C++ Tests...")
        
        if not self.env.is_build_available():
            return {
                'success': False,
                'message': 'Build artifacts not available',
                'details': 'Run make build first'
            }
        
        cpp_test_binary = self.env.get_cmake_dir() / "test" / "cpp" / "tiny_torch_cpp_tests"
        
        if not cpp_test_binary.exists():
            return {
                'success': False,
                'message': 'C++ test binary not found',
                'details': f'Expected: {cpp_test_binary}'
            }
        
        try:
            result = self.runner.run([str(cpp_test_binary)], check=False)
            return {
                'success': result.returncode == 0,
                'message': 'C++ tests completed',
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Failed to run C++ tests: {e}',
                'details': str(e)
            }
    
    def run_cuda_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run CUDA-specific tests."""
        print("🚀 Running CUDA Tests...")
        
        if not self.env.is_cuda_available():
            return {
                'success': True,
                'message': 'CUDA not available, skipping CUDA tests',
                'skipped': True
            }
        
        # Run CUDA unit tests
        return self._run_pytest(
            self.test_dir / "unit" / "test_cuda.py",
            "CUDA Tests",
            verbose=verbose
        )
    
    def run_performance_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run performance tests."""
        print("📊 Running Performance Tests...")
        return self._run_pytest(
            self.test_dir / "system" / "test_performance.py",
            "Performance Tests",
            verbose=verbose
        )
    
    def run_all_tests(self, verbose: bool = False, skip_slow: bool = False) -> Dict[str, Any]:
        """Run complete test suite."""
        print("🎯 Running Complete Test Suite...")
        
        results = {}
        overall_success = True
        
        # Run test categories
        test_categories = [
            ("unit", self.run_unit_tests),
            ("integration", self.run_integration_tests),
            ("system", self.run_system_tests),
            ("cpp", self.run_cpp_tests),
        ]
        
        if not skip_slow:
            test_categories.append(("performance", self.run_performance_tests))
        
        if self.env.is_cuda_available():
            test_categories.append(("cuda", self.run_cuda_tests))
        
        for category, test_func in test_categories:
            print(f"\n{'='*50}")
            result = test_func(verbose=verbose)
            results[category] = result
            
            if not result.get('success', False) and not result.get('skipped', False):
                overall_success = False
                
            self._print_result(category, result)
        
        results['overall'] = {
            'success': overall_success,
            'categories': len(test_categories),
            'passed': sum(1 for r in results.values() if r.get('success', False)),
            'skipped': sum(1 for r in results.values() if r.get('skipped', False))
        }
        
        return results
    
    def run_quick_check(self) -> Dict[str, Any]:
        """Run quick smoke tests."""
        print("⚡ Running Quick Check...")
        
        results = {}
        
        # Test 1: Basic imports
        try:
            import tiny_torch
            results['import'] = {'success': True, 'version': tiny_torch.__version__}
        except Exception as e:
            results['import'] = {'success': False, 'error': str(e)}
        
        # Test 2: CUDA check
        try:
            import tiny_torch.cuda
            cuda_available = tiny_torch.cuda.is_available()
            device_count = tiny_torch.cuda.device_count()
            results['cuda'] = {
                'success': True,
                'available': cuda_available,
                'device_count': device_count
            }
        except Exception as e:
            results['cuda'] = {'success': False, 'error': str(e)}
        
        # Test 3: Placeholder functions
        try:
            import tiny_torch
            tiny_torch.tensor([1, 2, 3])
            results['placeholder'] = {'success': False, 'error': 'Should raise NotImplementedError'}
        except NotImplementedError:
            results['placeholder'] = {'success': True}
        except Exception as e:
            results['placeholder'] = {'success': False, 'error': str(e)}
        
        overall_success = all(r.get('success', False) for r in results.values())
        results['overall'] = {'success': overall_success}
        
        return results
    
    def _run_pytest(self, path: Path, name: str, markers: Optional[List[str]] = None, 
                   verbose: bool = False) -> Dict[str, Any]:
        """Run pytest on specified path."""
        cmd = ["python", "-m", "pytest", str(path)]
        
        if markers:
            cmd.extend(markers)
        
        if verbose:
            cmd.append("-v")
        else:
            cmd.append("-q")
        
        # Add coverage if available
        cmd.extend(["--tb=short"])
        
        try:
            result = self.runner.run(cmd, cwd=str(PROJECT_ROOT), check=False)
            
            return {
                'success': result.returncode == 0,
                'message': f'{name} completed',
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Failed to run {name}: {e}',
                'error': str(e)
            }
    
    def _print_result(self, category: str, result: Dict[str, Any]):
        """Print test result summary."""
        if result.get('skipped'):
            print(f"⏭️  {category.title()}: SKIPPED - {result.get('message', '')}")
        elif result.get('success'):
            print(f"✅ {category.title()}: PASSED")
        else:
            print(f"❌ {category.title()}: FAILED - {result.get('message', '')}")
            if 'stderr' in result and result['stderr']:
                print(f"   Error: {result['stderr'][:200]}...")


def main():
    """Main test runner entry point."""
    parser = argparse.ArgumentParser(description='Tiny-Torch Test Runner')
    parser.add_argument('--category', choices=['unit', 'integration', 'system', 'cpp', 'cuda', 'performance', 'all', 'quick'], 
                       default='all', help='Test category to run')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--skip-slow', action='store_true', help='Skip slow tests')
    
    args = parser.parse_args()
    
    runner = TestRunner()
    
    print(f"🔬 Tiny-Torch Test Runner")
    print(f"Project: {PROJECT_ROOT}")
    print(f"CUDA Available: {runner.env.is_cuda_available()}")
    print(f"Build Available: {runner.env.is_build_available()}")
    print()
    
    start_time = time.time()
    
    # Run requested tests
    if args.category == 'unit':
        results = runner.run_unit_tests(verbose=args.verbose)
    elif args.category == 'integration':
        results = runner.run_integration_tests(verbose=args.verbose)
    elif args.category == 'system':
        results = runner.run_system_tests(verbose=args.verbose)
    elif args.category == 'cpp':
        results = runner.run_cpp_tests(verbose=args.verbose)
    elif args.category == 'cuda':
        results = runner.run_cuda_tests(verbose=args.verbose)
    elif args.category == 'performance':
        results = runner.run_performance_tests(verbose=args.verbose)
    elif args.category == 'quick':
        results = runner.run_quick_check()
    else:  # all
        results = runner.run_all_tests(verbose=args.verbose, skip_slow=args.skip_slow)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n{'='*50}")
    print(f"🏁 Test Results Summary")
    print(f"Duration: {duration:.2f}s")
    
    if args.category == 'all':
        overall = results.get('overall', {})
        print(f"Overall: {'✅ PASSED' if overall.get('success') else '❌ FAILED'}")
        print(f"Categories: {overall.get('passed', 0)}/{overall.get('categories', 0)} passed")
        if overall.get('skipped', 0) > 0:
            print(f"Skipped: {overall.get('skipped', 0)}")
    elif args.category == 'quick':
        overall = results.get('overall', {})
        print(f"Quick Check: {'✅ PASSED' if overall.get('success') else '❌ FAILED'}")
    else:
        print(f"Status: {'✅ PASSED' if results.get('success') else '❌ FAILED'}")
    
    # Exit with appropriate code
    if args.category == 'all':
        sys.exit(0 if results.get('overall', {}).get('success') else 1)
    elif args.category == 'quick':
        sys.exit(0 if results.get('overall', {}).get('success') else 1)
    else:
        sys.exit(0 if results.get('success') else 1)


if __name__ == '__main__':
    main()
