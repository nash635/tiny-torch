"""
Build system test utilities.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
from .common import TestEnvironment, CommandRunner


class BuildTestUtils:
    """Build system testing utilities."""
    
    @staticmethod
    def get_required_project_files() -> List[str]:
        """Get list of required project files."""
        return [
            'CMakeLists.txt',
            'setup.py', 
            'Makefile',
            'pyproject.toml',
            'requirements.txt',
            'requirements-dev.txt'
        ]
    
    @staticmethod
    def get_required_directories() -> List[str]:
        """Get list of required source directories."""
        return [
            'csrc',
            'tiny_torch',
            'test',
            'docs'
        ]
    
    @staticmethod
    def get_expected_build_artifacts() -> Dict[str, str]:
        """Get expected build artifacts and their locations."""
        cmake_dir = TestEnvironment.get_cmake_dir()
        return {
            'cpp_library': str(cmake_dir / 'libtiny_torch_cpp.a'),
            'python_extension': 'tiny_torch/_C.cpython-*.so',
            'ninja_build': str(cmake_dir / 'build.ninja'),
            'cmake_cache': str(cmake_dir / 'CMakeCache.txt'),
            'cpp_tests': str(cmake_dir / 'test' / 'cpp' / 'tiny_torch_cpp_tests')
        }
    
    @staticmethod
    def validate_project_structure() -> Dict[str, bool]:
        """Validate basic project structure."""
        results = {}
        project_root = TestEnvironment.get_project_root()
        
        # Check required files
        for file_name in BuildTestUtils.get_required_project_files():
            file_path = project_root / file_name
            results[f'file_{file_name}'] = file_path.exists()
        
        # Check required directories
        for dir_name in BuildTestUtils.get_required_directories():
            dir_path = project_root / dir_name
            results[f'dir_{dir_name}'] = dir_path.exists() and dir_path.is_dir()
        
        return results
    
    @staticmethod
    def validate_build_artifacts() -> Dict[str, bool]:
        """Validate build artifacts exist."""
        results = {}
        artifacts = BuildTestUtils.get_expected_build_artifacts()
        
        for name, pattern in artifacts.items():
            if '*' in pattern:
                # Handle glob patterns
                parent_dir = Path(pattern).parent
                pattern_name = Path(pattern).name
                if parent_dir.exists():
                    matches = list(parent_dir.glob(pattern_name))
                    results[name] = len(matches) > 0
                else:
                    results[name] = False
            else:
                # Handle exact paths
                results[name] = Path(pattern).exists()
        
        return results
    
    @staticmethod
    def test_module_imports() -> Dict[str, Any]:
        """Test that Python modules can be imported."""
        results = {
            'main_module': False,
            'submodules': {},
            'version_check': False,
            'errors': {}
        }
        
        # Test main module import
        try:
            import tiny_torch
            results['main_module'] = True
            
            # Test version
            if hasattr(tiny_torch, '__version__'):
                results['version_check'] = tiny_torch.__version__ == "0.1.0"
                results['version'] = tiny_torch.__version__
            else:
                results['errors']['version'] = 'No __version__ attribute found'
                
        except ImportError as e:
            results['errors']['main_module'] = str(e)
        
        # Test submodule imports
        submodules = ['nn', 'optim', 'autograd', 'cuda']
        for submodule in submodules:
            try:
                exec(f'import tiny_torch.{submodule}')
                results['submodules'][submodule] = True
            except ImportError as e:
                results['submodules'][submodule] = False
                results['errors'][f'submodule_{submodule}'] = str(e)
        
        return results
    
    @staticmethod
    def test_placeholder_functions() -> Dict[str, bool]:
        """Test that placeholder functions correctly raise NotImplementedError."""
        results = {}
        
        try:
            import tiny_torch
            
            # Test tensor function
            try:
                tiny_torch.tensor([1, 2, 3])
                results['tensor_placeholder'] = False  # Should have raised NotImplementedError
            except NotImplementedError:
                results['tensor_placeholder'] = True
            except Exception as e:
                results['tensor_placeholder'] = False
                
        except ImportError:
            results['tensor_placeholder'] = False
            
        return results


class BuildValidator:
    """Comprehensive build system validator."""
    
    def __init__(self):
        self.runner = CommandRunner()
        self.env = TestEnvironment()
    
    def run_full_validation(self) -> Dict[str, Any]:
        """Run complete build system validation."""
        results = {
            'structure_valid': False,
            'build_valid': False,
            'import_valid': False,
            'placeholder_valid': False,
            'overall_valid': False,
            'details': {}
        }
        
        # Test 1: Project structure
        structure_results = BuildTestUtils.validate_project_structure()
        results['structure_valid'] = all(structure_results.values())
        results['details']['structure'] = structure_results
        
        # Test 2: Build artifacts
        if self.env.is_build_available():
            build_results = BuildTestUtils.validate_build_artifacts()
            results['build_valid'] = all(build_results.values())
            results['details']['build'] = build_results
        else:
            results['build_valid'] = False
            results['details']['build'] = {'message': 'Build directory not found'}
        
        # Test 3: Module imports
        import_results = BuildTestUtils.test_module_imports()
        results['import_valid'] = (
            import_results['main_module'] and 
            import_results['version_check'] and
            all(import_results['submodules'].values())
        )
        results['details']['imports'] = import_results
        
        # Test 4: Placeholder functions
        placeholder_results = BuildTestUtils.test_placeholder_functions()
        results['placeholder_valid'] = all(placeholder_results.values())
        results['details']['placeholders'] = placeholder_results
        
        # Overall validation
        results['overall_valid'] = all([
            results['structure_valid'],
            results['build_valid'],
            results['import_valid'],
            results['placeholder_valid']
        ])
        
        return results
    
    def clean_build(self) -> bool:
        """Clean build directories."""
        try:
            build_dir = self.env.get_build_dir()
            if build_dir.exists():
                shutil.rmtree(build_dir)
            
            # Clean Python cache and extensions
            project_root = self.env.get_project_root()
            for pattern in ['**/__pycache__', '**/*.pyc', '**/*.so']:
                for path in project_root.glob(pattern):
                    if path.is_file():
                        path.unlink()
                    elif path.is_dir():
                        shutil.rmtree(path)
            
            return True
        except Exception:
            return False
    
    def rebuild(self) -> Dict[str, Any]:
        """Clean and rebuild the project."""
        results = {
            'clean_success': False,
            'build_success': False,
            'build_output': '',
            'build_errors': ''
        }
        
        # Clean first
        results['clean_success'] = self.clean_build()
        
        if results['clean_success']:
            # Rebuild
            try:
                project_root = self.env.get_project_root()
                result = self.runner.run(['make', 'build'], cwd=str(project_root), check=False)
                results['build_success'] = result.returncode == 0
                results['build_output'] = result.stdout
                results['build_errors'] = result.stderr
            except Exception as e:
                results['build_errors'] = str(e)
        
        return results
