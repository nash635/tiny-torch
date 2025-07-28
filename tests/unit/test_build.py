"""
Unit tests for build system components.
"""

from tests.utils import BuildTestBase, requires_build, BuildTestUtils


class TestProjectStructure(BuildTestBase):
    """Test basic project structure."""
    
    def test_required_files_exist(self):
        """Test that all required project files exist."""
        results = BuildTestUtils.validate_project_structure()
        
        required_files = BuildTestUtils.get_required_project_files()
        for file_name in required_files:
            key = f'file_{file_name}'
            assert key in results, f"Missing validation for {file_name}"
            assert results[key], f"Required file missing: {file_name}"
    
    def test_required_directories_exist(self):
        """Test that all required directories exist.""" 
        results = BuildTestUtils.validate_project_structure()
        
        required_dirs = BuildTestUtils.get_required_directories()
        for dir_name in required_dirs:
            key = f'dir_{dir_name}'
            assert key in results, f"Missing validation for {dir_name}"
            assert results[key], f"Required directory missing: {dir_name}"
    
    def test_source_code_structure(self):
        """Test source code directory structure."""
        project_root = self.env.get_project_root()
        
        # Check C++ source structure
        csrc_dir = project_root / 'csrc'
        assert csrc_dir.exists()
        assert (csrc_dir / 'api').exists()
        assert (csrc_dir / 'aten').exists()
        assert (csrc_dir / 'autograd').exists()
        
        # Check Python package structure
        python_dir = project_root / 'tiny_torch'
        assert python_dir.exists()
        assert (python_dir / '__init__.py').exists()
        assert (python_dir / 'nn').exists()
        assert (python_dir / 'optim').exists()
        assert (python_dir / 'autograd').exists()
        assert (python_dir / 'cuda').exists()


class TestBuildArtifacts(BuildTestBase):
    """Test build artifacts and outputs."""
    
    @requires_build
    def test_build_artifacts_exist(self):
        """Test that expected build artifacts exist."""
        results = BuildTestUtils.validate_build_artifacts()
        
        # At least some artifacts should exist
        assert any(results.values()), "No build artifacts found"
        
        # Check specific important artifacts
        if results.get('cpp_library'):
            # If C++ library exists, verify it's not empty
            artifacts = BuildTestUtils.get_expected_build_artifacts()
            lib_path = artifacts['cpp_library']
            assert self.env.get_project_root().joinpath(lib_path).stat().st_size > 0
    
    @requires_build  
    def test_cmake_build_files(self):
        """Test CMake build files exist."""
        cmake_dir = self.env.get_cmake_dir()
        
        # CMake should generate these files
        assert (cmake_dir / 'CMakeCache.txt').exists()
        assert (cmake_dir / 'cmake_install.cmake').exists()
        
        # Should have build configuration
        ninja_file = cmake_dir / 'build.ninja'
        if ninja_file.exists():
            # If using Ninja, check it's valid
            assert ninja_file.stat().st_size > 0


class TestModuleImports(BuildTestBase):
    """Test Python module imports after build."""
    
    def test_basic_imports(self):
        """Test basic module import functionality."""
        results = BuildTestUtils.test_module_imports()
        
        assert results['main_module'], "Failed to import main tiny_torch module"
        assert results['version_check'], f"Version mismatch: expected 0.1.0, got {results.get('version', 'None')}"
        
        # Check submodules
        for name, success in results['submodules'].items():
            assert success, f"Failed to import tiny_torch.{name}"
    
    def test_placeholder_behavior(self):
        """Test placeholder function behavior."""
        results = BuildTestUtils.test_placeholder_functions()
        
        for name, success in results.items():
            assert success, f"Placeholder function {name} did not behave correctly"
