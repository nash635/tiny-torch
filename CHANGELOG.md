# Tiny-Torch Changelog

All notable changes to this project will be documented in this file.

## v0.1.1 (2025-06-17) - Phase 1.1 Complete 🎉

### 🎯 Phase 1.1: Build System Setup - COMPLETED
**Major Milestone**: Successfully established complete build infrastructure and resolved all critical build errors.

### ✅ Build System Fixes
- **CMake Threading Support**: Added `find_package(Threads REQUIRED)` to resolve pthread linking errors
- **GoogleTest Simplification**: Replaced complex GoogleTest setup with simple assert-based testing framework
- **Python Extension Build**: Fixed variable scoping issues in setup.py (`WITH_CUDA` UnboundLocalError)
- **BLAS/LAPACK Integration**: Made dependencies optional with graceful fallback when not available
- **CUDA Source Management**: Properly separated CUDA sources between CMake and Python builds

### 🏗️ Build Artifacts Successfully Created
- `build/libtiny_torch_cpp.a` - C++ static library (27 source files compiled)
- `build/_C.cpython-310-x86_64-linux-gnu.so` - Python extension module
- `build/test/cpp/tiny_torch_cpp_tests` - C++ test executable
- Complete source file structure (20+ placeholder C++ files with proper namespaces)

### 🧪 Verification & Testing
- **Phase 1.1 Verification**: All 4 verification checks pass (files, directories, build, functionality)
- **Build Environment**: CMake + Python setuptools dual build system working
- **Test Framework**: C++ and Python testing infrastructure established
- **Import Testing**: Python module imports successfully with proper error handling

### 📁 Source Structure Completed
- **ATen Core**: Tensor, TensorImpl, Storage, Allocator implementations (placeholders)
- **Autograd Engine**: Variable, Function, Engine core components (placeholders)
- **API Bindings**: Python C API bindings for tensor operations (placeholders)
- **CUDA Support**: CUDA kernel implementations for core operations (placeholders)
- **Native Operations**: CPU implementations for arithmetic, linear algebra, activations (placeholders)

### 🛠️ Development Environment
- **Configuration Files**: .gitignore, .clang-format, .pre-commit-config.yaml, .editorconfig
- **CI/CD Pipeline**: GitHub Actions workflow for automated testing
- **Development Tools**: Makefile with convenient build/test/clean commands
- **Code Quality**: Pre-commit hooks for formatting and style checks

### 📊 Progress Tracking
- **File Count**: 27/27 required source files created and building
- **Directory Structure**: 14/14 required directories established
- **Build System**: 100% functional (CMake + Python setuptools)
- **Phase 1.1 Status**: ✅ COMPLETE - Ready for Phase 1.2 (Tensor Implementation)

### 🚀 Ready for Next Phase
- Phase 1.2: Implement actual Tensor and TensorImpl core functionality
- Core classes now have proper structure and namespace organization
- Build system proven to handle complex C++/CUDA/Python integration
- Development workflow established and verified

---

## v0.1.0 (2025-06-13) - Initial Setup

### Added
- Initial project structure
- CMake build system configuration
- Python package setup with pybind11
- CI/CD pipeline with GitHub Actions
- Code formatting and quality checks
- Basic project documentation
- Placeholder implementations for all major components

### Infrastructure
- Support for C++17 and CUDA compilation
- Multi-platform build support (Linux, macOS, Windows)
- Pre-commit hooks for code quality
- Comprehensive test framework setup
- Development environment configuration

### Build System
- CMake configuration with automatic dependency detection
- Python setuptools integration
- Environment variable configuration for build options
- Makefile for convenient development commands
- Build scripts for automated compilation

### Documentation
- Project README with detailed implementation roadmap
- Build instructions and dependency requirements
- API structure design documentation
- Contributing guidelines and code style configuration
