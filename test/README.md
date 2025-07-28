# Tiny-Torch Test Suite

This directory contains the comprehensive test suite for tiny-torch, organized into a structured hierarchy for better maintainability and clarity.

## Directory Structure

```
test/
├── unit/                   # Unit tests - isolated component testing
│   ├── test_core.py       # Core module functionality tests
│   ├── test_cuda.py       # CUDA functionality unit tests
│   └── test_build.py      # Build system unit tests
├── integration/            # Integration tests - component interaction
│   ├── test_build_cuda.py # Build system and CUDA integration
│   └── test_api.py        # API consistency and interaction tests
├── system/                 # System tests - end-to-end validation
│   ├── test_complete.py   # Complete system validation
│   └── test_performance.py # Performance and scalability tests
├── utils/                  # Test utilities and common code
│   ├── __init__.py        # Test utils package
│   ├── common.py          # Common test base classes and utilities
│   ├── cuda.py            # CUDA-specific test utilities
│   └── build.py           # Build system test utilities
├── fixtures/               # Test data and fixtures
├── cpp/                    # C++ tests
│   ├── CMakeLists.txt     # C++ test build configuration
│   ├── test_tensor.cpp    # C++ tensor tests
│   └── test_autograd.cpp  # C++ autograd tests
├── run_tests.py           # Unified test runner
├── setup_test_env.sh      # Test environment setup script
└── README.md              # This file
```

## Test Categories

### Unit Tests (`unit/`)
- **Purpose**: Test individual components in isolation
- **Scope**: Single functions, classes, or modules
- **Dependencies**: Minimal external dependencies
- **Speed**: Fast execution (< 1s per test)

#### Files:
- `test_core.py`: Core module structure, imports, API consistency
- `test_cuda.py`: CUDA detection, environment validation, basic functionality
- `test_build.py`: Build system components, project structure, artifacts

### Integration Tests (`integration/`)
- **Purpose**: Test component interactions and interfaces
- **Scope**: Multiple modules working together
- **Dependencies**: Requires built artifacts
- **Speed**: Medium execution (1-10s per test)

#### Files:
- `test_build_cuda.py`: Build system and CUDA integration
- `test_api.py`: API consistency across modules, error propagation

### System Tests (`system/`)
- **Purpose**: End-to-end system validation
- **Scope**: Complete system behavior
- **Dependencies**: Full build and environment
- **Speed**: Slower execution (10s+ per test)

#### Files:
- `test_complete.py`: Complete system validation, Phase 1.1 verification
- `test_performance.py`: Performance, scalability, resource usage

### Test Utilities (`utils/`)
- **Purpose**: Shared testing infrastructure
- **Scope**: Common utilities, base classes, helpers

#### Files:
- `common.py`: Base test classes, environment detection, fixtures
- `cuda.py`: CUDA-specific utilities, validation, environment info
- `build.py`: Build system utilities, validation, artifact checking

## Test Runner

### Usage

```bash
# Run all tests
python test/run_tests.py

# Run specific category
python test/run_tests.py --category unit
python test/run_tests.py --category integration  
python test/run_tests.py --category system

# Quick smoke test
python test/run_tests.py --category quick

# Verbose output
python test/run_tests.py --verbose

# Skip slow tests
python test/run_tests.py --skip-slow
```

### Available Categories

| Category | Description | Requirements |
|----------|-------------|--------------|
| `unit` | Unit tests | Basic imports |
| `integration` | Integration tests | Built artifacts |
| `system` | System tests | Full environment |
| `cpp` | C++ tests | C++ build artifacts |
| `cuda` | CUDA tests | CUDA environment |
| `performance` | Performance tests | Full environment |
| `quick` | Quick smoke test | Basic imports |
| `all` | All categories | Full environment |

## Test Base Classes

### TestBase
Base class for all tests with common setup/teardown and utilities.

```python
from test.utils import TestBase

class MyTest(TestBase):
    def test_something(self):
        # Test implementation
        pass
```

### Specialized Base Classes

- `UnitTestBase`: For unit tests
- `IntegrationTestBase`: For integration tests (requires build)
- `SystemTestBase`: For system tests (requires build)
- `CudaTestBase`: For CUDA tests (requires CUDA)
- `BuildTestBase`: For build system tests

### Test Markers and Decorators

```python
from test.utils import requires_cuda, requires_build, requires_nvcc

@requires_cuda
def test_cuda_functionality():
    """Test that requires CUDA runtime."""
    pass

@requires_build  
def test_build_artifacts():
    """Test that requires build artifacts."""
    pass

@requires_nvcc
def test_nvcc_compilation():
    """Test that requires nvcc compiler."""
    pass
```

## Environment Detection

The test suite automatically detects the available environment:

- **CUDA Runtime**: `nvidia-smi` availability
- **CUDA Compiler**: `nvcc` availability  
- **Build Artifacts**: `build/` directory and key files
- **Python Environment**: Module import capabilities

Tests are automatically skipped if requirements are not met.

## Writing New Tests

### 1. Choose the Right Category

- **Unit Test**: Testing a single function or class
- **Integration Test**: Testing module interactions
- **System Test**: Testing complete workflows

### 2. Use Appropriate Base Class

```python
from test.utils import UnitTestBase, requires_cuda

class TestNewFeature(UnitTestBase):
    @requires_cuda
    def test_cuda_feature(self):
        """Test CUDA functionality."""
        # Test implementation
        pass
```

### 3. Follow Naming Conventions

- Test files: `test_*.py`
- Test classes: `Test*`
- Test methods: `test_*`
- Descriptive names that explain what is being tested

### 4. Add Documentation

- Docstrings for test classes and methods
- Comments for complex test logic
- Update this README when adding new categories

## Migration from Legacy Tests

### Removed Files (Consolidated)
- `test_cuda.py` → `unit/test_cuda.py`
- `test_cuda_detection.py` → Merged into `unit/test_cuda.py`
- `test_cuda_comprehensive.py` → Merged into `unit/test_cuda.py`
- `test_build_system.py` → `unit/test_build.py`
- `test_build_comprehensive.py` → Merged into `unit/test_build.py`
- `test_core.py` → `unit/test_core.py`
- `test_integration.py` → `integration/test_api.py`
- `test_performance.py` → `system/test_performance.py`
- `test_utils.py` → `utils/common.py` (expanded)

### Consolidated Features
- **CUDA Testing**: All CUDA tests consolidated with better organization
- **Build Testing**: Build system tests unified with comprehensive validation
- **Test Utilities**: Expanded into a proper utilities package
- **Test Runner**: Single unified runner for all categories

## Best Practices

1. **Test Isolation**: Tests should not depend on each other
2. **Environment Handling**: Use environment detection and appropriate skips
3. **Error Messages**: Provide clear, actionable error messages
4. **Performance**: Keep unit tests fast, mark slow tests appropriately
5. **Documentation**: Document test purpose and requirements
6. **Fixtures**: Use the `fixtures/` directory for test data
7. **Cleanup**: Ensure proper cleanup in teardown methods

## Continuous Integration

The test suite is designed for CI environments:

- Automatic environment detection
- Graceful skipping of unavailable features
- Comprehensive error reporting
- Multiple test categories for different CI stages
- Performance regression detection

## Future Extensions

The test structure is designed to accommodate future development:

- **Phase 1.2**: Tensor implementation tests
- **Phase 2**: Operator tests
- **Phase 3**: Autograd tests
- **Phase 4**: Neural network module tests
- **Phase 5**: Distributed training tests

Add new test files to the appropriate category directory and extend the test runner as needed.
