# Contributing to Tiny-Torch

Thank you for your interest in contributing to Tiny-Torch! This document provides guidelines for contributing to the project.

## Development Setup

### Prerequisites
- Python 3.8+
- CMake 3.18+
- C++17 compatible compiler
- CUDA 11.0+ (optional, for GPU support)

### Setting up Development Environment

1. **Clone the repository**
```bash
git clone https://github.com/your-username/tiny-torch.git
cd tiny-torch
```

2. **Install development dependencies**
```bash
pip install -r requirements-dev.txt
```

3. **Set up pre-commit hooks**
```bash
pre-commit install
```

4. **Build the project**
```bash
make build
# or manually:
python setup.py build_ext --inplace
```

5. **Run tests**
```bash
make test
```

## Code Style

### Python Code
- Follow PEP 8 style guide
- Use Black for code formatting: `black torch/ test/`
- Use isort for import sorting: `isort torch/ test/`
- Type hints are required for public APIs
- Maximum line length: 88 characters

### C++ Code
- Follow the project's `.clang-format` configuration
- Use 2-space indentation
- Maximum line length: 80 characters
- Include guards should use `#pragma once`
- Follow PyTorch naming conventions

### Documentation
- All public functions and classes must have docstrings
- Use Google-style docstrings for Python
- Use Doxygen-style comments for C++
- Update README.md for significant changes

## Testing

### Running Tests
```bash
# All tests
make test

# Specific test categories
pytest test/ -m "not cuda"  # CPU tests only
pytest test/ -m cuda        # CUDA tests only
```

### Writing Tests
- Add tests for all new functionality
- Use pytest for Python tests
- Use Google Test for C++ tests
- Mock external dependencies when appropriate
- Aim for >90% test coverage

### Test Organization
```
test/
├── test_tensor.py          # Tensor operations
├── test_autograd.py        # Automatic differentiation
├── test_nn.py             # Neural network modules
├── test_optim.py          # Optimizers
└── cpp/                   # C++ tests
    ├── test_tensor.cpp
    └── test_autograd.cpp
```

## Pull Request Process

### Before Submitting
1. Ensure all tests pass
2. Run code formatting: `make format`
3. Run linting: `make lint`
4. Update documentation if needed
5. Add/update tests for new functionality

### PR Guidelines
1. **Create a descriptive title**
   - Format: `[category] Brief description`
   - Categories: `[feature]`, `[bugfix]`, `[docs]`, `[refactor]`, `[test]`

2. **Write a clear description**
   - Explain what changes were made and why
   - Reference any related issues
   - Include testing information

3. **Keep PRs focused**
   - One feature or fix per PR
   - Avoid mixing unrelated changes

4. **Update documentation**
   - Update docstrings for API changes
   - Update README if needed
   - Add changelog entry for significant changes

### Review Process
1. All PRs require at least one review
2. CI checks must pass
3. No merge conflicts
4. Maintainer approval required

## Issue Reporting

### Bug Reports
Please include:
- Python and system version
- Tiny-Torch version
- Minimal reproduction code
- Expected vs actual behavior
- Full error traceback

### Feature Requests
Please include:
- Clear description of the proposed feature
- Use case and motivation
- Proposed API (if applicable)
- Implementation ideas (if any)

## Architecture Guidelines

### Module Organization
```
tiny-torch/
├── torch/           # Python frontend
├── csrc/           # C++ backend
│   ├── api/        # Python bindings
│   ├── aten/       # Tensor library
│   └── autograd/   # Automatic differentiation
└── test/           # Test suite
```

### Design Principles
1. **API Compatibility**: Follow PyTorch conventions where possible
2. **Performance**: Optimize critical paths with C++/CUDA
3. **Modularity**: Clean separation between components
4. **Testability**: Write testable, mockable code
5. **Documentation**: Self-documenting code with clear interfaces

### Implementation Phases
Follow the roadmap in README.md:
1. Phase 1: Core infrastructure and tensor system
2. Phase 2: Core operators (CPU/CUDA)
3. Phase 3: Automatic differentiation
4. Phase 4: Python bindings
5. Phase 5: Neural network modules
6. Phase 6: Optimizers
7. Phase 7: Advanced features

## Communication

### Getting Help
- GitHub Issues for bugs and feature requests
- GitHub Discussions for general questions
- Check existing issues before creating new ones

### Maintainer Contact
- Create an issue for technical questions
- Use GitHub Discussions for design discussions

## License

By contributing to Tiny-Torch, you agree that your contributions will be licensed under the BSD 3-Clause License.

## Recognition

Contributors will be acknowledged in:
- CHANGELOG.md for significant contributions
- Project documentation
- Release notes

Thank you for contributing to Tiny-Torch! 🚀
