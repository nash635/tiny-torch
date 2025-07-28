# Tiny-Torch: A PyTorch-Inspired Deep Learning Framework

![Tiny-Torch](https://img.shields.io/badge/Tiny--Torch-v0.1.0-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.9%2B-green.svg)
![C++](https://img.shields.io/badge/C%2B%2B-17-orange.svg)
![Build Status](https://img.shields.io/badge/Build-Passing-brightgreen.svg)
![Phase](https://img.shields.io/badge/Phase-1.1%20Complete-success.svg)

Tiny-Torch is a deep learning framework implemented from scratch, strictly following the architectural design and underlying implementation of [PyTorch](https://github.com/pytorch/pytorch). This project aims to provide deep understanding of modern deep learning framework internals through reimplementing PyTorch's core components.

## Table of Contents

- [Project Status](#project-status)
- [Quick Start](#quick-start)
- [Additional Features](#additional-features)
- [Architecture Design](#architecture-design)
- [Development Roadmap](#development-roadmap)
- [Testing and Validation](#testing-and-validation)
- [Build Instructions](#build-instructions)
- [Contributing](#contributing)
- [Learning Value](#learning-value)
- [License](#license)

## Project Status

**Current Version**: Phase 1.1 [Completed]

### Completed Features

- **Complete Build System** - CMake + Python setuptools dual build support
- **C++ Core Library** - Successfully compiled static library (`libtiny_torch_cpp.a`)
- **Python Extension** - Importable Python module (`tiny_torch._C`)
- **CUDA Support** - 6/6 CUDA source files compiled, GPU detection and management
- **CUDA Auto-detection** - Intelligent CUDA environment detection, automatic fallback to CPU build when CUDA unavailable
- **Testing Framework** - Complete C++ and Python testing environment setup
- **Development Toolchain** - Code formatting, CI/CD, pre-commit hooks configuration

### Development Progress

```
Phase 1: Core Infrastructure ████████████████████████████████ 100% [Completed]
├─ 1.1 Build System Setup   ████████████████████████████████ 100% [Completed]
├─ 1.2 Tensor Core Library  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0% [In Progress]
└─ 1.3 Low-level Tensor Impl ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0% [Planned]

Phase 2: Core Operators      ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0% [Planned]
Phase 3: Autograd Engine     ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0% [Planned]
Phase 4: Python Frontend     ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0% [Planned]
Phase 5: Advanced Features   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0% [Planned]

Overall Progress: ████████░░░░░░░░░░░░░░░░░░░░░░ 20%
```

## Quick Start

### Environment Requirements

- Python 3.8+
- CMake 3.18+
- C++17 compiler
- CUDA 11.0+ (optional, auto-detection supported)

### Installation Methods

#### Method 1: One-click Installation (Recommended)

```bash
git clone https://github.com/nash635/tiny-torch.git
cd tiny-torch
make install
```

#### Method 2: Docker Environment (Recommended for Beginners)

```bash
# Use pre-built image
docker pull crpi-rxpfp3shzt1yww56.cn-hangzhou.personal.cr.aliyuncs.com/tiny-torch/tiny-torch-gpu:latest

# Start development environment
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  crpi-rxpfp3shzt1yww56.cn-hangzhou.personal.cr.aliyuncs.com/tiny-torch/tiny-torch-gpu:latest

# Build inside container
cd /workspace && make install
```

#### Method 3: Step-by-step Installation

```bash
pip install -r requirements.txt
make build
pip install -e .
```

### Installation Verification

```bash
# Quick verification
python -c "import tiny_torch; print('Installation successful:', tiny_torch.__version__)"

# Complete verification
python tests/verify_phase1_1.py
```
## Additional Features

### CUDA Auto-detection

Tiny-Torch supports intelligent CUDA environment detection without manual configuration:

```bash
# Auto-detect CUDA availability
python setup.py build_ext --inplace

# Manual CUDA control
WITH_CUDA=1 python setup.py build_ext --inplace  # Force enable (with fallback)
WITH_CUDA=0 python setup.py build_ext --inplace  # Force disable

# Test CUDA detection logic
python test_cuda_detection.py
```

Build output explanations:
- `"CUDA automatically detected and enabled"` - CUDA detected and enabled
- `"CUDA not available, using CPU-only build"` - No CUDA, using CPU build
- `"Warning: CUDA requested but nvcc not found..."` - CUDA requested but unavailable, auto fallback

Details: [CUDA Auto-detection Documentation](docs/CUDA_AUTO_DETECTION.md)

### Currently Available Features

```python
import tiny_torch

# Basic features (Phase 1.1)
print(tiny_torch.__version__)              # 0.1.0
print(tiny_torch.cuda.is_available())      # CUDA detection
print(tiny_torch.cuda.device_count())      # GPU count

# Module structure in place
import tiny_torch.nn
import tiny_torch.optim  
import tiny_torch.autograd

# Tensor interface framework (Phase 1.2 will implement functionality)
try:
    tensor = tiny_torch.tensor([1, 2, 3])
except NotImplementedError:
    print("Tensor interface ready, awaiting Phase 1.2 implementation")
```

## Architecture Design

### Project Goals

- **Strictly Follow PyTorch** - Follow PyTorch API design and implementation patterns
- **Low-level Optimization** - Core operators implemented in C++/CUDA for performance
- **Education-oriented** - Provide clear code comments and implementation documentation
- **Modular Design** - Adopt PyTorch layered architecture for learning and extension

### Directory Structure

```
tiny-torch/
├── csrc/                      # C++/CUDA source code
│   ├── aten/                  # Tensor library (reference pytorch/aten)
│   ├── autograd/              # Automatic differentiation engine
│   └── api/                   # Python C API bindings
├── tiny_torch/                # Python frontend
│   ├── nn/                    # Neural network modules
│   ├── optim/                 # Optimizers
│   ├── autograd/              # Autograd interface
│   └── cuda/                  # CUDA support
├── tests/                      # Test suite
│   ├── cpp/                   # C++ tests
│   └── *.py                   # Python tests
└── docs/                      # Documentation
```

## Development Roadmap

### Phase 1: Core Infrastructure [Completed]
- **1.1 Build System** [Completed] - CMake, Python extensions, CUDA support
- **1.2 Tensor Core** [In Progress] - Tensor, TensorImpl, Storage classes
- **1.3 Low-level Implementation** [Planned] - Memory management, device abstraction, type system

### Phase 2: Core Operators [Planned]
- **CPU Operators** - Basic operations, linear algebra, activation functions
- **CUDA Operators** - GPU kernels, memory optimization, multi-GPU support
- **Operator Registration** - Dynamic dispatch, device-agnostic interfaces

### Phase 3: Autograd [Planned]
- **Computation Graph** - Dynamic graph construction, node management
- **Backpropagation** - Gradient computation, chain rule
- **Distributed Gradients** - Gradient synchronization, All-Reduce primitives
- **Advanced Features** - Higher-order gradients, JIT compilation

### Phase 4: Neural Network Modules [Planned]
- **Module Base Class** - Parameter management, state synchronization
- **Convolution Layers** - Conv2d, BatchNorm2d, pooling layers  
- **ResNet Components** - BasicBlock, Bottleneck, ResNet architecture
- **Loss Functions** - CrossEntropy, MSE, etc.

### Phase 5: Distributed Training Framework [Key Addition]
- **Communication Backend** - NCCL integration, process group management
- **Data Parallelism** - DistributedDataParallel (DDP)
- **Model Parallelism** - Large model sharding, pipeline parallelism
- **Hybrid Parallelism** - Data + model parallel strategies

### Phase 6: Optimizers and Tools Ecosystem [Planned]
- **Optimizers** - SGD, Adam, distributed optimizers
- **Data Loading** - DistributedSampler, multi-process loading
- **Model Saving** - Distributed checkpoints, model sharding

### Phase 7: ResNet50 Validation [Final Goal]
- **ImageNet Training** - Multi-node multi-GPU ResNet50 training
- **Performance Benchmarking** - Performance comparison with PyTorch
- **Accuracy Validation** - Top-1 accuracy > 75%


## Testing and Validation

### Quick Verification

```bash
# Complete test suite (recommended)
make test

# Basic functionality verification
python -c "import tiny_torch; print('Import successful')"

# Build diagnostics
make diagnose
```

### Detailed Testing

```bash
# Python test suite
python -m pytest tests/ -v

# CUDA functionality tests
python tests/test_cuda.py --env

# C++ unit tests
cd build/cmake && ./test/cpp/tiny_torch_cpp_tests

# Verify Phase 1.1 completion
python tests/verify_phase1_1.py
```

### Test Coverage

- **Build System**: 100% (CMake + Ninja + setuptools)
- **Python Integration**: 95% (module import, error handling, API)  
- **CUDA Support**: 90% (compilation, runtime, device management)
- **Infrastructure**: 95% (project structure, dependency management)

### Troubleshooting

```bash
# Build issue diagnostics
python tools/diagnose_build.py

# Environment check
python tools/check_env.py

# Reinstall
make clean && make install
```

## Build Instructions

### Build Commands

| Command | Function | Use Case |
|---------|----------|----------|
| `make install` | Complete installation | Production environment, first-time install |
| `make build` | Compile only | Development debugging, quick testing |
| `make test` | Run tests | Feature verification |
| `make clean` | Clean | Resolve build issues |
| `make diagnose` | Diagnose | Troubleshoot build problems |

### Build Options

```bash
# Environment variable build control
DEBUG=1 make install         # Debug mode
WITH_CUDA=0 make install     # Disable CUDA
USE_NINJA=1 make install     # Force use Ninja
VERBOSE=1 make install       # Verbose output
```

### Build Artifacts

| Artifact | Location | Description |
|----------|----------|-------------|
| **C++ Static Library** | `build/cmake/libtiny_torch_cpp.a` | Compiled C++ core library |
| **Python Extension** | `tiny_torch/_C.cpython-*.so` | Python-importable C++ extension |
| **CUDA Kernels** | `build/cmake/*.cu.o` | Compiled CUDA object files |
| **Test Programs** | `build/cmake/test/cpp/tiny_torch_cpp_tests` | C++ test programs |

## Contributing

### How to Contribute

1. Fork this project
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push branch (`git push origin feature/amazing-feature`)
5. Create Pull Request

### Development Standards

- **C++**: Follow PyTorch coding style
- **Python**: Follow PEP 8 standards
- **Testing**: Ensure all tests pass
- **Documentation**: Add necessary comments and documentation

## Learning Value

Through implementing Tiny-Torch, gain deep understanding of:

- **Low-level System Design** - Memory management, device abstraction, type system
- **Computation Graph and Automatic Differentiation** - Dynamic graph construction, backpropagation algorithms
- **High-Performance Computing** - SIMD optimization, GPU programming, memory hierarchy
- **System Integration** - Python C extensions, build systems, API design

## License

This project uses **Apache License 2.0**, ensuring open-source friendliness and commercial compatibility.

### License Features

- **Open Source Friendly**: Free to use, modify, and distribute
- **Commercial Compatible**: Suitable for commercial projects, including patent protection
- **Academic Research**: Suitable for educational and research purposes
- **Modern Standard**: Widely adopted modern open-source license

### Related Files

- [LICENSE](LICENSE) - Complete license text
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
- [CHANGELOG.md](CHANGELOG.md) - Version update history

When using this project, please retain copyright notices and license information. For detailed terms, please refer to the [LICENSE](LICENSE) file.

---

**Start Your Deep Learning Framework Exploration Journey!**

Through implementing Tiny-Torch, gain deep understanding of the underlying mechanisms of modern deep learning frameworks.
