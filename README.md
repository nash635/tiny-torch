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
python test/verify_phase1_1.py
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
├── test/                      # Test suite
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
- **Module基类** - Parameter management, state synchronization
- **卷积层** - Conv2d, BatchNorm2d, pooling layers  
- **ResNet组件** - BasicBlock, Bottleneck, ResNet architecture
- **损失函数** - CrossEntropy, MSE, etc.

### Phase 5: Distributed Training Framework [Key Addition]
- **通信后端** - NCCL integration, process group management
- **数据并行** - DistributedDataParallel (DDP)
- **模型并行** - Large model sharding, pipeline parallelism
- **混合并行** - Data + model parallel strategies

### Phase 6: Optimizers and Tools Ecosystem [Planned]
- **优化器** - SGD, Adam, distributed optimizers
- **数据加载** - DistributedSampler, multi-process loading
- **模型保存** - Distributed checkpoints, model sharding

### Phase 7: ResNet50 Validation [Final Goal]
- **ImageNet训练** - Multi-node multi-GPU ResNet50 training
- **性能对标** - Performance comparison with PyTorch
- **Accuracy validation** - Top-1准确率 > 75%


## Testing and Validation

### Quick Verification

```bash
# 完整测试套件（推荐）
make test

# 基础功能验证
python -c "import tiny_torch; print('导入成功')"

# 构建诊断
make diagnose
```

### Detailed Testing

```bash
# Python测试套件
python -m pytest test/ -v

# CUDA功能测试
python test/test_cuda.py --env

# C++单元测试
cd build/cmake && ./test/cpp/tiny_torch_cpp_tests

# 验证Phase 1.1完成度
python test/verify_phase1_1.py
```

### Test Coverage

- **构建系统**: 100% (CMake + Ninja + setuptools)
- **Python集成**: 95% (模块导入、错误处理、API)  
- **CUDA支持**: 90% (编译、运行时、设备管理)
- **基础架构**: 95% (项目结构、依赖管理)

### Troubleshooting

```bash
# 构建问题诊断
python tools/diagnose_build.py

# 环境检查
python tools/check_env.py

# 重新安装
make clean && make install
```

## Build Instructions

### Build Commands

| 命令 | 功能 | 适用场景 |
|------|------|----------|
| `make install` | 完整安装 | 生产环境、首次安装 |
| `make build` | 仅编译 | 开发调试、快速测试 |
| `make test` | 运行测试 | 验证功能 |
| `make clean` | 清理 | 解决构建问题 |
| `make diagnose` | 诊断 | 排查构建问题 |

### Build Options

```bash
# 环境变量控制构建
DEBUG=1 make install         # Debug模式
WITH_CUDA=0 make install     # 禁用CUDA
USE_NINJA=1 make install     # 强制使用Ninja
VERBOSE=1 make install       # 详细输出
```

### Build Artifacts

| 产物 | 位置 | 说明 |
|------|------|------|
| **C++静态库** | `build/cmake/libtiny_torch_cpp.a` | 编译后的C++核心库 |
| **Python扩展** | `tiny_torch/_C.cpython-*.so` | Python可导入的C++扩展 |
| **CUDA内核** | `build/cmake/*.cu.o` | 编译后的CUDA对象文件 |
| **测试程序** | `build/cmake/test/cpp/tiny_torch_cpp_tests` | C++测试程序 |

## Contributing

### How to Contribute

1. Fork本项目
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送分支 (`git push origin feature/amazing-feature`)
5. 创建Pull Request

### Development Standards

- **C++**: 遵循PyTorch代码风格
- **Python**: 遵循PEP 8规范
- **测试**: 确保所有测试通过
- **文档**: 添加必要的注释和文档

## Learning Value

通过实现Tiny-Torch，深入掌握：

- **底层系统设计** - Memory management, device abstraction, type system
- **计算图和自动微分** - 动态图构建、反向传播算法
- **高性能计算** - SIMD优化、GPU编程、内存层次
- **系统集成** - Python C扩展、构建系统、API设计

## License

本项目采用 **Apache License 2.0**，确保开源友好和商业兼容。

### License特点

- **开源友好**: 允许自由使用、修改和分发
- **商业兼容**: 可用于商业项目，包括专利保护
- **学术研究**: 适合教育和研究用途
- **现代标准**: 广泛采用的现代开源许可证

### Related Files

- [LICENSE](LICENSE) - 完整许可证文本
- [CONTRIBUTING.md](CONTRIBUTING.md) - 贡献指南
- [CHANGELOG.md](CHANGELOG.md) - 版本更新记录

在使用本项目时，请保留版权声明和许可证信息。详细条款请参阅 [LICENSE](LICENSE) 文件。

---

**开始您的深度学习框架探索之旅！**

通过实现Tiny-Torch，获得对现代深度学习框架底层机制的深刻理解。
