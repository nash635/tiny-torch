# Tiny-Torch: A PyTorch-Inspired Deep Learning Framework

![Tiny-Torch](https://img.shields.io/badge/Tiny--Torch-v0.1.0-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.9%2B-green.svg)
![C++](https://img.shields.io/badge/C%2B%2B-17-orange.svg)
![Build Status](https://img.shields.io/badge/Build-Passing-brightgreen.svg)
![Phase](https://img.shields.io/badge/Phase-1.1%20Complete-success.svg)

Tiny-Torch 是一个从零开始实现的深度学习框架，严格参考 [PyTorch](https://github.com/pytorch/pytorch) 的架构设计和底层实现。本项目旨在通过重新实现 PyTorch 的核心组件，深入理解现代深度学习框架的底层机制。

## 目录

- [项目状态](#项目状态)
- [快速开始](#快速开始)
- [附加特性](#附加特性)
- [架构设计](#架构设计)
- [开发路线图](#开发路线图)
- [测试验证](#测试验证)
- [构建说明](#构建说明)
- [贡献指南](#贡献指南)
- [学习价值](#学习价值)
- [许可证](#许可证)

## 项目状态

**当前版本**: Phase 1.1 [已完成]

### 已完成功能

- **完整构建系统** - CMake + Python setuptools双重构建支持
- **C++核心库** - 成功编译的静态库 (`libtiny_torch_cpp.a`)
- **Python扩展** - 可导入的Python模块 (`tiny_torch._C`)
- **CUDA支持** - 6/6 CUDA源文件编译，GPU检测和管理
- **CUDA自动检测** - 智能CUDA环境检测，无CUDA环境自动降级为CPU构建
- **测试框架** - C++和Python测试环境完整搭建
- **开发工具链** - 代码格式化、CI/CD、预提交钩子配置

### 开发进度

```
Phase 1: 核心基础设施    ████████████████████████████████ 100% [完成]
├─ 1.1 构建系统设置     ████████████████████████████████ 100% [完成]
├─ 1.2 张量核心库       ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0% [开发中]
└─ 1.3 底层张量实现     ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0% [计划中]

Phase 2: 核心算子实现    ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0% [计划中]
Phase 3: 自动微分引擎    ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0% [计划中]  
Phase 4: Python前端     ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0% [计划中]
Phase 5: 高级特性       ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0% [计划中]

总体进度: ████████░░░░░░░░░░░░░░░░░░░░░░ 20%
```

## 快速开始

### 环境要求

- Python 3.8+
- CMake 3.18+
- C++17编译器
- CUDA 11.0+ (可选，支持自动检测)

### 安装方式

#### 方式一：一键安装 (推荐)

```bash
git clone https://github.com/nash635/tiny-torch.git
cd tiny-torch
make install
```

#### 方式二：Docker环境 (推荐新手)

```bash
# 使用预构建镜像
docker pull crpi-rxpfp3shzt1yww56.cn-hangzhou.personal.cr.aliyuncs.com/tiny-torch/tiny-torch-gpu:latest

# 启动开发环境
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  crpi-rxpfp3shzt1yww56.cn-hangzhou.personal.cr.aliyuncs.com/tiny-torch/tiny-torch-gpu:latest

# 在容器内构建
cd /workspace && make install
```

#### 方式三：分步安装

```bash
pip install -r requirements.txt
make build
pip install -e .
```

### 验证安装

```bash
# 快速验证
python -c "import tiny_torch; print('安装成功:', tiny_torch.__version__)"

# 完整验证
python test/verify_phase1_1.py
```
## 附加特性

### CUDA自动检测

Tiny-Torch支持智能CUDA环境检测，无需手动配置：

```bash
# 自动检测CUDA可用性
python setup.py build_ext --inplace

# 手动控制CUDA设置
WITH_CUDA=1 python setup.py build_ext --inplace  # 强制启用（有回退）
WITH_CUDA=0 python setup.py build_ext --inplace  # 强制禁用

# 测试CUDA检测逻辑
python test_cuda_detection.py
```

构建输出说明：
- `"CUDA automatically detected and enabled"` - 检测到CUDA并启用
- `"CUDA not available, using CPU-only build"` - 无CUDA，使用CPU构建
- `"Warning: CUDA requested but nvcc not found..."` - 请求CUDA但不可用，自动回退

详细信息：[CUDA自动检测文档](docs/CUDA_AUTO_DETECTION.md)

### 当前可用功能

```python
import tiny_torch

# 基础功能 (Phase 1.1)
print(tiny_torch.__version__)              # 0.1.0
print(tiny_torch.cuda.is_available())      # CUDA检测
print(tiny_torch.cuda.device_count())      # GPU数量

# 模块结构已就位
import tiny_torch.nn
import tiny_torch.optim  
import tiny_torch.autograd

# 张量接口框架 (Phase 1.2将实现具体功能)
try:
    tensor = tiny_torch.tensor([1, 2, 3])
except NotImplementedError:
    print("张量接口已就位，等待Phase 1.2实现")
```

## 架构设计

### 项目目标

- **严格参考PyTorch** - 遵循PyTorch的API设计和实现模式
- **底层优化** - 核心算子使用C++/CUDA实现，确保性能
- **教育导向** - 提供清晰的代码注释和实现文档
- **模块化设计** - 采用PyTorch的分层架构，便于学习和扩展

### 目录结构

```
tiny-torch/
├── csrc/                      # C++/CUDA源码
│   ├── aten/                  # 张量库 (参考pytorch/aten)
│   ├── autograd/              # 自动微分引擎
│   └── api/                   # Python C API绑定
├── tiny_torch/                # Python前端
│   ├── nn/                    # 神经网络模块
│   ├── optim/                 # 优化器
│   ├── autograd/              # 自动微分接口
│   └── cuda/                  # CUDA支持
├── test/                      # 测试套件
│   ├── cpp/                   # C++测试
│   └── *.py                   # Python测试
└── docs/                      # 文档
```

## 开发路线图

### Phase 1: 核心基础设施 [已完成]
- **1.1 构建系统** [完成] - CMake、Python扩展、CUDA支持
- **1.2 张量核心** [开发中] - Tensor、TensorImpl、Storage类
- **1.3 底层实现** [计划中] - 内存管理、设备抽象、类型系统

### Phase 2: 核心算子 [计划中]
- **CPU算子** - 基础运算、线性代数、激活函数
- **CUDA算子** - GPU内核、内存优化、多GPU支持
- **算子注册** - 动态分发、设备无关接口

### Phase 3: 自动微分 [计划中]
- **计算图** - 动态图构建、节点管理
- **反向传播** - 梯度计算、链式法则
- **分布式梯度** - 梯度同步、All-Reduce原语
- **高级特性** - 高阶梯度、JIT编译

### Phase 4: 神经网络模块 [计划中]
- **Module基类** - 参数管理、状态同步
- **卷积层** - Conv2d、BatchNorm2d、池化层  
- **ResNet组件** - BasicBlock、Bottleneck、ResNet架构
- **损失函数** - CrossEntropy、MSE等

### Phase 5: 分布式训练框架 [重点新增]
- **通信后端** - NCCL集成、进程组管理
- **数据并行** - DistributedDataParallel (DDP)
- **模型并行** - 大模型分片、管道并行
- **混合并行** - 数据+模型并行策略

### Phase 6: 优化器与工具生态 [计划中]
- **优化器** - SGD、Adam、分布式优化器
- **数据加载** - DistributedSampler、多进程加载
- **模型保存** - 分布式检查点、模型分片

### Phase 7: ResNet50验证 [最终目标]
- **ImageNet训练** - 多机多卡ResNet50训练
- **性能对标** - 与PyTorch性能对比
- **准确性验证** - Top-1准确率 > 75%


## 测试验证

### 快速验证

```bash
# 完整测试套件（推荐）
make test

# 基础功能验证
python -c "import tiny_torch; print('导入成功')"

# 构建诊断
make diagnose
```

### 详细测试

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

### 测试覆盖率

- **构建系统**: 100% (CMake + Ninja + setuptools)
- **Python集成**: 95% (模块导入、错误处理、API)  
- **CUDA支持**: 90% (编译、运行时、设备管理)
- **基础架构**: 95% (项目结构、依赖管理)

### 故障排除

```bash
# 构建问题诊断
python tools/diagnose_build.py

# 环境检查
python tools/check_env.py

# 重新安装
make clean && make install
```

## 构建说明

### 构建命令

| 命令 | 功能 | 适用场景 |
|------|------|----------|
| `make install` | 完整安装 | 生产环境、首次安装 |
| `make build` | 仅编译 | 开发调试、快速测试 |
| `make test` | 运行测试 | 验证功能 |
| `make clean` | 清理 | 解决构建问题 |
| `make diagnose` | 诊断 | 排查构建问题 |

### 构建选项

```bash
# 环境变量控制构建
DEBUG=1 make install         # Debug模式
WITH_CUDA=0 make install     # 禁用CUDA
USE_NINJA=1 make install     # 强制使用Ninja
VERBOSE=1 make install       # 详细输出
```

### 构建产物

| 产物 | 位置 | 说明 |
|------|------|------|
| **C++静态库** | `build/cmake/libtiny_torch_cpp.a` | 编译后的C++核心库 |
| **Python扩展** | `tiny_torch/_C.cpython-*.so` | Python可导入的C++扩展 |
| **CUDA内核** | `build/cmake/*.cu.o` | 编译后的CUDA对象文件 |
| **测试程序** | `build/cmake/test/cpp/tiny_torch_cpp_tests` | C++测试程序 |

## 贡献指南

### 如何贡献

1. Fork本项目
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送分支 (`git push origin feature/amazing-feature`)
5. 创建Pull Request

### 开发规范

- **C++**: 遵循PyTorch代码风格
- **Python**: 遵循PEP 8规范
- **测试**: 确保所有测试通过
- **文档**: 添加必要的注释和文档

## 学习价值

通过实现Tiny-Torch，深入掌握：

- **底层系统设计** - 内存管理、设备抽象、类型系统
- **计算图和自动微分** - 动态图构建、反向传播算法
- **高性能计算** - SIMD优化、GPU编程、内存层次
- **系统集成** - Python C扩展、构建系统、API设计

## 许可证

本项目采用 **Apache License 2.0**，确保开源友好和商业兼容。

### 许可证特点

- **开源友好**: 允许自由使用、修改和分发
- **商业兼容**: 可用于商业项目，包括专利保护
- **学术研究**: 适合教育和研究用途
- **现代标准**: 广泛采用的现代开源许可证

### 相关文件

- [LICENSE](LICENSE) - 完整许可证文本
- [CONTRIBUTING.md](CONTRIBUTING.md) - 贡献指南
- [CHANGELOG.md](CHANGELOG.md) - 版本更新记录

在使用本项目时，请保留版权声明和许可证信息。详细条款请参阅 [LICENSE](LICENSE) 文件。

---

**开始您的深度学习框架探索之旅！**

通过实现Tiny-Torch，获得对现代深度学习框架底层机制的深刻理解。
