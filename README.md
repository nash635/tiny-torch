# Tiny-Torch: A PyTorch-Inspired Deep Learning Framework

![Tiny-Torch](https://img.shields.io/badge/Tiny--Torch-v0.1.0-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![CUDA](https://img.shields.io/badge/CUDA-11.0%2B-green.svg)
![C++](https://img.shields.io/badge/C%2B%2B-17-orange.svg)

Tiny-Torch 是一个从零开始实现的深度学习框架，严格参考 [PyTorch](https://github.com/pytorch/pytorch) 的架构设计和底层实现。本项目旨在通过重新实现 PyTorch 的核心组件，深入理解现代深度学习框架的底层机制。

## 🎯 项目目标

- **严格参考PyTorch**: 遵循PyTorch的API设计和实现模式
- **底层优化**: 核心算子使用C++/CUDA实现，确保性能
- **教育导向**: 提供清晰的代码注释和实现文档
- **模块化设计**: 采用PyTorch的分层架构，便于学习和扩展

## 📁 项目结构

```
tiny-torch/
├── README.md
├── CMakeLists.txt              # CMake构建配置
├── setup.py                   # Python包安装配置
├── pyproject.toml             # 现代Python项目配置
├── requirements.txt           # Python依赖
├── .gitignore
├── .clang-format              # C++代码格式化
├── csrc/                      # C++/CUDA源码 (参考pytorch/csrc)
│   ├── api/                   # Python C API绑定
│   │   ├── include/
│   │   └── src/
│   ├── aten/                  # ATen张量库 (参考pytorch/aten)
│   │   ├── src/
│   │   │   ├── ATen/
│   │   │   │   ├── core/       # 核心张量类
│   │   │   │   ├── native/     # CPU实现
│   │   │   │   ├── cuda/       # CUDA实现
│   │   │   │   └── ops/        # 算子定义
│   │   │   └── TH/             # 底层张量实现
│   │   └── include/
│   ├── autograd/              # 自动微分引擎
│   │   ├── engine.cpp
│   │   ├── function.cpp
│   │   └── variable.cpp
│   └── jit/                   # 即时编译 (简化版)
│       ├── codegen/
│       └── passes/
├── torch/                     # Python前端 (参考pytorch/torch)
│   ├── __init__.py
│   ├── _C/                    # C扩展绑定
│   ├── nn/                    # 神经网络模块
│   │   ├── __init__.py
│   │   ├── modules/           # 网络层实现
│   │   │   ├── __init__.py
│   │   │   ├── module.py      # 基础Module类
│   │   │   ├── linear.py      # 线性层
│   │   │   ├── conv.py        # 卷积层
│   │   │   ├── activation.py  # 激活函数
│   │   │   ├── loss.py        # 损失函数
│   │   │   ├── batchnorm.py   # 批归一化
│   │   │   └── container.py   # 容器类
│   │   ├── functional.py      # 函数式接口
│   │   ├── init.py           # 参数初始化
│   │   └── parameter.py       # 参数类
│   ├── optim/                 # 优化器
│   │   ├── __init__.py
│   │   ├── optimizer.py       # 基础优化器
│   │   ├── sgd.py            # SGD
│   │   ├── adam.py           # Adam
│   │   └── lr_scheduler.py    # 学习率调度
│   ├── autograd/             # 自动微分Python接口
│   │   ├── __init__.py
│   │   ├── function.py       # Function基类
│   │   ├── variable.py       # Variable(已废弃,兼容性)
│   │   └── functional.py     # 自动微分函数
│   ├── utils/                # 实用工具
│   │   ├── data/             # 数据处理
│   │   └── cpp_extension.py  # C++扩展工具
│   └── distributed/          # 分布式训练
├── test/                     # 测试套件
│   ├── test_tensor.py
│   ├── test_autograd.py
│   ├── test_nn.py
│   ├── test_optim.py
│   └── cpp/                  # C++测试
├── benchmarks/               # 性能基准测试
│   ├── tensor_ops.py
│   ├── autograd_benchmark.py
│   └── compare_with_pytorch.py
├── examples/                 # 示例代码
│   ├── mnist/
│   ├── cifar10/
│   └── transformer/
├── docs/                     # 文档
│   ├── design/               # 设计文档
│   ├── api/                  # API文档
│   └── tutorials/            # 教程
└── tools/                    # 构建和开发工具
    ├── build/
    ├── setup_helpers/
    └── codegen/              # 代码生成工具
```

## 🚀 实现阶段

### Phase 1: 核心基础设施 (Week 1-3)

#### 1.1 构建系统设置
```bash
# 参考 pytorch/CMakeLists.txt 和 setup.py
```
- [ ] 配置CMake构建系统，支持C++17和CUDA
- [ ] 设置Python C扩展编译
- [ ] 配置CI/CD流水线
- [ ] 建立代码风格和质量检查

#### 1.2 张量核心库 (ATen)
**参考**: `pytorch/aten/src/ATen/`

```cpp
// csrc/aten/src/ATen/core/Tensor.h
class TORCH_API Tensor {
public:
  // 参考pytorch实现张量核心接口
  Tensor(const Tensor& other);
  Tensor& operator=(const Tensor& other);
  
  // 基础属性
  IntArrayRef sizes() const;
  IntArrayRef strides() const;
  int64_t dim() const;
  ScalarType dtype() const;
  Device device() const;
  
  // 核心操作
  Tensor& add_(const Tensor& other);
  Tensor& mul_(const Tensor& other);
  Tensor& matmul_(const Tensor& other);
  
private:
  c10::intrusive_ptr<TensorImpl> impl_;
};
```

#### 1.3 底层张量实现
**参考**: `pytorch/c10/core/` 和 `pytorch/aten/src/TH/`

- [ ] 实现 `TensorImpl` 类 (张量数据存储)
- [ ] 实现 `Storage` 类 (内存管理)
- [ ] 实现 `Device` 和 `ScalarType` 枚举
- [ ] 实现基础的内存分配器

### Phase 2: 核心算子实现 (Week 4-8)

#### 2.1 CPU算子实现
**参考**: `pytorch/aten/src/ATen/native/`

```cpp
// csrc/aten/src/ATen/native/BinaryOps.cpp
namespace at { namespace native {

Tensor add_cpu(const Tensor& self, const Tensor& other, const Scalar& alpha) {
  // 参考pytorch的CPU实现
  return at::empty_like(self).add_(other, alpha);
}

Tensor& add_cpu_(Tensor& self, const Tensor& other, const Scalar& alpha) {
  // 就地操作实现
  auto iter = TensorIterator::binary_op(self, self, other);
  add_stub(iter.device_type(), iter, alpha);
  return self;
}

}}
```

- [ ] 基础算术运算: add, sub, mul, div
- [ ] 线性代数: mm, bmm, addmm
- [ ] 激活函数: relu, sigmoid, tanh, gelu
- [ ] 归约操作: sum, mean, max, min
- [ ] 索引操作: index_select, gather, scatter

#### 2.2 CUDA算子实现
**参考**: `pytorch/aten/src/ATen/native/cuda/`

```cuda
// csrc/aten/src/ATen/native/cuda/BinaryOps.cu
template<typename scalar_t>
__global__ void add_kernel(
    TensorIterator iter,
    scalar_t alpha) {
  // 参考pytorch的CUDA kernel实现
  GPU_LAMBDA(index) {
    scalar_t a = iter.data<scalar_t>()[iter.strides(1) * index];
    scalar_t b = iter.data<scalar_t>()[iter.strides(2) * index];
    iter.data<scalar_t>()[iter.strides(0) * index] = a + alpha * b;
  });
}

Tensor add_cuda(const Tensor& self, const Tensor& other, const Scalar& alpha) {
  auto result = at::empty_like(self);
  auto iter = TensorIterator::binary_op(result, self, other);
  AT_DISPATCH_ALL_TYPES(iter.dtype(), "add_cuda", [&]() {
    gpu_kernel(iter, add_kernel<scalar_t>, alpha.to<scalar_t>());
  });
  return result;
}
```

- [ ] CUDA内核实现基础运算
- [ ] 内存合并访问优化
- [ ] 多GPU支持
- [ ] CuBLAS和CuDNN集成

#### 2.3 算子注册机制
**参考**: `pytorch/aten/src/ATen/core/dispatch/`

```cpp
// csrc/aten/src/ATen/ops/add.cpp
TORCH_LIBRARY_IMPL(aten, CPU, m) {
  m.impl("add.Tensor", TORCH_FN(add_cpu));
  m.impl("add_.Tensor", TORCH_FN(add_cpu_));
}

TORCH_LIBRARY_IMPL(aten, CUDA, m) {
  m.impl("add.Tensor", TORCH_FN(add_cuda));
  m.impl("add_.Tensor", TORCH_FN(add_cuda_));
}
```

### Phase 3: 自动微分引擎 (Week 9-12)

#### 3.1 计算图构建
**参考**: `pytorch/torch/csrc/autograd/`

```cpp
// csrc/autograd/variable.cpp
struct TORCH_API AutogradMeta {
  Variable grad_;
  std::shared_ptr<Node> grad_fn_;
  std::weak_ptr<Node> grad_accumulator_;
  VariableVersion version_counter_;
  bool requires_grad_;
  bool retains_grad_;
  bool is_view_;
};
```

- [ ] 实现 `Variable` 类 (已合并到Tensor)
- [ ] 实现 `Node` 基类和计算图
- [ ] 实现梯度累积机制
- [ ] 实现view和in-place操作处理

#### 3.2 反向传播引擎
**参考**: `pytorch/torch/csrc/autograd/engine.cpp`

```cpp
// csrc/autograd/engine.cpp
class Engine {
public:
  variable_list execute(
      const edge_list& roots,
      const variable_list& inputs,
      bool keep_graph,
      bool create_graph,
      const edge_list& outputs = {});
      
private:
  void compute_dependencies(FunctionTask task, ReadyQueue& ready);
  void evaluate_function(
      std::shared_ptr<GraphTask>& graph_task,
      Node* func,
      InputBuffer& inputs,
      const std::shared_ptr<ReadyQueue>& cpu_ready_queue);
};
```

#### 3.3 梯度函数实现
**参考**: `pytorch/torch/csrc/autograd/functions/`

```cpp
// csrc/autograd/functions/basic_ops.cpp
struct AddBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  
  variable_list apply(variable_list&& grads) override {
    auto grad_input = grads[0];
    return {grad_input, grad_input * alpha};
  }
  
  Scalar alpha;
};
```

### Phase 4: Python绑定层 (Week 13-15)

#### 4.1 PyBind11集成
**参考**: `pytorch/torch/csrc/Module.cpp`

```cpp
// csrc/api/src/python_bindings.cpp
PYBIND11_MODULE(_C, m) {
  // 张量绑定
  py::class_<Tensor>(m, "Tensor")
    .def("add", &Tensor::add)
    .def("add_", &Tensor::add_)
    .def("backward", &Tensor::backward)
    .def_property_readonly("grad", &Tensor::grad);
    
  // 函数绑定
  m.def("add", &torch::add);
  m.def("mm", &torch::mm);
}
```

#### 4.2 Python前端实现
**参考**: `pytorch/torch/`

```python
# torch/__init__.py
from torch._C import *  # C扩展导入
from torch.tensor import Tensor
from torch.autograd import Variable  # 兼容性

# 导出主要接口
__all__ = [
    'Tensor', 'tensor', 'add', 'mm', 'nn', 'optim'
]

def tensor(data, dtype=None, device=None, requires_grad=False):
    """创建张量的Python接口"""
    return Tensor._make_subclass(data, dtype, device, requires_grad)
```

### Phase 5: 神经网络模块 (Week 16-20)

#### 5.1 Module基类
**参考**: `pytorch/torch/nn/modules/module.py`

```python
# torch/nn/modules/module.py
class Module:
    def __init__(self):
        self._parameters = OrderedDict()
        self._buffers = OrderedDict()
        self._modules = OrderedDict()
        self.training = True
        
    def forward(self, *input):
        raise NotImplementedError
        
    def __call__(self, *input, **kwargs):
        result = self.forward(*input, **kwargs)
        # Hook机制
        for hook in self._forward_hooks.values():
            hook_result = hook(self, input, result)
            if hook_result is not None:
                result = hook_result
        return result
        
    def register_parameter(self, name, param):
        if '_parameters' not in self.__dict__:
            raise AttributeError("cannot assign parameter before Module.__init__() call")
        self._parameters[name] = param
        
    def parameters(self, recurse=True):
        for name, param in self.named_parameters(recurse=recurse):
            yield param
```

#### 5.2 核心层实现
**参考**: `pytorch/torch/nn/modules/`

```python
# torch/nn/modules/linear.py
class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features)))
        if bias:
            self.bias = Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
    def forward(self, input):
        return F.linear(input, self.weight, self.bias)
        
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
```

### Phase 6: 优化器实现 (Week 21-22)

#### 6.1 优化器基类
**参考**: `pytorch/torch/optim/optimizer.py`

```python
# torch/optim/optimizer.py
class Optimizer:
    def __init__(self, params, defaults):
        self.defaults = defaults
        self.state = defaultdict(dict)
        self.param_groups = []
        
        if isinstance(params, torch.Tensor):
            raise TypeError("params argument given to the optimizer should be "
                          "an iterable of Tensors or dicts, but got " +
                          torch.typename(params))
                          
        self.add_param_group({'params': list(params)})
        
    def step(self, closure=None):
        raise NotImplementedError
        
    def zero_grad(self, set_to_none=False):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        if p.grad.grad_fn is not None:
                            p.grad.detach_()
                        else:
                            p.grad.requires_grad_(False)
                        p.grad.zero_()
```

#### 6.2 具体优化器
**参考**: `pytorch/torch/optim/`

```python
# torch/optim/sgd.py
class SGD(Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                       weight_decay=weight_decay, nesterov=nesterov)
        super(SGD, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            momentum_buffer_list = []
            
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    grads.append(p.grad)
                    
                    state = self.state[p]
                    if len(state) == 0:
                        state['momentum_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    momentum_buffer_list.append(state['momentum_buffer'])
            
            F.sgd(params_with_grad,
                  grads,
                  momentum_buffer_list,
                  weight_decay=group['weight_decay'],
                  momentum=group['momentum'],
                  lr=group['lr'],
                  dampening=group['dampening'],
                  nesterov=group['nesterov'])
        
        return loss
```

### Phase 7: 高级功能 (Week 23-26)

#### 7.1 JIT编译 (简化版)
**参考**: `pytorch/torch/jit/`

```python
# torch/jit/__init__.py
def script(obj, optimize=None, _frames_up=0, _rcb=None):
    """将Python函数/模块编译为TorchScript"""
    if isinstance(obj, torch.nn.Module):
        return torch.jit._script.script_module(obj, _frames_up=_frames_up + 1)
    else:
        return torch.jit._script.script(obj, _frames_up=_frames_up + 1)

def trace(func, example_inputs, optimize=None, check_trace=True, check_inputs=None, check_tolerance=1e-5, strict=True):
    """通过跟踪执行路径生成TorchScript"""
    return torch.jit._trace.trace(func, example_inputs, optimize, check_trace, check_inputs, check_tolerance, strict)
```

#### 7.2 分布式训练基础
**参考**: `pytorch/torch/distributed/`

```python
# torch/distributed/__init__.py
def init_process_group(backend, init_method=None, timeout=default_pg_timeout, 
                      world_size=-1, rank=-1, store=None, group_name=""):
    """初始化分布式进程组"""
    global _default_pg_init_method
    if init_method is None:
        init_method = _default_pg_init_method
    backend = Backend(backend)
    default_pg = _new_process_group_helper(world_size, rank, [], backend, store, timeout=timeout)
    _update_default_pg(default_pg)

class DistributedDataParallel(Module):
    def __init__(self, module, device_ids=None, output_device=None, 
                 dim=0, broadcast_buffers=True, process_group=None, 
                 bucket_cap_mb=25, find_unused_parameters=False, 
                 check_reduction=False, gradient_as_bucket_view=False):
        super(DistributedDataParallel, self).__init__()
        self.module = module
        self.device_ids = device_ids
        self.output_device = output_device
        self.dim = dim
        self.process_group = process_group
        # DDP具体实现...
```

## 🛠️ 构建和安装

### 环境要求
```bash
# 系统要求
Python >= 3.8
CMake >= 3.18
CUDA >= 11.0 (可选)
cuDNN >= 8.0 (可选)

# C++编译器
GCC >= 9.0 (Linux)
Clang >= 10.0 (macOS)
MSVC >= 2019 (Windows)
```

### 构建步骤
```bash
# 克隆项目
git clone https://github.com/nash635/tiny-torch.git
cd tiny-torch

# 创建构建目录
mkdir build && cd build

# 配置CMake
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DWITH_CUDA=ON \
         -DPYTHON_EXECUTABLE=$(which python)

# 编译
make -j$(nproc)

# 安装Python包
cd ..
pip install -e .
```

### 验证安装
```python
import torch

# 基础张量操作
x = torch.randn(2, 3, requires_grad=True)
y = x * 2 + 1
z = y.sum()
z.backward()
print(x.grad)

# 神经网络
import torch.nn as nn
model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 1)
)
output = model(torch.randn(1, 10))
```

## 📊 性能基准

### 与PyTorch对比
```python
# benchmarks/compare_with_pytorch.py
import time
import torch as pytorch
import tiny_torch as torch

def benchmark_matmul(size=1000, iterations=100):
    # PyTorch
    x_pt = pytorch.randn(size, size)
    y_pt = pytorch.randn(size, size)
    
    start = time.time()
    for _ in range(iterations):
        z_pt = pytorch.mm(x_pt, y_pt)
    pytorch_time = time.time() - start
    
    # Tiny-Torch
    x_tt = torch.randn(size, size)
    y_tt = torch.randn(size, size)
    
    start = time.time()
    for _ in range(iterations):
        z_tt = torch.mm(x_tt, y_tt)
    tiny_torch_time = time.time() - start
    
    print(f"PyTorch: {pytorch_time:.4f}s")
    print(f"Tiny-Torch: {tiny_torch_time:.4f}s")
    print(f"Relative performance: {pytorch_time/tiny_torch_time:.2f}x")

if __name__ == "__main__":
    benchmark_matmul()
```

## 🧪 测试

### 运行测试套件
```bash
# Python测试
python -m pytest test/ -v

# C++测试
cd build && make test

# 性能基准测试
python benchmarks/compare_with_pytorch.py
```

### 测试覆盖率
- 张量操作: 95%+
- 自动微分: 90%+
- 神经网络层: 95%+
- 优化器: 90%+

## 📚 学习资源

### 必读材料
1. [PyTorch Internals](http://blog.ezyang.com/2019/05/pytorch-internals/)
2. [PyTorch Autograd Deep Dive](https://pytorch.org/blog/computational-graphs-constructed-in-pytorch/)
3. [ATen: A TENsor library](https://github.com/pytorch/pytorch/tree/main/aten)
4. [PyTorch C++ API](https://pytorch.org/cppdocs/)

### 关键技术文档
1. **张量核心**: `pytorch/c10/` 和 `pytorch/aten/`
2. **自动微分**: `pytorch/torch/csrc/autograd/`
3. **JIT编译**: `pytorch/torch/csrc/jit/`
4. **分布式**: `pytorch/torch/csrc/distributed/`

### 参考实现
- [PyTorch源码](https://github.com/pytorch/pytorch)
- [PyTorch教程](https://pytorch.org/tutorials/)
- [Deep Learning with PyTorch](https://pytorch.org/assets/deep-learning/Deep-Learning-with-PyTorch.pdf)

## 🎓 核心学习目标

通过实现Tiny-Torch，您将深入掌握：

### 1. 底层系统设计
- **内存管理**: 张量存储和生命周期管理
- **设备抽象**: CPU/GPU统一编程模型
- **类型系统**: 动态类型和静态类型的结合

### 2. 计算图和自动微分
- **计算图构建**: 动态图的设计和实现
- **反向传播**: 梯度计算的具体算法
- **内存优化**: 梯度累积和释放策略

### 3. 高性能计算
- **向量化**: SIMD和并行计算优化
- **GPU编程**: CUDA kernel的编写和优化
- **内存层次**: 缓存友好的数据访问模式

### 4. 系统集成
- **Python C扩展**: 高性能Python库的开发
- **构建系统**: 复杂C++项目的组织和编译
- **API设计**: 易用性和性能的平衡

## 🤝 贡献指南

### 代码风格
- C++: 遵循PyTorch的代码风格 (基于Google Style Guide)
- Python: 遵循PEP 8和PyTorch约定
- 注释: 详细的实现注释和API文档

### 提交流程
1. Fork本项目
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 确保测试通过 (`python -m pytest test/`)
4. 提交代码 (`git commit -m 'Add amazing feature'`)
5. 推送分支 (`git push origin feature/amazing-feature`)
6. 创建Pull Request

### 开发环境设置
```bash
# 安装开发依赖
pip install -r requirements-dev.txt

# 安装pre-commit hooks
pre-commit install

# 格式化代码
black torch/ test/
clang-format -i csrc/**/*.cpp csrc/**/*.h
```

## 📄 许可证

本项目采用BSD 3-Clause许可证，与PyTorch保持一致。详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- **PyTorch团队**: 提供了卓越的深度学习框架设计
- **Facebook AI Research**: 开源了PyTorch的核心实现
- **NVIDIA**: 提供了CUDA和cuDNN支持
- **开源社区**: 为深度学习框架发展做出的贡献

---

**开始您的深度学习框架探索之旅！** 🚀

通过实现Tiny-Torch，您将获得对现代深度学习框架底层机制的深刻理解，这对于深度学习研究和工程实践都具有重要价值。

如果在实现过程中遇到问题，欢迎提交 [Issues](../../issues) 或参与 [Discussions](../../discussions)。
