# Tiny-Torch Docker Development Environment

本目录包含了Tiny-Torch项目的Docker开发环境配置，支持GPU（CUDA 12.9.1）和CPU两种开发模式，内置PyTorch 2.7.1支持。

## 🚀 快速开始

### 1. 构建Docker镜像

```bash
# 构建所有镜像（推荐）
./docker/build.sh all

# 仅构建CPU版本
./docker/build.sh cpu

# 仅构建GPU版本（需要NVIDIA Docker支持）
./docker/build.sh gpu
```

### 2. 启动开发环境

```bash
# 使用统一的build.sh脚本（推荐）
./docker/build.sh dev-cpu     # 启动CPU开发环境
./docker/build.sh dev-gpu     # 启动GPU开发环境

# 或直接使用docker-compose
docker-compose run --rm tiny-torch-cpu
docker-compose run --rm tiny-torch-dev
```

### 3. 运行测试

```bash
# 测试Docker环境
./docker/build.sh test all

# 在容器中构建和测试项目
./docker/build.sh project-build
./docker/build.sh project-test
```

## 📦 环境配置

### GPU版本 (tiny-torch:dev)
- **基础镜像**: nvidia/cuda:12.9.1-devel-ubuntu22.04
- **CUDA版本**: 12.9.1
- **PyTorch版本**: 2.7.1 (with CUDA 12.9 support)
- **Python版本**: 3.10+
- **GPU运行时**: 需要NVIDIA Docker runtime

### CPU版本 (tiny-torch:cpu)
- **基础镜像**: ubuntu:22.04
- **PyTorch版本**: 2.7.1 (CPU only)
- **Python版本**: 3.10+
- **用途**: 无GPU服务器开发

### 预装软件包

#### Python核心库
- PyTorch 2.7.1 + torchvision + torchaudio
- NumPy, typing-extensions
- pybind11 (用于C++绑定)

#### 开发工具
- CMake, Ninja Build
- pytest, pytest-cov (测试框架)
- black, flake8, mypy (代码质量)
- pre-commit, isort (代码格式化)

#### 实用工具
- Jupyter Notebook + IPython
- Git, vim, htop, tree
- 完整的C++编译工具链

## 🛠️ 开发工作流

### 1. 日常开发

```bash
# 启动开发环境
./docker/build.sh dev-cpu

# 在容器内
cd /workspace
make install          # 构建项目
make test             # 运行测试
python -c "import torch; print(torch.__version__)"
```

### 2. Jupyter开发

```bash
# 启动Jupyter Notebook
./docker/build.sh jupyter-cpu    # CPU版本
./docker/build.sh jupyter-gpu    # GPU版本

# 访问 http://localhost:8888
```

### 3. 代码调试

```bash
# 进入容器Shell
./docker/build.sh shell

# 在容器内使用开发工具
pytest test/                  # 运行测试
black .                      # 代码格式化
mypy torch/                  # 类型检查
```

## 📋 构建和测试说明

### 自动化构建

1. **构建Docker镜像**:
   ```bash
   cd /path/to/tiny-torch
   ./docker/build.sh build all
   ```

2. **验证构建**:
   ```bash
   ./docker/build.sh test all
   ```

3. **启动开发**:
   ```bash
   ./docker/build.sh dev-cpu
   ```

### 手动测试步骤

#### 环境测试
```bash
# 1. 测试Python环境
docker-compose run --rm tiny-torch-cpu python --version

# 2. 测试PyTorch安装
docker-compose run --rm tiny-torch-cpu python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"

# 3. 测试基本tensor操作
docker-compose run --rm tiny-torch-cpu python -c "
import torch
x = torch.randn(3, 3)
y = torch.randn(3, 3)
z = torch.matmul(x, y)
print(f'Matrix multiplication result shape: {z.shape}')
"
```

#### 项目构建测试
```bash
# 1. 检查项目结构
docker-compose run --rm tiny-torch-cpu ls -la /workspace

# 2. 测试构建系统
docker-compose run --rm tiny-torch-cpu bash -c "
cd /workspace
python3 tools/diagnose_build.py
"

# 3. 尝试构建项目
docker-compose run --rm tiny-torch-cpu bash -c "
cd /workspace
make clean
make install
"

# 4. 运行项目测试
docker-compose run --rm tiny-torch-cpu bash -c "
cd /workspace
make test
"
```

#### 开发工具测试
```bash
# 1. 测试CMake
docker-compose run --rm tiny-torch-cpu cmake --version

# 2. 测试C++编译器
docker-compose run --rm tiny-torch-cpu g++ --version

# 3. 测试Python开发工具
docker-compose run --rm tiny-torch-cpu python -c "
import pytest, black, mypy, pybind11
print('All development tools available')
"

# 4. 测试Jupyter
docker-compose run --rm tiny-torch-cpu jupyter --version
```

### GPU特定测试 (需要GPU服务器)

```bash
# 1. 测试NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.9.1-base-ubuntu22.04 nvidia-smi

# 2. 测试PyTorch GPU支持
docker-compose run --rm tiny-torch-dev python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'Current GPU: {torch.cuda.current_device()}')
    print(f'GPU name: {torch.cuda.get_device_name()}')
"

# 3. 测试GPU tensor操作
docker-compose run --rm tiny-torch-dev python -c "
import torch
if torch.cuda.is_available():
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print(f'GPU computation successful: {z.shape}')
else:
    print('GPU not available for testing')
"
```

## 🔧 统一管理脚本

### `./docker/build.sh` - 统一管理脚本

```bash
# 构建命令
./docker/build.sh build [gpu|cpu|all]     # 构建Docker镜像

# 开发命令
./docker/build.sh dev-cpu                 # 启动CPU开发环境
./docker/build.sh dev-gpu                 # 启动GPU开发环境
./docker/build.sh jupyter-cpu             # 启动Jupyter (CPU)
./docker/build.sh jupyter-gpu             # 启动Jupyter (GPU)
./docker/build.sh shell [cpu|gpu]         # 打开容器Shell

# 项目命令
./docker/build.sh project-build           # 构建项目
./docker/build.sh project-test            # 运行项目测试

# 测试命令
./docker/build.sh test [cpu|gpu|all]      # 测试环境
./docker/build.sh test-python             # 测试Python环境
./docker/build.sh test-pytorch            # 测试PyTorch安装
./docker/build.sh test-cuda               # 测试CUDA支持
./docker/build.sh test-project            # 测试项目功能

# 实用命令
./docker/build.sh clean                   # 清理容器和镜像
./docker/build.sh logs [cpu|gpu]          # 查看容器日志
./docker/build.sh status                  # 显示状态
./docker/build.sh help                    # 显示帮助
```

## 🐛 故障排除

### 常见问题

1. **GPU支持不可用**
   ```bash
   # 检查NVIDIA Docker runtime
   docker run --rm --gpus all nvidia/cuda:12.9.1-base-ubuntu22.04 nvidia-smi
   
   # 如果失败，安装nvidia-container-runtime
   # 参考: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
   ```

2. **构建失败**
   ```bash
   # 清理并重新构建
   ./docker/build.sh clean
   ./docker/build.sh build all
   ```

3. **权限问题**
   ```bash
   # 确保脚本可执行
   chmod +x docker/*.sh
   ```

4. **网络问题**
   ```bash
   # 使用国内PyTorch镜像源（如需要）
   # 编辑 Dockerfile 中的 pip install 命令
   pip3 install torch==2.7.0 -i https://pypi.tuna.tsinghua.edu.cn/simple/
   ```

### 调试模式

```bash
# 启用Docker构建详细输出
DOCKER_BUILDKIT=0 docker build -f docker/Dockerfile -t tiny-torch:dev .

# 检查容器内部状态
docker-compose run --rm tiny-torch-cpu bash -c "
echo 'Python version:' && python --version
echo 'PyTorch version:' && python -c 'import torch; print(torch.__version__)'
echo 'Working directory:' && pwd
echo 'Files:' && ls -la
"
```

## 📚 进阶使用

### 1. 数据卷挂载

```yaml
# 在docker-compose.yml中添加自定义卷
volumes:
  - ./data:/workspace/data
  - ~/.cache/pip:/home/tinytorch/.cache/pip
```

### 2. 环境变量配置

```bash
# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=0,1
docker-compose run --rm tiny-torch-dev

# 设置开发模式
export TINY_TORCH_DEV=1
```

### 3. 端口映射

```bash
# 映射额外端口 (TensorBoard等)
docker-compose run -p 6006:6006 --rm tiny-torch-dev
```

### 4. IDE集成

```bash
# VS Code Remote-Containers
# 添加 .devcontainer/devcontainer.json 配置

# PyCharm Professional
# 配置Docker作为远程Python解释器
```

## 📝 开发注意事项

1. **数据持久化**: 容器内的 `/workspace` 目录挂载到项目根目录，数据会持久化
2. **GPU内存**: GPU版本默认使用所有可用GPU，可通过环境变量限制
3. **网络端口**: Jupyter默认使用8888端口，TensorBoard使用6006端口
4. **用户权限**: 容器内使用非root用户 `tinytorch` 运行，避免权限问题
5. **依赖更新**: 如需添加新依赖，更新Dockerfile并重新构建镜像

## 🤝 贡献

如需改进Docker环境配置，请遵循以下步骤：

1. 修改相应的Dockerfile或脚本
2. 运行测试: `./docker/test.sh`
3. 更新文档
4. 提交Pull Request

---

**作者**: Tiny-Torch团队  
**更新日期**: 2025年6月  
**Docker版本要求**: 20.10+  
**nvidia-container-runtime**: 3.4.0+ (GPU支持)
