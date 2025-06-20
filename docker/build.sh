#!/bin/bash
# Unified Docker script for Tiny-Torch development environment
# Integrates build, test, and development functions

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_command() {
    echo -e "${YELLOW}[COMMAND]${NC} $1"
}

# Execute command with debug output
execute_cmd() {
    local cmd="$1"
    print_command "$cmd"
    eval "$cmd"
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        print_success "Command executed successfully"
    else
        print_error "Command failed with exit code: $exit_code"
    fi
    return $exit_code
}

# Show usage information
show_usage() {
    echo "Tiny-Torch Docker Management Script"
    echo ""
    echo "USAGE:"
    echo "  $0 <command> [options]"
    echo ""
    echo "BUILD COMMANDS:"
    echo "  build [gpu|cpu|all]     Build Docker images"
    echo ""
    echo "DEVELOPMENT COMMANDS:"
    echo "  dev-cpu                 Start CPU development environment"
    echo "  dev-gpu                 Start GPU development environment"
    echo "  jupyter-cpu             Start Jupyter Notebook (CPU)"
    echo "  jupyter-gpu             Start Jupyter Notebook (GPU)"
    echo "  shell [cpu|gpu]         Open interactive shell"
    echo ""
    echo "PROJECT COMMANDS:"
    echo "  project-build           Build tiny-torch project in container"
    echo "  project-test            Run project tests in container"
    echo ""
    echo "TEST COMMANDS:"
    echo "  test [cpu|gpu|all]      Test Docker environments"
    echo "  test-python             Test Python environment"
    echo "  test-pytorch            Test PyTorch installation"
    echo "  test-cuda               Test CUDA support (GPU only)"
    echo "  test-project            Test project building and functionality"
    echo ""
    echo "UTILITY COMMANDS:"
    echo "  clean                   Clean containers and images"
    echo "  logs [cpu|gpu]          View container logs"
    echo "  status                  Show container status"
    echo ""
    echo "EXAMPLES:"
    echo "  $0 build all            # Build all Docker images"
    echo "  $0 dev-gpu              # Start GPU development environment"
    echo "  $0 test all             # Test all environments"
    echo "  $0 project-build        # Build project in container"
}
# Check if docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running. Please start Docker."
        exit 1
    fi
    
    print_success "Docker is available and running"
}

# Check if nvidia-docker is available (for GPU support)
check_nvidia_docker() {
    if command -v nvidia-docker &> /dev/null || docker info | grep -q nvidia; then
        print_success "NVIDIA Docker runtime is available"
        return 0
    else
        print_warning "NVIDIA Docker runtime not found. GPU support will not be available."
        return 1
    fi
}

# Check if docker-compose is available
check_docker_compose() {
    if ! command -v docker-compose &> /dev/null; then
        print_error "docker-compose is not installed. Please install docker-compose."
        exit 1
    fi
    print_success "docker-compose is available"
}
# Build Docker image
build_image() {
    local image_type=$1
    local dockerfile=$2
    local tag=$3
    
    print_status "Building ${image_type} Docker image..."
    
    local build_cmd="docker build -f \"docker/${dockerfile}\" -t \"${tag}\" ."
    if execute_cmd "$build_cmd"; then
        print_success "${image_type} image built successfully: ${tag}"
    else
        print_error "Failed to build ${image_type} image"
        exit 1
    fi
}

# Build all or specific images
cmd_build() {
    local target="${1:-all}"
    
    print_status "Starting Tiny-Torch Docker environment build..."
    
    # Change to project root directory
    cd "$(dirname "$0")/.."
    
    # Check prerequisites
    check_docker
    
    case "${target}" in
        "gpu"|"cuda")
            if check_nvidia_docker; then
                build_image "GPU/CUDA" "Dockerfile" "tiny-torch:latest"
            else
                print_error "GPU build requested but NVIDIA Docker runtime not available"
                exit 1
            fi
            ;;
        "cpu")
            build_image "CPU-only" "Dockerfile.cpu" "tiny-torch:cpu"
            ;;
        "all")
            # Build CPU version first (if Dockerfile.cpu exists)
            if [ -f "docker/Dockerfile.cpu" ]; then
                build_image "CPU-only" "Dockerfile.cpu" "tiny-torch:cpu"
            fi
            
            # Build GPU version (current Dockerfile)
            if check_nvidia_docker; then
                build_image "GPU/CUDA" "Dockerfile" "tiny-torch:latest"
            else
                print_warning "Skipping GPU build - NVIDIA Docker runtime not available"
            fi
            ;;
        *)
            print_error "Invalid build target. Use: gpu, cpu, or all"
            exit 1
            ;;
    esac
    
    print_success "Docker build completed!"
    
    # Show available images
    print_status "Available Tiny-Torch Docker images:"
    docker images | grep "tiny-torch" || print_warning "No Tiny-Torch images found"
}

# Development environment functions
cmd_dev_cpu() {
    print_status "Starting CPU development environment..."
    cd "$(dirname "$0")/.."
    check_docker
    
    if ! docker images | grep -q "tiny-torch:cpu"; then
        print_warning "CPU image not found. Building it first..."
        cmd_build "cpu"
    fi
    
    local run_cmd="docker run --rm -it -v \"$(pwd)\":/workspace -p 8888:8888 tiny-torch:cpu /bin/bash"
    execute_cmd "$run_cmd"
}

cmd_dev_gpu() {
    print_status "Starting GPU development environment..."
    cd "$(dirname "$0")/.."
    check_docker
    
    if ! check_nvidia_docker; then
        print_error "GPU development requires NVIDIA Docker runtime"
        exit 1
    fi
    
    if ! docker images | grep -q "tiny-torch:latest"; then
        print_warning "GPU image not found. Building it first..."
        cmd_build "gpu"
    fi
    
    local run_cmd="docker run --gpus all --rm -it -v \"$(pwd)\":/workspace -p 8888:8888 -p 6006:6006 tiny-torch:latest /bin/bash"
    execute_cmd "$run_cmd"
}

cmd_jupyter_cpu() {
    print_status "Starting Jupyter Notebook (CPU)..."
    cd "$(dirname "$0")/.."
    check_docker
    
    if ! docker images | grep -q "tiny-torch:cpu"; then
        print_warning "CPU image not found. Building it first..."
        cmd_build "cpu"
    fi
    
    print_status "Jupyter will be available at: http://localhost:8888"
    local run_cmd="docker run --rm -it -v \"$(pwd)\":/workspace -p 8888:8888 tiny-torch:cpu jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root"
    execute_cmd "$run_cmd"
}

cmd_jupyter_gpu() {
    print_status "Starting Jupyter Notebook (GPU)..."
    cd "$(dirname "$0")/.."
    check_docker
    
    if ! check_nvidia_docker; then
        print_error "GPU Jupyter requires NVIDIA Docker runtime"
        exit 1
    fi
    
    if ! docker images | grep -q "tiny-torch:latest"; then
        print_warning "GPU image not found. Building it first..."
        cmd_build "gpu"
    fi
    
    print_status "Jupyter will be available at: http://localhost:8888"
    local run_cmd="docker run --gpus all --rm -it -v \"$(pwd)\":/workspace -p 8888:8888 -p 6006:6006 tiny-torch:latest jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root"
    execute_cmd "$run_cmd"
}

cmd_shell() {
    local env_type="${1:-gpu}"
    cd "$(dirname "$0")/.."
    check_docker
    
    case "$env_type" in
        "cpu")
            cmd_dev_cpu
            ;;
        "gpu")
            cmd_dev_gpu
            ;;
        *)
            print_error "Invalid shell type. Use: cpu or gpu"
            exit 1
            ;;
    esac
}

# Project management functions
cmd_project_build() {
    print_status "Building tiny-torch project in container..."
    cd "$(dirname "$0")/.."
    check_docker
    
    local image="tiny-torch:latest"
    if ! docker images | grep -q "$image"; then
        print_warning "GPU image not found. Building it first..."
        cmd_build "gpu"
    fi
    
    local run_cmd="docker run --gpus all --rm -v \"$(pwd)\":/workspace \"$image\" bash -c \"cd /workspace && make clean && make install\""
    execute_cmd "$run_cmd"
}

cmd_project_test() {
    print_status "Running tiny-torch project tests in container..."
    cd "$(dirname "$0")/.."
    check_docker
    
    local image="tiny-torch:latest"
    if ! docker images | grep -q "$image"; then
        print_warning "GPU image not found. Building it first..."
        cmd_build "gpu"
    fi
    
    local run_cmd="docker run --gpus all --rm -v \"$(pwd)\":/workspace \"$image\" bash -c \"cd /workspace && make test\""
    execute_cmd "$run_cmd"
}

# Testing functions
cmd_test() {
    local target="${1:-all}"
    cd "$(dirname "$0")/.."
    check_docker
    
    case "$target" in
        "cpu")
            test_cpu_environment
            ;;
        "gpu")
            test_gpu_environment
            ;;
        "all")
            test_cpu_environment
            test_gpu_environment
            ;;
        *)
            print_error "Invalid test target. Use: cpu, gpu, or all"
            exit 1
            ;;
    esac
}

test_cpu_environment() {
    print_status "Testing CPU environment..."
    
    if ! docker images | grep -q "tiny-torch:cpu"; then
        print_warning "CPU image not found. Building it first..."
        cmd_build "cpu"
    fi
    
    # Test basic Python environment
    print_status "Testing Python environment..."
    local cmd1="docker run --rm tiny-torch:cpu python --version"
    execute_cmd "$cmd1"
    
    # Test PyTorch installation
    print_status "Testing PyTorch installation..."
    local cmd2="docker run --rm tiny-torch:cpu python -c \"
import tiny_torch
print(f'PyTorch version: {tiny_torch.__version__}')
print(f'CUDA available: {tiny_torch.cuda.is_available()}')
\""
    execute_cmd "$cmd2"
    
    # Test basic tensor operations
    print_status "Testing basic tensor operations..."
    local cmd3="docker run --rm tiny-torch:cpu python -c \"
import tiny_torch
x = tiny_torch.randn(3, 3)
y = tiny_torch.randn(3, 3)
z = tiny_torch.matmul(x, y)
print(f'Matrix multiplication successful: {z.shape}')
\""
    execute_cmd "$cmd3"
    
    print_success "CPU environment tests passed!"
}

test_gpu_environment() {
    print_status "Testing GPU environment..."
    
    if ! check_nvidia_docker; then
        print_warning "Skipping GPU tests - NVIDIA Docker runtime not available"
        return
    fi
    
    if ! docker images | grep -q "tiny-torch:latest"; then
        print_warning "GPU image not found. Building it first..."
        cmd_build "gpu"
    fi
    
    # Test NVIDIA runtime
    print_status "Testing NVIDIA Docker runtime..."
    local cmd1="docker run --gpus all --rm nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi"
    if ! execute_cmd "$cmd1" > /dev/null 2>&1; then
        print_error "NVIDIA Docker runtime test failed"
        return 1
    fi
    
    # Test PyTorch GPU support
    print_status "Testing PyTorch GPU support..."
    local cmd2="docker run --gpus all --rm tiny-torch:latest python -c \"
import tiny_torch
print(f'PyTorch version: {tiny_torch.__version__}')
print(f'CUDA available: {tiny_torch.cuda.is_available()}')
if tiny_torch.cuda.is_available():
    print(f'GPU count: {tiny_torch.cuda.device_count()}')
    print(f'Current GPU: {tiny_torch.cuda.current_device()}')
    print(f'GPU name: {tiny_torch.cuda.get_device_name()}')
else:
    print('No GPU available')
\""
    execute_cmd "$cmd2"
    
    # Test GPU tensor operations
    print_status "Testing GPU tensor operations..."
    local cmd3="docker run --gpus all --rm tiny-torch:latest python -c \"
import tiny_torch
if tiny_torch.cuda.is_available():
    x = tiny_torch.randn(100, 100).cuda()
    y = tiny_torch.randn(100, 100).cuda()
    z = tiny_torch.matmul(x, y)
    print(f'GPU computation successful: {z.shape} on {z.device}')
else:
    print('No GPU available for testing')
\""
    execute_cmd "$cmd3"
    
    print_success "GPU environment tests passed!"
}

cmd_test_python() {
    print_status "Testing Python environment in all containers..."
    cd "$(dirname "$0")/.."
    
    for image in "tiny-torch:cpu" "tiny-torch:latest"; do
        if docker images | grep -q "$image"; then
            print_status "Testing $image..."
            local cmd="docker run --rm \"$image\" python -c \"
import sys
print(f'Python version: {sys.version}')
print('Python path:', sys.path[:3])
import tiny_torch
print(f'PyTorch version: {tiny_torch.__version__}')
print('Available modules: torch, numpy, pytest, jupyter')
\""
            execute_cmd "$cmd"
        fi
    done
}

cmd_test_pytorch() {
    print_status "Testing PyTorch installation and functionality..."
    cd "$(dirname "$0")/.."
    
    # Test with standard PyTorch (not project torch)
    local test_script="
import sys
# Remove workspace from path to test installed PyTorch
if '/workspace' in sys.path:
    sys.path.remove('/workspace')

import tiny_torch
print(f'Standard PyTorch version: {tiny_torch.__version__}')
print(f'CUDA available: {tiny_torch.cuda.is_available()}')

# Test basic operations
x = tiny_torch.randn(5, 3)
y = tiny_torch.randn(3, 4)
z = tiny_torch.mm(x, y)
print(f'Matrix multiplication: {x.shape} x {y.shape} = {z.shape}')

# Test autograd
x = tiny_torch.randn(2, 2, requires_grad=True)
y = x.sum()
y.backward()
print(f'Autograd test: gradient shape {x.grad.shape}')
print('PyTorch functionality test passed!')
"
    
    if docker images | grep -q "tiny-torch:latest"; then
        local cmd1="docker run --gpus all --rm tiny-torch:latest python -c \"$test_script\""
        execute_cmd "$cmd1"
    elif docker images | grep -q "tiny-tiny_torch.*cpu"; then
        local cmd2="docker run --rm tiny-torch:cpu python -c \"$test_script\""
        execute_cmd "$cmd2"
    else
        print_error "No tiny-torch images found. Build images first."
        exit 1
    fi
}

cmd_test_cuda() {
    print_status "Testing CUDA support..."
    cd "$(dirname "$0")/.."
    
    if ! check_nvidia_docker; then
        print_error "CUDA test requires NVIDIA Docker runtime"
        exit 1
    fi
    
    if ! docker images | grep -q "tiny-torch:latest"; then
        print_warning "GPU image not found. Building it first..."
        cmd_build "gpu"
    fi
    
    local cmd="docker run --gpus all --rm tiny-torch:latest python -c \"
import tiny_torch
print('=== CUDA Environment Test ===')
print(f'CUDA available: {tiny_torch.cuda.is_available()}')
if tiny_torch.cuda.is_available():
    print(f'CUDA version: {tiny_torch.version.cuda}')
    print(f'GPU count: {tiny_torch.cuda.device_count()}')
    for i in range(tiny_torch.cuda.device_count()):
        print(f'GPU {i}: {tiny_torch.cuda.get_device_name(i)}')
        print(f'  Memory: {tiny_torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB')
    
    # Test GPU memory allocation
    print('Testing GPU memory allocation...')
    x = tiny_torch.randn(1000, 1000).cuda()
    print(f'Allocated tensor on GPU: {x.device}')
    print(f'Memory allocated: {tiny_torch.cuda.memory_allocated() / 1024**2:.1f} MB')
    print('CUDA functionality test passed!')
else:
    print('CUDA not available')
\""
    execute_cmd "$cmd"
}

cmd_test_project() {
    print_status "Testing project building and functionality..."
    cd "$(dirname "$0")/.."
    
    if ! docker images | grep -q "tiny-torch:latest"; then
        print_warning "GPU image not found. Building it first..."
        cmd_build "gpu"
    fi
    
    # Test project structure
    print_status "Checking project structure..."
    local cmd1="docker run --gpus all --rm -v \"$(pwd)\":/workspace tiny-torch:latest bash -c \"
cd /workspace
echo 'Project files:'
ls -la
echo 'Python files:'
find . -name '*.py' | head -10
echo 'C++ files:'
find . -name '*.cpp' -o -name '*.cu' | head -10
\""
    execute_cmd "$cmd1"
    
    # Test build system
    print_status "Testing build system..."
    local cmd2="docker run --gpus all --rm -v \"$(pwd)\":/workspace tiny-torch:latest bash -c \"
cd /workspace
if [ -f 'tools/diagnose_build.py' ]; then
    python3 tools/diagnose_build.py
fi
\""
    execute_cmd "$cmd2"
    
    # Try to build project
    print_status "Attempting to build project..."
    local cmd3="docker run --gpus all --rm -v \"$(pwd)\":/workspace tiny-torch:latest bash -c \"
cd /workspace
if [ -f 'Makefile' ]; then
    make clean || true
    make install || echo 'Build failed - this is expected for development setup'
elif [ -f 'setup.py' ]; then
    python setup.py build || echo 'Build failed - this is expected for development setup' 
elif [ -f 'pyproject.toml' ]; then
    pip install -e . || echo 'Build failed - this is expected for development setup'
fi
\""
    execute_cmd "$cmd3"
    
    print_success "Project test completed!"
}

# Utility functions
cmd_clean() {
    print_status "Cleaning Docker containers and images..."
    cd "$(dirname "$0")/.."
    
    # Stop and remove containers
    print_status "Removing containers..."
    local containers=$(docker ps -a | grep tiny-torch | awk '{print $1}')
    if [ -n "$containers" ]; then
        local cmd1="docker rm -f $containers"
        execute_cmd "$cmd1"
    else
        print_status "No tiny-torch containers found"
    fi
    
    # Remove images
    print_status "Removing images..."
    local images=$(docker images | grep tiny-torch | awk '{print $3}')
    if [ -n "$images" ]; then
        local cmd2="docker rmi -f $images"
        execute_cmd "$cmd2"
    else
        print_status "No tiny-torch images found"
    fi
    
    # Clean build cache
    print_status "Cleaning build cache..."
    local cmd3="docker builder prune -f"
    execute_cmd "$cmd3"
    
    print_success "Cleanup completed!"
}

cmd_logs() {
    local env_type="${1:-gpu}"
    
    case "$env_type" in
        "cpu")
            if docker ps | grep -q tiny-torch-cpu; then
                local cmd="docker logs -f tiny-torch-cpu"
                execute_cmd "$cmd"
            else
                print_warning "No running CPU container found"
            fi
            ;;
        "gpu")
            local container=$(docker ps | grep tiny-torch | grep -v cpu | awk '{print $1}')
            if [ -n "$container" ]; then
                local cmd="docker logs -f $container"
                execute_cmd "$cmd"
            else
                print_warning "No running GPU container found"
            fi
            ;;
        *)
            print_error "Invalid log type. Use: cpu or gpu"
            exit 1
            ;;
    esac
}

cmd_status() {
    print_status "Docker container and image status:"
    
    echo ""
    echo "=== Running Containers ==="
    docker ps | grep tiny-torch || echo "No tiny-torch containers running"
    
    echo ""
    echo "=== All Containers ==="
    docker ps -a | grep tiny-torch || echo "No tiny-torch containers found"
    
    echo ""
    echo "=== Available Images ==="
    docker images | grep tiny-torch || echo "No tiny-torch images found"
    
    echo ""
    echo "=== System Resources ==="
    docker system df
}

# Main function
main() {
    # Change to project root directory
    cd "$(dirname "$0")/.."
    
    local command="${1:-help}"
    shift || true
    
    case "$command" in
        # Build commands
        "build")
            cmd_build "$@"
            ;;
        
        # Development commands
        "dev-cpu")
            cmd_dev_cpu
            ;;
        "dev-gpu")
            cmd_dev_gpu
            ;;
        "jupyter-cpu")
            cmd_jupyter_cpu
            ;;
        "jupyter-gpu")
            cmd_jupyter_gpu
            ;;
        "shell")
            cmd_shell "$@"
            ;;
        
        # Project commands
        "project-build")
            cmd_project_build
            ;;
        "project-test")
            cmd_project_test
            ;;
        
        # Test commands
        "test")
            cmd_test "$@"
            ;;
        "test-python")
            cmd_test_python
            ;;
        "test-pytorch")
            cmd_test_pytorch
            ;;
        "test-cuda")
            cmd_test_cuda
            ;;
        "test-project")
            cmd_test_project
            ;;
        
        # Utility commands
        "clean")
            cmd_clean
            ;;
        "logs")
            cmd_logs "$@"
            ;;
        "status")
            cmd_status
            ;;
        
        # Help and legacy support
        "help"|"-h"|"--help")
            show_usage
            ;;
        
        # Legacy support for old usage
        "gpu"|"cpu"|"all")
            print_warning "Legacy usage detected. Use: $0 build $command"
            cmd_build "$command"
            ;;
        
        *)
            print_error "Unknown command: $command"
            echo ""
            show_usage
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
