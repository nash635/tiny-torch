#!/bin/bash
# Memory Debug Tools Setup Script
# 显存调试工具安装和设置脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS_DIR="$SCRIPT_DIR"

echo "🚀 Memory Debug Tools Setup"
echo "=" | awk '{printf "%.50s\n", $0}'

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印消息函数
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

# 检查Python版本
check_python() {
    print_status "Checking Python version..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        print_error "Python not found. Please install Python 3.6 or later."
        exit 1
    fi
    
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
    print_success "Python version: $PYTHON_VERSION"
    
    # 检查Python版本是否满足要求（>= 3.6）
    if $PYTHON_CMD -c "import sys; exit(0 if sys.version_info >= (3, 6) else 1)"; then
        print_success "Python version is compatible"
    else
        print_error "Python 3.6 or later is required"
        exit 1
    fi
}

# 检查pip
check_pip() {
    print_status "Checking pip..."
    
    if ! $PYTHON_CMD -m pip --version &> /dev/null; then
        print_error "pip not found. Please install pip."
        exit 1
    fi
    
    PIP_VERSION=$($PYTHON_CMD -m pip --version | awk '{print $2}')
    print_success "pip version: $PIP_VERSION"
}

# 安装依赖
install_dependencies() {
    print_status "Installing Python dependencies..."
    
    if [ -f "$TOOLS_DIR/requirements.txt" ]; then
        $PYTHON_CMD -m pip install -r "$TOOLS_DIR/requirements.txt"
        print_success "Dependencies installed successfully"
    else
        print_warning "requirements.txt not found, installing dependencies manually..."
        $PYTHON_CMD -m pip install numpy matplotlib psutil
        print_success "Basic dependencies installed"
    fi
}

# 检查NVIDIA工具
check_nvidia() {
    print_status "Checking NVIDIA tools..."
    
    if command -v nvidia-smi &> /dev/null; then
        NVIDIA_VERSION=$(nvidia-smi --version | grep "NVIDIA-SMI" | awk '{print $3}')
        print_success "nvidia-smi found: $NVIDIA_VERSION"
        
        # 检查GPU数量
        GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
        print_success "Found $GPU_COUNT GPU(s)"
        
        return 0
    else
        print_warning "nvidia-smi not found. GPU monitoring may not work properly."
        print_warning "Please install NVIDIA drivers if you plan to monitor GPUs."
        return 1
    fi
}

# 运行测试
run_tests() {
    print_status "Running basic tests..."
    
    cd "$TOOLS_DIR"
    
    if [ -f "test_memory_tools.py" ]; then
        if $PYTHON_CMD test_memory_tools.py; then
            print_success "All tests passed!"
            return 0
        else
            print_error "Some tests failed. Please check the output above."
            return 1
        fi
    else
        print_warning "Test script not found, skipping tests"
        return 0
    fi
}

# 创建启动脚本
create_launcher() {
    print_status "Creating launcher script..."
    
    LAUNCHER_SCRIPT="$TOOLS_DIR/memory-debug"
    
    cat > "$LAUNCHER_SCRIPT" << EOF
#!/bin/bash
# Memory Debug Tools Launcher
# 显存调试工具启动器

SCRIPT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
cd "\$SCRIPT_DIR"

$PYTHON_CMD memory_debug_cli.py "\$@"
EOF
    
    chmod +x "$LAUNCHER_SCRIPT"
    print_success "Launcher script created: $LAUNCHER_SCRIPT"
}

# 显示使用说明
show_usage() {
    echo ""
    print_status "Setup completed successfully!"
    echo ""
    echo "🎯 Quick Start Guide:"
    echo ""
    echo "1. Basic memory profiling:"
    echo "   $PYTHON_CMD memory_debug_cli.py profile --duration 60 --plot"
    echo ""
    echo "2. OOM detection:"
    echo "   $PYTHON_CMD memory_debug_cli.py oom --threshold 85 --monitor"
    echo ""
    echo "3. Memory leak detection:"
    echo "   $PYTHON_CMD memory_debug_cli.py leak --enable-tracking --duration 300"
    echo ""
    echo "4. Fragmentation analysis:"
    echo "   $PYTHON_CMD memory_debug_cli.py fragment --duration 120 --plot"
    echo ""
    echo "5. Interactive mode:"
    echo "   $PYTHON_CMD memory_debug_cli.py interactive"
    echo ""
    echo "📄 For detailed documentation, see: README.md"
    echo ""
    echo "🔧 If you created the launcher script, you can also use:"
    echo "   ./memory-debug profile --duration 60"
    echo ""
}

# 主函数
main() {
    echo "Starting setup process..."
    echo ""
    
    # 检查系统环境
    check_python
    check_pip
    
    # 安装依赖
    install_dependencies
    
    # 检查NVIDIA工具（可选）
    check_nvidia || true  # 不强制要求NVIDIA工具
    
    # 运行测试
    if run_tests; then
        # 创建启动脚本
        create_launcher
        
        # 显示使用说明
        show_usage
    else
        print_error "Setup completed with warnings. Some functionality may not work properly."
        exit 1
    fi
}

# 处理命令行参数
case "${1:-}" in
    --help|-h)
        echo "Memory Debug Tools Setup Script"
        echo ""
        echo "Usage: $0 [options]"
        echo ""
        echo "Options:"
        echo "  --help, -h     Show this help message"
        echo "  --test-only    Only run tests without installation"
        echo "  --no-test      Skip tests during setup"
        echo ""
        exit 0
        ;;
    --test-only)
        check_python
        run_tests
        exit $?
        ;;
    --no-test)
        print_status "Skipping tests as requested"
        check_python
        check_pip
        install_dependencies
        check_nvidia || true
        create_launcher
        show_usage
        ;;
    "")
        main
        ;;
    *)
        print_error "Unknown option: $1"
        echo "Use --help for usage information"
        exit 1
        ;;
esac
