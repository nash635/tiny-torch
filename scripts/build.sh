#!/bin/bash
# scripts/build.sh
# 构建脚本 (参考 pytorch 构建流程)

set -e  # 出错时退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查命令是否存在
check_command() {
    if ! command -v $1 &> /dev/null; then
        log_error "$1 could not be found"
        return 1
    fi
    return 0
}

# 获取脚本目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 解析命令行参数
BUILD_TYPE="Release"
CLEAN_BUILD=0
VERBOSE=0
JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            BUILD_TYPE="Debug"
            DEBUG=1
            shift
            ;;
        --clean)
            CLEAN_BUILD=1
            shift
            ;;
        --verbose)
            VERBOSE=1
            shift
            ;;
        --jobs)
            JOBS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --debug     Build in debug mode"
            echo "  --clean     Clean build directory first"
            echo "  --verbose   Verbose output"
            echo "  --jobs N    Use N parallel jobs"
            echo "  --help      Show this help"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

log_info "Starting Tiny-Torch build..."
log_info "Build type: $BUILD_TYPE"
log_info "Jobs: $JOBS"

# 检查依赖
log_info "Checking build dependencies..."
check_command python3 || exit 1
check_command cmake || exit 1

# 检查Python版本
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
log_info "Python version: $PYTHON_VERSION"

# 检查环境
cd "$PROJECT_ROOT"
log_info "Checking build environment..."
python3 tools/setup_helpers/env.py

# 设置环境变量
export CMAKE_BUILD_TYPE="$BUILD_TYPE"
if [[ "$VERBOSE" == "1" ]]; then
    export VERBOSE=1
fi

# 清理构建目录
if [[ "$CLEAN_BUILD" == "1" ]]; then
    log_info "Cleaning build directory..."
    rm -rf build/
    rm -rf dist/
    rm -rf *.egg-info/
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -delete 2>/dev/null || true
    find . -name "*.so" -delete 2>/dev/null || true
    find . -name "*.dylib" -delete 2>/dev/null || true
fi

# 创建构建目录
BUILD_DIR="$PROJECT_ROOT/build"
mkdir -p "$BUILD_DIR"

# CMake配置
log_info "Configuring with CMake..."
cd "$BUILD_DIR"

CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    -DBUILD_TESTS=ON
    -DWITH_CUDA="${WITH_CUDA:-ON}"
    -DWITH_MKL="${WITH_MKL:-OFF}"
    -DWITH_OPENMP="${WITH_OPENMP:-ON}"
    -DPYTHON_EXECUTABLE="$(which python3)"
)

if [[ "$VERBOSE" == "1" ]]; then
    CMAKE_ARGS+=(-DCMAKE_VERBOSE_MAKEFILE=ON)
fi

cmake "${CMAKE_ARGS[@]}" ..

# 编译
log_info "Building with $JOBS jobs..."
if [[ "$VERBOSE" == "1" ]]; then
    make -j"$JOBS" VERBOSE=1
else
    make -j"$JOBS"
fi

# 回到项目根目录
cd "$PROJECT_ROOT"

# Python扩展编译
log_info "Building Python extensions..."
if [[ "$VERBOSE" == "1" ]]; then
    python3 setup.py build_ext --inplace --verbose
else
    python3 setup.py build_ext --inplace
fi

log_info "Build completed successfully!"
log_info "You can now install with: pip install -e ."
