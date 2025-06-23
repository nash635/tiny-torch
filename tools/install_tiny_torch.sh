#!/bin/bash
# Tiny-Torch Installation Script
# Handles both host and container environments

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Check if we're in a container
is_container() {
    if [ -f /.dockerenv ] || [ -n "${CONTAINER}" ] || [ -n "${DOCKER_CONTAINER}" ]; then
        return 0
    else
        return 1
    fi
}

# Detect pip flags needed
get_pip_flags() {
    if is_container; then
        echo "--break-system-packages"
    else
        echo ""
    fi
}

main() {
    print_status "Starting Tiny-Torch installation..."
    
    # Detect environment
    if is_container; then
        print_status "Container environment detected"
        PIP_FLAGS="--break-system-packages"
    else
        print_status "Host environment detected"
        PIP_FLAGS=""
    fi
    
    # Clean previous installations
    print_status "Cleaning previous installation..."
    make clean || true
    
    # Install Python dependencies
    print_status "Installing Python dependencies..."
    if ! pip install -r requirements.txt $PIP_FLAGS; then
        print_error "Failed to install requirements"
        exit 1
    fi
    
    # Build C++ extensions
    print_status "Building C++ extensions..."
    if ! python setup.py build_ext --inplace; then
        print_error "Failed to build C++ extensions"
        exit 1
    fi
    
    # Install in development mode
    print_status "Installing in development mode..."
    if ! pip install -e . --no-deps $PIP_FLAGS; then
        print_error "Failed to install in development mode"
        exit 1
    fi
    
    # Verify installation
    print_status "Verifying installation..."
    if python -c "import tiny_torch; print('✅ tiny_torch imported successfully')"; then
        print_success "Installation completed successfully!"
    else
        print_error "Installation verification failed"
        exit 1
    fi
    
    print_status "Running basic tests..."
    if python -c "
import tiny_torch
print(f'Tiny-Torch version: {tiny_torch.__version__}')
print(f'CUDA available: {tiny_torch.cuda.is_available()}')
print('✅ Basic functionality verified')
"; then
        print_success "All tests passed!"
    else
        print_warning "Some tests failed, but installation appears successful"
    fi
    
    print_success "Tiny-Torch installation complete!"
    echo ""
    echo "Next steps:"
    echo "  • Run tests: make test"
    echo "  • Check diagnostics: make diagnose"
    echo "  • Start development: python -c 'import tiny_torch'"
}

main "$@"
