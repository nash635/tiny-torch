#!/bin/bash
# Build script for Tiny-Torch Docker environments

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

# Build Docker image
build_image() {
    local image_type=$1
    local dockerfile=$2
    local tag=$3
    
    print_status "Building ${image_type} Docker image..."
    
    if docker build -f "docker/${dockerfile}" -t "${tag}" .; then
        print_success "${image_type} image built successfully: ${tag}"
    else
        print_error "Failed to build ${image_type} image"
        exit 1
    fi
}

# Main build function
main() {
    print_status "Starting Tiny-Torch Docker environment build..."
    
    # Change to project root directory
    cd "$(dirname "$0")/.."
    
    # Check prerequisites
    check_docker
    
    # Check command line arguments
    case "${1:-all}" in
        "gpu"|"cuda")
            if check_nvidia_docker; then
                build_image "GPU/CUDA" "Dockerfile" "tiny-torch:dev"
            else
                print_error "GPU build requested but NVIDIA Docker runtime not available"
                exit 1
            fi
            ;;
        "cpu")
            build_image "CPU-only" "Dockerfile.cpu" "tiny-torch:cpu"
            ;;
        "all")
            # Build CPU version first
            build_image "CPU-only" "Dockerfile.cpu" "tiny-torch:cpu"
            
            # Try to build GPU version if available
            if check_nvidia_docker; then
                build_image "GPU/CUDA" "Dockerfile" "tiny-torch:dev"
            else
                print_warning "Skipping GPU build - NVIDIA Docker runtime not available"
            fi
            ;;
        *)
            print_error "Invalid argument. Use: gpu, cpu, or all"
            echo "Usage: $0 [gpu|cpu|all]"
            exit 1
            ;;
    esac
    
    print_success "Docker build completed!"
    
    # Show available images
    print_status "Available Tiny-Torch Docker images:"
    docker images | grep "tiny-torch" || print_warning "No Tiny-Torch images found"
    
    print_status "To run the development environment:"
    echo "  GPU version:  docker-compose run --rm tiny-torch-dev"
    echo "  CPU version:  docker-compose run --rm tiny-torch-cpu"
    echo "  Or use the test script: ./docker/test.sh"
}

# Run main function
main "$@"
