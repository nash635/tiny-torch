#!/bin/bash
# Test setup script for tiny-torch

set -e

echo "Setting up test environment for tiny-torch..."

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python version
python_version=$(python --version 2>&1 | cut -d' ' -f2)
echo "Python version: $python_version"

# Install basic requirements
echo "Installing basic requirements..."
pip install --upgrade pip wheel setuptools

# Install requirements from files
if [ -f "requirements.txt" ]; then
    echo "Installing requirements.txt..."
    pip install -r requirements.txt
fi

if [ -f "requirements-dev.txt" ]; then
    echo "Installing requirements-dev.txt..."
    pip install -r requirements-dev.txt
fi

# Install test dependencies
echo "Installing test dependencies..."
pip install pytest pytest-cov pytest-xdist pytest-mock pytest-timeout

# Install additional test dependencies
pip install psutil  # For performance tests

# Check if CUDA is available
if command_exists nvcc; then
    echo "CUDA compiler found: $(nvcc --version | head -1)"
    echo "CUDA environment available"
else
    echo "CUDA compiler not found - will use CPU-only mode"
fi

# Check if cmake is available
if command_exists cmake; then
    echo "CMake found: $(cmake --version | head -1)"
else
    echo "Warning: CMake not found - some build tests may fail"
fi

# Check if ninja is available
if command_exists ninja; then
    echo "Ninja found: $(ninja --version)"
else
    echo "Warning: Ninja not found - will use make instead"
fi

# Install tiny-torch in development mode
echo "Installing tiny-torch in development mode..."
pip install -e .

# Verify installation
echo "Verifying installation..."
python -c "import tiny_torch; print(f'tiny-torch {tiny_torch.__version__} installed successfully')"

# Run a quick test
echo "Running quick test..."
python -c "
from tiny_torch import cuda
print(f'CUDA available: {cuda.is_available()}')
print(f'Device count: {cuda.device_count()}')
"

echo "Test environment setup complete!"
echo ""
echo "You can now run tests with:"
echo "  python tests/run_tests.py all          # Run all tests"
echo "  python tests/run_tests.py unit         # Run unit tests only"
echo "  python tests/run_tests.py --check-env  # Check environment"
echo "  pytest tests/ -v                       # Direct pytest"
