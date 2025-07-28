# Makefile for Tiny-Torch (reference pytorch/Makefile)
# Simplified build interface for developer convenience

.PHONY: help clean build install test lint format docs benchmark

# Default target
help:
	@echo "Tiny-Torch Build System"
	@echo "======================="
	@echo "Available targets:"
	@echo "  help        - Show this help message"
	@echo "  status      - Quick status check"
	@echo "  diagnose    - Run build diagnostics"
	@echo "  clean       - Clean build artifacts"
	@echo "  build       - Build C++ extensions only"
	@echo "  install     - Complete installation (recommended)"
	@echo "  install-full- Full installation with diagnostics"
	@echo "  test        - Run tests (includes phase verification)"
	@echo "  test-cuda   - Run CUDA tests (auto-skip if no CUDA)"
	@echo "  test-no-cuda- Run tests excluding CUDA tests"
	@echo "  lint        - Run code quality checks"
	@echo "  format      - Format code"
	@echo "  docs        - Build documentation"
	@echo "  benchmark   - Run benchmarks"
	@echo ""
	@echo "Recommended workflow:"
	@echo "  pip install -r requirements.txt  # Install dependencies once"
	@echo "  make build    # Development/quick compilation"
	@echo "  make test     # Verify functionality"
	@echo ""
	@echo "Environment variables:"
	@echo "  WITH_CUDA=1   - Enable CUDA support"
	@echo "  WITH_MKL=1    - Enable Intel MKL"
	@echo "  DEBUG=1       - Debug build"
	@echo "  USE_NINJA=1   - Force Ninja backend"
	@echo "  VERBOSE=1      - Verbose output"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf tiny_torch/_C*.so
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	find . -name "*.so" -delete
	@echo "Clean completed."

# Diagnose build issues
diagnose:
	@echo "Running build diagnostics..."
	python3 tools/diagnose_build.py

# Quick status check
status:
	@echo "Quick status check..."
	python3 tools/diagnose_build.py --quick

# Build project
build:
	@echo "Building Tiny-Torch..."
	python3 setup.py build_ext --inplace

# Install project (improved version handles multiple .egg-info directories issue)
install: clean
	@echo "Installing Tiny-Torch (with cleanup)..."
	@if [ -f /.dockerenv ] || [ -n "$(CONTAINER)" ]; then \
		echo "Detected container environment, using --break-system-packages"; \
		pip install -r requirements.txt --break-system-packages; \
		python setup.py build_ext --inplace; \
		pip install -e . --no-deps --break-system-packages; \
	else \
		pip install -r requirements.txt; \
		python setup.py build_ext --inplace; \
		pip install -e . --no-deps; \
	fi

# 完整安装 (使用专用脚本)
install-full:
	@echo "Full installation with diagnostics..."
	chmod +x tools/install_tiny_torch.sh
	./tools/install_tiny_torch.sh

# 运行测试
test: build
	@echo "Running tests..."
	python -m pytest tests/ -v

# 运行CUDA测试
test-cuda: build
	@echo "Running CUDA tests..."
	@if python tools/check_cuda.py; then \
		echo "CUDA available - running CUDA-specific tests"; \
		python -m pytest tests/ -v -m cuda; \
	else \
		echo "CUDA not available - skipping CUDA tests"; \
		echo "Use 'make test' to run all non-CUDA tests"; \
	fi

# 运行非CUDA测试
test-no-cuda: build
	@echo "Running tests without CUDA..."
	python -m pytest tests/ -v -m "not cuda"

# 代码质量检查
lint:
	@echo "Running code quality checks..."
	pre-commit run --all-files

# 代码格式化
format:
	@echo "Formatting code..."
	black tiny_torch/ test/ examples/
	isort tiny_torch/ test/ examples/
	clang-format -i csrc/**/*.cpp csrc/**/*.h csrc/**/*.cu

# 构建文档
docs:
	@echo "Building documentation..."
	cd docs && make html

# 运行性能测试
benchmark: install
	@echo "Running benchmarks..."
	python benchmarks/compare_with_pytorch.py

# 开发环境设置
setup-dev:
	@echo "Setting up development environment..."
	pip install -r requirements-dev.txt
	pre-commit install
	@echo "Development environment ready!"

# 检查构建环境
check-env:
	@echo "Checking build environment..."
	@echo "Python: $$(python --version)"
	@echo "CMake: $$(cmake --version | head -1)"
	@if command -v nvcc >/dev/null 2>&1; then \
		echo "CUDA: $$(nvcc --version | grep release)"; \
	else \
		echo "CUDA: Not found"; \
	fi
	@echo "Build flags:"
	@echo "  WITH_CUDA: $${WITH_CUDA:-0}"
	@echo "  WITH_MKL: $${WITH_MKL:-0}"
	@echo "  DEBUG: $${DEBUG:-0}"
