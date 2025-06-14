# Makefile for Tiny-Torch (参考 pytorch/Makefile)
# 简化的构建接口，方便开发者使用

.PHONY: help clean build install test lint format docs benchmark

# 默认目标
help:
	@echo "Tiny-Torch Build System"
	@echo "======================="
	@echo "Available targets:"
	@echo "  help       - Show this help message"
	@echo "  clean      - Clean build artifacts"
	@echo "  build      - Build the project"
	@echo "  install    - Install the package"
	@echo "  test       - Run tests"
	@echo "  test-cuda  - Run CUDA tests"
	@echo "  lint       - Run code quality checks"
	@echo "  format     - Format code"
	@echo "  docs       - Build documentation"
	@echo "  benchmark  - Run benchmarks"
	@echo ""
	@echo "Environment variables:"
	@echo "  WITH_CUDA=1    - Enable CUDA support"
	@echo "  WITH_MKL=1     - Enable Intel MKL"
	@echo "  DEBUG=1        - Debug build"
	@echo "  VERBOSE=1      - Verbose output"

# 清理构建产物
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf torch/_C*.so
	rm -rf **/__pycache__/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	find . -name "*.pyc" -delete
	find . -name "*.so" -delete
	find . -name "*.dylib" -delete
	@echo "Clean completed."

# 构建项目
build:
	@echo "Building Tiny-Torch..."
	python setup.py build_ext --inplace

# 安装项目
install: build
	@echo "Installing Tiny-Torch..."
	pip install -e .

# 运行测试
test: install
	@echo "Running tests..."
	pytest test/ -v

# 运行CUDA测试
test-cuda: install
	@echo "Running CUDA tests..."
	pytest test/ -v -m cuda

# 代码质量检查
lint:
	@echo "Running code quality checks..."
	pre-commit run --all-files

# 代码格式化
format:
	@echo "Formatting code..."
	black torch/ test/ examples/
	isort torch/ test/ examples/
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
