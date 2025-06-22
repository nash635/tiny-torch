# Tiny-Torch Docker Development Environment
# Based on NVIDIA CUDA 12.8 with PyTorch GPU support

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
ENV PATH=$CUDA_HOME/bin:$PATH
ENV PYTHONPATH=/workspace:$PYTHONPATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    ninja-build \
    git \
    wget \
    curl \
    vim \
    htop \
    tree \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Skip pip upgrade due to Ubuntu 24.04 externally managed environment
# Use system pip directly

# Install PyTorch with CUDA 12.8 support (matching base image)
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --break-system-packages

# Install additional Python dependencies from requirements files
COPY requirements.txt requirements-dev.txt ./
RUN pip3 install -r requirements.txt -r requirements-dev.txt --break-system-packages

# Install additional development tools
RUN pip3 install \
    jupyter \
    ipython \
    jupyterlab \
    matplotlib \
    seaborn \
    pandas \
    --break-system-packages

# Set working directory
WORKDIR /workspace

# Copy project files
COPY . .

# Keep root user for development convenience
# (Comment out user creation to use root)
# RUN groupadd -r tinytorch && useradd -r -g tinytorch -m tinytorch
# RUN chown -R tinytorch:tinytorch /workspace
# USER tinytorch

# Expose Jupyter port
EXPOSE 8888

# Default command
CMD ["/bin/bash"]
