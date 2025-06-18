cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# Fix conda environment library conflicts - set global policies
if(DEFINED ENV{CONDA_PREFIX})
    message(STATUS "Conda environment detected: $ENV{CONDA_PREFIX}")
    # Set policies to handle conda library conflicts
    cmake_policy(SET CMP0060 NEW)  # Link libraries by full path
    if(POLICY CMP0135)
        cmake_policy(SET CMP0135 NEW)  # Use download timestamp
    endif()
    
    # Disable automatic RPATH generation for conda environments
    set(CMAKE_SKIP_BUILD_RPATH TRUE)
    set(CMAKE_SKIP_INSTALL_RPATH TRUE)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
    
    message(STATUS "Applied conda-specific CMake policies to avoid library conflicts")
endif()

# Improve build output formatting
set(CMAKE_COLOR_MAKEFILE ON)
set(CMAKE_VERBOSE_MAKEFILE OFF)

# Force colored output for better readability
# Enable for all generators, not just Ninja
set(CMAKE_COLOR_DIAGNOSTICS ON)
if(CMAKE_GENERATOR STREQUAL "Ninja")
    # Additional Ninja-specific optimizations can go here
endif()

# Project definition
project(tiny-torch 
    VERSION 0.1.0
    DESCRIPTION "A PyTorch-inspired deep learning framework"
    LANGUAGES CXX C
)

# 仅支持 Linux 平台 - 在 project() 之后检查
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "This project only supports Linux. Current platform: ${CMAKE_SYSTEM_NAME}")
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build options
option(WITH_CUDA "Enable CUDA support" ON)
option(WITH_MKL "Enable Intel MKL support" OFF)
option(WITH_OPENMP "Enable OpenMP support" ON)
option(BUILD_TESTS "Build tests" ON)
option(BUILD_BENCHMARKS "Build benchmarks" OFF)

# 启用彩色输出 - 对所有编译器和生成器
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # GCC彩色诊断
    add_compile_options(-fdiagnostics-color=always)
    message(STATUS "Enabled GCC color diagnostics for all generators")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # Clang彩色诊断
    add_compile_options(-fcolor-diagnostics)
    message(STATUS "Enabled Clang color diagnostics for all generators")
endif()

# 为CUDA编译器设置彩色输出
if(WITH_CUDA AND CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fdiagnostics-color=always")
    message(STATUS "Enabled CUDA compiler color diagnostics")
endif()

# Ninja generator specific optimizations
if(CMAKE_GENERATOR STREQUAL "Ninja")
    message(STATUS "Using Ninja generator - enabling optimizations")
    
    # 优化并行编译
    if(NOT DEFINED CMAKE_BUILD_PARALLEL_LEVEL)
        include(ProcessorCount)
        ProcessorCount(N)
        if(NOT N EQUAL 0)
            set(CMAKE_BUILD_PARALLEL_LEVEL ${N} CACHE STRING "Number of parallel build jobs")
            message(STATUS "Setting parallel build jobs to ${N}")
        endif()
    endif()
endif()

# Set default build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
        -Wall -Wextra -Wpedantic
        -Wno-unused-parameter
        -Wno-missing-field-initializers
        -fPIC
    )
    
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-g -O0)
    else()
        add_compile_options(-O3 -DNDEBUG)
    endif()
elseif(MSVC)
    add_compile_options(/W4 /wd4267 /wd4244 /wd4996)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

# Find packages
find_package(Threads REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Find pybind11
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE _tmp_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ECHO STDOUT
)
list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")
find_package(pybind11 CONFIG REQUIRED)

# CUDA support
if(WITH_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    
    if(NOT DEFINED CMAKE_CUDA_STANDARD)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    endif()
    
    # CUDA flags
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIC")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
    
    # Find cuBLAS and cuDNN
    find_library(CUBLAS_LIBRARIES cublas HINTS ${CUDAToolkit_LIBRARY_DIR})
    find_library(CUDNN_LIBRARIES cudnn HINTS ${CUDAToolkit_LIBRARY_DIR})
    
    add_definitions(-DWITH_CUDA)
endif()

# OpenMP support
if(WITH_OPENMP)
    find_package(OpenMP REQUIRED)
    add_definitions(-DWITH_OPENMP)
endif()

# Intel MKL support
if(WITH_MKL)
    find_package(MKL QUIET)
    if(MKL_FOUND)
        add_definitions(-DWITH_MKL)
    else()
        message(WARNING "MKL not found, falling back to standard BLAS")
        find_package(BLAS REQUIRED)
        find_package(LAPACK REQUIRED)
    endif()
else()
    # Try to find standard BLAS/LAPACK
    find_package(BLAS QUIET)
    find_package(LAPACK QUIET)
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/csrc)
include_directories(${CMAKE_SOURCE_DIR}/csrc/api/include)
include_directories(${CMAKE_SOURCE_DIR}/csrc/aten/include)

# Source files
set(ATEN_SOURCES
    # Core tensor implementation
    csrc/aten/src/ATen/core/Tensor.cpp
    csrc/aten/src/ATen/core/TensorImpl.cpp
    csrc/aten/src/ATen/core/Storage.cpp
    csrc/aten/src/ATen/core/Allocator.cpp
    
    # Native CPU operations
    csrc/aten/src/ATen/native/BinaryOps.cpp
    csrc/aten/src/ATen/native/UnaryOps.cpp
    csrc/aten/src/ATen/native/LinearAlgebra.cpp
    csrc/aten/src/ATen/native/Activation.cpp
    csrc/aten/src/ATen/native/Reduction.cpp
    
    # TH layer
    csrc/aten/src/TH/THTensor.cpp
    csrc/aten/src/TH/THStorage.cpp
    csrc/aten/src/TH/THAllocator.cpp
)

set(AUTOGRAD_SOURCES
    csrc/autograd/engine.cpp
    csrc/autograd/function.cpp
    csrc/autograd/variable.cpp
    csrc/autograd/functions/basic_ops.cpp
)

set(API_SOURCES
    csrc/api/src/python_bindings.cpp
    csrc/api/src/tensor_api.cpp
    csrc/api/src/autograd_api.cpp
)

# CUDA sources
if(WITH_CUDA)
    set(ATEN_CUDA_SOURCES
        csrc/aten/src/ATen/cuda/CUDAContext.cu
        csrc/aten/src/ATen/native/cuda/BinaryOps.cu
        csrc/aten/src/ATen/native/cuda/UnaryOps.cu
        csrc/aten/src/ATen/native/cuda/LinearAlgebra.cu
        csrc/aten/src/ATen/native/cuda/Activation.cu
        csrc/aten/src/ATen/native/cuda/Reduction.cu
    )
    list(APPEND ATEN_SOURCES ${ATEN_CUDA_SOURCES})
    
    # Set CUDA-specific compiler options to reduce warnings
    set_property(SOURCE ${ATEN_CUDA_SOURCES} PROPERTY COMPILE_OPTIONS
        $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-declarations>
        $<$<COMPILE_LANGUAGE:CUDA>:-Wno-unknown-pragmas>
        $<$<COMPILE_LANGUAGE:CUDA>:--disable-warnings>
    )
endif()

# Create the main library
add_library(tiny_torch_cpp STATIC ${ATEN_SOURCES} ${AUTOGRAD_SOURCES})

# Link libraries
target_link_libraries(tiny_torch_cpp PUBLIC ${Python3_LIBRARIES})

if(WITH_CUDA)
    target_link_libraries(tiny_torch_cpp PUBLIC 
        CUDA::cudart 
        CUDA::cublas 
        CUDA::curand
    )
    if(CUDNN_LIBRARIES)
        target_link_libraries(tiny_torch_cpp PUBLIC ${CUDNN_LIBRARIES})
    endif()
endif()

if(WITH_OPENMP)
    target_link_libraries(tiny_torch_cpp PUBLIC OpenMP::OpenMP_CXX)
endif()

if(MKL_FOUND)
    target_link_libraries(tiny_torch_cpp PUBLIC ${MKL_LIBRARIES})
elseif(BLAS_FOUND AND LAPACK_FOUND)
    target_link_libraries(tiny_torch_cpp PUBLIC ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
endif()

# Python extension module
pybind11_add_module(_C ${API_SOURCES})
target_link_libraries(_C PRIVATE tiny_torch_cpp)

# Fix runtime library path conflicts - comprehensive approach
set_target_properties(_C PROPERTIES
    CXX_VISIBILITY_PRESET "hidden"
    CUDA_VISIBILITY_PRESET "hidden"
    # Disable automatic RPATH generation to avoid conflicts
    SKIP_BUILD_RPATH TRUE
    SKIP_INSTALL_RPATH TRUE
    # Use explicit linking without RPATH dependencies
    LINK_WHAT_YOU_USE TRUE
)

# Handle OpenMP linking explicitly to avoid conda conflicts
if(WITH_OPENMP AND OpenMP_CXX_FOUND)
    # Link OpenMP directly without RPATH dependencies
    target_link_libraries(_C PRIVATE OpenMP::OpenMP_CXX)
    # Set explicit compiler flags instead of relying on RPATH
    target_compile_options(_C PRIVATE ${OpenMP_CXX_FLAGS})
    target_link_options(_C PRIVATE ${OpenMP_CXX_FLAGS})
endif()

# Compiler-specific properties
target_compile_definitions(_C PRIVATE VERSION_INFO=${PROJECT_VERSION})

# Install targets
install(TARGETS _C DESTINATION torch)

# Tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(test/cpp)
endif()

# Benchmarks
if(BUILD_BENCHMARKS)
    add_subdirectory(benchmarks/cpp)
endif()

# Print configuration summary
message(STATUS "=== Tiny-Torch Configuration Summary ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Python executable: ${Python3_EXECUTABLE}")
message(STATUS "Python version: ${Python3_VERSION}")
message(STATUS "CUDA support: ${WITH_CUDA}")
if(WITH_CUDA)
    message(STATUS "CUDA version: ${CUDAToolkit_VERSION}")
    message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()
message(STATUS "OpenMP support: ${WITH_OPENMP}")
message(STATUS "MKL support: ${WITH_MKL}")
message(STATUS "Build tests: ${BUILD_TESTS}")
message(STATUS "Build benchmarks: ${BUILD_BENCHMARKS}")
message(STATUS "=========================================")
