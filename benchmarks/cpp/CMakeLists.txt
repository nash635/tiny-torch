/*
 * benchmarks/cpp/CMakeLists.txt
 * C++性能测试构建配置
 */

cmake_minimum_required(VERSION 3.18)

# 添加Google Benchmark
include(FetchContent)
FetchContent_Declare(
  googlebenchmark
  URL https://github.com/google/benchmark/archive/v1.8.3.zip
)
FetchContent_MakeAvailable(googlebenchmark)

# 基准测试源文件
set(BENCHMARK_SOURCES
    tensor_benchmark.cpp
    ops_benchmark.cpp
)

# 创建基准测试可执行文件
add_executable(tiny_torch_benchmarks ${BENCHMARK_SOURCES})

# 链接库
target_link_libraries(tiny_torch_benchmarks 
    PRIVATE 
    tiny_torch_cpp
    benchmark::benchmark
    benchmark::benchmark_main
)

# 设置编译选项
target_compile_options(tiny_torch_benchmarks PRIVATE -O3)
