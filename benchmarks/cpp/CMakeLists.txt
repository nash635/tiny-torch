# benchmarks/cpp/CMakeLists.txt
# C++性能测试构建配置

cmake_minimum_required(VERSION 3.18)

# 暂时禁用C++基准测试，等待后续实现
# 基准测试源文件
# set(BENCHMARK_SOURCES
#     tensor_benchmark.cpp
#     ops_benchmark.cpp
# )

# # 创建基准测试可执行文件
# add_executable(tiny_torch_benchmarks ${BENCHMARK_SOURCES})

# # 链接库
# target_link_libraries(tiny_torch_benchmarks 
#     PRIVATE 
#     tiny_torch_cpp
# )

# # 设置编译选项
# target_compile_options(tiny_torch_benchmarks PRIVATE -O3)

message(STATUS "C++ benchmarks are disabled. Enable them later when core library is ready.")
