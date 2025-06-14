/*
 * test/cpp/CMakeLists.txt
 * C++测试构建配置
 */

cmake_minimum_required(VERSION 3.18)

# 添加Google Test
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50f33f9142f310f5e3f08b5afced9f.zip
)
FetchContent_MakeAvailable(googletest)

# 测试源文件
set(TEST_SOURCES
    test_tensor.cpp
    test_autograd.cpp
)

# 创建测试可执行文件
add_executable(tiny_torch_cpp_tests ${TEST_SOURCES})

# 链接库
target_link_libraries(tiny_torch_cpp_tests 
    PRIVATE 
    tiny_torch_cpp
    gtest_main
    gmock_main
)

# 添加测试
include(GoogleTest)
gtest_discover_tests(tiny_torch_cpp_tests)
