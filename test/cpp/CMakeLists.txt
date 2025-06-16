# test/cpp/CMakeLists.txt
# C++测试构建配置

cmake_minimum_required(VERSION 3.18)

# 查找线程支持
find_package(Threads REQUIRED)

# 简化的测试配置 - 暂时不使用 GoogleTest
# 创建一个简单的测试主程序

# 测试源文件
set(TEST_SOURCES
    test_tensor.cpp
    test_autograd.cpp
    simple_test_main.cpp
)

# 创建测试可执行文件
add_executable(tiny_torch_cpp_tests ${TEST_SOURCES})

# 设置C++17标准
target_compile_features(tiny_torch_cpp_tests PRIVATE cxx_std_17)

# 链接线程库
target_link_libraries(tiny_torch_cpp_tests 
    PRIVATE 
    Threads::Threads
)

# 如果tiny_torch_cpp库存在，则链接它
if(TARGET tiny_torch_cpp)
    target_link_libraries(tiny_torch_cpp_tests PRIVATE tiny_torch_cpp)
endif()

# 添加包含目录
target_include_directories(tiny_torch_cpp_tests PRIVATE 
    ${CMAKE_SOURCE_DIR}/csrc
    ${CMAKE_SOURCE_DIR}/csrc/aten/include
)
