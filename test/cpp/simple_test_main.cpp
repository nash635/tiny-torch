// simple_test_main.cpp
// 简单的测试主程序，不依赖 GoogleTest

#include <iostream>
#include <cassert>

// 前向声明测试函数
void test_tensor_basic();
void test_autograd_basic();

int main() {
    std::cout << "Running tiny-torch C++ tests..." << std::endl;
    
    try {
        std::cout << "Testing tensor operations..." << std::endl;
        test_tensor_basic();
        std::cout << "✓ Tensor tests passed" << std::endl;
        
        std::cout << "Testing autograd operations..." << std::endl;
        test_autograd_basic();
        std::cout << "✓ Autograd tests passed" << std::endl;
        
        std::cout << "All tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}
