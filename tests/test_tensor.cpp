#include <iostream>
#include <vector>
#include "../include/tensor.hpp"
#include "../include/ops.hpp"

int main() {
    // 1. 构造 Tensor
    Tensor t({2, 3, 4});  // 2x3x4 Tensor
    std::cout << "Shape: ";
    for (auto s : t.shape()) std::cout << s << " ";
    std::cout << "\nNumel: " << t.numel() << std::endl;

    // 2. 访问和修改元素
    t({1, 2, 3}) = 42.0f;      // 修改最后一个元素
    std::cout << "t(1,2,3) = " << t({1, 2, 3}) << std::endl;

    t({0, 0, 0}) = 1.0f;
    std::cout << "t(0,0,0) = " << t({0, 0, 0}) << std::endl;

    // 3. reshape
    t.reshape({4, 3, 2});      // 4x3x2
    std::cout << "After reshape, shape: ";
    for (auto s : t.shape()) std::cout << s << " ";
    std::cout << "\nNumel: " << t.numel() << std::endl;

    // 4. 读取一些元素，确保 reshape 后顺序没变
    std::cout << "t(3,2,1) = " << t({3,2,1}) << std::endl;  // 原来的 {1,2,3} 对应位置
    std::cout << "t(0,0,0) = " << t({0,0,0}) << std::endl;

    // 5. 测试逐元素除法
    Tensor a({3, 3}, 8.0f);
    Tensor b({3, 3}, 2.0f);
    Tensor c = a / b;
    std::cout << "Element-wise division result:" << std::endl;
    for(size_t i = 0; i < c.numel(); ++i){
        std::cout << c.data()[i] << " ";
    }
    std::cout << std::endl;


    return 0;
}