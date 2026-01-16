#include "tensor.hpp"
#include "ops.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

// 辅助函数：比较两个 float 是否足够接近
bool near(float a, float b, float tol = 1e-5) {
    return std::abs(a - b) < tol;
}

void test_div_broadcasting() {
    std::cout << "[Test] Division with Broadcasting..." << std::endl;
    // a: [3], {10, 20, 30}
    Tensor a({3}, {10.0f, 20.0f, 30.0f}, true);
    // b: [1], {2} -> 会广播成 {2, 2, 2}
    Tensor b({1}, {2.0f}, true);

    auto c = a / b; // {5, 10, 15}
    
    // 验证前向
    assert(near(c[0], 5.0f));
    assert(near(c[2], 15.0f));

    c.backward();

    // 验证梯度
    // da = 1/b = 1/2 = 0.5
    assert(near(a.grad()[0], 0.5f));
    assert(near(a.grad()[2], 0.5f));

    // db = sum(-a / b^2) = -(10/4 + 20/4 + 30/4) = -(2.5 + 5 + 7.5) = -15
    assert(near(b.grad()[0], -15.0f));
    
    std::cout << "  -> Pass!" << std::endl;
}

void test_matmul_basic() {
    std::cout << "[Test] Matrix Multiplication..." << std::endl;
    // A: 2x3
    // [1, 2, 3]
    // [4, 5, 6]
    Tensor A({2, 3}, {1, 2, 3, 4, 5, 6}, true);

    // B: 3x2
    // [7,  8]
    // [9,  10]
    // [11, 12]
    Tensor B({3, 2}, {7, 8, 9, 10, 11, 12}, true);

    auto C = matmul(A, B); // 结果应为 2x2
    // C[0,0] = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
    // C[0,1] = 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
    assert(near(C[0], 58.0f));
    assert(near(C[1], 64.0f));

    C.backward();

    // 验证 A 的梯度: G_A = G_out * B^T
    // 假设 G_out 是 [1, 1, 1, 1] (backward 默认初始值)
    // G_A[0,0] = G_out[0,0]*B[0,0] + G_out[0,1]*B[0,1] = 1*7 + 1*8 = 15
    assert(near(A.grad()[0], 15.0f)); 
    // G_A[1,2] = G_out[1,0]*B[2,0] + G_out[1,1]*B[2,1] = 1*11 + 1*12 = 23
    assert(near(A.grad()[5], 23.0f));

    // 验证 B 的梯度: G_B = A^T * G_out
    // G_B[0,0] = A[0,0]*G_out[0,0] + A[1,0]*G_out[1,0] = 1*1 + 4*1 = 5
    assert(near(B.grad()[0], 5.0f));

    std::cout << "  -> Pass!" << std::endl;
}

int main() {
    try {
        test_div_broadcasting();
        test_matmul_basic();
        std::cout << "\nAll advanced tests passed!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}