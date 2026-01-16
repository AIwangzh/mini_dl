#include <iostream>
#include <vector>

#include "tensor.hpp"
#include "autograd.hpp"
#include "grad_fn.hpp"
#include "ops.hpp"

int main() {
    std::cout << "=== Day5 Tensor-level backward test (add / sub / neg) ===\n";

    // 非 scalar Tensor（长度 3）
    Tensor a({3}, true);
    Tensor b({3}, true);

    a[0] = 1.0f; a[1] = 2.0f; a[2] = 3.0f;
    b[0] = 4.0f; b[1] = 5.0f; b[2] = 6.0f;


// Tensor c = a + b;
// std::cout << "c shape size: " << c.grad().size() << std::endl;

// Tensor neg_c = -c;
// std::cout << "neg_c shape size: " << neg_c.grad().size() << std::endl;

// Tensor d = c - neg_c;
// std::cout << "d shape size: " << d.grad().size() << std::endl;

// std::cout << "Starting backward..." << std::endl;
// d.backward();

    // c = a + b
    Tensor c = a + b;

    //Tensor negc = -c;
    // d = c - (-c) = 2c 
    Tensor d = c - (-c);
    d.grad().assign(d.numel(), 1.0f);

    // 触发反向传播（tensor backward）
    d.backward();

    // ===== 输出梯度 =====
    std::cout << "a.grad(): ";
    for (float v : a.grad()) std::cout << v << " ";
    std::cout << "\n";

    std::cout << "b.grad(): ";
    for (float v : b.grad()) std::cout << v << " ";
    std::cout << "\n";

    std::cout << "c.grad(): ";
    for (float v : c.grad()) std::cout << v << " ";
    std::cout << "\n";

    // ===== 期望结果 =====
    std::cout << "\nExpected:\n";
    std::cout << "a.grad() = 2 2 2\n";
    std::cout << "b.grad() = 2 2 2\n";
    std::cout << "c.grad() = 2 2 2\n";

    // 测试广播乘法梯度
    Tensor e({2, 2}, {1, 2, 3, 4}, true); // [[1, 2], [3, 4]]
    Tensor f({2, 1}, {10, 20}, true);     // [[10], [20]] -> 广播为 [[10, 10], [20, 20]]
    auto g = e * f;                       // [[10, 20], [60, 80]]
    g.backward();

    std::cout << "e.grad(): ";
    for (float v : e.grad()) std::cout << v << " ";
    std::cout << "\n";

    std::cout << "f.grad(): ";
    for (float v : f.grad()) std::cout << v << " ";
    std::cout << "\n";

    // 期待结果：
    // e.grad 应为 [[10, 10], [20, 20]] (即 f 广播后的样子)
    // f.grad 应为 [[1+2], [3+4]] = [[3], [7]] (即 e 在行上求和后的结果)

    return 0;
}
