#include <iostream>
#include "tensor.hpp"
#include "autograd.hpp"
#include "ops.hpp"

int main() {
    std::cout << "=== Minimal AddGradFn backward test ===\n";

    // 创建两个 Tensor，并开启 autograd
    Tensor a({1}, true);
    a[0] = 2.0f;

    Tensor b({1}, true);
    b[0] = 3.0f;

    // 第一次加法，生成 c
    Tensor c = a + b; // AddGradFn 会绑定 a 和 b
    // 第二次加法，生成 d
    Tensor d = c + c; // AddGradFn 会绑定 c 和 c

    Tensor e = d + d + d;

    // 触发反向传播
    e.backward();

    // 输出结果
    std::cout << "a.grad() = ";
    for (auto v : a.grad()) std::cout << v << " ";
    std::cout << "\n";

    std::cout << "b.grad() = ";
    for (auto v : b.grad()) std::cout << v << " ";
    std::cout << "\n";

    std::cout << "c.grad() = ";
    for (auto v : c.grad()) std::cout << v << " ";
    std::cout << "\n";

    for (auto v : d.grad()) std::cout << v << " ";

    return 0;
}
