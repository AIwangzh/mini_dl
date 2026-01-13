#include <iostream>
#include "tensor.hpp"
#include "autograd.hpp"
#include "ops.hpp"

int main() {

    /* =======================
     *  AddGradFn 测试（原样保留）
     * ======================= */
    std::cout << "=== AddGradFn test ===\n";

    Tensor a({1}, true);
    a[0] = 2.0f;

    Tensor b({1}, true);
    b[0] = 3.0f;

    Tensor c = a + b;   // c = 5
    Tensor d = c + c;   // d = 10

    d.backward();

    std::cout << "a.grad() = ";
    for (auto v : a.grad()) std::cout << v << " ";
    std::cout << "\n";

    std::cout << "b.grad() = ";
    for (auto v : b.grad()) std::cout << v << " ";
    std::cout << "\n";

    std::cout << "c.grad() = ";
    for (auto v : c.grad()) std::cout << v << " ";
    std::cout << "\n\n";


    /* =======================
     *  SubGradFn 测试
     * ======================= */
    std::cout << "=== SubGradFn test ===\n";

    Tensor x({1}, true);
    x[0] = 5.0f;

    Tensor y({1}, true);
    y[0] = 2.0f;

    Tensor z = x - y;   // z = 3
    Tensor w = z - z;   // w = 0

    w.backward();

    std::cout << "x.grad() = ";
    for (auto v : x.grad()) std::cout << v << " ";
    std::cout << "\n";

    std::cout << "y.grad() = ";
    for (auto v : y.grad()) std::cout << v << " ";
    std::cout << "\n";

    std::cout << "z.grad() = ";
    for (auto v : z.grad()) std::cout << v << " ";
    std::cout << "\n\n";


    /* =======================
     *  NegGradFn 测试
     * ======================= */
    std::cout << "=== NegGradFn test ===\n";

    Tensor p({1}, true);
    p[0] = 4.0f;

    Tensor q = -p;      // q = -4
    Tensor r = q + q;   // r = -8

    r.backward();

    std::cout << "p.grad() = ";
    for (auto v : p.grad()) std::cout << v << " ";
    std::cout << "\n";

    std::cout << "q.grad() = ";
    for (auto v : q.grad()) std::cout << v << " ";
    std::cout << "\n";

    return 0;
}
