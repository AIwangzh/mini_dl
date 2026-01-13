#include <iostream>
#include "../include/tensor.hpp"
#include "../include/ops.hpp"

void print_tensor(const Tensor& t, const std::string& name) {
    std::cout << name << ": ";
    for (auto v : t.data()) std::cout << v << " ";
    std::cout << std::endl;
}

void test_tensor_tensor_ops() {
    std::cout << "=== Tensor x Tensor ===" << std::endl;
    Tensor a({2,2}, 4.0f);
    Tensor b({2,2}, 2.0f);

    print_tensor(add(a,b), "a + b");    // 6 6 6 6
    print_tensor(sub(a,b), "a - b");    // 2 2 2 2
    print_tensor(mul(a,b), "a * b");    // 8 8 8 8
    print_tensor(div(a,b), "a / b");    // 2 2 2 2
}

void test_tensor_scalar_ops() {
    std::cout << "=== Tensor x scalar ===" << std::endl;
    Tensor t({2,3});
    t.data() = {1,2,3,4,5,6};

    print_tensor(add(t, 1.0f), "t + 1");     // 2 3 4 5 6 7
    print_tensor(sub(t, 1.0f), "t - 1");     // 0 1 2 3 4 5
    print_tensor(mul(t, 2.0f), "t * 2");     // 2 4 6 8 10 12
    print_tensor(div(t, 2.0f), "t / 2");     // 0.5 1 1.5 2 2.5 3
}

void test_scalar_tensor_ops() {
    std::cout << "=== scalar x Tensor ===" << std::endl;
    Tensor t({2,3});
    t.data() = {1,2,3,4,5,6};

    print_tensor(add(1.0f, t), "1 + t");     // 2 3 4 5 6 7
    print_tensor(sub(10.0f, t), "10 - t");   // 9 8 7 6 5 4
    print_tensor(mul(2.0f, t), "2 * t");     // 2 4 6 8 10 12
    print_tensor(div(12.0f, t), "12 / t");   // 12 6 4 3 2.4 2
}

int main() {
    test_tensor_tensor_ops();
    test_tensor_scalar_ops();
    test_scalar_tensor_ops();

    return 0;
}
