#include <iostream>
#include "../include/tensor.hpp"
#include "../include/ops.hpp"

void print_tensor(const Tensor& t) {
    const auto& data = t.data();
    for (size_t i = 0; i < t.numel(); ++i) {
        std::cout << data[i] << " ";
        if ((i + 1) % t.shape().back() == 0) std::cout << "\n";
    }
    std::cout << std::endl;
}

void test_matmul() {
    std::cout << "=== test matmul ===\n";
    Tensor a({2,3});
    a.data() = {1,2,3,4,5,6};
    Tensor b({3,2});
    b.data() = {7,8,9,10,11,12};

    Tensor c = matmul(a, b);
    print_tensor(c);

    // 正确结果手算：{{58,64},{139,154}}
}

void test_transpose() {
    std::cout << "=== test transpose ===\n";
    Tensor t({2,3});
    t.data() = {1,2,3,4,5,6};

    Tensor t_T = transpose(t);
    print_tensor(t_T);

    // 正确结果手算：{{1,4},{2,5},{3,6}}
}

void test_flatten() {
    std::cout << "=== test flatten ===\n";
    Tensor t({2,3});
    t.data() = {1,2,3,4,5,6};

    Tensor f = t;
    f.flatten();
    print_tensor(f);

    // 正确结果手算：{1,2,3,4,5,6}
}

int main() {
    test_matmul();
    test_transpose();
    test_flatten();
    return 0;
}
