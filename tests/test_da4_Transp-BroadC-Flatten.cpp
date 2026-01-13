#include <iostream>
#include <vector>
#include "tensor.hpp"
#include "ops.hpp"

using std::cout;
using std::endl;

// 简单打印 Tensor（线性）
void print_tensor(const Tensor& t) {
    cout << "shape = (";
    for (size_t i = 0; i < t.shape().size(); ++i) {
        cout << t.shape()[i];
        if (i + 1 < t.shape().size()) cout << ",";
    }
    cout << ")\nvalues = ";

    for (size_t i = 0; i < t.numel(); ++i) {
        cout << t[i] << " ";
    }
    cout << "\n\n";
}

/* ===================== 1. Broadcasting ===================== */

void test_broadcast_ops() {
    cout << "=== Test Broadcasting (add / sub / mul / div) ===\n";

    Tensor a({2,3,1});
    Tensor b({1,3,4});

    for (size_t i = 0; i < a.numel(); ++i) a[i] = 1.0f;
    for (size_t i = 0; i < b.numel(); ++i) b[i] = static_cast<float>(i + 1);

    Tensor c_add = add(a, b);
    Tensor c_sub = sub(a, b);
    Tensor c_mul = mul(a, b);
    Tensor c_div = div(a, b);

    cout << "[add]\n"; print_tensor(c_add);
    cout << "[sub]\n"; print_tensor(c_sub);
    cout << "[mul]\n"; print_tensor(c_mul);
    cout << "[div]\n"; print_tensor(c_div);
}

/* ===================== 2. Index Access ===================== */

void test_indexing() {
    cout << "=== Test Indexing ===\n";

    Tensor t({2,3});
    t.data() = {1,2,3,4,5,6};

    cout << "t[4] = " << t[4] << endl;
    cout << "t({1,2}) = " << t({1,2}) << endl;

    cout << "\n";
}

/* ===================== 3. High-Dim Transpose ===================== */

void test_transpose() {
    cout << "=== Test High-Dim Transpose ===\n";

    Tensor t({2,3,4});
    for (size_t i = 0; i < t.numel(); ++i)
        t[i] = static_cast<float>(i);

    Tensor t_perm = t.transpose({1,0,2});

    print_tensor(t_perm);
}

/* ===================== 4. Flatten ===================== */

void test_flatten() {
    cout << "=== Test Flatten ===\n";

    Tensor t({2,3,4});
    for (size_t i = 0; i < t.numel(); ++i)
        t[i] = static_cast<float>(i + 1);

    cout << "[original]\n";
    print_tensor(t);

    Tensor part = t.flatten(1,2);
    cout << "[flatten(1,2)]\n";
    print_tensor(part);

    Tensor f = t;
    f.flatten();
    cout << "[flatten all]\n";
    print_tensor(f);
}

/* ===================== main ===================== */

int main() {
    test_broadcast_ops();
    test_indexing();
    test_transpose();
    test_flatten();
    return 0;
}
