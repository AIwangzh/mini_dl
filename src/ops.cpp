#include "ops.hpp"
#include "tensor_utils.hpp"
#include "autograd.hpp"
#include "grad_fn.hpp"
#include <vector>
#include <stdexcept>
#include <cassert>
#include <cmath>

// ---------------- Tensor × Tensor (广播机制) ----------------

Tensor add(const Tensor& a, const Tensor& b) {
    auto out_shape = broadcast_shape(a.shape(), b.shape());
    Tensor out(out_shape);

    for (size_t i = 0; i < out.numel(); ++i) {
        auto idx = unravel_index(i, out_shape);
        size_t ia = ravel_index_broadcast(idx, a.shape());
        size_t ib = ravel_index_broadcast(idx, b.shape());
        out[i] = a[ia] + b[ib];
    }

    // ===== Autograd 绑定 =====
    // 此时传入的 a, b 是 Tensor 句柄，内部 shared_ptr 会自动增加引用计数
    if (a.requires_grad() || b.requires_grad()) {
        out.set_requires_grad(true);
        out.set_grad_fn(new AddGradFn(a, b));
    }

    return out;
}

Tensor sub(const Tensor& a, const Tensor& b) {
    auto out_shape = broadcast_shape(a.shape(), b.shape());
    Tensor out(out_shape);

    for (size_t i = 0; i < out.numel(); ++i) {
        auto idx = unravel_index(i, out_shape);
        size_t ia = ravel_index_broadcast(idx, a.shape());
        size_t ib = ravel_index_broadcast(idx, b.shape());
        out[i] = a[ia] - b[ib];
    }

    // ===== Autograd 绑定 =====
    if (a.requires_grad() || b.requires_grad()) {
        out.set_requires_grad(true);
        out.set_grad_fn(new SubGradFn(a, b));
    }

    return out;
}

Tensor mul(const Tensor& a, const Tensor& b) {
    // 1. 确定输出形状（处理广播）
    auto out_shape = broadcast_shape(a.shape(), b.shape());
    Tensor out(out_shape);

    // 2. 前向计算：逐元素相乘，并处理广播索引映射
    for (size_t i = 0; i < out.numel(); ++i) {
        auto idx = unravel_index(i, out_shape);
        size_t ia = ravel_index_broadcast(idx, a.shape());
        size_t ib = ravel_index_broadcast(idx, b.shape());
        
        // 实际计算
        out[i] = a[ia] * b[ib];
    }

    // 3. Autograd 绑定：
    // 只要其中一个输入需要梯度，结果就需要梯度，并挂载 MulGradFn
    if (a.requires_grad() || b.requires_grad()) {
        out.set_requires_grad(true);
        // 这里传入 a 和 b 的句柄，MulGradFn 会自动通过 shared_ptr 延长它们的生命周期
        out.set_grad_fn(new MulGradFn(a, b)); 
    }

    return out;
}

Tensor div(const Tensor& a, const Tensor& b) {
    auto out_shape = broadcast_shape(a.shape(), b.shape());
    Tensor out(out_shape);

    for (size_t i = 0; i < out.numel(); ++i) {
        auto idx = unravel_index(i, out_shape);
        size_t ia = ravel_index_broadcast(idx, a.shape());
        size_t ib = ravel_index_broadcast(idx, b.shape());
        if (b[ib] == 0) throw std::runtime_error("Division by zero");
        out[i] = a[ia] / b[ib];
    }

    // 3. 绑定 Autograd 逻辑
    if (a.requires_grad() || b.requires_grad()) {
        out.set_requires_grad(true);
        out.set_grad_fn(new DivGradFn(a, b)); 
    }

    return out;
}

Tensor neg(const Tensor& a) {
    Tensor out(a.shape());

    for (size_t i = 0; i < a.numel(); ++i) {
        out[i] = -a[i];
    }

    // ===== Autograd 绑定 =====
    if (a.requires_grad()) {
        out.set_requires_grad(true);
        out.set_grad_fn(new NegGradFn(a));
    }

    return out;
}

// ---------------- Tensor × Scalar 混合运算 ----------------

Tensor add(const Tensor& t, float scalar) {
    Tensor out(t.shape());
    for (size_t i = 0; i < t.numel(); ++i) out[i] = t[i] + scalar;
    if (t.requires_grad()) out.set_requires_grad(true);
    return out;
}

Tensor add(float scalar, const Tensor& t) { return add(t, scalar); }

Tensor sub(const Tensor& t, float scalar) {
    Tensor out(t.shape());
    for (size_t i = 0; i < t.numel(); ++i) out[i] = t[i] - scalar;
    if (t.requires_grad()) out.set_requires_grad(true);
    return out;
}

Tensor sub(float scalar, const Tensor& t) {
    Tensor out(t.shape());
    for (size_t i = 0; i < t.numel(); ++i) out[i] = scalar - t[i];
    if (t.requires_grad()) out.set_requires_grad(true);
    return out;
}

Tensor mul(const Tensor& t, float scalar) {
    Tensor out(t.shape());
    for (size_t i = 0; i < t.numel(); ++i) out[i] = t[i] * scalar;
    if (t.requires_grad()) out.set_requires_grad(true);
    return out;
}

Tensor mul(float scalar, const Tensor& t) { return mul(t, scalar); }

Tensor div(const Tensor& t, float scalar) {
    if (scalar == 0) throw std::runtime_error("Division by zero");
    Tensor out(t.shape());
    for (size_t i = 0; i < t.numel(); ++i) out[i] = t[i] / scalar;
    if (t.requires_grad()) out.set_requires_grad(true);
    return out;
}

Tensor div(float scalar, const Tensor& t) {
    Tensor out(t.shape());
    for (size_t i = 0; i < t.numel(); ++i) {
        if (t[i] == 0) throw std::runtime_error("Division by zero");
        out[i] = scalar / t[i];
    }
    if (t.requires_grad()) out.set_requires_grad(true);
    return out;
}

// ---------------- 矩阵与转置 ----------------

Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.shape().size() != 2 || b.shape().size() != 2) {
        throw std::runtime_error("matmul only supports 2D tensors");
    }

    size_t m = a.shape()[0];
    size_t k = a.shape()[1];
    size_t k2 = b.shape()[0];
    size_t n = b.shape()[1];

    if (k != k2) throw std::runtime_error("matmul shape mismatch");

    Tensor out({m, n});
    const auto& A = a.data();
    const auto& B = b.data();
    auto& C = out.data();

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (size_t p = 0; p < k; ++p) {
                sum += A[i * k + p] * B[p * n + j];
            }
            C[i * n + j] = sum;
        }
    }
    
    // 如果需要矩阵求导，在此绑定 MatMulGradFn
    if (a.requires_grad() || b.requires_grad()) {
        out.set_requires_grad(true);
        out.set_grad_fn(new MatMulGradFn(a, b));
    }

    return out;
}

Tensor transpose(const Tensor& t) {
    if (t.shape().size() != 2) {
        throw std::runtime_error("transpose only supports 2D tensors");
    }

    size_t m = t.shape()[0];
    size_t n = t.shape()[1];
    Tensor out({n, m});

    const auto& src = t.data();
    auto& dst = out.data();

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            dst[j * m + i] = src[i * n + j];
        }
    }
    
    if (t.requires_grad()) out.set_requires_grad(true);

    return out;
}

// #include "ops.hpp"
// #include "tensor_utils.hpp"
// #include "autograd.hpp"
// #include "grad_fns.hpp"
// #include <vector>
// #include <stdexcept>
// #include <cassert>



// // ---------------- Tensor × Tensor （广播机制） ----------------
// Tensor add(const Tensor& a, const Tensor& b) {
//     auto out_shape = broadcast_shape(a.shape(), b.shape());
//     Tensor out(out_shape);

//     for(size_t i = 0; i < out.numel(); ++i){
//         auto idx = unravel_index(i, out_shape);
//         size_t ia = ravel_index_broadcast(idx, a.shape());
//         size_t ib = ravel_index_broadcast(idx, b.shape());
//         out[i] = a[ia] + b[ib];
//     }

//     // ===== Autograd 绑定（Day 5 新增）=====
//     if (a.requires_grad() || b.requires_grad()) {
//         out.set_requires_grad(true);
//         out.set_grad_fn(
//             new AddGradFn(a, b)
//         );
//     }

//     return out;
// }

// Tensor sub(const Tensor& a, const Tensor& b) {
//     auto out_shape = broadcast_shape(a.shape(), b.shape());
//     Tensor out(out_shape);

//     for (size_t i = 0; i < out.numel(); ++i) {
//         auto idx = unravel_index(i, out_shape);
//         size_t ia = ravel_index_broadcast(idx, a.shape());
//         size_t ib = ravel_index_broadcast(idx, b.shape());
//         out[i] = a[ia] - b[ib];
//     }

//     // ===== Autograd 绑定 =====
//     if (a.requires_grad() || b.requires_grad()) {
//         out.set_requires_grad(true);
//         out.set_grad_fn(
//             new SubGradFn(a, b)
//         );
//     }

//     return out;
// }

// Tensor mul(const Tensor& a, const Tensor& b) {
//     auto out_shape = broadcast_shape(a.shape(), b.shape());
//     Tensor out(out_shape);

//     for(size_t i = 0; i < out.numel(); ++i){
//         auto idx = unravel_index(i, out_shape);
//         size_t ia = ravel_index_broadcast(idx, a.shape());
//         size_t ib = ravel_index_broadcast(idx, b.shape());
//         out[i] = a[ia] * b[ib];
//     }
//     return out;
// }

// Tensor div(const Tensor& a, const Tensor& b) {
//     auto out_shape = broadcast_shape(a.shape(), b.shape());
//     Tensor out(out_shape);

//     for(size_t i = 0; i < out.numel(); ++i){
//         auto idx = unravel_index(i, out_shape);
//         size_t ia = ravel_index_broadcast(idx, a.shape());
//         size_t ib = ravel_index_broadcast(idx, b.shape());
//         out[i] = a[ia] / b[ib];
//     }
//     return out;
// }

// Tensor neg(const Tensor& a) {
//     Tensor out(a.shape());

//     for (size_t i = 0; i < a.numel(); ++i) {
//         out[i] = -a[i];
//     }

//     if (a.requires_grad()) {
//         out.set_requires_grad(true);
//         out.set_grad_fn(
//             new NegGradFn(a)
//         );
//     }

//     return out;
// }

// // ---------------- Tensor × scalar / scalar × Tensor ----------------
// Tensor add(const Tensor& t, float scalar) {
//     Tensor out(t.shape());
//     for(size_t i = 0; i < t.numel(); ++i) out[i] = t[i] + scalar;
//     return out;
// }

// Tensor add(float scalar, const Tensor& t) { return add(t, scalar); }

// Tensor sub(const Tensor& t, float scalar) {
//     Tensor out(t.shape());
//     for(size_t i = 0; i < t.numel(); ++i) out[i] = t[i] - scalar;
//     return out;
// }

// Tensor sub(float scalar, const Tensor& t) {
//     Tensor out(t.shape());
//     for(size_t i = 0; i < t.numel(); ++i) out[i] = scalar - t[i];
//     return out;
// }

// Tensor mul(const Tensor& t, float scalar) {
//     Tensor out(t.shape());
//     for(size_t i = 0; i < t.numel(); ++i) out[i] = t[i] * scalar;
//     return out;
// }

// Tensor mul(float scalar, const Tensor& t) { return mul(t, scalar); }

// Tensor div(const Tensor& t, float scalar) {
//     Tensor out(t.shape());
//     for(size_t i = 0; i < t.numel(); ++i) out[i] = t[i] / scalar;
//     return out;
// }

// Tensor div(float scalar, const Tensor& t) {
//     Tensor out(t.shape());
//     for(size_t i = 0; i < t.numel(); ++i) out[i] = scalar / t[i];
//     return out;
// }

// // ---------------- Tensor × Tensor ----------------
// Tensor add(const Tensor& a, const Tensor& b) {
//     return a + b; // 调用 Tensor::operator+
// }

// Tensor sub(const Tensor& a, const Tensor& b) {
//     return a - b; // 调用 Tensor::operator-
// }

// Tensor mul(const Tensor& a, const Tensor& b) {
//     return a * b; // 调用 Tensor::operator*（逐元素乘）
// }

// Tensor div(const Tensor& a, const Tensor& b) {
//     return a / b;
// }

// // ---------------- Tensor × scalar ----------------
// Tensor add(const Tensor& t, float scalar) {
//     Tensor result = t;
//     for (auto& v : result.data()) v += scalar;
//     return result;
// }

// Tensor add(float scalar, const Tensor& t) {
//     return add(t, scalar);
// }

// Tensor sub(const Tensor& t, float scalar) {
//     Tensor result = t;
//     for (auto& v : result.data()) v -= scalar;
//     return result;
// }

// Tensor sub(float scalar, const Tensor& t) {
//     Tensor result = t;
//     auto& dr = result.data();
//     const auto& d = t.data();
//     for (size_t i = 0; i < dr.size(); ++i) dr[i] = scalar - d[i];
//     return result;
// }

// Tensor mul(const Tensor& t, float scalar) {
//     Tensor result = t;
//     for (auto& v : result.data()) v *= scalar;
//     return result;
// }

// Tensor mul(float scalar, const Tensor& t) {
//     return mul(t, scalar);
// }

// Tensor div(const Tensor& t, float scalar) {
//     assert(scalar != 0 && "Division by zero");
//     Tensor result = t;
//     for (auto& v : result.data()) v /= scalar;
//     return result;
// }

// Tensor div(float scalar, const Tensor& t) {
//     Tensor result(t.shape());
//     const auto& d = t.data();
//     auto& dr = result.data();
//     for (size_t i = 0; i < d.size(); ++i) {
//         assert(d[i] != 0 && "Division by zero");
//         dr[i] = scalar / d[i];
//     }
//     return result;
// }

// Tensor matmul(const Tensor& a, const Tensor& b) {
//     // 只支持 2D
//     if (a.shape().size() != 2 || b.shape().size() != 2) {
//         throw std::runtime_error("matmul only supports 2D tensors");
//     }

//     size_t m = a.shape()[0];
//     size_t k = a.shape()[1];
//     size_t k2 = b.shape()[0];
//     size_t n = b.shape()[1];

//     if (k != k2) {
//         throw std::runtime_error("matmul shape mismatch");
//     }

//     Tensor out({m, n});

//     const auto& A = a.data();
//     const auto& B = b.data();
//     auto& C = out.data();

//     // row-major
//     for (size_t i = 0; i < m; ++i) {
//         for (size_t j = 0; j < n; ++j) {
//             float sum = 0.0f;
//             for (size_t p = 0; p < k; ++p) {
//                 sum += A[i * k + p] * B[p * n + j];
//             }
//             C[i * n + j] = sum;
//         }
//     }

//     return out;
// }

// Tensor transpose(const Tensor& t) {
//     if (t.shape().size() != 2) {
//         throw std::runtime_error("transpose only supports 2D tensors");
//     }

//     size_t m = t.shape()[0];
//     size_t n = t.shape()[1];

//     Tensor out({n, m});

//     const auto& src = t.data();
//     auto& dst = out.data();

//     for (size_t i = 0; i < m; ++i) {
//         for (size_t j = 0; j < n; ++j) {
//             dst[j * m + i] = src[i * n + j];
//         }
//     }

//     return out;
// }




// // 广播add
// Tensor add(const Tensor& a, const Tensor& b){
//     auto out_shape = broadcast_shape(a.shape(), b.shape());
//     Tensor out(out_shape);

//     size_t ndim = out_shape.size();

//     for(size_t i=0; i < out.numel(); ++i){
//         auto idx = unravel_index(i, out_shape);

//         size_t ia = ravel_index_broadcast(idx, a.shape());
//         size_t ib = ravel_index_broadcast(idx, b.shape());

//         out[i] = a[ia] + b[ib];
//     }
//     return out;
// }

