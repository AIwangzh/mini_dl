#include "autograd.hpp"
#include "ops.hpp"
#include "tensor.hpp" 
#include "grad_fn.hpp" 

// Add 实现
void AddGradFn::backward(const std::vector<float>& grad_out) {
    if (a_.requires_grad())  accumulate(&a_, grad_out);
    if (b_.requires_grad())  accumulate(&b_, grad_out);
}
std::vector<Tensor*> AddGradFn::parents() { return {&a_, &b_}; }

// Sub 实现
void SubGradFn::backward(const std::vector<float>& grad_out) {
    if (a_.requires_grad()) {
            accumulate(&a_, grad_out);
        }
        
        if (b_.requires_grad()) {
            // 对 grad_out 取反
            std::vector<float> neg_grad = grad_out;
            for (auto& v : neg_grad) v = -v;
            accumulate(&b_, neg_grad);
        }
}
std::vector<Tensor*> SubGradFn::parents() { return { const_cast<Tensor*>(&a_), const_cast<Tensor*>(&b_) }; }

// Neg 实现
void NegGradFn::backward(const std::vector<float>& grad_out) {
    if (a_.requires_grad()) {
            std::vector<float> neg = grad_out;
            for (auto& v : neg) v = -v;
            accumulate(&a_, neg);
        }
}
std::vector<Tensor*> NegGradFn::parents() { return { const_cast<Tensor*>(&a_) }; }

// Mul 实现
void MulGradFn::backward(const std::vector<float>& grad_out) {
    // 1. 获取前向传播时的广播形状
    auto out_shape = broadcast_shape(a_.shape(), b_.shape());
    
    // 2. 初始化输入张量的梯度容器（大小与输入一致，初始为0）
    std::vector<float> grad_a(a_.numel(), 0.0f);
    std::vector<float> grad_b(b_.numel(), 0.0f);

    // 3. 遍历输出梯度，将其分摊（累加）回输入梯度
    for (size_t i = 0; i < grad_out.size(); ++i) {
        // 使用你的工具函数找到当前输出点对应的输入点索引
        auto idx = unravel_index(i, out_shape);
        size_t ia = ravel_index_broadcast(idx, a_.shape());
        size_t ib = ravel_index_broadcast(idx, b_.shape());

        // 根据乘法法则：da = d_out * b, db = d_out * a
        if (a_.requires_grad()) {
            grad_a[ia] += grad_out[i] * b_[ib]; 
        }
        if (b_.requires_grad()) {
            grad_b[ib] += grad_out[i] * a_[ia];
        }
    }

    // 4. 调用你定义的辅助函数更新 TensorImpl 里的 grad 数组
    if (a_.requires_grad()) accumulate(&a_, grad_a);
    if (b_.requires_grad()) accumulate(&b_, grad_b);
}

std::vector<Tensor*> MulGradFn::parents() {
        return { const_cast<Tensor*>(&a_), const_cast<Tensor*>(&b_) };
    }


// Div 实现
void DivGradFn::backward(const std::vector<float>& grad_out) {
    auto out_shape = broadcast_shape(a_.shape(), b_.shape());
    std::vector<float> grad_a(a_.numel(), 0.0f);
    std::vector<float> grad_b(b_.numel(), 0.0f);

    for (size_t i = 0; i < grad_out.size(); ++i) {
        auto idx = unravel_index(i, out_shape);
        size_t ia = ravel_index_broadcast(idx, a_.shape());
        size_t ib = ravel_index_broadcast(idx, b_.shape());

        float a_val = a_.data()[ia];
        float b_val = b_.data()[ib];

        if (a_.requires_grad()) {
            grad_a[ia] += grad_out[i] * (1.0f / b_val);
        }
        if (b_.requires_grad()) {
            grad_b[ib] += grad_out[i] * (-a_val / (b_val * b_val));
        }
    }

    if (a_.requires_grad()) accumulate(&a_, grad_a);
    if (b_.requires_grad()) accumulate(&b_, grad_b);
}

std::vector<Tensor*> DivGradFn::parents() {
    return { const_cast<Tensor*>(&a_), const_cast<Tensor*>(&b_) };
}

// MatMul 实现
void MatMulGradFn::backward(const std::vector<float>& grad_out) {
    // 构造 grad_out 的 Tensor 视图 [m, n]
    size_t m = a_.shape()[0];
    size_t n = b_.shape()[1];
    Tensor g_out({m, n}, grad_out);

    if (a_.requires_grad()) {
        // dL/dA = G_out * B^T
        Tensor b_t = transpose(b_); 
        Tensor g_a = matmul(g_out, b_t);
        accumulate(&a_, g_a.data());
    }

    if (b_.requires_grad()) {
        // dL/dB = A^T * G_out
        Tensor a_t = transpose(a_);
        Tensor g_b = matmul(a_t, g_out);
        accumulate(&b_, g_b.data());
    }
}

std::vector<Tensor*> MatMulGradFn::parents() {
    return { const_cast<Tensor*>(&a_), const_cast<Tensor*>(&b_) };
}