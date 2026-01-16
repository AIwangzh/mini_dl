// include/grad_fns.hpp
#pragma once
#include <vector>
#include <memory>
#include "autograd.hpp"
#include "tensor.hpp" // 这里必须包含完整的 Tensor 定义

// --- Add ---
struct AddGradFn : public GradFn {
    Tensor a_, b_;
    AddGradFn(Tensor a, Tensor b) : a_(a), b_(b) {}
    void backward(const std::vector<float>& grad_out) override; // 只留声明，去掉花括号实现
    std::vector<Tensor*> parents() override;
};

// --- Sub ---
struct SubGradFn : public GradFn {
    Tensor a_, b_;
    SubGradFn(Tensor a, Tensor b) : a_(a), b_(b) {}
    void backward(const std::vector<float>& grad_out) override;
    std::vector<Tensor*> parents() override;
};

// --- Neg ---
struct NegGradFn : public GradFn {
    Tensor a_;
    explicit NegGradFn(Tensor a) : a_(a) {}
    void backward(const std::vector<float>& grad_out) override;
    std::vector<Tensor*> parents() override;
};

// --- Mul ---
struct MulGradFn : public GradFn {
    Tensor a_, b_;
    MulGradFn(Tensor a, Tensor b) : a_(a), b_(b) {}

    void backward(const std::vector<float>& grad_out) override;
    std::vector<Tensor*> parents() override;
};

// --- Div ---
struct DivGradFn : public GradFn {
    Tensor a_, b_;
    DivGradFn(Tensor a, Tensor b) : a_(a), b_(b) {}
    void backward(const std::vector<float>& grad_out) override;
    std::vector<Tensor*> parents() override; // 仅声明
};

// --- MatMul ---
struct MatMulGradFn : public GradFn {
    Tensor a_, b_;
    MatMulGradFn(Tensor a, Tensor b) : a_(a), b_(b) {}
    void backward(const std::vector<float>& grad_out) override;
    std::vector<Tensor*> parents() override; // 仅声明
};
