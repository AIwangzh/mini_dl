#pragma once
#include <vector>
#include <memory>

// 前向声明 Tensor
class Tensor;

// 抽象基类：反向传播函数
struct GradFn {
    virtual ~GradFn() = default;

    // grad_out: 当前 Tensor 的梯度
    // 调用 backward 时，将梯度传递给依赖的 Tensor
    virtual void backward(const std::vector<float>& grad_out) = 0;

protected:
    // 派生类通过这个接口访问 Tensor 的 protected accumulate_grad
    void accumulate(Tensor* t, const std::vector<float>& g);
};

// 示例：加法的反向传播
struct AddGradFn : public GradFn {
    Tensor* a_;
    Tensor* b_;

    AddGradFn(Tensor* a, Tensor* b) : a_(a), b_(b) {}

    void backward(const std::vector<float>& grad_out) override;
};