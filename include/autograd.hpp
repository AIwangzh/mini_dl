#pragma once
#include <vector>
#include <memory>

class Tensor;

struct GradFn {
    virtual ~GradFn() = default;
    virtual void backward(const std::vector<float>& grad_out) = 0;
    virtual std::vector<Tensor*> parents() const = 0;

protected:
    void accumulate(Tensor* t, const std::vector<float>& g);
};

// --- Add ---
struct AddGradFn : public GradFn {
    Tensor *a_, *b_;
    AddGradFn(Tensor* a, Tensor* b) : a_(a), b_(b) {}
    void backward(const std::vector<float>& grad_out) override; // 只留声明，去掉花括号实现
    std::vector<Tensor*> parents() const override;
};

// --- Sub ---
struct SubGradFn : public GradFn {
    Tensor *a_, *b_;
    SubGradFn(Tensor* a, Tensor* b) : a_(a), b_(b) {}
    void backward(const std::vector<float>& grad_out) override;
    std::vector<Tensor*> parents() const override;
};

// --- Neg ---
struct NegGradFn : public GradFn {
    Tensor* a_;
    explicit NegGradFn(Tensor* a) : a_(a) {}
    void backward(const std::vector<float>& grad_out) override;
    std::vector<Tensor*> parents() const override;
};