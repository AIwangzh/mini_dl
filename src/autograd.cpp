#include "autograd.hpp"
#include "tensor.hpp" // 这里包含了完整定义，所以 a_->requires_grad() 合法了

void GradFn::accumulate(Tensor* t, const std::vector<float>& g) {
    if (t && t->requires_grad()) {
        t->accumulate_grad(g);
    }
}

// // Add 实现
// void AddGradFn::backward(const std::vector<float>& grad_out) {
//     if (a_.requires_grad())  accumulate(&a_, grad_out);
//     if (b_.requires_grad())accumulate(&b_, grad_out);
// }
// std::vector<Tensor*> AddGradFn::parents() const { return {&a_, &b_}; }

// // Sub 实现
// void SubGradFn::backward(const std::vector<float>& grad_out) {
//     if (a_.requires_grad()) {
//             accumulate(&a_, grad_out);
//         }
        
//         if (b_.requires_grad()) {
//             // 对 grad_out 取反
//             std::vector<float> neg_grad = grad_out;
//             for (auto& v : neg_grad) v = -v;
//             accumulate(&b_, neg_grad);
//         }
// }
// std::vector<Tensor*> SubGradFn::parents() const { return { const_cast<Tensor*>(&a_), const_cast<Tensor*>(&b_) }; }

// // Neg 实现
// void NegGradFn::backward(const std::vector<float>& grad_out) {
//     if (a_.requires_grad()) {
//             std::vector<float> neg = grad_out;
//             for (auto& v : neg) v = -v;
//             accumulate(&a_, neg);
//         }
// }
// std::vector<Tensor*> NegGradFn::parents() const { return { const_cast<Tensor*>(&a_) }; }