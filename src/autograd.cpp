#include "autograd.hpp"
#include "tensor.hpp"
#include <algorithm> // std::transform

// AddGradFn 的 backward 实现
// void AddGradFn::backward(const std::vector<float>& grad_out) {
//     if (a_) accumulate(a_, grad_out);  // 通过 GradFn::accumulate 调用
//     if (b_) accumulate(b_, grad_out);
// }

// GradFn 的 accumulate 实现
void GradFn::accumulate(Tensor* t, const std::vector<float>& g) {
    if (t && t->requires_grad()) {
        t->accumulate_grad(g);  // ✅ 合法：GradFn 是 friend
    }
}




