#include "tensor.hpp"
#include "tensor_utils.hpp"
#include "autograd.hpp"
#include "ops.hpp"
#include <cassert>
#include <numeric>
#include <unordered_set>
#include <functional>
#include <stdexcept>

/*Tensor::Tensor(const std::vector<size_t>& shape)
    : shape_(shape)
{
    data_.resize(numel());
}

Tensor::Tensor(const std::vector<size_t>& shape, float value)
    : shape_(shape)
{
    data_.resize(numel(), value);
}

Tensor::Tensor(const std::vector<size_t>& shape, bool requires_grad)
    : shape_(shape),
      requires_grad_(requires_grad) 
{
    data_.resize(numel());

    if (requires_grad_) {
        grad_.resize(numel(), 0.0f);
    }
}*/
/*
Tensor::Tensor(const std::vector<size_t>& shape, float value)
    : Tensor(shape, value, false) {}

Tensor::Tensor(const std::vector<size_t>& shape, bool requires_grad)
    : Tensor(shape, 0.0f, requires_grad) {}

Tensor::Tensor(const std::vector<size_t>& shape, float value, bool requires_grad)
    : shape_(shape),
      requires_grad_(requires_grad)
{
    data_.resize(numel(), value);

    if (requires_grad_) {
        grad_.resize(numel(), 0.0f);
    }
}
*/
Tensor::Tensor(const std::vector<size_t>& shape, bool requires_grad)
    : shape_(shape), 
      requires_grad_(requires_grad),
      grad_fn_(nullptr) // 明确初始化智能指针
{
    size_t n = numel();
    data_.resize(n, 0.0f); // 默认填充 0
    if (requires_grad_) {
        grad_.resize(n, 0.0f);
    }
}

Tensor::Tensor(const std::vector<size_t>& shape, float value, bool requires_grad)
    : shape_(shape), 
      requires_grad_(requires_grad),
      grad_fn_(nullptr)
{
    size_t n = numel();
    data_.resize(n, value);
    if (requires_grad_) {
        grad_.resize(n, 0.0f);
    }
}

const std::vector<size_t>& Tensor::shape() const {
    return shape_;
}

size_t Tensor::numel() const {
    return std::accumulate(
        shape_.begin(),
        shape_.end(),
        static_cast<size_t>(1),
        std::multiplies<size_t>());
}

float& Tensor::operator()(const std::vector<size_t>& indices) {
    return data_[calcOffset(indices)];
}

float Tensor::operator()(const std::vector<size_t>& indices) const {
    return data_[calcOffset(indices)];
}

void Tensor::reshape(const std::vector<size_t>& new_shape) {
    size_t new_numel = std::accumulate(
        new_shape.begin(),
        new_shape.end(),
        static_cast<size_t>(1),
        std::multiplies<size_t>());

    assert(new_numel == numel()); //检查新形状与原形状元素个数是否一致，不允许改变元素个数
    shape_ = new_shape;
}

void Tensor::flatten() {
    reshape({numel()});
}

// Tensor Tensor::operator+(const Tensor& other) const {
//     assert(shape_ == other.shape_);
//     Tensor out(shape_);
//     for (size_t i = 0; i < numel(); ++i) {
//         out.data_[i] = data_[i] + other.data_[i];
//     }
//     return out;
// }

Tensor Tensor::operator+(const Tensor& other) const {
    return add(*this, other);
}

Tensor Tensor::operator-(const Tensor& other) const {
    return sub(*this, other);
}

Tensor Tensor::operator*(const Tensor& other) const {
    assert(shape_ == other.shape_);
    Tensor out(shape_);
    for (size_t i = 0; i < numel(); ++i) {
        out.data_[i] = data_[i] * other.data_[i];
    }
    return out;
}

Tensor Tensor::operator/(const Tensor& other) const {
    assert(shape_ == other.shape_ && "Shapes must match for element-wise division");
    Tensor result(shape_);
    for (size_t i = 0; i < data_.size(); ++i) {
        assert(other.data_[i] != 0 && "Division by zero");
        result.data_[i] = data_[i] / other.data_[i];
    }
    return result;
}

Tensor Tensor::operator-() const {
    return neg(*this);
}


size_t Tensor::calcOffset(const std::vector<size_t>& indices) const {
    assert(indices.size() == shape_.size());

    size_t offset = 0;
    size_t stride = 1;

    // row-major
    for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
        offset += indices[i] * stride;
        stride *= shape_[i];
    }
    return offset;
}

Tensor Tensor::transpose(const std::vector<size_t>& perm) const {
    assert(perm.size() == shape_.size());

    std::vector<size_t> new_shape(shape_.size());
    for (size_t i = 0; i < perm.size(); ++i) {
        new_shape[i] = shape_[perm[i]];
    }

    Tensor out(new_shape);

    for (size_t i = 0; i < numel(); ++i) {
        auto idx = unravel_index(i, shape_);
        std::vector<size_t> new_idx(idx.size());

        for (size_t k = 0; k < perm.size(); ++k) {
            new_idx[k] = idx[perm[k]];
        }

        size_t j = ravel_index(new_idx, new_shape);
        out[j] = data_[i];
    }
    return out;
}

Tensor Tensor::flatten(size_t start_dim, size_t end_dim) const {
    assert(start_dim <= end_dim);
    assert(end_dim < shape_.size());

    std::vector<size_t> new_shape;

    for (size_t i = 0; i < start_dim; ++i)
        new_shape.push_back(shape_[i]);

    size_t flat = 1;
    for (size_t i = start_dim; i <= end_dim; ++i)
        flat *= shape_[i];
    new_shape.push_back(flat);

    for (size_t i = end_dim + 1; i < shape_.size(); ++i)
        new_shape.push_back(shape_[i]);

    Tensor out = *this;   // 复制数据
    out.reshape(new_shape);
    return out;
}

void Tensor::accumulate_grad(const std::vector<float>& g) {
    // 只对 requires_grad 的 Tensor 生效
    if (!requires_grad_) return;

    // 第一次收到梯度时分配空间
    if (grad_.empty()) {
        grad_.resize(numel(), 0.0f);
    }

    // 梯度累加（而不是赋值）
    for (size_t i = 0; i < grad_.size(); ++i) {
        grad_[i] += g[i];
    }
}

void Tensor::zero_grad() {
    if (!requires_grad_) return;

    std::fill(grad_.begin(), grad_.end(), 0.0f);
}

namespace {
    void build_topo(Tensor* t,
                std::vector<Tensor*>& topo,
                std::unordered_set<Tensor*>& visited) {
    if (!t || visited.count(t)) return;
    visited.insert(t);

    if (t->grad_fn()) {
        for (Tensor* parent : t->grad_fn()->parents()) {
            build_topo(parent, topo, visited);
        }
    }

    topo.push_back(t);
}
}


void Tensor::backward() {
    // 1) 必须需要梯度
    if (!requires_grad_) {
        return;
    }

    // 2) 当前阶段：仅支持 scalar
    if (numel() != 1) {
        throw std::runtime_error(
            "Tensor::backward(): only scalar Tensor is supported currently"
        );
    }

    // 3) 初始化自身梯度为 1（dL/dL = 1）
    if (grad_.empty()) {
        grad_.resize(1, 1.0f);
    }

    // 4) 构建拓扑序
    std::vector<Tensor*> topo;
    std::unordered_set<Tensor*> visited;
    build_topo(this, topo, visited);

    // 5) 反向执行（从后往前）
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        Tensor* t = *it;
        if (t->grad_fn()) {
            t->grad_fn()->backward(t->grad());
        }
    }
}
// void Tensor::backward() {
//     // 1. 只有需要梯度的 Tensor 才能 backward
//     if (!requires_grad_) {
//         return;
//     }

//     // 2. 当前阶段：只允许对 scalar Tensor backward
//     if (numel() != 1) {
//         throw std::runtime_error(
//             "backward() only supported for scalar Tensor currently"
//         );
//     }

//     // 3. 如果 grad 还没初始化，初始化为 1
//     if (grad_.empty()) {
//         grad_.resize(1, 1.0f);
//     }

//     // 4. 调用 grad_fn 向前传播
//     if (grad_fn_) {
//         grad_fn_->backward(grad_);
//     }
// }



