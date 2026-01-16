#include "tensor.hpp"
#include "ops.hpp"
#include "autograd.hpp"
#include "grad_fn.hpp"
#include <numeric>
#include <algorithm>
#include <queue>
#include <unordered_set>

// --- 构造函数 ---
Tensor::Tensor(const std::vector<size_t>& shape, bool requires_grad)
    : impl_(std::make_shared<TensorImpl>(shape, requires_grad)) {}

Tensor::Tensor(const std::vector<size_t>& shape, float value, bool requires_grad)
    : impl_(std::make_shared<TensorImpl>(shape, requires_grad)) {
    std::fill(impl_->data_.begin(), impl_->data_.end(), value);
}

// 实现 1: 接收 vector
Tensor::Tensor(const std::vector<size_t>& shape, const std::vector<float>& data, bool requires_grad)
    : impl_(std::make_shared<TensorImpl>(shape, requires_grad)) {
    if (data.size() != numel()) {
        throw std::runtime_error("Data size does not match tensor shape");
    }
    impl_->data_ = data;
}

// 实现 2: 接收 initializer_list (支持大括号直接传值)
Tensor::Tensor(const std::vector<size_t>& shape, std::initializer_list<float> data, bool requires_grad)
    : impl_(std::make_shared<TensorImpl>(shape, requires_grad)) {
    if (data.size() != numel()) {
        throw std::runtime_error("Data size does not match tensor shape");
    }
    impl_->data_ = std::vector<float>(data);
}

// --- 基础信息 ---
size_t Tensor::numel() const {
    if (!impl_) return 0;
    size_t n = 1;
    for (auto s : impl_->shape_) n *= s;
    return n;
}

void Tensor::set_requires_grad(bool r) {
    impl_->requires_grad_ = r;
    if (r && impl_->grad_.empty()) {
        impl_->grad_.assign(numel(), 0.0f);
    }
}

void Tensor::zero_grad() {
    if (impl_ && !impl_->grad_.empty()) {
        std::fill(impl_->grad_.begin(), impl_->grad_.end(), 0.0f);
    }
}

void Tensor::set_grad_fn(GradFn* fn) {
    // 将原始指针封装进共享指针，管理其生命周期
    impl_->grad_fn_ = std::shared_ptr<GradFn>(fn);
}

void Tensor::accumulate_grad(const std::vector<float>& g) {
    if (!impl_ || !impl_->requires_grad_) return;
    if (impl_->grad_.empty()) impl_->grad_.assign(numel(), 0.0f);
    for (size_t i = 0; i < g.size(); ++i) {
        impl_->grad_[i] += g[i];
    }
}

// --- 索引访问 ---
size_t Tensor::calcOffset(const std::vector<size_t>& indices) const {
    if (indices.size() != impl_->shape_.size()) {
        throw std::runtime_error("Index dimension mismatch");
    }
    size_t offset = 0;
    size_t stride = 1;
    for (int i = (int)impl_->shape_.size() - 1; i >= 0; --i) {
        offset += indices[i] * stride;
        stride *= impl_->shape_[i];
    }
    return offset;
}

float& Tensor::operator()(const std::vector<size_t>& indices) {
    return impl_->data_[calcOffset(indices)];
}

float Tensor::operator()(const std::vector<size_t>& indices) const {
    return impl_->data_[calcOffset(indices)];
}

// --- 变换操作 (修改 Impl 状态) ---
void Tensor::reshape(const std::vector<size_t>& new_shape) {
    // 简单校验
    size_t n = 1;
    for (auto s : new_shape) n *= s;
    if (n != numel()) throw std::runtime_error("Reshape size mismatch");
    impl_->shape_ = new_shape;
}

void Tensor::flatten() {
    impl_->shape_ = { numel() };
}

// --- 运算符桥接 (调用 ops.hpp 中的全局函数) ---
// 注意：需要在文件顶层包含 "ops.hpp"
// Tensor Tensor::operator+(const Tensor& other) const { return add(*this, other); }
// Tensor Tensor::operator-(const Tensor& other) const { return sub(*this, other); }
// Tensor Tensor::operator*(const Tensor& other) const { return mul(*this, other); }
// Tensor Tensor::operator/(const Tensor& other) const { return div(*this, other); }
// Tensor Tensor::operator-() const { return neg(*this); }

// --- Backward 核心逻辑 ---
void Tensor::backward() {
    if (!requires_grad()) return;

    // 1. 初始化种子梯度 (如果是标量或未初始化)
    if (impl_->grad_.empty()) impl_->grad_.assign(numel(), 1.0f);
    std::fill(impl_->grad_.begin(), impl_->grad_.end(), 1.0f);

    // 2. 拓扑排序 (DFS)
    std::vector<Tensor> topo;
    std::unordered_set<TensorImpl*> visited;
    std::function<void(Tensor)> dfs = [&](Tensor t) {
        if (!t.impl_ || visited.count(t.impl_.get())) return;
        visited.insert(t.impl_.get());
        if (t.grad_fn()) {
            for (auto* p_raw : t.grad_fn()->parents()) {
                dfs(*p_raw); // 递归访问父节点
            }
        }
        topo.push_back(t);
    };
    dfs(*this);

    // 3. 计算入度 (pending count)
    for (auto& t : topo) t.impl_->grad_pending_ = 0;
    for (auto& t : topo) {
        if (t.grad_fn()) {
            for (auto* p_raw : t.grad_fn()->parents()) {
                p_raw->impl_->grad_pending_++;
            }
        }
    }

    // 4. 广度优先触发 (队列)
    std::queue<Tensor> q;
    q.push(*this);

    while (!q.empty()) {
        Tensor t = q.front();
        q.pop();

        if (t.grad_fn()) {
            // 执行当前节点的反向传播，将梯度传给 parents
            t.grad_fn()->backward(t.grad());
            
            for (auto* p_raw : t.grad_fn()->parents()) {
                p_raw->impl_->grad_pending_--;
                if (p_raw->impl_->grad_pending_ == 0) {
                    q.push(*p_raw);
                }
            }
        }
    }
}

// --- 补充 transpose 成员函数 ---
Tensor Tensor::transpose(const std::vector<size_t>& perm) const {
    // 调用 ops.hpp 中的全局 transpose 逻辑，或者在这里直接写实现
    // 由于你之前的 ops.hpp 中 transpose 暂不支持 perm 参数，
    // 如果你只需要 2D 转置，可以先这样写：
    if (perm.size() == 2 && perm[0] == 1 && perm[1] == 0) {
        return ::transpose(*this); // 调用 ops.cpp 里的那个全局 transpose
    }
    throw std::runtime_error("Advanced transpose with perm not implemented yet");
}

// --- 补充部分维度 flatten 成员函数 ---
Tensor Tensor::flatten(size_t start_dim, size_t end_dim) const {
    const auto& old_shape = this->shape();
    if (start_dim >= old_shape.size() || end_dim >= old_shape.size() || start_dim > end_dim) {
        throw std::runtime_error("Invalid flatten dimensions");
    }

    std::vector<size_t> new_shape;
    // 保持 start_dim 之前的维度
    for (size_t i = 0; i < start_dim; ++i) new_shape.push_back(old_shape[i]);
    
    // 合并中间的维度
    size_t flattened_size = 1;
    for (size_t i = start_dim; i <= end_dim; ++i) flattened_size *= old_shape[i];
    new_shape.push_back(flattened_size);
    
    // 保持 end_dim 之后的维度
    for (size_t i = end_dim + 1; i < old_shape.size(); ++i) new_shape.push_back(old_shape[i]);

    // 创建一个共享同一个数据的 Tensor (或者深拷贝数据)
    // 简单起见，这里创建新 Tensor 并拷贝数据
    Tensor out(new_shape, this->requires_grad());
    out.data() = this->data(); // 拷贝数据
    return out;
}


// #include "tensor.hpp"
// #include "tensor_utils.hpp"
// #include "autograd.hpp"
// #include "ops.hpp"
// #include <cassert>
// #include <numeric>
// #include <unordered_set>
// #include <functional>
// #include <stdexcept>
// #include <queue>

// /*Tensor::Tensor(const std::vector<size_t>& shape)
//     : shape_(shape)
// {
//     data_.resize(numel());
// }

// Tensor::Tensor(const std::vector<size_t>& shape, float value)
//     : shape_(shape)
// {
//     data_.resize(numel(), value);
// }

// Tensor::Tensor(const std::vector<size_t>& shape, bool requires_grad)
//     : shape_(shape),
//       requires_grad_(requires_grad) 
// {
//     data_.resize(numel());

//     if (requires_grad_) {
//         grad_.resize(numel(), 0.0f);
//     }
// }*/
// /*
// Tensor::Tensor(const std::vector<size_t>& shape, float value)
//     : Tensor(shape, value, false) {}

// Tensor::Tensor(const std::vector<size_t>& shape, bool requires_grad)
//     : Tensor(shape, 0.0f, requires_grad) {}

// Tensor::Tensor(const std::vector<size_t>& shape, float value, bool requires_grad)
//     : shape_(shape),
//       requires_grad_(requires_grad)
// {
//     data_.resize(numel(), value);

//     if (requires_grad_) {
//         grad_.resize(numel(), 0.0f);
//     }
// }
// */
// Tensor::Tensor(const std::vector<size_t>& shape, bool requires_grad)
//     : shape_(shape), 
//       requires_grad_(requires_grad),
//       grad_fn_(nullptr) // 明确初始化智能指针
// {
//     size_t n = numel();
//     data_.resize(n, 0.0f); // 默认填充 0
//     if (requires_grad_) {
//         grad_.resize(n, 0.0f);
//     }
// }

// Tensor::Tensor(const std::vector<size_t>& shape, float value, bool requires_grad)
//     : shape_(shape), 
//       requires_grad_(requires_grad),
//       grad_fn_(nullptr)
// {
//     size_t n = numel();
//     data_.resize(n, value);
//     if (requires_grad_) {
//         grad_.resize(n, 0.0f);
//     }
// }

// const std::vector<size_t>& Tensor::shape() const {
//     return shape_;
// }

// size_t Tensor::numel() const {
//     return std::accumulate(
//         shape_.begin(),
//         shape_.end(),
//         static_cast<size_t>(1),
//         std::multiplies<size_t>());
// }

// float& Tensor::operator()(const std::vector<size_t>& indices) {
//     return data_[calcOffset(indices)];
// }

// float Tensor::operator()(const std::vector<size_t>& indices) const {
//     return data_[calcOffset(indices)];
// }

// void Tensor::reshape(const std::vector<size_t>& new_shape) {
//     size_t new_numel = std::accumulate(
//         new_shape.begin(),
//         new_shape.end(),
//         static_cast<size_t>(1),
//         std::multiplies<size_t>());

//     assert(new_numel == numel()); //检查新形状与原形状元素个数是否一致，不允许改变元素个数
//     shape_ = new_shape;
// }

// void Tensor::flatten() {
//     reshape({numel()});
// }

// // Tensor Tensor::operator+(const Tensor& other) const {
// //     assert(shape_ == other.shape_);
// //     Tensor out(shape_);
// //     for (size_t i = 0; i < numel(); ++i) {
// //         out.data_[i] = data_[i] + other.data_[i];
// //     }
// //     return out;
// // }

// Tensor Tensor::operator+(const Tensor& other) const {
//     return add(*this, other);
// }

// Tensor Tensor::operator-(const Tensor& other) const {
//     return sub(*this, other);
// }

// Tensor Tensor::operator*(const Tensor& other) const {
//     assert(shape_ == other.shape_);
//     Tensor out(shape_);
//     for (size_t i = 0; i < numel(); ++i) {
//         out.data_[i] = data_[i] * other.data_[i];
//     }
//     return out;
// }

// Tensor Tensor::operator/(const Tensor& other) const {
//     assert(shape_ == other.shape_ && "Shapes must match for element-wise division");
//     Tensor result(shape_);
//     for (size_t i = 0; i < data_.size(); ++i) {
//         assert(other.data_[i] != 0 && "Division by zero");
//         result.data_[i] = data_[i] / other.data_[i];
//     }
//     return result;
// }

// Tensor Tensor::operator-() const {
//     return neg(*this);
// }


// size_t Tensor::calcOffset(const std::vector<size_t>& indices) const {
//     assert(indices.size() == shape_.size());

//     size_t offset = 0;
//     size_t stride = 1;

//     // row-major
//     for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
//         offset += indices[i] * stride;
//         stride *= shape_[i];
//     }
//     return offset;
// }

// Tensor Tensor::transpose(const std::vector<size_t>& perm) const {
//     assert(perm.size() == shape_.size());

//     std::vector<size_t> new_shape(shape_.size());
//     for (size_t i = 0; i < perm.size(); ++i) {
//         new_shape[i] = shape_[perm[i]];
//     }

//     Tensor out(new_shape);

//     for (size_t i = 0; i < numel(); ++i) {
//         auto idx = unravel_index(i, shape_);
//         std::vector<size_t> new_idx(idx.size());

//         for (size_t k = 0; k < perm.size(); ++k) {
//             new_idx[k] = idx[perm[k]];
//         }

//         size_t j = ravel_index(new_idx, new_shape);
//         out[j] = data_[i];
//     }
//     return out;
// }

// Tensor Tensor::flatten(size_t start_dim, size_t end_dim) const {
//     assert(start_dim <= end_dim);
//     assert(end_dim < shape_.size());

//     std::vector<size_t> new_shape;

//     for (size_t i = 0; i < start_dim; ++i)
//         new_shape.push_back(shape_[i]);

//     size_t flat = 1;
//     for (size_t i = start_dim; i <= end_dim; ++i)
//         flat *= shape_[i];
//     new_shape.push_back(flat);

//     for (size_t i = end_dim + 1; i < shape_.size(); ++i)
//         new_shape.push_back(shape_[i]);

//     Tensor out = *this;   // 复制数据
//     out.reshape(new_shape);
//     return out;
// }

// GradFn* Tensor::grad_fn() const { 
//     return grad_fn_.get(); 
// }

// void Tensor::set_grad_fn(GradFn* fn) { 
//     grad_fn_.reset(fn); // 这里调用 reset 时，编译器已经通过 autograd.hpp 知道如何销毁 fn 了
// }

// void Tensor::accumulate_grad(const std::vector<float>& g) {
//     // 只对 requires_grad 的 Tensor 生效
//     if (!requires_grad_) return;

//     // 第一次收到梯度时分配空间
//     if (grad_.empty()) {
//         grad_.resize(numel(), 0.0f);
//     }

//     // 梯度累加（而不是赋值）
//     for (size_t i = 0; i < grad_.size(); ++i) {
//         grad_[i] += g[i];
//     }
// }

// void Tensor::zero_grad() {
//     if (!requires_grad_) return;

//     std::fill(grad_.begin(), grad_.end(), 0.0f);
// }

// namespace {
//     void build_topo(Tensor* t,
//                 std::vector<Tensor*>& topo,
//                 std::unordered_set<Tensor*>& visited) {
//     if (!t || visited.count(t)) return;
//     visited.insert(t);

//     if (t->grad_fn()) {
//         for (Tensor* parent : t->grad_fn()->parents()) {
//             build_topo(parent, topo, visited);
//         }
//     }

//     topo.push_back(t);
// }
// }

// void Tensor::backward() {
//     if (!requires_grad_) return;

//     // 1. 初始化输出梯度（默认 dL/dself = 1）
//     if (grad_.empty()) {
//         grad_.assign(numel(), 1.0f);
//     }

//     // 2. 收集计算图中的所有 Tensor（反向 DFS）
//     std::vector<Tensor*> topo;
//     std::unordered_set<Tensor*> visited;

//     std::function<void(Tensor*)> dfs = [&](Tensor* t) {
//         if (!t || visited.count(t)) return;
//         visited.insert(t);

//         if (t->grad_fn()) {
//             for (auto* p : t->grad_fn()->parents()) {
//                 dfs(p);
//             }
//         }
//         topo.push_back(t);
//     };

//     dfs(this);

//     // 3. 初始化 pending 计数
//     for (auto* t : topo) {
//         t->grad_pending_ = 0;
//     }
//     for (auto* t : topo) {
//         if (t->grad_fn()) {
//             for (auto* p : t->grad_fn()->parents()) {
//                 p->grad_pending_++;
//             }
//         }
//     }

//     // 4. 反向拓扑传播
//     std::queue<Tensor*> q;
//     q.push(this);

//     while (!q.empty()) {
//         Tensor* t = q.front();
//         q.pop();

//         if (t->grad_fn()) {
//             t->grad_fn()->backward(t->grad());
//             for (auto* p : t->grad_fn()->parents()) {
//                 p->grad_pending_--;
//                 if (p->grad_pending_ == 0) {
//                     q.push(p);
//                 }
//             }
//         }
//     }
// }


// /*
// void Tensor::backward() {
//     // 1) 必须需要梯度
//     if (!requires_grad_) {
//         return;
//     }

//     // 2) 当前阶段：仅支持 scalar
//     if (numel() != 1) {
//         throw std::runtime_error(
//             "Tensor::backward(): only scalar Tensor is supported currently"
//         );
//     }

//     // 3) 初始化自身梯度为 1（dL/dL = 1）
//     if (grad_.empty()) {
//         grad_.resize(1, 1.0f);
//     }

//     // 4) 构建拓扑序
//     std::vector<Tensor*> topo;
//     std::unordered_set<Tensor*> visited;
//     build_topo(this, topo, visited);

//     // 5) 反向执行（从后往前）
//     for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
//         Tensor* t = *it;
//         if (t->grad_fn()) {
//             t->grad_fn()->backward(t->grad());
//         }
//     }
// }
// */

