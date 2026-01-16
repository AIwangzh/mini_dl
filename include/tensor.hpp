#pragma once
#include "tensor_utils.hpp"
#include <vector>
#include <memory>
#include <functional>
#include <string>
#include <stdexcept>
#include <iostream>

struct GradFn;

struct TensorImpl {
    /* === 数据本体 === */
    std::vector<float> data_;
    std::vector<size_t> shape_;

    /* === Autograd 内部状态 === */
    std::vector<float> grad_;
    bool requires_grad_{false};
    std::shared_ptr<GradFn> grad_fn_; // 保持使用 shared_ptr 管理 grad_fn
    int grad_pending_{0};             // 用于拓扑排序的依赖计数

    // 构造函数
    TensorImpl(const std::vector<size_t>& shape, bool requires_grad)
        : shape_(shape), requires_grad_(requires_grad) {
        size_t n = 1;
        for (auto s : shape) n *= s;
        data_.assign(n, 0.0f);
        if (requires_grad_) {
            grad_.assign(n, 0.0f);
        }
    }
};

// --- 外壳：Tensor 句柄 ---
class Tensor {
public:
    Tensor() = default;
    explicit Tensor(const std::vector<size_t>& shape, bool requires_grad = false);
    Tensor(const std::vector<size_t>& shape, float value, bool requires_grad = false);
    // 增加支持 std::vector 初始化数据的构造函数
    Tensor(const std::vector<size_t>& shape, const std::vector<float>& data, bool requires_grad = false);

    // (可选) 增加支持大括号 {} 初始化的构造函数，这样写起来更像 PyTorch
    Tensor(const std::vector<size_t>& shape, std::initializer_list<float> data, bool requires_grad = false);
    
    // 拷贝构造与赋值：现在是浅拷贝（遥控器拷贝）
    Tensor(const Tensor& other) : impl_(other.impl_) {}
    Tensor& operator=(const Tensor& other) {
        if (this != &other) impl_ = other.impl_;
        return *this;
    }
    ~Tensor() = default;

    // 基本信息
    const std::vector<size_t>& shape() const { return impl_->shape_; }
    size_t numel() const;

    // 数据访问
    std::vector<float>& data() { return impl_->data_; }
    const std::vector<float>& data() const { return impl_->data_; }
    std::vector<float>& grad() { return impl_->grad_; }
    const std::vector<float>& grad() const { return impl_->grad_; }

    // 索引访问
    float& operator[](size_t i) { return impl_->data_[i]; }
    const float& operator[](size_t i) const { return impl_->data_[i]; }
    float& operator()(const std::vector<size_t>& indices);
    float operator()(const std::vector<size_t>& indices) const;

    // 变换算子
    void reshape(const std::vector<size_t>& new_shape);
    void flatten();
    Tensor transpose(const std::vector<size_t>& perm) const;
    Tensor flatten(size_t start_dim, size_t end_dim) const;

    // 逐元素运算符重载
    // Tensor operator+(const Tensor& other) const;
    // Tensor operator-(const Tensor& other) const;
    // Tensor operator*(const Tensor& other) const;
    // Tensor operator/(const Tensor& other) const;
    // Tensor operator-() const;

    /* === Autograd 接口 === */
    friend struct GradFn;
    bool requires_grad() const { return impl_ ? impl_->requires_grad_ : false; }
    void set_requires_grad(bool r);
    void zero_grad();
    void backward(); 
    
    GradFn* grad_fn() const { return impl_->grad_fn_.get(); }
    void set_grad_fn(GradFn* fn);

    // 暴露内部 pending 给拓扑排序使用
    int& grad_pending() { return impl_->grad_pending_; }

protected:
    void accumulate_grad(const std::vector<float>& g);

private:
    std::shared_ptr<TensorImpl> impl_;
    size_t calcOffset(const std::vector<size_t>& indices) const;
};

// class Tensor {
// public:
//     Tensor() = default;  //默认构造函数
//     explicit Tensor(const std::vector<size_t>& shape, bool requires_grad = false);
//     Tensor(const std::vector<size_t>& shape, float value, bool requires_grad = false);
//     //Tensor(const std::vector<size_t>& shape, float value); //创建一个指定shape的Tensor，并把所有元素初始化为value
    
//     Tensor(const Tensor& other)
//     : data_(other.data_),
//       shape_(other.shape_),
//       grad_(other.grad_),
//       requires_grad_(other.requires_grad_),
//       grad_fn_(other.grad_fn_) {}

//     ~Tensor() = default;

//     Tensor& operator=(const Tensor& other) {
//     if (this != &other) {
//         data_ = other.data_;
//         shape_ = other.shape_;
//         grad_ = other.grad_;
//         requires_grad_ = other.requires_grad_;
//         grad_fn_ = other.grad_fn_; // shared_ptr 会自动处理引用计数
//     }
//     return *this;
// }


//     // 基本信息
//     const std::vector<size_t>& shape() const; //Tensor的形状
//     size_t numel() const; //Tensor中元素的个数

//     // 只读访问（const Tensor）
//     const std::vector<float>& data() const { return data_; }
//     // 可写访问（非 const Tensor）
//     std::vector<float>& data() { return data_; }

//     // 索引访问（多维）
//     float& operator()(const std::vector<size_t>& indices);
//     float operator()(const std::vector<size_t>& indices) const;

//     float& operator[](size_t i) { return data_[i]; }
//     const float& operator[](size_t i) const { return data_[i]; }

//     // reshape（不改变数据）
//     void reshape(const std::vector<size_t>& new_shape);
//     // 展开成一维
//     void flatten();

//     // 逐元素算子
//     Tensor operator+(const Tensor& other) const;
//     Tensor operator-(const Tensor& other) const;
//     Tensor operator*(const Tensor& other) const;
//     Tensor operator/(const Tensor& other) const; 
//     Tensor operator-() const;

//     // 高维transpose
//     Tensor transpose(const std::vector<size_t>& perm) const;
//     // 部分维度flatten
//     Tensor flatten(size_t start_dim, size_t end_dim) const;

//     /* === Autograd 工具 === */
//     friend struct GradFn; // 反向传播函数基类的友元声明
//     void backward(); // 反向传播调用接口，仅限scalar
//     //void backward(const std::vector<float>& grad_out = {}); // 适用于非scalar Tensor
//     const std::vector<float>& grad() const { return grad_; } // 梯度访问接口
//     std::vector<float>& grad() { return grad_; }
//     void zero_grad(); // 梯度清零
//     bool requires_grad() const {return requires_grad_;} // 构造算子是判断是否需要建图(计算梯度)
//     void set_requires_grad(bool r) {
//         requires_grad_ = r;
//         if (requires_grad_ && grad_.empty()) {
//             grad_.resize(numel(), 0.0f);
//         }
//     } // 设置requires_grad标志
   

//     int grad_pending_ = 0; // 用于追踪反向传播中未处理的依赖数
//     GradFn* grad_fn() const;
//     void set_grad_fn(GradFn* fn);


// protected:
//     /* === Autograd 内部工具 === */
//     void accumulate_grad(const std::vector<float>& g);

// private:
//     /* ===数据本体=== */
//     std::vector<float> data_;
//     std::vector<size_t> shape_;
//     /* === Autograd内部状态 === */
//     std::vector<float> grad_;
//     bool requires_grad_{false};

//     void backward_internal();

//     std::shared_ptr<GradFn> grad_fn_;

//     size_t calcOffset(const std::vector<size_t>& indices) const;
// };
