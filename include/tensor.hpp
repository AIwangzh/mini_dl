#pragma once

#include "tensor_utils.hpp"

#include <vector>
#include <cstddef>

class Tensor {
public:
    Tensor();  //默认构造函数
    explicit Tensor(const std::vector<size_t>& shape); //根据shape分配空间的构造函数，用于创建未赋值Tensor
    Tensor(const std::vector<size_t>& shape, float value); //创建一个指定shape的Tensor，并把所有元素初始化为value
    
    Tensor(const Tensor& other)
    : data_(other.data_),
      shape_(other.shape_),
      grad_(other.grad_),
      requires_grad_(other.requires_grad_),
      grad_fn_(other.grad_fn_) {}

    Tensor(const std::vector<size_t>& shape, bool requires_grad);
    Tensor(const std::vector<size_t>& shape, float value, bool requires_grad);

    Tensor& operator=(const Tensor& other) {
        if (this != &other) {
            data_ = other.data_;
            shape_ = other.shape_;
        }
        return *this;
    }


    // 基本信息
    const std::vector<size_t>& shape() const; //Tensor的形状
    size_t numel() const; //Tensor中元素的个数

    // 只读访问（const Tensor）
    const std::vector<float>& data() const { return data_; }
    // 可写访问（非 const Tensor）
    std::vector<float>& data() { return data_; }

    // 索引访问（多维）
    float& operator()(const std::vector<size_t>& indices);
    float operator()(const std::vector<size_t>& indices) const;

    float& operator[](size_t i) { return data_[i]; }
    const float& operator[](size_t i) const { return data_[i]; }

    // reshape（不改变数据）
    void reshape(const std::vector<size_t>& new_shape);
    // 展开成一维
    void flatten();

    // 逐元素算子
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const; 
    Tensor operator-() const;

    // 高维transpose
    Tensor transpose(const std::vector<size_t>& perm) const;
    // 部分维度flatten
    Tensor flatten(size_t start_dim, size_t end_dim) const;

    /* === Autograd 工具 === */
    friend struct GradFn; // 反向传播函数基类的友元声明
    void backward(); // 反向传播调用接口，仅限scalar
    void backward(const std::vector<float>& grad_out); // 适用于非scalar Tensor
    const std::vector<float>& grad() const { return grad_; } // 梯度访问接口
    void zero_grad(); // 梯度清零
    bool requires_grad() const {return requires_grad_;} // 构造算子是判断是否需要建图(计算梯度)
    void set_requires_grad(bool r) { requires_grad_ = r; } // 设置requires_grad标志

    int grad_pending_ = 0; // 用于追踪反向传播中未处理的依赖数
    GradFn* grad_fn() const { return grad_fn_; }
    void set_grad_fn(GradFn* fn) { grad_fn_ = fn; }


protected:
    /* === Autograd 内部工具 === */
    void accumulate_grad(const std::vector<float>& g);

private:
    /* ===数据本体=== */
    std::vector<float> data_;
    std::vector<size_t> shape_;
    /* === Autograd内部状态 === */
    std::vector<float> grad_;
    bool requires_grad_ = false;

    void backward_internal();

    struct GradFn* grad_fn_ = nullptr;

    size_t calcOffset(const std::vector<size_t>& indices) const;
};
