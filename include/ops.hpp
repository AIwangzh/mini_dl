#pragma once
#include "tensor.hpp"
#include "tensor_utils.hpp"
#include <stdexcept>

// ---------------- Tensor × Tensor (广播机制) ----------------
Tensor add(const Tensor& a, const Tensor& b);
Tensor sub(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b); // 逐元素乘
Tensor div(const Tensor& a, const Tensor& b); // 逐元素除

// ---------------- Tensor × scalar / scalar × Tensor ----------------
Tensor mul(const Tensor& t, float scalar);
Tensor mul(float scalar, const Tensor& t);

Tensor add(const Tensor& t, float scalar);
Tensor add(float scalar, const Tensor& t);

Tensor sub(const Tensor& t, float scalar);
Tensor sub(float scalar, const Tensor& t);

Tensor div(const Tensor& t, float scalar);
Tensor div(float scalar, const Tensor& t);

// ---------------- inline operator overload ----------------
inline Tensor operator*(const Tensor& t, float scalar) { return mul(t, scalar); }
inline Tensor operator*(float scalar, const Tensor& t) { return mul(scalar, t); }

inline Tensor operator+(const Tensor& t, float scalar) { return add(t, scalar); }
inline Tensor operator+(float scalar, const Tensor& t) { return add(scalar, t); }

inline Tensor operator-(const Tensor& t, float scalar) { return sub(t, scalar); }
inline Tensor operator-(float scalar, const Tensor& t) { return sub(scalar, t); }

inline Tensor operator/(const Tensor& t, float scalar) { return div(t, scalar); }
inline Tensor operator/(float scalar, const Tensor& t) { return div(scalar, t); }

// 矩阵相乘算子
Tensor matmul(const Tensor& a, const Tensor& b);
// 转置算子
Tensor transpose(const Tensor& t);


