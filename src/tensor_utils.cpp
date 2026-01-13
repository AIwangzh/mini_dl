#include "tensor_utils.hpp"
#include <cassert>
#include <stdexcept>

// 广播机制
// 广播机制辅助函数，得到广播后的形状
std::vector<size_t> broadcast_shape(
    const std::vector<size_t>& a,
    const std::vector<size_t>& b) 
{
    size_t ndim = std::max(a.size(), b.size());
    std::vector<size_t> out(ndim);

    for (int i = 0; i < ndim; ++i) {
        size_t da = (i < ndim - a.size()) ? 1 : a[i - (ndim - a.size())];
        size_t db = (i < ndim - b.size()) ? 1 : b[i - (ndim - b.size())];

        if (da != db && da != 1 && db != 1) {
            throw std::runtime_error("broadcast shape mismatch");
        }
        out[i] = std::max(da, db);
    }
    return out;
}

std::vector<size_t> unravel_index(
    size_t linear_idx,
    const std::vector<size_t>& shape)
{
    std::vector<size_t> idx(shape.size());

    for (int i = (int)shape.size() - 1; i >= 0; --i) {
        idx[i] = linear_idx % shape[i];
        linear_idx /= shape[i];
    }
    return idx;
}

size_t ravel_index(
    const std::vector<size_t>& idx,
    const std::vector<size_t>& shape)
{
    assert(idx.size() == shape.size());

    size_t linear = 0;
    size_t stride = 1;

    for (int i = (int)shape.size() - 1; i >= 0; --i) {
        linear += idx[i] * stride;
        stride *= shape[i];
    }
    return linear;
}

size_t ravel_index_broadcast(
    const std::vector<size_t>& out_idx,
    const std::vector<size_t>& in_shape)
{
    size_t ndim_out = out_idx.size();
    size_t ndim_in  = in_shape.size();

    std::vector<size_t> in_idx(ndim_in);

    // 右对齐
    for (size_t i = 0; i < ndim_in; ++i) {
        size_t out_dim = ndim_out - ndim_in + i;

        if (in_shape[i] == 1) {
            in_idx[i] = 0;                  // 广播
        } else {
            in_idx[i] = out_idx[out_dim];   // 正常索引
        }
    }

    return ravel_index(in_idx, in_shape);
}