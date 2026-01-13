#pragma once
#include <vector>
#include <cstddef>


std::vector<size_t> broadcast_shape(
    const std::vector<size_t>& a,
    const std::vector<size_t>& b);

std::vector<size_t> unravel_index(size_t linear_idx,
                                  const std::vector<size_t>& shape);

size_t ravel_index(
    const std::vector<size_t>& idx,
    const std::vector<size_t>& shape);

size_t ravel_index_broadcast(
    const std::vector<size_t>& out_idx,
    const std::vector<size_t>& in_shape);


