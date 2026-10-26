#pragma once

#include <array>
#include <cstddef>

// The allocator only needs fixed-size storage. Keep the production allocator
// declaration, replacing the unrelated HPX/container dependency graph.
template <class Value, int count>
using Array = std::array<Value, count>;
