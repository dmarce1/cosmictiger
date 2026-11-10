/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <atomic>

#include <cosmictiger/core/safe_io.hpp>
#include <cosmictiger/gravity/fmm_order.hpp>

namespace detail {
std::atomic<int> selectedFmmOrder{kFmmDefaultOrder};
}    // namespace detail

/// Read the currently selected runtime FMM order.
int fmmOrder() {
    return detail::selectedFmmOrder.load(std::memory_order_relaxed);
}

/// Validate the requested order against the compiled range and publish it for subsequent
/// work.
void setFmmOrder(int order) {
    if (order < kFmmMinOrder || order > kFmmMaxOrder) {
        THROW_ERROR(
            "Potential expansion degree %i is unavailable; this executable supports "
            "degrees %i through %i. Configure COSMICTIGER_EXPANSION_DEGREE to change the range.\n",
            order - 1, kFmmMinOrder - 1, kFmmMaxOrder - 1);
    }
    detail::selectedFmmOrder.store(order, std::memory_order_relaxed);
}
