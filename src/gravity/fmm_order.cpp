/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <atomic>

#include <cosmictiger/core/safe_io.hpp>
#include <cosmictiger/gravity/fmm_order.hpp>

namespace {
std::atomic<int> runtimeOrder{kFmmDefaultOrder};
}

int fmmOrder() {
    return runtimeOrder.load(std::memory_order_relaxed);
}

void setFmmOrder(int order) {
    if (order < kFmmMinOrder || order > kFmmMaxOrder) {
        THROW_ERROR("gravity.fmmOrder must be between %i and %i; got %i\n", kFmmMinOrder,
            kFmmMaxOrder, order);
    }
    runtimeOrder.store(order, std::memory_order_relaxed);
}
