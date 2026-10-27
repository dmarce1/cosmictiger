/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <cosmictiger/core/safe_io.hpp>
#include <cosmictiger/gravity/fmm_order.hpp>

void setFmmOrder(int order) {
    if (order != kFmmDefaultOrder) {
        THROW_ERROR("This executable was built for FMM order %i; requested order %i. "
                    "Rebuild with COSMICTIGER_WITH_ORDER=%i to change the order.\n",
            kFmmDefaultOrder, order, order);
    }
}
