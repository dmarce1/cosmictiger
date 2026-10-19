/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 This program is free software; you can redistribute it and/or
 modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation; either version 2
 of the License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */

#include <atomic>

#include <cosmictiger/fmm_order.hpp>
#include <cosmictiger/safe_io.hpp>

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
