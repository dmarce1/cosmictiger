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

#include <array>

#include <cosmictiger/cuda.hpp>
#include <cosmictiger/fmm_order.hpp>
#include <cosmictiger/kick.hpp>
#include <cosmictiger/kick_cuda.hpp>
#include <cosmictiger/options.hpp>

// Runtime-order dispatch to the per-order CUDA kick instantiations.

KickReturn cudaExecuteKicks(KickParameters params, Fixed32* devX, Fixed32* devY, Fixed32* devZ,
    TreeNode* devTreeNodes, Vector<KickWorkitem> workitems) {
    return fmmDispatch([&](auto order) {
        return cudaExecuteKicksImpl<order.value>(
            params, devX, devY, devZ, devTreeNodes, std::move(workitems));
    });
}

CudaKickArenaLayout const& kickArenaLayout() {
    return fmmDispatch([](auto order) -> CudaKickArenaLayout const& {
        return kickArenaLayoutImpl<order.value>();
    });
}

// Blocks per launch; also the CPU/GPU handoff divisor in kick.cpp, unless
// compute.cudaKickHandoffBlocks pins that divisor for benchmarking. The CPU
// walk calls this at every node, so the device query is cached per order.
int kickBlockCount() {
    int const handoff = getOptions().compute.cudaKickHandoffBlocks;
    if (handoff > 0) {
        return handoff;
    }
    static std::array<int, kFmmMaxOrder + 1> cached{};
    int const order = fmmOrder();
    if (cached[order] == 0) {
        cached[order] = kickArenaLayout().residency * cudaSmpCount();
    }
    return cached[order];
}
