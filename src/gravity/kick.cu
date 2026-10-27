/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <algorithm>
#include <cmath>

#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/cuda/cuda.hpp>
#include <cosmictiger/gravity/fmm_order.hpp>
#include <cosmictiger/gravity/gravity.hpp>
#include <cosmictiger/gravity/kick.hpp>
#include <cosmictiger/gravity/kick_cuda.hpp>

// Dispatch to the single compiled CUDA kick order.

KickReturn cudaExecuteKicks(KickParameters params, Position* devX, Position* devY, Position* devZ,
    TreeNode* devTreeNodes, Vector<KickWorkitem> workitems, detail::FmmForceOutput output) {
    return kickRunCudaLaunch([&]() {
        return fmmDispatch([&](auto order) {
            return cudaExecuteKicksImpl<order.value>(
                params, devX, devY, devZ, devTreeNodes, std::move(workitems), output);
        });
    });
}

CudaKickArenaLayout const& kickArenaLayout() {
    return fmmDispatch([](auto order) -> CudaKickArenaLayout const& {
        return kickArenaLayoutImpl<order.value>();
    });
}

// CPU/GPU handoff divisor: the CPU walk hands a subtree to the GPU once it
// holds at most max(particles / kickBlockCount(particles), bucket) particles
// (capped by CUDA_KICK_PARTS_MAX). It is deliberately not the launch size
// (residency times the multiprocessor count): tying it to the launch made the
// arena's eight resident blocks per multiprocessor hand off 50-particle
// subtrees, and the two-thread CPU walk then dominated the kick (bucket 32 ran
// 2.7x slower than with the baseline's 492-block divisor). Measured
// (docs/cuda_kick_shared_memory.md): at 32,768 particles two blocks per
// multiprocessor (200-particle subtrees) is the fastest division for every
// fixture at 2 to 12 worker threads; at 262,144 particles the GPU needs
// twice the workitems, and four blocks per multiprocessor (800-particle
// subtrees) is 10-25% faster on the clustered and adaptive fixtures from four
// threads up while eight blocks (400-particle subtrees) leaves the CPU walk
// dominant again; at 2,097,152 particles six blocks per multiprocessor
// (4,262-particle subtrees) is 4-9% faster on average than four and never
// more than 3% slower, while eight or more blocks gain 16-27% on the
// clustered and adaptive fixtures but lose 17-28% on the lattice at 8 and 12
// threads alike. The divisor therefore grows as the cube root of the
// particle count from two blocks per multiprocessor at 32,768 particles to
// six at 884,736 and holds the endpoint values outside that range;
// compute.cudaKickHandoffBlocks pins it for experiments. The CPU walk calls
// this at every node, so the device query is cached.
int kickBlockCount(ParticleInteger particles) {
    int const handoff = getOptions().compute.cudaKickHandoffBlocks;
    if (handoff > 0) {
        return handoff;
    }
    static int const referenceBlocks = kKickHandoffBlocksPerMultiprocessor * cudaSmpCount();
    double const scale = std::clamp(
        std::cbrt(double(particles) / kKickHandoffReferenceParticles), 1.0, kKickHandoffMaxScale);
    addCpuFlops(4);    // Division, cube root, scaling, and rounding in automatic handoff sizing.
    return int(std::lround(referenceBlocks * scale));
}

// Handoff divisor for a node whose Ewald checklist is empty. A finer cut
// gains where uneven workitems set the kernel's tail, but a CPU node that
// still carries Ewald checks evaluates them as cell-cell interactions at
// about 4 us each, while the kernel absorbs them at the workitem roots at no
// measurable cost: on the uniform lattice at 2,097,152 particles the Ewald
// lists resolve at the box/8 cells, and a cut below them made the kick
// 0.82x of the coarse one (docs/cuda_kick_shared_memory.md, "Where the finer
// division loses on the lattice"). The CPU walk therefore applies this
// divisor only to nodes without Ewald checks and kickBlockCount to the rest.
// compute.cudaKickFineHandoffBlocks sets it; zero, or a count below
// kickBlockCount, leaves every node at kickBlockCount. It is off by default:
// at 2,097,152 particles fine divisors of 656-1312 keep the lattice at the
// coarse cut and gain 1.06-1.15x on the clustered fixture, but lose 5-14% on
// the adaptive one, whose large Ewald-carrying cells stay whole while the
// finer cut makes more workitems than resident blocks, and the kernel takes
// workitems in tree order ("Ewald-aware handoff" in the same document).
int kickFineBlockCount(ParticleInteger particles) {
    return std::max(kickBlockCount(particles), getOptions().compute.cudaKickFineHandoffBlocks);
}

// The CUDA operators are order templates in gravity_cuda.hpp; this unit owns
// the interaction counters they share.

__managed__ double iclose;
__managed__ double idirect;
__managed__ double numberOfProcessedSinkParticles;
__managed__ bool useGravityCounters = false;

void resetGravityCounters() {
    if (!cudaEnabled()) {
        return;
    }
    iclose = idirect = numberOfProcessedSinkParticles = 0;
}

void setGravityCounterUse(bool code) {
    if (!cudaEnabled()) {
        return;
    }
    useGravityCounters = code;
}

void getGravityCounters(double& close, double& direct) {
    if (!cudaEnabled()) {
        close = direct = 0.0;
        return;
    }
    if (numberOfProcessedSinkParticles == 0.0) {
        close = direct = 0.0;
    } else {
        close = iclose / numberOfProcessedSinkParticles;
        direct = idirect / numberOfProcessedSinkParticles;
    }
}
