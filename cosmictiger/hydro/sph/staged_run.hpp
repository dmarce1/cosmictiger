/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

#include <cstddef>

#include <cosmictiger/hydro/sph/sph_kernel.hpp>

// Run existing SPH kernels on caller-owned, already-packed host arrays. Sources
// are an immutable snapshot for this call and must not alias mutable sinks. Each
// target leaf maps its particleRange in the source arrays to a disjoint, equally
// sized sinkParticleRange in the packed sink arrays. sinkParticleIndices is the
// identity map into sinkParticleRungs; both have sinkCount entries. Optional
// arrays retain the null-pointer semantics of SphRunData and the selected phase.
//
// The caller keeps all host arrays alive until this synchronous function returns.
// CPU workers are joined on failure. CUDA stages sources and sinks separately,
// copies modified sink arrays back, and drains the stream before freeing storage.
// No global particle arrays, trees, sorting, or source preparation are performed.
SphRunReturn sphRunStaged(SphRunParameters const& parameters, SphRunData data, size_t sourceCount,
    size_t sinkCount, size_t treeCount, size_t neighborCount, bool useCuda);
