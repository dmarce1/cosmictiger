/*
CosmicTiger - A cosmological N-Body code
Copyright (C) 2021  Dominic C. Marcello

Distributed under the Boost Software License, Version 1.0.
See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
*/

#pragma once

#ifndef __CUDACC__

#include <hpx/async_base/scheduling_properties.hpp>
#include <hpx/hpx.hpp>
#include <hpx/hpx_finalize.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/run_as.hpp>
#include <hpx/modules/executors.hpp>
#include <hpx/parallel/algorithms/copy.hpp>
#include <hpx/parallel/algorithms/fill.hpp>
#include <hpx/parallel/algorithms/sort.hpp>
#include <hpx/serialization/unordered_map.hpp>

#endif

#include <cosmictiger/core/containers.hpp>

#ifndef __CUDACC__

#define HPX_PRIORITY_HI hpx::execution::parallel_executor(hpx::threads::thread_priority::high)
#define PAR_EXECUTION_POLICY hpx::execution::par(hpx::execution::task)
#define hpx_copy hpx::copy
#define hpx_fill hpx::fill

// Propagate the phase name to the tasks generated inside parallel algorithms.
inline auto annotatedParallelPolicy(char const* name) {
    auto policy = hpx::execution::par(hpx::execution::task);
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
    return policy.on(hpx::execution::experimental::with_annotation(policy.executor(), name));
#else
    (void) name;
    return policy;
#endif
}

Vector<hpx::id_type> const& hpxLocalities();
Vector<hpx::id_type> const& hpxChildren();
void hpxInit();
void hpxInitCuda();

using MutexType = hpx::mutex;
using SpinlockType = hpx::spinlock;
using SharedMutexType = hpx::shared_mutex;

#endif

int hpxRank();
int hpxSize();
void hpxYield();
int hpxHardwareConcurrency();
