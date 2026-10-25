/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <mutex>

#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/gravity/kick.hpp>

#ifdef USE_CUDA
KickReturn kickRunCudaLaunch(std::function<KickReturn()> const& launch) {
    // The per-order launch packet and list slab are reused across launches.
    // Tickets preserve submission order without holding a mutex over an HPX
    // event wait, so later ready batches cannot starve an earlier packet.
    static MutexType launchMutex;
    static hpx::shared_future<void> launchTail = hpx::make_ready_future().share();
    hpx::promise<void> completion;
    auto const completed = completion.get_future().share();
    hpx::shared_future<void> predecessor;
    {
        std::lock_guard<MutexType> lock(launchMutex);
        predecessor = launchTail;
        launchTail = completed;
    }
    KickReturn result;
    try {
        predecessor.wait();
        result = launch();
    } catch (...) {
        // The launch's stream guard has drained before the next ticket runs.
        completion.set_value();
        throw;
    }
    completion.set_value();
    return result;
}
#endif
