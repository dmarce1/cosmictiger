/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

#include <exception>
#include <memory>
#include <mutex>
#include <unordered_map>

#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/sph/sph_particles.hpp>

namespace detail {

struct SphCacheLineId {
    int proc;
    ParticleInteger index;
    inline bool operator==(SphCacheLineId other) const {
        return proc == other.proc && index == other.index;
    }
};

struct SphCacheLineHash {
    inline size_t operator()(SphCacheLineId id) const {
        ParticleInteger const lineSize = getOptions().compute.particleCacheLineSize;
        ParticleInteger const i = id.index / lineSize;
        return i * (hpxSize() - 1) + ((id.proc < hpxRank()) ? id.proc : id.proc - 1);
    }
};

struct SphCacheLineHashLo {
    inline size_t operator()(SphCacheLineId id) const {
        SphCacheLineHash hash;
        return hash(id) % kParticleCacheSize;
    }
};

struct SphCacheLineHashHi {
    inline size_t operator()(SphCacheLineId id) const {
        SphCacheLineHash hash;
        return hash(id) / kParticleCacheSize;
    }
};

template <typename Action, typename T>
T const* readSphCacheLine(SphCacheLineId lineId,
    Array<std::unordered_map<SphCacheLineId, std::shared_ptr<hpx::shared_future<Vector<T>>>,
              SphCacheLineHashHi>,
        kParticleCacheSize>& cache,
    Array<SpinlockType, kParticleCacheSize>& locks) {
    size_t const bin = SphCacheLineHashLo()(lineId);
    std::shared_ptr<hpx::shared_future<Vector<T>>> future;
    {
        std::lock_guard<SpinlockType> lock(locks[bin]);
        auto const entry = cache[bin].find(lineId);
        if (entry != cache[bin].end()) {
            future = entry->second;
        }
    }
    if (!future) {
        auto promise = std::make_shared<hpx::promise<Vector<T>>>();
        auto candidate =
            std::make_shared<hpx::shared_future<Vector<T>>>(promise->get_future().share());
        bool inserted;
        {
            std::lock_guard<SpinlockType> lock(locks[bin]);
            auto const entry = cache[bin].emplace(lineId, candidate);
            inserted = entry.second;
            future = entry.first->second;
        }
        if (inserted) {
            try {
                hpx::post([promise, lineId]() {
                    try {
                        auto remote =
                            hpx::async<Action>(hpxLocalities()[lineId.proc], lineId.index);
                        promise->set_value(remote.get());
                    } catch (...) {
                        promise->set_exception(std::current_exception());
                    }
                });
            } catch (...) {
                promise->set_exception(std::current_exception());
            }
        }
    }
    return future->get().data();
}

}    // namespace detail
