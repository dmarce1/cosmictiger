/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#include <algorithm>
#include <cstring>
#include <limits>
#include <mutex>
#include <type_traits>
#include <utility>
#include <vector>

#include <cosmictiger/particles/particle_mesh_sort.hpp>
#include <cosmictiger/sph/sph_particles.hpp>

namespace detail {

template <class Value>
void swapParticleBytes(Value& first, Value& second) {
    static_assert(std::is_trivially_copyable_v<Value>);
    unsigned char temporary[sizeof(Value)];
    std::memcpy(temporary, &first, sizeof(Value));
    std::memcpy(&first, &second, sizeof(Value));
    std::memcpy(&second, temporary, sizeof(Value));
}

template <class Function>
void particleMeshParallelFor(size_t count, size_t workers, Function const& function) {
    workers = std::min(count, workers);
    if (workers <= 1) {
        for (size_t i = 0; i < count; ++i) {
            function(i);
        }
        return;
    }
    std::vector<hpx::future<void>> futures;
    futures.reserve(workers);
    try {
        for (size_t worker = 0; worker < workers; ++worker) {
            futures.push_back(hpx::async([&, worker] {
                for (size_t i = worker; i < count; i += workers) {
                    function(i);
                }
            }));
        }
    } catch (...) {
        // References captured by already launched work must outlive that work.
        hpx::wait_all(futures.begin(), futures.end());
        throw;
    }
    hpx::wait_all(futures.begin(), futures.end());
    for (auto& future : futures) {
        future.get();
    }
}

size_t particleMeshChunkBoundary(size_t count, size_t chunk, size_t chunks) {
    return (count / chunks) * chunk + std::min(count % chunks, chunk);
}

uint64_t particleMeshKey(size_t index, uint32_t meshSize) {
    uint64_t const x = particleMeshCoordinate(particlesX[XDIM][index].raw(), meshSize);
    uint64_t const y = particleMeshCoordinate(particlesX[YDIM][index].raw(), meshSize);
    uint64_t const z = particleMeshCoordinate(particlesX[ZDIM][index].raw(), meshSize);
    return (x * meshSize + y) * meshSize + z;
}

struct ParticleMeshPartition {
    size_t begin;
    size_t middle;
    size_t end;
    uint64_t lowerMinimum = std::numeric_limits<uint64_t>::max();
    uint64_t lowerMaximum = 0;
    uint64_t upperMinimum = std::numeric_limits<uint64_t>::max();
    uint64_t upperMaximum = 0;
};

ParticleMeshPartition partitionParticleMeshRange(
    size_t begin, size_t end, uint64_t split, uint32_t meshSize) {
    ParticleMeshPartition result{begin, begin, end};
    auto const record = [&](uint64_t key) {
        auto& minimum = key < split ? result.lowerMinimum : result.upperMinimum;
        auto& maximum = key < split ? result.lowerMaximum : result.upperMaximum;
        minimum = std::min(minimum, key);
        maximum = std::max(maximum, key);
    };
    size_t lower = begin;
    size_t upper = end;
    while (lower < upper) {
        uint64_t const key = particleMeshKey(lower, meshSize);
        record(key);
        if (key < split) {
            ++lower;
        } else {
            while (lower < upper) {
                --upper;
                if (upper == lower) {
                    break;
                }
                uint64_t const otherKey = particleMeshKey(upper, meshSize);
                record(otherKey);
                if (otherKey < split) {
                    particlesSwap(
                        static_cast<ParticleInteger>(lower), static_cast<ParticleInteger>(upper));
                    ++lower;
                    break;
                }
            }
        }
    }
    result.middle = lower;
    return result;
}

void sortParticleMeshRange(size_t begin, size_t end, uint64_t minimum, uint64_t maximum,
    size_t workers, uint32_t meshSize) {
    size_t const count = end - begin;
    if (count < 2 || minimum >= maximum) {
        return;
    }
    constexpr size_t kMinimumChunkSize = 2048;
    workers = std::min(workers, std::max(size_t(1), count / kMinimumChunkSize));
    uint64_t const split = minimum + (maximum - minimum) / 2 + 1;
    if (workers == 1) {
        auto const partition = partitionParticleMeshRange(begin, end, split, meshSize);
        sortParticleMeshRange(
            begin, partition.middle, partition.lowerMinimum, partition.lowerMaximum, 1, meshSize);
        sortParticleMeshRange(
            partition.middle, end, partition.upperMinimum, partition.upperMaximum, 1, meshSize);
        return;
    }

    // Each chunk is first partitioned independently. Only O(workers) boundary
    // metadata is needed to repair the combined partition in place.
    std::vector<ParticleMeshPartition> partitions(workers);
    particleMeshParallelFor(workers, workers, [&](size_t chunk) {
        size_t const chunkBegin = begin + particleMeshChunkBoundary(count, chunk, workers);
        size_t const chunkEnd = begin + particleMeshChunkBoundary(count, chunk + 1, workers);
        partitions[chunk] = partitionParticleMeshRange(chunkBegin, chunkEnd, split, meshSize);
    });
    size_t middle = begin;
    uint64_t lowerMinimum = std::numeric_limits<uint64_t>::max();
    uint64_t upperMinimum = std::numeric_limits<uint64_t>::max();
    uint64_t lowerMaximum = 0;
    uint64_t upperMaximum = 0;
    for (auto const& partition : partitions) {
        middle += partition.middle - partition.begin;
        lowerMinimum = std::min(lowerMinimum, partition.lowerMinimum);
        lowerMaximum = std::max(lowerMaximum, partition.lowerMaximum);
        upperMinimum = std::min(upperMinimum, partition.upperMinimum);
        upperMaximum = std::max(upperMaximum, partition.upperMaximum);
    }
    using IndexRange = std::pair<size_t, size_t>;
    std::vector<IndexRange> misplacedLower;
    std::vector<IndexRange> misplacedUpper;
    misplacedLower.reserve(workers);
    misplacedUpper.reserve(workers);
    for (auto const& partition : partitions) {
        size_t const lowerEnd = std::min(partition.end, middle);
        if (partition.middle < lowerEnd) {
            misplacedLower.emplace_back(partition.middle, lowerEnd);
        }
        size_t const upperBegin = std::max(partition.begin, middle);
        if (upperBegin < partition.middle) {
            misplacedUpper.emplace_back(upperBegin, partition.middle);
        }
    }
    struct SwapRange {
        size_t first;
        size_t second;
        size_t count;
    };
    std::vector<SwapRange> swaps;
    swaps.reserve(2 * workers);
    size_t lowerRun = 0;
    size_t upperRun = 0;
    while (lowerRun < misplacedLower.size() && upperRun < misplacedUpper.size()) {
        auto& lower = misplacedLower[lowerRun];
        auto& upper = misplacedUpper[upperRun];
        size_t const length = std::min(lower.second - lower.first, upper.second - upper.first);
        swaps.push_back({lower.first, upper.first, length});
        lower.first += length;
        upper.first += length;
        lowerRun += lower.first == lower.second;
        upperRun += upper.first == upper.second;
    }
    // The paired ranges are disjoint, so no atomics or particle locks are needed.
    particleMeshParallelFor(swaps.size(), workers, [&](size_t run) {
        auto const& swap = swaps[run];
        for (size_t i = 0; i < swap.count; ++i) {
            particlesSwap(static_cast<ParticleInteger>(swap.first + i),
                static_cast<ParticleInteger>(swap.second + i));
        }
    });
    if (middle == begin) {
        sortParticleMeshRange(begin, end, upperMinimum, upperMaximum, workers, meshSize);
    } else if (middle == end) {
        sortParticleMeshRange(begin, end, lowerMinimum, lowerMaximum, workers, meshSize);
    } else {
        size_t const lowerWorkers =
            std::clamp((middle - begin) / (count / workers + 1), size_t(1), workers - 1);
        particleMeshParallelFor(2, 2, [&](size_t side) {
            if (side == 0) {
                sortParticleMeshRange(
                    begin, middle, lowerMinimum, lowerMaximum, lowerWorkers, meshSize);
            } else {
                sortParticleMeshRange(
                    middle, end, upperMinimum, upperMaximum, workers - lowerWorkers, meshSize);
            }
        });
    }
}

}    // namespace detail

void particlesSwap(ParticleInteger first, ParticleInteger second) {
    if (first == second) {
        return;
    }
    CHECK_PART_BOUNDS(first);
    CHECK_PART_BOUNDS(second);
    for (int dim = 0; dim < NDIM; ++dim) {
        detail::swapParticleBytes(particlesX[dim][first], particlesX[dim][second]);
    }
    detail::swapParticleBytes(particlesV[first], particlesV[second]);
    detail::swapParticleBytes(particlesR[first], particlesR[second]);
    auto const& options = getOptions();
    if (options.analysis.saveForce) {
        for (int dim = 0; dim < NDIM; ++dim) {
            detail::swapParticleBytes(particlesGravityAccelerations[dim][first],
                particlesGravityAccelerations[dim][second]);
        }
        detail::swapParticleBytes(particlesP[first], particlesP[second]);
    }
    if (options.analysis.groups) {
        detail::swapParticleBytes(particlesLgrp[first], particlesLgrp[second]);
    }
    if (particlesGrp != nullptr) {
        auto const firstGroup = particlesGrp[first].load(std::memory_order_relaxed);
        auto const secondGroup = particlesGrp[second].load(std::memory_order_relaxed);
        particlesGrp[first].store(secondGroup, std::memory_order_relaxed);
        particlesGrp[second].store(firstGroup, std::memory_order_relaxed);
    }
    if (options.analysis.tracers) {
        detail::swapParticleBytes(particlesTr[first], particlesTr[second]);
    }
    if (options.hydro.enabled) {
        detail::swapParticleBytes(particlesSph[first], particlesSph[second]);
        detail::swapParticleBytes(particlesTy[first], particlesTy[second]);
    }
}

Vector<ParticleMeshCell> particleMeshSort(int meshSize) {
    uint64_t const cellCount = detail::particleMeshCellCount(meshSize);
    size_t const count = static_cast<size_t>(particlesSize());
    size_t const workers = static_cast<size_t>(std::max(1, hpxHardwareConcurrency()));
    std::unique_lock<SharedMutexType> lock(particlesSharedMutex());
    detail::sortParticleMeshRange(0, count, 0, cellCount - 1, workers, meshSize);
    if (getOptions().hydro.enabled) {
        // Gas arrays remain in their own ordering; rebuild their reverse map.
        detail::particleMeshParallelFor(workers, workers, [&](size_t worker) {
            size_t const begin = detail::particleMeshChunkBoundary(count, worker, workers);
            size_t const end = detail::particleMeshChunkBoundary(count, worker + 1, workers);
            for (size_t i = begin; i < end; ++i) {
                if (particlesType(i) == SPH_TYPE) {
                    ParticleInteger const gasIndex = particlesCatIndex(i);
                    if (gasIndex < 0 || gasIndex >= sphParticlesSize()) {
                        throw std::runtime_error("Invalid gas index after particle mesh sort");
                    }
                    sphParticlesParticleIndex(gasIndex) = static_cast<ParticleInteger>(i);
                }
            }
        });
    }
    particlesCacheFreeLocal();

    // Two parallel scans construct only the occupied-cell directory. There is
    // no particle-sized key array, permutation, or second particle allocation.
    std::vector<size_t> offsets(workers + 1, 0);
    detail::particleMeshParallelFor(workers, workers, [&](size_t worker) {
        size_t const begin = detail::particleMeshChunkBoundary(count, worker, workers);
        size_t const end = detail::particleMeshChunkBoundary(count, worker + 1, workers);
        uint64_t previous = begin == 0 ? std::numeric_limits<uint64_t>::max() :
                                         detail::particleMeshKey(begin - 1, meshSize);
        for (size_t i = begin; i < end; ++i) {
            uint64_t const key = detail::particleMeshKey(i, meshSize);
            offsets[worker + 1] += key != previous;
            previous = key;
        }
    });
    for (size_t worker = 0; worker < workers; ++worker) {
        offsets[worker + 1] += offsets[worker];
    }
    Vector<ParticleMeshCell> cells(offsets.back());
    detail::particleMeshParallelFor(workers, workers, [&](size_t worker) {
        size_t const begin = detail::particleMeshChunkBoundary(count, worker, workers);
        size_t const end = detail::particleMeshChunkBoundary(count, worker + 1, workers);
        uint64_t previous = begin == 0 ? std::numeric_limits<uint64_t>::max() :
                                         detail::particleMeshKey(begin - 1, meshSize);
        size_t entry = offsets[worker];
        for (size_t i = begin; i < end; ++i) {
            uint64_t const key = detail::particleMeshKey(i, meshSize);
            if (key != previous) {
                cells[entry++] = {key, static_cast<ParticleInteger>(i), 0};
            }
            previous = key;
        }
    });
    detail::particleMeshParallelFor(cells.size(), workers, [&](size_t entry) {
        cells[entry].end =
            entry + 1 < cells.size() ? cells[entry + 1].begin : static_cast<ParticleInteger>(count);
    });
    return cells;
}
