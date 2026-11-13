/* CosmicTiger - common particle permutation; BSL-1.0. */

#include <algorithm>
#include <cstring>
#include <tuple>
#include <type_traits>
#include <utility>

#include <cosmictiger/particles/particle_partition.hpp>
#if !COSMICTIGER_WITH_SPH
#include <cosmictiger/soa_iterator.hpp>
#endif
#if COSMICTIGER_WITH_SPH
#include <cosmictiger/sph/particles.hpp>
#endif

namespace detail {

/// Swap two trivially copyable records through byte storage, preserving all representation
/// bits.
template <class Value>
void swapParticleBytes(Value& first, Value& second) {
    static_assert(std::is_trivially_copyable_v<Value>);
    unsigned char temporary[sizeof(Value)];
    std::memcpy(temporary, &first, sizeof(Value));
    std::memcpy(&first, &second, sizeof(Value));
    std::memcpy(&second, temporary, sizeof(Value));
}

#if !COSMICTIGER_WITH_SPH
template <bool acceleration, bool potential, size_t... dimensions>
ParticleInteger partitionParticleArrays(Pair<ParticleInteger> range, Position split, int dimension,
    std::index_sequence<dimensions...>) {
    auto arrays = std::tuple{particlesX[dimensions]..., particlesV, particlesR};
    auto forceArrays = [] {
        if constexpr (acceleration) {
            return std::tuple{particlesGravityAccelerations[dimensions]...};
        } else {
            return std::tuple{};
        }
    }();
    auto potentialArrays = [] {
        if constexpr (potential) {
            return std::tuple{particlesP};
        } else {
            return std::tuple{};
        }
    }();
    auto const first = std::apply([](auto*... fields) { return soa::Iterator(fields...); },
        std::tuple_cat(arrays, forceArrays, potentialArrays));
    // Select the axis once, outside the element loop. All coordinates are fields
    // of the same proxy, so iter_swap moves the entire particle together.
    auto partition = [&]<size_t axis>() {
        auto const tail = std::ranges::partition(first + range.first, first + range.second,
            [split](auto const& row) { return get<axis>(row) < split; });
        return ParticleInteger(tail.begin() - first);
    };
    ParticleInteger result = range.first;
    ((dimension == int(dimensions) ?
             (result = partition.template operator()<dimensions>(), void()) :
             void()),
        ...);
    return result;
}
#endif

}    // namespace detail

/// Partition [first, second), preserving every enabled particle attribute.
ParticleInteger particlesPartition(Pair<ParticleInteger> range, Position split, int dimension) {
#if !COSMICTIGER_WITH_SPH
    // Repeated tree builds usually retain the previous partition. Scan those
    // prefixes/suffixes using just the key array before constructing row proxies
    // or dispatching on optional fields. No particle fields move in this scan.
    auto const* positions = particlesX[dimension];
    while (range.first < range.second && positions[range.first] < split) {
        ++range.first;
    }
    while (range.first < range.second && positions[range.second - 1] >= split) {
        --range.second;
    }
    if (range.first == range.second) {
        return range.first;
    }
    auto const& options = getOptions();
    // Species maps, atomic group labels and uncommon analysis sidecars continue
    // through particlesSwap until their iterator adapters are implemented.
    if (!options.hydro.enabled && !options.starFormation.enabled && !options.analysis.groups &&
        !options.analysis.tracers && options.gravity.errorInterval <= 0 &&
        particlesGrp == nullptr) {
        auto const dimensions = std::make_index_sequence<kSpatialDimensions>{};
        if (saveParticleAcceleration(options.analysis)) {
            if (saveParticlePotential(options.analysis)) {
                return detail::partitionParticleArrays<true, true>(
                    range, split, dimension, dimensions);
            }
            return detail::partitionParticleArrays<true, false>(
                range, split, dimension, dimensions);
        }
        if (saveParticlePotential(options.analysis)) {
            return detail::partitionParticleArrays<false, true>(
                range, split, dimension, dimensions);
        }
        return detail::partitionParticleArrays<false, false>(range, split, dimension, dimensions);
    }
#endif
    // Keep the original scan and the full linked-field swap for SPH builds and
    // optional sidecars. In particular, no element is read for an empty range.
    ParticleInteger lo = range.first;
    ParticleInteger hi = range.second;
    auto* fallbackPositions = particlesX[dimension];
    while (lo < hi) {
        if (fallbackPositions[lo] >= split) {
            while (lo != hi) {
                --hi;
                if (fallbackPositions[hi] < split) {
                    particlesSwap(hi, lo);
                    break;
                }
            }
        }
        ++lo;
    }
    return hi;
}

/// Swap every enabled particle field and maintain linked species/catalogue state during
/// reordering.
void particlesSwap(ParticleInteger first, ParticleInteger second) {
    if (first == second) {
        return;
    }
    CHECK_PART_BOUNDS(first);
    CHECK_PART_BOUNDS(second);
    for (int dim = 0; dim < kSpatialDimensions; ++dim) {
        detail::swapParticleBytes(particlesX[dim][first], particlesX[dim][second]);
    }
    detail::swapParticleBytes(particlesV[first], particlesV[second]);
    detail::swapParticleBytes(particlesR[first], particlesR[second]);
    auto const& options = getOptions();
    if (saveParticleAcceleration(options.analysis)) {
        for (int dim = 0; dim < kSpatialDimensions; ++dim) {
            detail::swapParticleBytes(particlesGravityAccelerations[dim][first],
                particlesGravityAccelerations[dim][second]);
        }
    }

    if (saveParticlePotential(options.analysis)) {
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
    if (options.gravity.errorInterval > 0) {
        detail::swapParticleBytes(
            particlesGravitySampleIndices[first], particlesGravitySampleIndices[second]);
    }
    if (options.hydro.enabled) {
        detail::swapParticleBytes(particlesSph[first], particlesSph[second]);
        detail::swapParticleBytes(particlesTy[first], particlesTy[second]);
#if COSMICTIGER_WITH_SPH
        // Keep the gas-to-particle map in step under every integrator: an SPH tree build
        // rebuilds each particle's gas index from this map (sphParticlesResolveWithParticles),
        // so a stale entry pairs gas fields with another particle, for example after the
        // domain rebound partitions and before the gas migration.
        if (particlesTy[first] == kSphType) {
            sphParticlesParticleIndex(particlesSph[first]) = first;
        }
        if (particlesTy[second] == kSphType) {
            sphParticlesParticleIndex(particlesSph[second]) = second;
        }
#endif
    }
    if (options.starFormation.enabled) {
        detail::swapParticleBytes(
            particlesStarFormationStates[first], particlesStarFormationStates[second]);
    }
}
