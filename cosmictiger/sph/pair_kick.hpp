/* CosmicTiger - conservative SPH pair impulses; BSL-1.0. */
#pragma once

#include <atomic>

#include <cosmictiger/cuda/cuda_export.hpp>
#include <cosmictiger/math/vector3.hpp>
#include <cosmictiger/particles/particles.hpp>

// Event-local identities remain valid until the frozen pass and its reverse
// exchange finish. No particle migration or reordering may intervene.
struct SphPairSourceId {
    int rank;
    ParticleInteger index;

    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & rank;
        archive & index;
    }
};

// Scratch increments, not force history. Double accumulation limits reduction
// roundoff; committing to the particle's float velocity still rounds once.
struct SphPairUpdate {
    double velocity[kDimensions] = {};
    double entropy = 0.0;

    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & velocity;
        archive & entropy;
    }
};

CUDA_EXPORT inline bool sphOwnsPairKick(SphPairSourceId const& target,
    SphPairSourceId const& neighbor, int targetRung, int neighborRung) {
    if (targetRung != neighborRung) {
        return targetRung > neighborRung;
    }
    return target.rank < neighbor.rank ||
        (target.rank == neighbor.rank && target.index < neighbor.index);
}

CUDA_EXPORT inline void sphPairAtomicAdd(double& destination, double increment) {
#ifdef __CUDA_ARCH__
    atomicAdd(&destination, increment);
#else
    std::atomic_ref<double>(destination).fetch_add(increment, std::memory_order_relaxed);
#endif
}

// clang-format off
/*
 * Derived discrete kick for the antisymmetric SPH pair force of Hopkins (2013),
 * equation (12), https://arxiv.org/abs/1206.5006:
 *   Δv_a = Δτ a_ab, Δv_b = −Δv_a, ΔA_a = Δτ Ȧ_ab, ΔA_b = Δτ Ȧ_ba.
 * All production gas masses are equal. Evaluate a pair once at the midpoint of
 * its finer endpoint's interval and reuse the identical impulse with opposite
 * signs. Only nonadiabatic energy exchange contributes to Ȧ; directed viscous heating
 * can differ between endpoints. The scheduling and accumulation are local
 * derivations, not a time integrator prescribed by the cited paper.
 * Symbol directory:
 *   Δτ → timestep; a_ab → acceleration; Ȧ_ab/Ȧ_ba → target/neighborEntropyRate;
 *   Δv_a → increment; a/b → target/neighbor.
 */
// clang-format on
CUDA_EXPORT inline void sphAccumulatePairKick(SphPairUpdate& target, SphPairUpdate& neighbor,
    Vector3<float> const& acceleration, double targetEntropyRate, double neighborEntropyRate,
    double timestep) {
    for (int dimension = 0; dimension < kDimensions; ++dimension) {
        double const increment = timestep * double(acceleration[dimension]);
        sphPairAtomicAdd(target.velocity[dimension], increment);
        sphPairAtomicAdd(neighbor.velocity[dimension], -increment);
    }
    sphPairAtomicAdd(target.entropy, timestep * targetEntropyRate);
    sphPairAtomicAdd(neighbor.entropy, timestep * neighborEntropyRate);
}

CUDA_EXPORT inline void sphAccumulateExternalKick(
    SphPairUpdate& target, Vector3<float> const& acceleration, double timestep) {
    for (int dimension = 0; dimension < kDimensions; ++dimension) {
        sphPairAtomicAdd(target.velocity[dimension], timestep * double(acceleration[dimension]));
    }
}
