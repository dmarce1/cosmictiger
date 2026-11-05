/* CosmicTiger - conservative SPH pair impulses; BSL-1.0. */
#pragma once

#include <atomic>
#include <bit>
#include <cstdint>

#include <cosmictiger/core/flops.hpp>
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

// A sum and its Kahan correction are two floats packed into one integer CAS.
// Keeping both in the same atomic transaction prevents lost corrections when
// CPU workers or CUDA threads update the same particle. No FP64 arithmetic.
class SphCompensatedSum {
    using Bits = unsigned long long;
    alignas(8) Bits bits_ = 0;

    CUDA_EXPORT static uint32_t floatBits(float value) {
#ifdef __CUDA_ARCH__
        return __float_as_uint(value);
#else
        return std::bit_cast<uint32_t>(value);
#endif
    }
    CUDA_EXPORT static float fromBits(uint32_t bits) {
#ifdef __CUDA_ARCH__
        return __uint_as_float(bits);
#else
        return std::bit_cast<float>(bits);
#endif
    }

public:
    // Read only after the accumulation pass has joined/synchronized.
    CUDA_EXPORT operator float() const {
        return fromBits(uint32_t(bits_));
    }
    CUDA_EXPORT float correction() const {
        return fromBits(uint32_t(bits_ >> 32));
    }
    CUDA_EXPORT bool isZero() const {
        return float(*this) == 0.0f && correction() == 0.0f;
    }

    // Kahan (1965), Further remarks on reducing truncation errors, CACM 8(1), 40,
    // https://doi.org/10.1145/363707.363723 (unnumbered algorithm).
    // Derived atomic application, with the opposite sign convention for c:
    //   y = x − c; t = s + y; c′ = (t − s) − y; s′ = t.
    // Symbol directory: x -> increment, c -> oldCorrection, s -> oldSum,
    // y -> adjusted, t -> nextSum, c′ -> nextCorrection.
    // Explicit round-to-nearest CUDA operations and volatile CPU intermediates
    // preserve the four separate operations under the production fast-math flags.
    // Return actual FP32 work, including attempts retried after contention.
    CUDA_EXPORT FlopCount add(float increment) {
#ifdef __CUDA_ARCH__
        Bits expected = atomicCAS(&bits_, Bits(0), Bits(0));
#else
        std::atomic_ref<Bits> destination(bits_);
        Bits expected = destination.load(std::memory_order_relaxed);
#endif
        FlopCount operations = 0;
        for (;;) {
            float const oldSum = fromBits(uint32_t(expected));
            float const oldCorrection = fromBits(uint32_t(expected >> 32));
#ifdef __CUDA_ARCH__
            float const adjusted = __fsub_rn(increment, oldCorrection);
            float const nextSum = __fadd_rn(oldSum, adjusted);
            float const nextCorrection = __fsub_rn(__fsub_rn(nextSum, oldSum), adjusted);
#else
            float volatile const adjusted = increment - oldCorrection;
            float const volatile nextSum = oldSum + adjusted;
            float const volatile rounding = nextSum - oldSum;
            float const nextCorrection = rounding - adjusted;
#endif
            Bits const desired = Bits(floatBits(nextSum)) | (Bits(floatBits(nextCorrection)) << 32);
            operations += 4;
#ifdef __CUDA_ARCH__
            Bits const observed = atomicCAS(&bits_, expected, desired);
            if (observed == expected) {
                return operations;
            }
            expected = observed;
#else
            if (destination.compare_exchange_weak(
                    expected, desired, std::memory_order_relaxed, std::memory_order_relaxed)) {
                return operations;
            }
#endif
        }
    }

    // A partial sum represents s-c. Preserve its residual across leaf/rank
    // merges by feeding both terms through the same compensated accumulator.
    CUDA_EXPORT FlopCount merge(SphCompensatedSum const& other) {
        FlopCount const operations = add(float(other));
        return operations + add(-other.correction());
    }

    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & bits_;
    }
};

static_assert(sizeof(SphCompensatedSum) == 2 * sizeof(float));

// Event-local scratch; both the sum and compensation travel in reverse exchanges.
struct SphPairUpdate {
    SphCompensatedSum velocity[kDimensions] = {};
    SphCompensatedSum entropy;

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
CUDA_EXPORT inline FlopCount sphAccumulatePairKick(SphPairUpdate& target, SphPairUpdate& neighbor,
    Vector3<float> const& acceleration, float targetEntropyRate, float neighborEntropyRate,
    float timestep) {
    FlopCount operations = kDimensions + 2;    // Impulse and entropy products.
    for (int dimension = 0; dimension < kDimensions; ++dimension) {
        float const increment = timestep * acceleration[dimension];
        operations += target.velocity[dimension].add(increment);
        operations += neighbor.velocity[dimension].add(-increment);
    }
    operations += target.entropy.add(timestep * targetEntropyRate);
    operations += neighbor.entropy.add(timestep * neighborEntropyRate);
    return operations;
}

CUDA_EXPORT inline FlopCount sphAccumulateExternalKick(
    SphPairUpdate& target, Vector3<float> const& acceleration, float timestep) {
    FlopCount operations = kDimensions;
    for (int dimension = 0; dimension < kDimensions; ++dimension) {
        operations += target.velocity[dimension].add(timestep * acceleration[dimension]);
    }
    return operations;
}
