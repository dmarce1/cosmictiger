/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

#include <cstdint>

#include <cosmictiger/core/assert.hpp>
#include <cosmictiger/core/defs.hpp>
#include <cosmictiger/cuda/cuda_export.hpp>

// One byte per gas particle, with a fixed checkpoint/migration layout:
// bit 0 = density convergence, bit 1 = semiactive, bits 2–7 = proposed hydro rung.
// All proposals are actual nonnegative rungs; there is no unset sentinel.
// Explicit masks avoid implementation-defined bit-field ordering and references.
// Writers own one particle at a time; CUDA threads synchronize around shared reads.
struct SphParticleFlags {
    uint8_t bits = 0;

    /// Read the packed smoothing-solve convergence flag.
    CUDA_EXPORT bool converged() const {
        return (bits & 1U) != 0;
    }

    /// Read the packed neighbor-induced activity flag.
    CUDA_EXPORT bool semiactive() const {
        return (bits & 2U) != 0;
    }

    /// Decode the timestep-rung proposal from the upper flag bits.
    CUDA_EXPORT int proposedRung() const {
        return bits >> 2;
    }

    /// Test whether the encoded rung proposal lies below the supported maximum.
    CUDA_EXPORT bool valid() const {
        return proposedRung() < kMaxRung;
    }

    /// Update the convergence bit while preserving the other particle flags.
    CUDA_EXPORT void setConverged(bool value) {
        bits = uint8_t((bits & ~1U) | uint8_t(value));
    }

    /// Update the semiactive bit while preserving the other particle flags.
    CUDA_EXPORT void setSemiactive(bool value) {
        bits = uint8_t((bits & ~2U) | (uint8_t(value) << 1));
    }

    /// Validate and store a rung proposal while preserving convergence and activity
    /// bits.
    CUDA_EXPORT void setProposedRung(int rung) {
        ALWAYS_ASSERT(rung >= 0 && rung < kMaxRung);
        bits = uint8_t((bits & 3U) | (uint8_t(rung) << 2));
    }
};

static_assert(sizeof(SphParticleFlags) == 1);
static_assert(kMaxRung <= 64, "Packed SPH rungs must fit in six bits");
