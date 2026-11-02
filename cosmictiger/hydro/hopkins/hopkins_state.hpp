/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#pragma once

#include <array>
#include <cstdint>
#include <type_traits>

// Persistent gas state for meshless kicks, independent of the positional drift.
struct HopkinsParticleState {
    uint64_t identifier = UINT64_MAX;
    double mass = 0.0;
    double totalEnergy = 0.0;
    std::array<double, 3> momentum{};
    std::array<double, 3> meshVelocity{};
    std::array<double, 3> sourceAcceleration{};
    std::array<double, 3> pendingGravityWork{};
    double sourceTime = 0.0;
    double expansionCoefficient = 0.0;
    bool initialized = false;
    // Primitive and gradient anchors for inactive-particle prediction: rho, P, vx, vy, vz.
    std::array<double, 5> predictor{};
    std::array<std::array<double, 3>, 5> gradients{};
    // Low 16 bits of the finest-clock tick at the primitive/gradient anchor.
    uint16_t lastKickTick = 0;
    bool predictorInitialized = false;
    // Coordinates live only in the authoritative shared Fixed64 particle array.
    // The tree retains a float support mirror, while the solver keeps this radius.
    double supportRadius = 0.0;

    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & identifier & mass & totalEnergy & momentum & meshVelocity & sourceAcceleration &
            pendingGravityWork & sourceTime & expansionCoefficient & initialized & predictor &
            gradients & lastKickTick & predictorInitialized & supportRadius;
    }
};

static_assert(std::is_trivially_copyable_v<HopkinsParticleState>);
static_assert(std::is_standard_layout_v<HopkinsParticleState>);
