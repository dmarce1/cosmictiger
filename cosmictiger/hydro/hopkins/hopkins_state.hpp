/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#pragma once

#include <array>
#include <cstdint>
#include <type_traits>

#include <cosmictiger/math/vector3.hpp>

// Own-step conserved anchor and conservative transfers from completed pair intervals.
// Pending transfers never replace an inactive cell's reconstruction/trajectory anchor.
struct HopkinsParticleState {
    uint64_t identifier = UINT64_MAX;
    double mass = 0.0;
    double totalEnergy = 0.0;
    // Dual-energy auxiliary: the cell's entropic function P / rho^gamma in the code's
    // comoving variables (Bryan et al. 1995). MFM carries it unchanged between
    // closures; the expansion source, the thermal sources and the update switch refresh it.
    double entropy = 0.0;
    Vector3<double> momentum{};
    double pendingMass = 0.0;
    double pendingEnergy = 0.0;
    Vector3<double> pendingMomentum{};
    double anchorVolume = 0.0;
    Vector3<double> meshVelocity{};
    Vector3<double> meshAcceleration{};
    Vector3<double> sourceAcceleration{};
    Vector3<double> pendingGravityIntegral{};
    Vector3<double> pendingGravityWork{};
    double sourceTime = 0.0;
    double endTime = 0.0;
    double expansionCoefficient = 0.0;
    uint64_t beginTick = 0;
    uint64_t endTick = 0;
    bool initialized = false;
    std::array<Vector3<double>, 5> gradients{};
    bool predictorInitialized = false;
    // Coordinates live only in the authoritative shared Fixed64 particle array.
    double supportRadius = 0.0;

    // Conserved accounting includes transfers already accepted by either endpoint.
    // Reconstruction continues to use the own-step anchor above.
    CUDA_EXPORT double accountedMass() const {
        return mass + pendingMass;
    }

    CUDA_EXPORT double accountedEnergy() const {
        return totalEnergy + pendingEnergy;
    }

    CUDA_EXPORT double accountedMomentum(int dimension) const {
        return momentum[dimension] + pendingMomentum[dimension];
    }

    CUDA_EXPORT double driftVelocity(int dimension, double time) const {
        return meshVelocity[dimension] + (time - sourceTime) * meshAcceleration[dimension];
    }

    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & identifier & mass & totalEnergy & entropy & momentum & pendingMass &
            pendingEnergy & pendingMomentum & anchorVolume & meshVelocity & meshAcceleration &
            sourceAcceleration & pendingGravityIntegral & pendingGravityWork & sourceTime &
            endTime & expansionCoefficient & beginTick & endTick & initialized & gradients &
            predictorInitialized & supportRadius;
    }
};

static_assert(std::is_trivially_copyable_v<HopkinsParticleState>);
static_assert(std::is_standard_layout_v<HopkinsParticleState>);

static_assert(sizeof(Vector3<double>) == sizeof(std::array<double, 3>));
static_assert(alignof(Vector3<double>) == alignof(std::array<double, 3>));
