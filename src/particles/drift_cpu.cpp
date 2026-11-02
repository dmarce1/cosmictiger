/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#include <cmath>

#if !defined(COSMICTIGER_WITH_GRAVITY) || COSMICTIGER_WITH_GRAVITY
#include <chealpix.h>
#endif

#include <cosmictiger/math/simd_math.hpp>
#include <cosmictiger/math/vector3.hpp>
#include <cosmictiger/particles/drift_cpu.hpp>

#if !defined(COSMICTIGER_WITH_GRAVITY) || COSMICTIGER_WITH_GRAVITY
namespace detail {

void appendDriftLightconeEntries(Vector3<double> const& before, Vector3<double> const& after,
    Vector3<float> const& velocity, Vector3<float> const& outputVelocity, char type,
    DriftParameters const& parameters, std::vector<LightconeEntry>& entries,
    HopkinsParticleState const* meshState) {
    double const inverseScale = 1.0 / parameters.scale;
    double const inverseTauMax = 1.0 / parameters.tauMax;
    // CUDA uses float physical velocities for the crossing interpolation.
    float const vx =
        (meshState != nullptr ? meshState->meshVelocity[kXDimension] : velocity[kXDimension]) *
        inverseScale;
    float const vy =
        (meshState != nullptr ? meshState->meshVelocity[kYDimension] : velocity[kYDimension]) *
        inverseScale;
    float const vz =
        (meshState != nullptr ? meshState->meshVelocity[kZDimension] : velocity[kZDimension]) *
        inverseScale;
    for (int imageX = -1; imageX <= 0; ++imageX) {
        double const x0 = before[kXDimension] + imageX;
        double const x1 = after[kXDimension] + imageX;
        for (int imageY = -1; imageY <= 0; ++imageY) {
            double const y0 = before[kYDimension] + imageY;
            double const y1 = after[kYDimension] + imageY;
            for (int imageZ = -1; imageZ <= 0; ++imageZ) {
                double const z0 = before[kZDimension] + imageZ;
                double const z1 = after[kZDimension] + imageZ;
                double const radius0 = std::sqrt(x0 * x0 + y0 * y0 + z0 * z0);
                double const radius1 = std::sqrt(x1 * x1 + y1 * y1 + z1 * z1);
                int const shell0 = (parameters.tau0 + radius0) * inverseTauMax;
                int const shell1 = (parameters.tau1 + radius1) * inverseTauMax;
                if (shell0 == shell1 || (radius0 > 1.0 && radius1 > 1.0)) {
                    continue;
                }
                double const crossingTau = (shell0 + 1.0) * parameters.tauMax;
                double const tauOffset = parameters.tau0 - crossingTau;
                double const quadratic = 1.0 - (vx * vx + vy * vy + vz * vz);
                double const linear = 2.0 * (tauOffset - (vx * x0 + vy * y0 + vz * z0));
                double const constant = tauOffset * tauOffset - (x0 * x0 + y0 * y0 + z0 * z0);
                double const time =
                    -(linear + std::sqrt(linear * linear - 4.0 * quadratic * constant)) /
                    (2.0 * quadratic);
                Vector3<double> crossing = {x0 + vx * time, y0 + vy * time, z0 + vz * time};
                double const radiusSquared = crossing[kXDimension] * crossing[kXDimension] +
                    crossing[kYDimension] * crossing[kYDimension] +
                    crossing[kZDimension] * crossing[kZDimension];
                if (radiusSquared <= 1.0) {
                    long pixel;
                    vec2pix_nest(parameters.mapSize, crossing.data(), &pixel);
                    LightconeEntry entry;
                    entry.x = crossing[kXDimension];
                    entry.y = crossing[kYDimension];
                    entry.z = crossing[kZDimension];
                    entry.vx = outputVelocity[kXDimension];
                    entry.vy = outputVelocity[kYDimension];
                    entry.vz = outputVelocity[kZDimension];
                    entry.pix = pixel;
                    entry.type = type;
                    if (meshState != nullptr) {
                        entry.mass = float(meshState->mass);
                    }
                    entries.push_back(entry);
                }
            }
        }
    }
}

}    // namespace detail
#endif

void cpuDriftKernel(Position* x, Position* y, Position* z, Vector3<float>* velocities,
    char const* rungs, char const* types, size_t count, DriftParameters const& parameters,
    std::vector<LightconeEntry>* entries, HopkinsParticleState* meshStates,
    ParticleInteger const* gasIndices) {
    if (count == 0) {
        return;
    }
#if !defined(COSMICTIGER_WITH_GRAVITY) || COSMICTIGER_WITH_GRAVITY
    bool const lightcone = entries != nullptr && parameters.lightcone && parameters.tauMax > 0.0 &&
        parameters.tauMax - parameters.tau1 < 1.0;
#else
    bool const lightcone = false;
#endif
    if (parameters.timestep == 0.0 && !lightcone) {
        return;
    }
    double const driftFactor = parameters.timestep / parameters.scale;
    double const damping = 1.0 / (1.0 + parameters.drag * parameters.timestep);
    auto const meshState = [&](size_t particle) -> HopkinsParticleState* {
        if (meshStates != nullptr && types != nullptr && types[particle] == kSphType) {
            ALWAYS_ASSERT(gasIndices != nullptr && gasIndices[particle] >= 0);
            auto* state = meshStates + gasIndices[particle];
            return state->initialized ? state : nullptr;
        }
        return nullptr;
    };
    auto const meshDriftVelocity = [&](size_t particle, int dimension) {
        auto const* state = meshState(particle);
        return state != nullptr ? state->meshVelocity[dimension] :
                                  double(velocities[particle][dimension]);
    };
    Position* positions[kDimensions] = {x, y, z};
    size_t index = 0;
    if (!lightcone) {
        using DriftPack = NativeSimdVector<double>;
        constexpr size_t kWidth = DriftPack::size();
        for (; count - index >= kWidth; index += kWidth) {
            for (int dim = 0; dim < kDimensions; ++dim) {
                DriftPack const velocity([&](auto lane) {
                    size_t const particle = index + lane;
                    return parameters.rung < 0 || rungs[particle] == parameters.rung ?
                        meshDriftVelocity(particle, dim) :
                        0.0;
                });
                for (size_t lane = 0; lane < kWidth; ++lane) {
                    size_t const particle = index + lane;
                    if (parameters.rung < 0 || rungs[particle] == parameters.rung) {
                        auto* state = meshState(particle);
                        if constexpr (kPositionBits == 64) {
                            // Add in fixed point so conversion of the absolute
                            // position to double cannot discard its low bits.
                            positions[dim][particle] += Position(velocity[lane] * driftFactor);
                        } else {
                            positions[dim][particle].raw() += static_cast<PositionInteger>(
                                std::llrint(std::remainder(velocity[lane] * driftFactor, 1.0) * 0x1p32));
                        }
                        if (parameters.drag != 0.0 && state == nullptr) {
                            velocities[particle][dim] = velocity[lane] * damping;
                        }
                    }
                }
            }
        }
    }
    for (; index < count; ++index) {
        if (parameters.rung >= 0 && rungs[index] != parameters.rung) {
            continue;
        }
        Vector3<double> before;
        Vector3<double> after;
        Vector3<double> displacement;
#if !defined(COSMICTIGER_WITH_GRAVITY) || COSMICTIGER_WITH_GRAVITY
        auto const driftVelocity = velocities[index];
#endif
        auto* state = meshState(index);
        for (int dim = 0; dim < kDimensions; ++dim) {
            before[dim] = positions[dim][index].toDouble();
            after[dim] = std::fma(meshDriftVelocity(index, dim), driftFactor, before[dim]);
            displacement[dim] = meshDriftVelocity(index, dim) * driftFactor;
            if (parameters.drag != 0.0 && state == nullptr) {
                velocities[index][dim] *= damping;
            }
        }
#if !defined(COSMICTIGER_WITH_GRAVITY) || COSMICTIGER_WITH_GRAVITY
        if (lightcone) {
            char const type = types != nullptr ? types[index] : char(kDarkMatterType);
            detail::appendDriftLightconeEntries(
                before, after, driftVelocity, velocities[index], type, parameters, *entries, state);
        }
#endif
        for (int dim = 0; dim < kDimensions; ++dim) {
            if constexpr (kPositionBits == 64) {
                positions[dim][index] += Position(displacement[dim]);
            } else {
                positions[dim][index].raw() += static_cast<PositionInteger>(
                    std::llrint(std::remainder(displacement[dim], 1.0) * 0x1p32));
            }
        }
    }
}
