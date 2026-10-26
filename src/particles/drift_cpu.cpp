/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#include <cmath>

#include <chealpix.h>

#include <cosmictiger/math/simd_math.hpp>
#include <cosmictiger/particles/drift_cpu.hpp>

namespace detail {

void appendDriftLightconeEntries(Array<double, kDimensions> const& before,
    Array<double, kDimensions> const& after, Array<float, kDimensions> const& velocity,
    DriftParameters const& parameters, std::vector<LightconeEntry>& entries) {
    double const inverseScale = 1.0 / parameters.scale;
    double const inverseTauMax = 1.0 / parameters.tauMax;
    // CUDA uses float physical velocities for the crossing interpolation.
    float const vx = velocity[kXDimension] * inverseScale;
    float const vy = velocity[kYDimension] * inverseScale;
    float const vz = velocity[kZDimension] * inverseScale;
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
                double crossing[kDimensions] = {x0 + vx * time, y0 + vy * time, z0 + vz * time};
                double const radiusSquared = crossing[kXDimension] * crossing[kXDimension] +
                    crossing[kYDimension] * crossing[kYDimension] +
                    crossing[kZDimension] * crossing[kZDimension];
                if (radiusSquared <= 1.0) {
                    long pixel;
                    vec2pix_nest(parameters.mapSize, crossing, &pixel);
                    LightconeEntry entry;
                    entry.x = crossing[kXDimension];
                    entry.y = crossing[kYDimension];
                    entry.z = crossing[kZDimension];
                    entry.vx = velocity[kXDimension];
                    entry.vy = velocity[kYDimension];
                    entry.vz = velocity[kZDimension];
                    entry.pix = pixel;
                    entries.push_back(entry);
                }
            }
        }
    }
}

}    // namespace detail

void cpuDriftKernel(Position* x, Position* y, Position* z, Array<float, kDimensions>* velocities,
    char const* rungs, size_t count, DriftParameters const& parameters,
    std::vector<LightconeEntry>* entries) {
    if (count == 0) {
        return;
    }
    bool const lightcone = entries != nullptr && parameters.lightcone && parameters.tauMax > 0.0 &&
        parameters.tauMax - parameters.tau1 < 1.0;
    if (parameters.timestep == 0.0 && !lightcone) {
        return;
    }
    double const driftFactor = parameters.timestep / parameters.scale;
    double const damping = 1.0 / (1.0 + parameters.drag * parameters.timestep);
    Position* positions[kDimensions] = {x, y, z};
    size_t index = 0;
    if (!lightcone) {
        using DriftPack = NativeSimdVector<double>;
        constexpr size_t kWidth = DriftPack::size();
        DriftPack const factor(driftFactor);
        DriftPack const zero(0.0);
        DriftPack const one(1.0);
        for (; count - index >= kWidth; index += kWidth) {
            for (int dim = 0; dim < kDimensions; ++dim) {
                DriftPack const position(
                    [&](auto lane) { return positions[dim][index + lane].toDouble(); });
                DriftPack const velocity([&](auto lane) {
                    size_t const particle = index + lane;
                    return parameters.rung < 0 || rungs[particle] == parameters.rung ?
                        double(velocities[particle][dim]) :
                        0.0;
                });
                DriftPack const displaced = simdFma(velocity, factor, position);
                DriftPack const fraction = displaced - simdFloor(displaced);
                // Very small negative displacements can round their fraction to one.
                DriftPack const wrapped = simdSelect(fraction >= one, zero, fraction);
                for (size_t lane = 0; lane < kWidth; ++lane) {
                    size_t const particle = index + lane;
                    if (parameters.rung < 0 || rungs[particle] == parameters.rung) {
                        if constexpr (kPositionBits == 64) {
                            // Add in fixed point so conversion of the absolute
                            // position to double cannot discard its low bits.
                            positions[dim][particle] += Position(velocity[lane] * driftFactor);
                        } else {
                            positions[dim][particle] = wrapped[lane];
                        }
                        if (parameters.drag != 0.0) {
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
        Array<double, kDimensions> before;
        Array<double, kDimensions> after;
        Array<double, kDimensions> displacement;
        for (int dim = 0; dim < kDimensions; ++dim) {
            before[dim] = positions[dim][index].toDouble();
            after[dim] = std::fma(double(velocities[index][dim]), driftFactor, before[dim]);
            displacement[dim] = double(velocities[index][dim]) * driftFactor;
            if (parameters.drag != 0.0) {
                velocities[index][dim] *= damping;
            }
        }
        if (lightcone) {
            detail::appendDriftLightconeEntries(
                before, after, velocities[index], parameters, *entries);
        }
        for (int dim = 0; dim < kDimensions; ++dim) {
            if constexpr (kPositionBits == 64) {
                positions[dim][index] += Position(displacement[dim]);
            } else {
                double const fraction = after[dim] - std::floor(after[dim]);
                positions[dim][index] = fraction >= 1.0 ? 0.0 : fraction;
            }
        }
    }
}
