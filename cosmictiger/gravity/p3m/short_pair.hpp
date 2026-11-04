/* CosmicTiger - P3M short-range pair field with softening and cutoff; BSL-1.0. */
#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <numbers>

#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/cuda/cuda_export.hpp>
#include <cosmictiger/gravity/kernels/gaussian_screening.hpp>
#include <cosmictiger/math/radial_kernel.hpp>
#include <cosmictiger/math/vector3.hpp>

namespace cosmictiger::gravity::p3m {

// Source or sink record in box units. Periodic images are selected by the
// caller, who shifts the source position by whole box lengths before pairing.
struct ShortParticle {
    Vector3<double> position{};
    double mass = 0;
    int rank = 0;
    int64_t index = 0;

    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & position & mass & rank & index;
    }
};

struct ShortSourceRange {
    size_t begin = 0;
    size_t end = 0;
};

// Pair geometry shared by the CPU, SIMD and CUDA backends. alpha = 1/(2 r_s).
struct ShortPairParameters {
    double alpha = 0;
    double cutoffSquared = 0;
    double smoothingLength = 0;
    double supportSquared = 0;
    double inverseSupport = 0;
    double inverseSupportCubed = 0;
};

CUDA_EXPORT inline ShortPairParameters makeShortPairParameters(
    double splitRadius, double cutoffRadius, double smoothingLength) {
    ShortPairParameters parameters;
    parameters.alpha = 0.5 / splitRadius;
    parameters.cutoffSquared = cutoffRadius * cutoffRadius;
    parameters.smoothingLength = smoothingLength;
    parameters.supportSquared = smoothingLength * smoothingLength;
    parameters.inverseSupport = 1.0 / smoothingLength;
    parameters.inverseSupportCubed =
        parameters.inverseSupport * parameters.inverseSupport * parameters.inverseSupport;
    return parameters;
}

// Accumulate the short-range field of one source of the given mass into
// field = {potential, ax, ay, az} (without G). displacement is sink minus
// source. Outside the compact support the pair uses the screened Newtonian
// kernel -erfc(alpha r)/r; inside it uses the physical softened force and
// potential minus the smooth long component -erf(alpha r)/r, so the short plus
// mesh total reproduces the softened pair law. Pairs beyond the cutoff radius
// are dropped: their short force is the documented truncation of the split.
// Distinct coincident particles keep the finite softened-minus-long potential
// and exert no force; identical primary particles must be excluded by the caller.
template <class Kernel>
CUDA_EXPORT inline FlopCount accumulateShortPair(double dx, double dy, double dz, double mass,
    ShortPairParameters const& parameters, bool doPotential, Kernel const& kernel,
    std::array<double, 4>& field) {
    double const r2 = dx * dx + dy * dy + dz * dz;
    if (r2 >= parameters.cutoffSquared) {
        return 6;
    }
    GaussianScreening<double> const screening(parameters.alpha);
    if (r2 > parameters.supportSquared) {
        auto const radial = evaluateGaussianRadial<GaussianComponent::Short, 1>(r2, screening);
        // coefficient<1,0>() = r * A_1 with A_1 = (1/r) d/dr g, so a = -A_1 d.
        double const inward = -mass * radial.template coefficient<1, 0>() * radial.inverseRadius();
        field[1] += inward * dx;
        field[2] += inward * dy;
        field[3] += inward * dz;
        if (doPotential) {
            field[0] += mass * radial.template coefficient<0, 0>();
        }
        return 6 + radial.flops() + 8 + (doPotential ? 2 : 0);
    }
    if (r2 == 0.0) {
        if (doPotential) {
            double const longOrigin = -2.0 * parameters.alpha * std::numbers::inv_sqrtpi;
            field[0] += mass *
                (-radial_kernel::potential(kernel, 0.0) * parameters.inverseSupport - longOrigin);
        }
        return 6 + (doPotential ? 6 : 0);
    }
    double const r = std::sqrt(r2);
    double const q = r * parameters.inverseSupport;
    double const softenedFactor =
        mass * parameters.inverseSupportCubed * radial_kernel::forceFactor(kernel, q);
    auto const radial = evaluateGaussianRadial<GaussianComponent::Long, 1>(r2, screening);
    double const longFactor = mass * radial.template coefficient<1, 0>() * radial.inverseRadius();
    double const inward = -softenedFactor + longFactor;
    field[1] += inward * dx;
    field[2] += inward * dy;
    field[3] += inward * dz;
    if (doPotential) {
        field[0] += mass *
            (-radial_kernel::potential(kernel, q) * parameters.inverseSupport -
                radial.template coefficient<0, 0>());
    }
    return 6 + 4 + radial.flops() + 10 + (doPotential ? 5 : 0);
}

// Pair with identity exclusion; the displacement is sink minus shifted source.
template <class Kernel>
CUDA_EXPORT inline FlopCount accumulateShortPair(ShortParticle const& sink,
    ShortParticle const& source, ShortPairParameters const& parameters, bool doPotential,
    Kernel const& kernel, std::array<double, 4>& field) {
    if (source.rank == sink.rank && source.index == sink.index) {
        return 0;
    }
    return 3 +
        accumulateShortPair(sink.position[0] - source.position[0],
            sink.position[1] - source.position[1], sink.position[2] - source.position[2],
            source.mass, parameters, doPotential, kernel, field);
}

}    // namespace cosmictiger::gravity::p3m
