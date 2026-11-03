/* CosmicTiger - retained source geometry for individual SPH timesteps; BSL-1.0. */

#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include <cosmictiger/hydro/source_forest.hpp>
#include <cosmictiger/hydro/sph/hierarchical_state.hpp>

namespace sph_hierarchical {

// Unwrapped unit-periodic coordinates; upper must not be below lower. A box
// straddling zero can be expressed as [-epsilon, epsilon], without splitting it.
struct SourceBox {
    std::array<double, 3> lower{};
    std::array<double, 3> upper{};

    operator hydro_source::Box() const {
        return {lower, upper};
    }

    template <typename Archive>
    void serialize(Archive&& archive, unsigned) {
        archive & lower;
        archive & upper;
    }
};

using SourceCounters = hydro_source::ForestCounters;

namespace detail {

    inline bool finite(double value) {
        // This check must survive production builds that enable fast-math.
        return (std::bit_cast<uint64_t>(value) & 0x7ff0000000000000ULL) != 0x7ff0000000000000ULL;
    }

    inline double predictedSupport(SourceState const& source, double interval) {
        double const logarithm =
            std::log(source.supportRadius) - source.logDensityRate * interval / 3.0;
        if (logarithm <= std::log(source.minimumSupportRadius)) {
            return source.minimumSupportRadius;
        }
        if (logarithm >= std::log(source.maximumSupportRadius)) {
            return source.maximumSupportRadius;
        }
        return std::exp(logarithm);
    }

    inline void validateSource(SourceState const& source) {
        bool valid = source.rung >= 0 && finite(source.epoch) && finite(source.validUntil) &&
            source.validUntil >= source.epoch && finite(source.validUntil - source.epoch) &&
            finite(source.driftScale) && source.driftScale > 0.0 && finite(source.internalEnergy) &&
            source.internalEnergy > 0.0 && finite(source.internalEnergyRate) &&
            finite(source.density) && source.density > 0.0 && finite(source.logDensityRate) &&
            finite(source.physicalDivergence) && finite(source.minimumSupportRadius) &&
            source.minimumSupportRadius > 0.0 && finite(source.maximumSupportRadius) &&
            source.maximumSupportRadius <= 0.5 && finite(source.supportRadius) &&
            source.minimumSupportRadius <= source.supportRadius &&
            source.supportRadius <= source.maximumSupportRadius && finite(source.gradHCorrection) &&
            finite(source.gradHCorrectionRate) && finite(source.balsaraFactor) &&
            finite(source.viscosityAlpha) && finite(source.pressureEnergyRatio) &&
            source.pressureEnergyRatio > 0.0 && finite(source.adiabaticIndex) &&
            source.adiabaticIndex > 0.0;
        for (size_t dimension = 0; dimension < 3; ++dimension) {
            valid = valid && finite(source.position[dimension]) &&
                finite(source.driftVelocity[dimension]) &&
                finite(source.synchronizedVelocity[dimension]) &&
                finite(source.hydroAcceleration[dimension]);
        }
        if (!valid) {
            throw std::invalid_argument("Invalid retained SPH source state");
        }
    }

    inline double soundSpeed(SourceState const& source) {
        double const squared =
            source.adiabaticIndex * source.pressureEnergyRatio * source.internalEnergy;
        if (!finite(squared) || squared <= 0.0) {
            throw std::range_error("Retained SPH source sound speed is invalid");
        }
        return std::sqrt(squared);
    }

    inline double periodicDistance(
        std::array<double, 3> const& point, hydro_source::Box const& box) {
        std::array<double, 3> gap{};
        for (size_t dimension = 0; dimension < 3; ++dimension) {
            double const width = box.upper[dimension] - box.lower[dimension];
            if (width >= 1.0) {
                continue;
            }
            double const center = box.lower[dimension] + 0.5 * width;
            double distance = point[dimension] - center;
            distance -= std::round(distance);
            gap[dimension] = std::max(0.0, std::abs(distance) - 0.5 * width);
        }
        // Round toward a lower bound, including cancellation at periodic faces.
        // Each periodic gap is at most 1/2. Direct squares cannot overflow;
        // any underflow only makes this lower bound more conservative.
        double const distance = std::sqrt(gap[0] * gap[0] + gap[1] * gap[1] + gap[2] * gap[2]);
        return std::max(0.0, distance - 32.0 * std::numeric_limits<double>::epsilon());
    }

    inline uint64_t sourceFingerprint(std::vector<SourceState> const& sources) {
        uint64_t hash = 14695981039346656037ULL;
        auto const append = [&hash](uint64_t word) {
            for (int byte = 0; byte < 8; ++byte) {
                hash ^= (word >> (8 * byte)) & 255U;
                hash *= 1099511628211ULL;
            }
        };
        for (auto const& source : sources) {
            append(source.id);
            append(uint64_t(source.rung));
            for (auto const& vector : {source.position, source.driftVelocity,
                     source.synchronizedVelocity, source.hydroAcceleration}) {
                for (double value : vector) {
                    append(std::bit_cast<uint64_t>(value));
                }
            }
            for (double value : {source.internalEnergy, source.internalEnergyRate, source.density,
                     source.logDensityRate, source.physicalDivergence, source.supportRadius,
                     source.minimumSupportRadius, source.maximumSupportRadius,
                     source.gradHCorrection, source.gradHCorrectionRate, source.balsaraFactor,
                     source.viscosityAlpha, source.pressureEnergyRatio, source.adiabaticIndex,
                     source.epoch, source.validUntil, source.driftScale}) {
                append(std::bit_cast<uint64_t>(value));
            }
        }
        return hash;
    }

}    // namespace detail

// The returned packet is a prediction at time; the immutable input remains the
// anchor. Density/support evolve together and grad-H follows its saved slope.
// Divergence, viscosity and EOS coefficients are retained estimates, not a claim
// that a fresh density solve has occurred.
inline SourceState predictSource(SourceState const& source, double time) {
    detail::validateSource(source);
    if (!detail::finite(time) || time < source.epoch || time > source.validUntil) {
        throw std::out_of_range("Retained SPH source prediction exceeds its valid interval");
    }
    double const interval = time - source.epoch;
    SourceState result = source;
    for (size_t dimension = 0; dimension < 3; ++dimension) {
        double const position = source.position[dimension] +
            source.driftVelocity[dimension] * (interval / source.driftScale);
        result.synchronizedVelocity[dimension] = std::fma(
            source.hydroAcceleration[dimension], interval, source.synchronizedVelocity[dimension]);
        if (!detail::finite(position) || !detail::finite(result.synchronizedVelocity[dimension])) {
            throw std::range_error("Retained SPH source motion is not finite");
        }
        result.position[dimension] = position - std::floor(position);
    }
    result.internalEnergy = std::fma(source.internalEnergyRate, interval, source.internalEnergy);
    result.density = std::exp(std::log(source.density) + source.logDensityRate * interval);
    result.supportRadius = detail::predictedSupport(source, interval);
    if (result.supportRadius <= source.minimumSupportRadius ||
        result.supportRadius >= source.maximumSupportRadius) {
        // A support bound holds H fixed, so its adaptive grad-H term vanishes.
        result.gradHCorrection = 0.0;
        result.gradHCorrectionRate = 0.0;
    } else {
        result.gradHCorrection =
            std::fma(source.gradHCorrectionRate, interval, source.gradHCorrection);
    }
    if (!detail::finite(result.internalEnergy) || result.internalEnergy <= 0.0 ||
        !detail::finite(result.density) || result.density <= 0.0 ||
        !detail::finite(result.supportRadius) || !detail::finite(result.gradHCorrection)) {
        throw std::range_error("Retained SPH source prediction has invalid hydro state");
    }
    result.epoch = time;
    return result;
}

// Adapts the retained SPH anchor to the method-agnostic source forest: the
// motion envelope, the predicted packet, the float-staged support test and the
// signal-bound kinematics are the only SPH-specific pieces of the forest.
struct SphSourcePolicy {
    // The pair kernels stage floats; their rounded supports/positions can
    // include a pair just outside the double geometric envelope.
    static constexpr double kQueryPadding = 8.0 * std::numeric_limits<float>::epsilon();

    static void validate(SourceState const& source) {
        detail::validateSource(source);
    }

    static uint64_t identity(SourceState const& source) {
        return source.id;
    }

    static hydro_source::Source geometry(SourceState const& source) {
        hydro_source::Source result;
        result.id = source.id;
        result.rung = source.rung;
        result.position = source.position;
        result.epoch = source.epoch;
        result.validUntil = source.validUntil;
        result.signalVelocity = source.synchronizedVelocity;
        result.signalAcceleration = source.hydroAcceleration;
        for (size_t dimension = 0; dimension < 3; ++dimension) {
            result.velocity[dimension] = source.driftVelocity[dimension] / source.driftScale;
        }
        double const interval = source.validUntil - source.epoch;
        result.supportUpper =
            std::max(source.supportRadius, detail::predictedSupport(source, interval)) +
            8.0 * std::numeric_limits<float>::epsilon();
        auto hottest = source;
        hottest.internalEnergy = std::max(source.internalEnergy,
            std::fma(source.internalEnergyRate, interval, source.internalEnergy));
        result.signalUpper = detail::soundSpeed(hottest);
        return result;
    }

    static SourceState predict(SourceState const& source, double time) {
        return predictSource(source, time);
    }

    static bool withinSupport(
        SourceState const& source, hydro_source::Box const& sink, double sinkSupport) {
        double const support = std::max(sinkSupport, source.supportRadius) + kQueryPadding;
        return detail::periodicDistance(source.position, sink) <= support;
    }

    static hydro_source::SignalState signal(SourceState const& source) {
        return {source.position, source.synchronizedVelocity, source.supportRadius,
            detail::soundSpeed(source)};
    }

    static uint64_t fingerprint(std::vector<SourceState> const& sources) {
        return detail::sourceFingerprint(sources);
    }
};

using SourceForest = hydro_source::Forest<SourceState, SphSourcePolicy>;

}    // namespace sph_hierarchical
