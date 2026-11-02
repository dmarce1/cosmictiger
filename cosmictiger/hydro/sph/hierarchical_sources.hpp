/* CosmicTiger - retained source geometry for individual SPH timesteps; BSL-1.0. */

#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include <cosmictiger/hydro/source_tree.hpp>
#include <cosmictiger/hydro/sph/hierarchical_state.hpp>

namespace sph_hierarchical {

// Unwrapped unit-periodic coordinates; upper must not be below lower. A box
// straddling zero can be expressed as [-epsilon, epsilon], without splitting it.
struct SourceBox {
    std::array<double, 3> lower{};
    std::array<double, 3> upper{};

    template <typename Archive>
    void serialize(Archive&& archive, unsigned) {
        archive & lower;
        archive & upper;
    }
};

struct SourceCounters {
    uint64_t sourceRecordsSorted = 0;
    uint64_t sourceRecordsQueried = 0;
    uint64_t sourceRecordsPredicted = 0;
    uint64_t nodesVisited = 0;
    uint64_t leavesVisited = 0;
    uint64_t cohortsVisited = 0;
    uint64_t signalQueries = 0;
    uint64_t signalNodesPruned = 0;
};

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

    inline bool periodicIntersection(
        SourceBox const& first, SourceBox const& second, double radius) {
        for (size_t dimension = 0; dimension < 3; ++dimension) {
            double const lower = first.lower[dimension] - radius - second.upper[dimension];
            double const upper = first.upper[dimension] + radius - second.lower[dimension];
            // There is overlap iff some integer translation of the second interval
            // intersects the first. Spans exceeding a complete period work directly.
            if (std::ceil(lower) > std::floor(upper)) {
                return false;
            }
        }
        return true;
    }

    inline double soundSpeed(SourceState const& source) {
        double const squared =
            source.adiabaticIndex * source.pressureEnergyRatio * source.internalEnergy;
        if (!finite(squared) || squared <= 0.0) {
            throw std::range_error("Retained SPH source sound speed is invalid");
        }
        return std::sqrt(squared);
    }

    inline double periodicDistance(std::array<double, 3> const& point, SourceBox const& box) {
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

class SourceForest {
    struct Record {
        SourceState anchor;
        mutable SourceState prediction;
        mutable double predictionTime = -std::numeric_limits<double>::infinity();
    };
    struct Cohort {
        std::vector<uint64_t> ids;
        uint64_t generation = 0;
    };
    std::map<uint64_t, Record> records_;
    std::map<int, Cohort> cohorts_;
    hydro_source::SourceTree tree_;
    double signalSoundFloor_ = 0.0;
    uint64_t nextGeneration_ = 1;
    mutable SourceCounters counters_;

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

    SourceState const& prediction(uint64_t id, double time) const {
        auto const& record = records_.at(id);
        if (record.predictionTime != time) {
            record.prediction = predictSource(record.anchor, time);
            record.predictionTime = time;
            ++counters_.sourceRecordsPredicted;
        }
        return record.prediction;
    }

public:
    explicit SourceForest(size_t leafCapacity = 32)
      : tree_(leafCapacity) {}

    void setSignalSoundFloor(double sound) {
        if (!detail::finite(sound) || sound < 0.0) {
            throw std::invalid_argument("Invalid thermal-source signal bound");
        }
        signalSoundFloor_ = sound;
    }

    // Replace the active payloads without sorting or changing tree membership.
    // Slow records and their prediction caches are untouched. Migration or
    // conversion changes membership explicitly; only that operation rebuilds.
    void replaceActive(int minimumRung, std::vector<SourceState> sources) {
        if (minimumRung < 0) {
            throw std::invalid_argument("Retained SPH source minimum rung is negative");
        }
        std::map<int, Cohort> replacement;
        std::map<uint64_t, SourceState const*> incoming;
        std::vector<hydro_source::Source> geometryUpdates;
        geometryUpdates.reserve(sources.size());
        for (auto const& source : sources) {
            detail::validateSource(source);
            if (source.rung < minimumRung || !incoming.emplace(source.id, &source).second) {
                throw std::invalid_argument("Invalid or duplicate active SPH source");
            }
            auto const previous = records_.find(source.id);
            if (previous != records_.end() && previous->second.anchor.rung < minimumRung) {
                throw std::invalid_argument("Replacement overwrites an inactive SPH source");
            }
            replacement[source.rung].ids.push_back(source.id);
            geometryUpdates.push_back(geometry(source));
        }
        std::vector<uint64_t> removed;
        for (auto it = cohorts_.lower_bound(minimumRung); it != cohorts_.end(); ++it) {
            for (auto id : it->second.ids) {
                if (!incoming.contains(id)) {
                    removed.push_back(id);
                }
            }
        }
        tree_.replace(geometryUpdates, removed);
        for (auto id : removed) {
            records_.erase(id);
        }
        for (auto& source : sources) {
            records_[source.id] = Record{std::move(source)};
        }
        cohorts_.erase(cohorts_.lower_bound(minimumRung), cohorts_.end());
        for (auto& [rung, cohort] : replacement) {
            cohort.generation = nextGeneration_++;
        }
        cohorts_.merge(replacement);
    }

    std::vector<SourceState> query(SourceBox const& sink, double sinkSupport, double time,
        int maximumSourceRung = std::numeric_limits<int>::max()) const {
        if (!detail::finite(sinkSupport) || sinkSupport < 0.0 || sinkSupport > 0.5) {
            throw std::invalid_argument("Invalid retained SPH source query support");
        }
        // The pair kernels stage floats; their rounded supports/positions can
        // include a pair just outside the double geometric envelope.
        constexpr double kPadding = 8.0 * std::numeric_limits<float>::epsilon();
        auto ids =
            tree_.query({sink.lower, sink.upper}, sinkSupport + kPadding, time, maximumSourceRung);
        // Canonical summation order survives a checkpoint rebuild of topology.
        std::sort(ids.begin(), ids.end());
        std::vector<SourceState> result;
        result.reserve(ids.size());
        for (auto id : ids) {
            auto const& source = prediction(id, time);
            double const support = std::max(sinkSupport, source.supportRadius) + kPadding;
            if (detail::periodicDistance(source.position, sink) <= support) {
                result.push_back(source);
            }
        }
        return result;
    }

    // GADGET-4 equation (79), using current predicted fluid velocities, not
    // the mesh/drift velocity. All sources contribute to the incoming-signal
    // bound, including those outside the present compact support.
    double signalTimestep(
        SourceState const& sink, double time, double courant, double scale) const {
        if (!detail::finite(courant) || courant <= 0.0 || !detail::finite(scale) || scale <= 0.0) {
            throw std::invalid_argument("Invalid retained SPH signal timestep scale");
        }
        auto const predictedSink = predictSource(sink, time);
        double const sound = std::max(detail::soundSpeed(predictedSink), signalSoundFloor_);
        double shortest = predictedSink.supportRadius / sound;
        ++counters_.signalQueries;
        tree_.visit(
            time,
            [&](hydro_source::Bounds const& node) {
                if (time < node.latestEpoch || time > node.earliestValidUntil) {
                    throw std::out_of_range(
                        "Retained SPH signal source exceeds its valid interval");
                }
                double maximumSignal = sound + std::max(node.maxSignal, signalSoundFloor_);
                for (size_t dimension = 0; dimension < 3; ++dimension) {
                    maximumSignal +=
                        std::max(std::abs(predictedSink.synchronizedVelocity[dimension] -
                                     node.velocity.lower[dimension]),
                            std::abs(predictedSink.synchronizedVelocity[dimension] -
                                node.velocity.upper[dimension]));
                }
                double const distance =
                    hydro_source::periodicDistance(predictedSink.position, node.position);
                double const lowerLimit =
                    (2.0 * predictedSink.supportRadius + distance) / maximumSignal;
                bool const keep =
                    lowerLimit <= shortest * (1.0 + 32.0 * std::numeric_limits<double>::epsilon());
                counters_.signalNodesPruned += !keep;
                return keep;
            },
            [&](hydro_source::Source const& anchor, auto const&, auto const&) {
                if (anchor.id == predictedSink.id) {
                    return;
                }
                auto const& source = prediction(anchor.id, time);
                std::array<double, 3> separation{}, relativeVelocity{};
                double radialProduct = 0.0;
                for (size_t dimension = 0; dimension < 3; ++dimension) {
                    double distance =
                        predictedSink.position[dimension] - source.position[dimension];
                    distance -= std::round(distance);
                    separation[dimension] = distance;
                    relativeVelocity[dimension] = predictedSink.synchronizedVelocity[dimension] -
                        source.synchronizedVelocity[dimension];
                    radialProduct += distance * relativeVelocity[dimension];
                }
                double const distance = std::hypot(separation[0], separation[1], separation[2]);
                double const approach = distance > 0.0 ?
                    -radialProduct / distance :
                    std::hypot(relativeVelocity[0], relativeVelocity[1], relativeVelocity[2]);
                double const signal =
                    sound + std::max(detail::soundSpeed(source), signalSoundFloor_) + approach;
                if (!detail::finite(signal)) {
                    throw std::range_error("Retained SPH pair signal speed is not finite");
                }
                if (signal > 0.0) {
                    shortest =
                        std::min(shortest, (2.0 * predictedSink.supportRadius + distance) / signal);
                }
            });
        double const timestep = courant * scale * shortest;
        if (!detail::finite(timestep) || timestep <= 0.0) {
            throw std::range_error("Retained SPH signal timestep is not positive and finite");
        }
        return timestep;
    }

    uint64_t generation(int rung) const {
        auto const found = cohorts_.find(rung);
        return found == cohorts_.end() ? 0 : found->second.generation;
    }
    uint64_t fingerprint(int rung) const {
        auto const found = cohorts_.find(rung);
        if (found == cohorts_.end()) {
            return 0;
        }
        std::vector<SourceState> sources;
        for (auto id : found->second.ids) {
            sources.push_back(records_.at(id).anchor);
        }
        return detail::sourceFingerprint(sources);
    }
    size_t sourceCount(int rung) const {
        auto const found = cohorts_.find(rung);
        return found == cohorts_.end() ? 0 : found->second.ids.size();
    }
    SourceCounters counters() const {
        auto result = counters_;
        auto const geometryCounters = tree_.counters();
        result.sourceRecordsSorted = geometryCounters.sourcesSorted;
        result.sourceRecordsQueried = geometryCounters.sourcesVisited;
        result.nodesVisited = geometryCounters.nodesVisited;
        return result;
    }
    void resetCounters() {
        counters_ = {};
        tree_.resetCounters();
    }
};

}    // namespace sph_hierarchical
