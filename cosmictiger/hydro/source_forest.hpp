/* CosmicTiger - method-agnostic forest of retained hydro sources; BSL-1.0. */

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

#include <cosmictiger/hydro/source_tree.hpp>

namespace hydro_source {

struct ForestCounters {
    uint64_t sourceRecordsSorted = 0;
    uint64_t sourceRecordsQueried = 0;
    uint64_t sourceRecordsPredicted = 0;
    uint64_t nodesVisited = 0;
    uint64_t leavesVisited = 0;
    uint64_t cohortsVisited = 0;
    uint64_t signalQueries = 0;
    uint64_t signalNodesPruned = 0;
};

// Predicted kinematics that a payload exposes to the incoming-signal bound:
// its position, fluid velocity, compact support radius and sound speed.
struct SignalState {
    Vector position{};
    Vector velocity{};
    double radius = 0.0;
    double speed = 0.0;
};

// Retained per-id anchors with cached predictions, grouped into per-rung
// cohorts, over the persistent SourceTree geometry. The hydro method supplies
// a Policy with these static members:
//   void validate(Payload const& anchor);                   throws when invalid
//   uint64_t identity(Payload const& payload);
//   Source geometry(Payload const& anchor);                  id, rung, envelope
//   Payload predict(Payload const& anchor, double time);
//   bool withinSupport(Payload const& predicted, Box const& sink, double support);
//   SignalState signal(Payload const& predicted);
//   uint64_t fingerprint(std::vector<Payload> const& anchors);
//   static constexpr double kQueryPadding;                   envelope slack
template <class Payload, class Policy>
class Forest {
    struct Record {
        Payload anchor;
        int rung = 0;
        mutable Payload prediction;
        mutable double predictionTime = -std::numeric_limits<double>::infinity();
    };
    struct Cohort {
        std::vector<uint64_t> ids;
        uint64_t generation = 0;
    };
    std::map<uint64_t, Record> records_;
    std::map<int, Cohort> cohorts_;
    SourceTree tree_;
    double signalSoundFloor_ = 0.0;
    uint64_t nextGeneration_ = 1;
    mutable ForestCounters counters_;

    Payload const& prediction(uint64_t id, double time) const {
        auto const& record = records_.at(id);
        if (record.predictionTime != time) {
            record.prediction = Policy::predict(record.anchor, time);
            record.predictionTime = time;
            ++counters_.sourceRecordsPredicted;
        }
        return record.prediction;
    }

public:
    explicit Forest(size_t leafCapacity = 32)
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
    void replaceActive(int minimumRung, std::vector<Payload> sources) {
        if (minimumRung < 0) {
            throw std::invalid_argument("Retained hydro source minimum rung is negative");
        }
        std::map<int, Cohort> replacement;
        std::map<uint64_t, Payload const*> incoming;
        std::vector<Source> geometryUpdates;
        geometryUpdates.reserve(sources.size());
        for (auto const& source : sources) {
            Policy::validate(source);
            auto envelope = Policy::geometry(source);
            if (envelope.rung < minimumRung || !incoming.emplace(envelope.id, &source).second) {
                throw std::invalid_argument("Invalid or duplicate active hydro source");
            }
            auto const previous = records_.find(envelope.id);
            if (previous != records_.end() && previous->second.rung < minimumRung) {
                throw std::invalid_argument("Replacement overwrites an inactive hydro source");
            }
            replacement[envelope.rung].ids.push_back(envelope.id);
            geometryUpdates.push_back(std::move(envelope));
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
        for (size_t index = 0; index < sources.size(); ++index) {
            auto const& envelope = geometryUpdates[index];
            records_[envelope.id] = Record{std::move(sources[index]), envelope.rung};
        }
        cohorts_.erase(cohorts_.lower_bound(minimumRung), cohorts_.end());
        for (auto& [rung, cohort] : replacement) {
            cohort.generation = nextGeneration_++;
        }
        cohorts_.merge(replacement);
    }

    // Evaluate the tree envelopes once at this time before a batch of queries
    // at exactly that time. Results are unchanged; only the per-node interval
    // arithmetic is amortized over the batch.
    void prepare(double time) {
        tree_.prepare(time);
    }

    std::vector<Payload> query(Box const& sink, double sinkSupport, double time,
        int maximumSourceRung = std::numeric_limits<int>::max()) const {
        if (!detail::finite(sinkSupport) || sinkSupport < 0.0 || sinkSupport > 0.5) {
            throw std::invalid_argument("Invalid retained hydro source query support");
        }
        auto ids = tree_.query(sink, sinkSupport + Policy::kQueryPadding, time, maximumSourceRung);
        // Canonical summation order survives a checkpoint rebuild of topology.
        std::sort(ids.begin(), ids.end());
        std::vector<Payload> result;
        result.reserve(ids.size());
        for (auto id : ids) {
            auto const& source = prediction(id, time);
            if (Policy::withinSupport(source, sink, sinkSupport)) {
                result.push_back(source);
            }
        }
        return result;
    }

    // GADGET-4 equation (79), using current predicted fluid velocities, not
    // the mesh/drift velocity. All sources contribute to the incoming-signal
    // bound, including those outside the present compact support.
    double signalTimestep(Payload const& sink, double time, double courant, double scale) const {
        if (!detail::finite(courant) || courant <= 0.0 || !detail::finite(scale) || scale <= 0.0) {
            throw std::invalid_argument("Invalid retained hydro signal timestep scale");
        }
        auto const predictedSink = Policy::predict(sink, time);
        uint64_t const sinkId = Policy::identity(predictedSink);
        auto const target = Policy::signal(predictedSink);
        double const sound = std::max(target.speed, signalSoundFloor_);
        double shortest = target.radius / sound;
        ++counters_.signalQueries;
        tree_.visit(
            time,
            [&](Bounds const& node) {
                if (time < node.latestEpoch || time > node.earliestValidUntil) {
                    throw std::out_of_range(
                        "Retained hydro signal source exceeds its valid interval");
                }
                double maximumSignal = sound + std::max(node.maxSignal, signalSoundFloor_);
                for (size_t dimension = 0; dimension < 3; ++dimension) {
                    maximumSignal += std::max(
                        std::abs(target.velocity[dimension] - node.velocity.lower[dimension]),
                        std::abs(target.velocity[dimension] - node.velocity.upper[dimension]));
                }
                double const distance = periodicDistance(target.position, node.position);
                double const lowerLimit = (2.0 * target.radius + distance) / maximumSignal;
                bool const keep =
                    lowerLimit <= shortest * (1.0 + 32.0 * std::numeric_limits<double>::epsilon());
                counters_.signalNodesPruned += !keep;
                return keep;
            },
            [&](Source const& anchor, auto const&, auto const&) {
                if (anchor.id == sinkId) {
                    return;
                }
                auto const source = Policy::signal(prediction(anchor.id, time));
                std::array<double, 3> separation{}, relativeVelocity{};
                double radialProduct = 0.0;
                for (size_t dimension = 0; dimension < 3; ++dimension) {
                    double distance = target.position[dimension] - source.position[dimension];
                    distance -= std::round(distance);
                    separation[dimension] = distance;
                    relativeVelocity[dimension] =
                        target.velocity[dimension] - source.velocity[dimension];
                    radialProduct += distance * relativeVelocity[dimension];
                }
                double const distance = std::hypot(separation[0], separation[1], separation[2]);
                double const approach = distance > 0.0 ?
                    -radialProduct / distance :
                    std::hypot(relativeVelocity[0], relativeVelocity[1], relativeVelocity[2]);
                double const signal = sound + std::max(source.speed, signalSoundFloor_) + approach;
                if (!detail::finite(signal)) {
                    throw std::range_error("Retained hydro pair signal speed is not finite");
                }
                if (signal > 0.0) {
                    shortest = std::min(shortest, (2.0 * target.radius + distance) / signal);
                }
            });
        double const timestep = courant * scale * shortest;
        if (!detail::finite(timestep) || timestep <= 0.0) {
            throw std::range_error("Retained hydro signal timestep is not positive and finite");
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
        std::vector<Payload> sources;
        for (auto id : found->second.ids) {
            sources.push_back(records_.at(id).anchor);
        }
        return Policy::fingerprint(sources);
    }
    size_t sourceCount(int rung) const {
        auto const found = cohorts_.find(rung);
        return found == cohorts_.end() ? 0 : found->second.ids.size();
    }
    size_t size() const {
        return records_.size();
    }
    ForestCounters counters() const {
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

}    // namespace hydro_source
