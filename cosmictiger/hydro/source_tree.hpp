/* CosmicTiger - persistent, time-dependent hydro source geometry; BSL-1.0. */

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace hydro_source {

using Vector = std::array<double, 3>;

// Unwrapped unit-periodic coordinates. A seam-crossing box may be [-e,+e].
struct Box {
    Vector lower{};
    Vector upper{};
};

// Geometry is independent of mass and of the method's conserved-state ledger.
// Position follows x(epoch) + velocity*dt + acceleration*dt^2/2. The caller
// supplies coordinate-time coefficients, including its cosmological drift factors.
struct Source {
    uint64_t id = 0;
    Vector position{};
    Vector velocity{};
    Vector acceleration{};
    // Fluid velocity used for signal bounds need not equal generator motion.
    Vector signalVelocity{};
    Vector signalAcceleration{};
    double epoch = 0.0;
    double validUntil = 0.0;
    double supportUpper = 0.0;
    double signalUpper = 0.0;
    int rung = 0;
};

struct Bounds {
    Box position;
    Box velocity;
    Box motionVelocity;
    double maxSupport = 0.0;
    double maxSignal = 0.0;
    double latestEpoch = -std::numeric_limits<double>::infinity();
    double earliestValidUntil = std::numeric_limits<double>::infinity();
    int minRung = std::numeric_limits<int>::max();
    int maxRung = 0;
};

struct Counters {
    uint64_t sourcesSorted = 0;
    uint64_t sourcesUpdated = 0;
    uint64_t nodesRefitted = 0;
    uint64_t topologyRebuilds = 0;
    uint64_t nodesVisited = 0;
    uint64_t sourcesVisited = 0;
};

namespace detail {

    // Relaxed atomic tallies, so concurrent const queries count safely. Copies
    // and assignments take a snapshot; the fields need no mutual ordering.
    struct AtomicCounters {
        std::atomic<uint64_t> sourcesSorted{0};
        std::atomic<uint64_t> sourcesUpdated{0};
        std::atomic<uint64_t> nodesRefitted{0};
        std::atomic<uint64_t> topologyRebuilds{0};
        std::atomic<uint64_t> nodesVisited{0};
        std::atomic<uint64_t> sourcesVisited{0};

        AtomicCounters() = default;
        AtomicCounters(AtomicCounters const& other) {
            store(other.snapshot());
        }
        AtomicCounters& operator=(AtomicCounters const& other) {
            if (this != &other) {
                store(other.snapshot());
            }
            return *this;
        }

        Counters snapshot() const {
            Counters result;
            result.sourcesSorted = sourcesSorted.load(std::memory_order_relaxed);
            result.sourcesUpdated = sourcesUpdated.load(std::memory_order_relaxed);
            result.nodesRefitted = nodesRefitted.load(std::memory_order_relaxed);
            result.topologyRebuilds = topologyRebuilds.load(std::memory_order_relaxed);
            result.nodesVisited = nodesVisited.load(std::memory_order_relaxed);
            result.sourcesVisited = sourcesVisited.load(std::memory_order_relaxed);
            return result;
        }

        void store(Counters const& values) {
            sourcesSorted.store(values.sourcesSorted, std::memory_order_relaxed);
            sourcesUpdated.store(values.sourcesUpdated, std::memory_order_relaxed);
            nodesRefitted.store(values.nodesRefitted, std::memory_order_relaxed);
            topologyRebuilds.store(values.topologyRebuilds, std::memory_order_relaxed);
            nodesVisited.store(values.nodesVisited, std::memory_order_relaxed);
            sourcesVisited.store(values.sourcesVisited, std::memory_order_relaxed);
        }

        void reset() {
            store({});
        }
    };

    inline bool finite(double value) {
        return (std::bit_cast<uint64_t>(value) & 0x7ff0000000000000ULL) != 0x7ff0000000000000ULL;
    }

    inline double down(double value) {
        return std::nextafter(value, -std::numeric_limits<double>::infinity());
    }

    inline double up(double value) {
        return std::nextafter(value, std::numeric_limits<double>::infinity());
    }

    struct Interval {
        double lower = 0.0;
        double upper = 0.0;
    };

    inline Interval add(Interval first, Interval second) {
        return {down(first.lower + second.lower), up(first.upper + second.upper)};
    }

    inline Interval multiply(Interval first, Interval second) {
        std::array<double, 4> products{first.lower * second.lower, first.lower * second.upper,
            first.upper * second.lower, first.upper * second.upper};
        auto const limits = std::minmax_element(products.begin(), products.end());
        return {down(*limits.first), up(*limits.second)};
    }

    inline Interval scalar(double value) {
        return {value, value};
    }

    inline Interval difference(double first, double second) {
        return {down(first - second), up(first - second)};
    }

    inline Box emptyBox() {
        Box box;
        box.lower.fill(std::numeric_limits<double>::infinity());
        box.upper.fill(-std::numeric_limits<double>::infinity());
        return box;
    }

    inline void include(Box& destination, Box const& source) {
        for (size_t dimension = 0; dimension < 3; ++dimension) {
            destination.lower[dimension] =
                std::min(destination.lower[dimension], source.lower[dimension]);
            destination.upper[dimension] =
                std::max(destination.upper[dimension], source.upper[dimension]);
        }
    }

    inline void validateBox(Box const& box) {
        for (size_t dimension = 0; dimension < 3; ++dimension) {
            if (!finite(box.lower[dimension]) || !finite(box.upper[dimension]) ||
                box.lower[dimension] > box.upper[dimension]) {
                throw std::invalid_argument("Invalid hydro source query box");
            }
        }
    }

    inline void validateSource(Source const& source) {
        if (!finite(source.epoch) || !finite(source.validUntil) ||
            source.validUntil < source.epoch || !finite(source.supportUpper) ||
            source.supportUpper < 0.0 || !finite(source.signalUpper) || source.signalUpper < 0.0 ||
            source.rung < 0) {
            throw std::invalid_argument("Invalid hydro source envelope");
        }
        for (auto const& vector : {source.position, source.velocity, source.acceleration,
                 source.signalVelocity, source.signalAcceleration}) {
            for (double value : vector) {
                if (!finite(value)) {
                    throw std::invalid_argument("Nonfinite hydro source trajectory");
                }
            }
        }
    }

    inline void validateTime(Source const& source, double time) {
        if (time < source.epoch || time > source.validUntil) {
            throw std::out_of_range("Hydro source query exceeds its anchor interval");
        }
    }

}    // namespace detail

inline bool periodicIntersection(Box const& first, Box const& second, double radius) {
    for (size_t dimension = 0; dimension < 3; ++dimension) {
        auto const lower =
            detail::add(detail::difference(first.lower[dimension], second.upper[dimension]),
                detail::scalar(-radius));
        auto const upper =
            detail::add(detail::difference(first.upper[dimension], second.lower[dimension]),
                detail::scalar(radius));
        if (std::ceil(lower.lower) > std::floor(upper.upper)) {
            return false;
        }
    }
    return true;
}

inline double periodicDistance(Vector const& point, Box const& box) {
    double squared = 0.0;
    for (size_t dimension = 0; dimension < 3; ++dimension) {
        double const width = box.upper[dimension] - box.lower[dimension];
        if (width >= 1.0) {
            continue;
        }
        double const center = box.lower[dimension] + 0.5 * width;
        double separation = point[dimension] - center;
        separation -= std::round(separation);
        double const rounding = 32.0 * std::numeric_limits<double>::epsilon() *
            (1.0 + std::abs(point[dimension]) + std::abs(center));
        double const gap = std::max(0.0, std::abs(separation) - 0.5 * width - rounding);
        squared = std::max(0.0, detail::down(squared + detail::down(gap * gap)));
    }
    return std::max(0.0, detail::down(std::sqrt(squared)));
}

class SourceTree {
    static constexpr size_t kNoNode_ = std::numeric_limits<size_t>::max();

    struct Coefficients {
        std::array<Box, 3> position;
        std::array<Box, 2> signal;
    };

    struct Record {
        Source source;
        Coefficients coefficients;
        size_t leaf = kNoNode_;
        Vector imageOffset{};
    };

    struct Node {
        Coefficients coefficients;
        Bounds metadata;
        size_t begin = 0;
        size_t end = 0;
        size_t parent = kNoNode_;
        size_t left = kNoNode_;
        size_t right = kNoNode_;
        size_t depth = 0;
    };

    std::vector<Record> records_;
    std::vector<size_t> order_;
    std::vector<Node> nodes_;
    std::unordered_map<uint64_t, size_t> indices_;
    size_t leafCapacity_;
    double referenceTime_ = 0.0;
    mutable detail::AtomicCounters counters_;
    // Node envelopes and source positions evaluated once at a prepared query
    // time (prepare); queries at exactly that time reuse them instead of
    // re-evaluating the interval polynomials per visited node. Refits keep the
    // prepared values current; a rebuild discards them.
    std::vector<Bounds> preparedBounds_;
    std::vector<Vector> preparedPositions_;
    double preparedTime_ = 0.0;
    bool prepared_ = false;

    static Coefficients emptyCoefficients() {
        Coefficients result;
        result.position.fill(detail::emptyBox());
        result.signal.fill(detail::emptyBox());
        return result;
    }

    static Coefficients coefficients(
        Source const& source, double referenceTime, Vector const& imageOffset = {}) {
        Coefficients result;
        auto const age = detail::difference(referenceTime, source.epoch);
        for (size_t dimension = 0; dimension < 3; ++dimension) {
            auto const acceleration = detail::scalar(source.acceleration[dimension]);
            auto const halfAcceleration = detail::multiply(detail::scalar(0.5), acceleration);
            auto const anchor = detail::add(
                detail::scalar(source.position[dimension]), detail::scalar(imageOffset[dimension]));
            std::array<detail::Interval, 3> position{
                detail::add(anchor,
                    detail::multiply(age,
                        detail::add(detail::scalar(source.velocity[dimension]),
                            detail::multiply(age, halfAcceleration)))),
                detail::add(detail::scalar(source.velocity[dimension]),
                    detail::multiply(age, acceleration)),
                halfAcceleration};
            std::array<detail::Interval, 2> signal{
                detail::add(detail::scalar(source.signalVelocity[dimension]),
                    detail::multiply(age, detail::scalar(source.signalAcceleration[dimension]))),
                detail::scalar(source.signalAcceleration[dimension])};
            for (size_t coefficient = 0; coefficient < position.size(); ++coefficient) {
                auto const value = position[coefficient];
                if (!detail::finite(value.lower) || !detail::finite(value.upper)) {
                    throw std::range_error("Hydro source position coefficients overflow");
                }
                result.position[coefficient].lower[dimension] = value.lower;
                result.position[coefficient].upper[dimension] = value.upper;
            }
            for (size_t coefficient = 0; coefficient < signal.size(); ++coefficient) {
                auto const value = signal[coefficient];
                if (!detail::finite(value.lower) || !detail::finite(value.upper)) {
                    throw std::range_error("Hydro source signal coefficients overflow");
                }
                result.signal[coefficient].lower[dimension] = value.lower;
                result.signal[coefficient].upper[dimension] = value.upper;
            }
        }
        return result;
    }

    static void includeCoefficients(Coefficients& destination, Coefficients const& source) {
        for (size_t coefficient = 0; coefficient < destination.position.size(); ++coefficient) {
            detail::include(destination.position[coefficient], source.position[coefficient]);
        }
        for (size_t coefficient = 0; coefficient < destination.signal.size(); ++coefficient) {
            detail::include(destination.signal[coefficient], source.signal[coefficient]);
        }
    }

    static void includeMetadata(Bounds& destination, Bounds const& source) {
        destination.maxSupport = std::max(destination.maxSupport, source.maxSupport);
        destination.maxSignal = std::max(destination.maxSignal, source.maxSignal);
        destination.latestEpoch = std::max(destination.latestEpoch, source.latestEpoch);
        destination.earliestValidUntil =
            std::min(destination.earliestValidUntil, source.earliestValidUntil);
        destination.minRung = std::min(destination.minRung, source.minRung);
        destination.maxRung = std::max(destination.maxRung, source.maxRung);
    }

    void refitNode(size_t index) {
        auto& node = nodes_[index];
        node.coefficients = emptyCoefficients();
        node.metadata = {};
        if (node.left != kNoNode_) {
            for (size_t child : {node.left, node.right}) {
                includeCoefficients(node.coefficients, nodes_[child].coefficients);
                includeMetadata(node.metadata, nodes_[child].metadata);
            }
        } else {
            for (size_t slot = node.begin; slot < node.end; ++slot) {
                auto const& record = records_[order_[slot]];
                includeCoefficients(node.coefficients, record.coefficients);
                Bounds metadata;
                metadata.maxSupport = record.source.supportUpper;
                metadata.maxSignal = record.source.signalUpper;
                metadata.latestEpoch = record.source.epoch;
                metadata.earliestValidUntil = record.source.validUntil;
                metadata.minRung = metadata.maxRung = record.source.rung;
                includeMetadata(node.metadata, metadata);
            }
        }
        counters_.nodesRefitted.fetch_add(1, std::memory_order_relaxed);
    }

    size_t buildNode(size_t begin, size_t end, size_t parent) {
        size_t const index = nodes_.size();
        nodes_.emplace_back();
        nodes_[index].begin = begin;
        nodes_[index].end = end;
        nodes_[index].parent = parent;
        nodes_[index].depth = parent == kNoNode_ ? 0 : nodes_[parent].depth + 1;
        if (end - begin > leafCapacity_) {
            Box box = detail::emptyBox();
            for (size_t slot = begin; slot < end; ++slot) {
                detail::include(box, records_[order_[slot]].coefficients.position[0]);
            }
            size_t dimension = 0;
            for (size_t candidate = 1; candidate < 3; ++candidate) {
                if (box.upper[candidate] - box.lower[candidate] >
                    box.upper[dimension] - box.lower[dimension]) {
                    dimension = candidate;
                }
            }
            size_t const middle = begin + (end - begin) / 2;
            std::nth_element(order_.begin() + begin, order_.begin() + middle, order_.begin() + end,
                [&](size_t first, size_t second) {
                    double const left = records_[first].coefficients.position[0].lower[dimension];
                    double const right = records_[second].coefficients.position[0].lower[dimension];
                    return left == right ? records_[first].source.id < records_[second].source.id :
                                           left < right;
                });
            size_t const left = buildNode(begin, middle, index);
            size_t const right = buildNode(middle, end, index);
            nodes_[index].left = left;
            nodes_[index].right = right;
        } else {
            std::sort(
                order_.begin() + begin, order_.begin() + end, [&](size_t first, size_t second) {
                    return records_[first].source.id < records_[second].source.id;
                });
            for (size_t slot = begin; slot < end; ++slot) {
                records_[order_[slot]].leaf = index;
            }
        }
        refitNode(index);
        return index;
    }

    Bounds bounds(Node const& node, double time) const {
        Bounds result = node.metadata;
        auto const age = detail::difference(time, referenceTime_);
        for (size_t dimension = 0; dimension < 3; ++dimension) {
            auto const interval = [&](Box const& box) {
                return detail::Interval{box.lower[dimension], box.upper[dimension]};
            };
            auto const first = interval(node.coefficients.position[1]);
            auto const second = interval(node.coefficients.position[2]);
            auto const position = detail::add(interval(node.coefficients.position[0]),
                detail::multiply(age, detail::add(first, detail::multiply(age, second))));
            auto const motion = detail::add(
                first, detail::multiply(detail::scalar(2.0), detail::multiply(age, second)));
            auto const velocity = detail::add(interval(node.coefficients.signal[0]),
                detail::multiply(age, interval(node.coefficients.signal[1])));
            if (!detail::finite(position.lower) || !detail::finite(position.upper) ||
                !detail::finite(motion.lower) || !detail::finite(motion.upper) ||
                !detail::finite(velocity.lower) || !detail::finite(velocity.upper)) {
                throw std::range_error("Hydro source bounds overflow at query time");
            }
            result.position.lower[dimension] = position.lower;
            result.position.upper[dimension] = position.upper;
            result.motionVelocity.lower[dimension] = motion.lower;
            result.motionVelocity.upper[dimension] = motion.upper;
            result.velocity.lower[dimension] = velocity.lower;
            result.velocity.upper[dimension] = velocity.upper;
        }
        return result;
    }

    static Vector positionUnchecked(Source const& source, double time) {
        Vector result;
        double const age = time - source.epoch;
        for (size_t dimension = 0; dimension < 3; ++dimension) {
            double const coordinate = std::fma(age,
                std::fma(0.5 * age, source.acceleration[dimension], source.velocity[dimension]),
                source.position[dimension]);
            if (!detail::finite(coordinate)) {
                throw std::range_error("Hydro source position overflow");
            }
            result[dimension] = coordinate - std::floor(coordinate);
        }
        return result;
    }

    static Vector velocityUnchecked(Source const& source, double time) {
        Vector result;
        for (size_t dimension = 0; dimension < 3; ++dimension) {
            result[dimension] = std::fma(
                time - source.epoch, source.acceleration[dimension], source.velocity[dimension]);
        }
        return result;
    }

    void queryNode(size_t index, Box const& sink, double support, double time, int maximumRung,
        std::vector<uint64_t>& result) const {
        counters_.nodesVisited.fetch_add(1, std::memory_order_relaxed);
        auto const& node = nodes_[index];
        if (node.metadata.minRung > maximumRung) {
            return;
        }
        bool const cached = prepared_ && time == preparedTime_;
        Bounds computed;
        if (!cached) {
            computed = bounds(node, time);
        }
        Bounds const& envelope = cached ? preparedBounds_[index] : computed;
        if (!periodicIntersection(
                sink, envelope.position, std::max(support, envelope.maxSupport))) {
            return;
        }
        if (node.left != kNoNode_) {
            queryNode(node.left, sink, support, time, maximumRung, result);
            queryNode(node.right, sink, support, time, maximumRung, result);
        } else {
            for (size_t slot = node.begin; slot < node.end; ++slot) {
                size_t const record = order_[slot];
                auto const& source = records_[record].source;
                if (source.rung > maximumRung) {
                    continue;
                }
                counters_.sourcesVisited.fetch_add(1, std::memory_order_relaxed);
                Vector const position =
                    cached ? preparedPositions_[record] : positionUnchecked(source, time);
                if (periodicDistance(position, sink) <= std::max(support, source.supportUpper)) {
                    detail::validateTime(source, time);
                    result.push_back(source.id);
                }
            }
        }
    }

    template <class Predicate, class Visitor>
    void visitNode(size_t index, double time, Predicate& predicate, Visitor& visitor) const {
        counters_.nodesVisited.fetch_add(1, std::memory_order_relaxed);
        auto const& node = nodes_[index];
        if (!predicate(bounds(node, time))) {
            return;
        }
        if (node.left != kNoNode_) {
            visitNode(node.left, time, predicate, visitor);
            visitNode(node.right, time, predicate, visitor);
        } else {
            for (size_t slot = node.begin; slot < node.end; ++slot) {
                auto const& source = records_[order_[slot]].source;
                counters_.sourcesVisited.fetch_add(1, std::memory_order_relaxed);
                detail::validateTime(source, time);
                visitor(source, positionUnchecked(source, time), velocityUnchecked(source, time));
            }
        }
    }

public:
    explicit SourceTree(size_t leafCapacity = 16)
      : leafCapacity_(leafCapacity) {
        if (leafCapacity == 0) {
            throw std::invalid_argument("Hydro source leaf capacity must be positive");
        }
    }

    // Membership changes rebuild explicitly. In-place updates retain connectivity.
    void reset(std::vector<Source> sources) {
        SourceTree replacement(leafCapacity_);
        if (!sources.empty()) {
            replacement.referenceTime_ = sources.front().epoch;
        }
        for (auto const& source : sources) {
            detail::validateSource(source);
            replacement.referenceTime_ = std::min(replacement.referenceTime_, source.epoch);
        }
        replacement.records_.reserve(sources.size());
        for (auto& source : sources) {
            size_t const index = replacement.records_.size();
            if (!replacement.indices_.emplace(source.id, index).second) {
                throw std::invalid_argument("Duplicate hydro source ID");
            }
            auto const polynomial = coefficients(source, replacement.referenceTime_);
            replacement.records_.push_back({std::move(source), polynomial, kNoNode_});
        }
        replacement.order_.resize(sources.size());
        std::iota(replacement.order_.begin(), replacement.order_.end(), size_t(0));
        if (!sources.empty()) {
            replacement.buildNode(0, sources.size(), kNoNode_);
        }
        replacement.counters_ = counters_;
        replacement.counters_.sourcesSorted.fetch_add(sources.size(), std::memory_order_relaxed);
        replacement.counters_.topologyRebuilds.fetch_add(1, std::memory_order_relaxed);
        *this = std::move(replacement);
    }

    void update(std::span<Source const> sources) {
        struct Change {
            size_t index;
            Coefficients coefficients;
            Vector imageOffset;
        };
        std::vector<Change> changes;
        changes.reserve(sources.size());
        std::unordered_set<uint64_t> seen;
        double referenceTime = referenceTime_;
        bool const rebase = !sources.empty() && sources.size() == records_.size();
        if (rebase) {
            referenceTime = sources.front().epoch;
        }
        for (auto const& source : sources) {
            detail::validateSource(source);
            auto const found = indices_.find(source.id);
            if (found == indices_.end() || !seen.insert(source.id).second) {
                throw std::invalid_argument("Missing or duplicate hydro source update ID");
            }
            changes.push_back({found->second, {}, {}});
            if (rebase) {
                referenceTime = std::min(referenceTime, source.epoch);
            }
        }
        for (size_t change = 0; change < changes.size(); ++change) {
            auto& updated = changes[change];
            auto const& previous = records_[updated.index];
            auto const& source = sources[change];
            double const age = source.epoch - previous.source.epoch;
            for (size_t dimension = 0; dimension < 3; ++dimension) {
                double const coordinate = std::fma(age,
                    std::fma(0.5 * age, previous.source.acceleration[dimension],
                        previous.source.velocity[dimension]),
                    previous.source.position[dimension]);
                updated.imageOffset[dimension] = std::round(
                    coordinate + previous.imageOffset[dimension] - source.position[dimension]);
            }
            updated.coefficients = coefficients(source, referenceTime, updated.imageOffset);
        }
        // Collect only changed ancestor paths, then refit deepest first. Each
        // node is refitted once even when a complete active cohort is replaced.
        std::unordered_set<size_t> dirty;
        std::vector<std::vector<size_t>> byDepth;
        for (auto const& change : changes) {
            auto const& record = records_[change.index];
            for (size_t node = record.leaf; node != kNoNode_; node = nodes_[node].parent) {
                if (!dirty.insert(node).second) {
                    break;
                }
                size_t const depth = nodes_[node].depth;
                if (byDepth.size() <= depth) {
                    byDepth.resize(depth + 1);
                }
                byDepth[depth].push_back(node);
            }
        }
        // All validation and allocations precede mutation. A complete update
        // can rebase polynomial time without inspecting any inactive records.
        referenceTime_ = referenceTime;
        for (size_t change = 0; change < changes.size(); ++change) {
            auto const& updated = changes[change];
            auto& record = records_[updated.index];
            record.source = sources[change];
            record.coefficients = updated.coefficients;
            record.imageOffset = updated.imageOffset;
        }
        for (size_t depth = byDepth.size(); depth > 0; --depth) {
            for (size_t node : byDepth[depth - 1]) {
                refitNode(node);
            }
        }
        if (prepared_) {
            for (auto const& change : changes) {
                preparedPositions_[change.index] =
                    positionUnchecked(records_[change.index].source, preparedTime_);
            }
            for (size_t node : dirty) {
                preparedBounds_[node] = bounds(nodes_[node], preparedTime_);
            }
        }
        counters_.sourcesUpdated.fetch_add(sources.size(), std::memory_order_relaxed);
    }

    // Evaluate every node envelope and source position at this time once, so
    // that queries at exactly this time skip the per-node interval arithmetic.
    // Later in-place updates refresh the prepared values; a rebuild clears them.
    void prepare(double time) {
        if (!detail::finite(time)) {
            throw std::invalid_argument("Nonfinite hydro source preparation time");
        }
        preparedTime_ = time;
        preparedBounds_.resize(nodes_.size());
        for (size_t index = 0; index < nodes_.size(); ++index) {
            preparedBounds_[index] = bounds(nodes_[index], time);
        }
        preparedPositions_.resize(records_.size());
        for (size_t index = 0; index < records_.size(); ++index) {
            preparedPositions_[index] = positionUnchecked(records_[index].source, time);
        }
        prepared_ = true;
    }

    // Apply removals and upserts together, rebuilding at most once. An upsert
    // wins when its ID also occurs in erasedIds. Existing-only changes refit.
    void replace(std::span<Source const> sources, std::span<uint64_t const> erasedIds) {
        bool membershipChanged = false;
        std::unordered_set<uint64_t> seen;
        std::unordered_set<uint64_t> removed(erasedIds.begin(), erasedIds.end());
        for (auto const& source : sources) {
            detail::validateSource(source);
            if (!seen.insert(source.id).second) {
                throw std::invalid_argument("Duplicate hydro source upsert ID");
            }
            membershipChanged = membershipChanged || !indices_.contains(source.id);
        }
        for (uint64_t id : removed) {
            membershipChanged = membershipChanged || (indices_.contains(id) && !seen.contains(id));
        }
        if (!membershipChanged) {
            update(sources);
            return;
        }
        std::vector<Source> replacement;
        replacement.reserve(records_.size() + sources.size());
        for (auto const& record : records_) {
            if (!seen.contains(record.source.id) && !removed.contains(record.source.id)) {
                replacement.push_back(record.source);
            }
        }
        replacement.insert(replacement.end(), sources.begin(), sources.end());
        reset(std::move(replacement));
    }

    void upsert(std::span<Source const> sources) {
        replace(sources, {});
    }

    void erase(std::span<uint64_t const> ids) {
        replace({}, ids);
    }

    bool erase(uint64_t id) {
        bool const found = contains(id);
        erase(std::span<uint64_t const>(&id, 1));
        return found;
    }

    void clear() {
        reset({});
    }

    bool contains(uint64_t id) const {
        return indices_.contains(id);
    }

    size_t size() const {
        return records_.size();
    }

    Source const& source(uint64_t id) const {
        return records_.at(indices_.at(id)).source;
    }

    Vector position(uint64_t id, double time) const {
        if (!detail::finite(time)) {
            throw std::invalid_argument("Nonfinite hydro source time");
        }
        auto const& record = source(id);
        detail::validateTime(record, time);
        return positionUnchecked(record, time);
    }

    std::vector<uint64_t> query(Box const& sink, double support, double time,
        int maximumSourceRung = std::numeric_limits<int>::max()) const {
        detail::validateBox(sink);
        if (!detail::finite(support) || support < 0.0 || !detail::finite(time)) {
            throw std::invalid_argument("Invalid hydro source query");
        }
        std::vector<uint64_t> result;
        if (!nodes_.empty()) {
            queryNode(0, sink, support, time, maximumSourceRung, result);
        }
        return result;
    }

    // Predicate returns true to descend. Bounds.velocity is the fluid signal
    // velocity; the visitor's last argument is the generator-motion velocity.
    // Touched source intervals are checked; pruned branches remain untouched.
    template <class Predicate, class Visitor>
    void visit(double time, Predicate&& predicate, Visitor&& visitor) const {
        if (!detail::finite(time)) {
            throw std::invalid_argument("Nonfinite hydro source traversal time");
        }
        if (!nodes_.empty()) {
            visitNode(0, time, predicate, visitor);
        }
    }

    Counters counters() const {
        return counters_.snapshot();
    }

    void resetCounters() const {
        counters_.reset();
    }
};

}    // namespace hydro_source
