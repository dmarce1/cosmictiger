/* CosmicTiger - retained SPH source geometry tests; BSL-1.0. */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include <cosmictiger/hydro/sph/hierarchical_sources.hpp>

namespace detail {

using SourceState = sph_hierarchical::SourceState;
using SourceBox = sph_hierarchical::SourceBox;
using SourceForest = sph_hierarchical::SourceForest;

void require(bool condition, char const* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void requireNear(double actual, double expected, char const* message) {
    require(std::abs(actual - expected) <= 2.0e-12 * std::max(1.0, std::abs(expected)), message);
}

template <typename Function>
void requireThrows(Function&& function, char const* message) {
    bool threw = false;
    try {
        function();
    } catch (std::exception const&) {
        threw = true;
    }
    require(threw, message);
}

SourceState makeSource(uint64_t id, int rung, std::array<double, 3> position) {
    SourceState source;
    source.id = id;
    source.rung = rung;
    source.position = position;
    source.validUntil = 1.0;
    source.supportRadius = 0.02;
    return source;
}

SourceBox pointBox(std::array<double, 3> point) {
    return {point, point};
}

void testPrediction() {
    auto source = makeSource(1, 2, {0.9, 0.1, 0.5});
    source.epoch = 2.0;
    source.validUntil = 3.0;
    source.driftScale = 2.0;
    source.driftVelocity = {0.8, -0.6, 0.0};
    source.synchronizedVelocity = {1.0, 2.0, 3.0};
    source.hydroAcceleration = {0.1, -0.2, 0.3};
    source.internalEnergy = 4.0;
    source.internalEnergyRate = -0.5;
    source.density = 2.0;
    source.logDensityRate = std::log(8.0);
    source.physicalDivergence = -0.75;
    source.supportRadius = 0.1;
    source.gradHCorrection = -0.05;
    source.gradHCorrectionRate = 0.01;
    auto const predicted = sph_hierarchical::predictSource(source, 3.0);
    requireNear(predicted.position[0], 0.3, "positive periodic drift");
    requireNear(predicted.position[1], 0.8, "negative periodic drift");
    requireNear(predicted.synchronizedVelocity[0], 1.1, "velocity uses synchronized anchor");
    requireNear(predicted.internalEnergy, 3.5, "linear internal energy prediction");
    requireNear(predicted.density, 16.0, "exponential density prediction");
    requireNear(predicted.supportRadius, 0.05, "support follows density prediction");
    requireNear(predicted.physicalDivergence, -0.75, "retained physical divergence");
    requireNear(predicted.gradHCorrection, -0.04, "linear grad-H prediction");
    requireNear(predicted.gradHCorrectionRate, 0.01, "retained grad-H slope");
    requireNear(predicted.epoch, 3.0, "predicted packet epoch");
    requireNear(source.position[0], 0.9, "predictor preserves source anchor");
    requireNear(source.internalEnergy, 4.0, "predictor preserves source energy");
    requireNear(source.gradHCorrection, -0.05, "predictor preserves source grad-H anchor");
    source.logDensityRate = -30.0;
    auto const upperBound = sph_hierarchical::predictSource(source, 3.0);
    requireNear(upperBound.supportRadius, 0.5, "support prediction respects upper bound");
    requireNear(upperBound.gradHCorrection, 0.0, "upper support bound disables grad-H");
    requireNear(upperBound.gradHCorrectionRate, 0.0, "upper support bound disables grad-H slope");
    source.logDensityRate = 30.0;
    auto const lowerBound = sph_hierarchical::predictSource(source, 3.0);
    requireNear(lowerBound.supportRadius, source.minimumSupportRadius,
        "support prediction respects lower bound");
    requireNear(lowerBound.gradHCorrection, 0.0, "lower support bound disables grad-H");
    requireNear(lowerBound.gradHCorrectionRate, 0.0, "lower support bound disables grad-H slope");
    requireNear(source.gradHCorrectionRate, 0.01, "support clipping preserves source slope");
}

void testDerivedStatePreservation() {
    auto slow = makeSource(1, 1, {0.1, 0.1, 0.1});
    slow.physicalDivergence = -1.25;
    slow.gradHCorrection = 0.2;
    slow.gradHCorrectionRate = -0.1;
    SourceForest forest(1);
    forest.replaceActive(0, {slow});
    uint64_t const fingerprint = forest.fingerprint(1);
    forest.replaceActive(3, {makeSource(2, 3, {0.8, 0.8, 0.8})});
    auto const predicted = forest.query(pointBox(slow.position), 0.01, 0.5);
    require(predicted.size() == 1, "derived-state query returned wrong cohort");
    requireNear(predicted[0].physicalDivergence, -1.25, "cohort retains physical divergence");
    requireNear(predicted[0].gradHCorrection, 0.15, "cohort predicts evolving grad-H");
    requireNear(predicted[0].gradHCorrectionRate, -0.1, "cohort retains grad-H slope");
    require(forest.fingerprint(1) == fingerprint, "derived-state prediction mutated snapshot");
    slow.physicalDivergence = 0.0;
    forest.replaceActive(1, {slow});
    uint64_t const divergenceFingerprint = forest.fingerprint(1);
    require(divergenceFingerprint != fingerprint, "fingerprint omitted physical divergence");
    slow.gradHCorrectionRate = 0.0;
    forest.replaceActive(1, {slow});
    require(forest.fingerprint(1) != divergenceFingerprint, "fingerprint omitted grad-H slope");
}

void testPeriodicAndGrowingSupportCoverage() {
    SourceForest forest(1);
    auto wraps = makeSource(1, 1, {0.98, 0.4, 0.4});
    wraps.driftVelocity[0] = 0.2;
    auto manyWraps = makeSource(2, 1, {0.2, 0.5, 0.5});
    manyWraps.driftVelocity[0] = 3.6;
    auto grows = makeSource(3, 2, {0.5, 0.7, 0.7});
    grows.logDensityRate = -3.0 * std::log(10.0);
    forest.replaceActive(0, {wraps, manyWraps, grows});
    auto const wrapped = forest.query({{-0.03, 0.39, 0.39}, {0.03, 0.41, 0.41}}, 0.01, 0.2);
    require(std::any_of(
                wrapped.begin(), wrapped.end(), [](auto const& source) { return source.id == 1; }),
        "wrapped sink box finds swept source");
    auto const several = forest.query(pointBox({0.8, 0.5, 0.5}), 0.01, 1.0);
    require(std::any_of(
                several.begin(), several.end(), [](auto const& source) { return source.id == 2; }),
        "multi-period drift remains conservatively enclosed");
    auto const growing = forest.query(pointBox({0.68, 0.7, 0.7}), 0.01, 1.0);
    require(std::any_of(
                growing.begin(), growing.end(), [](auto const& source) { return source.id == 3; }),
        "source support envelope includes later growth");
}

void testBruteForceCoverage() {
    std::mt19937_64 random(854209);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::uniform_real_distribution<double> signedUnit(-1.0, 1.0);
    std::vector<SourceState> sources;
    for (uint64_t index = 0; index < 257; ++index) {
        auto source = makeSource(index, int(index % 5), {unit(random), unit(random), unit(random)});
        source.supportRadius = 0.01 + 0.04 * unit(random);
        source.logDensityRate = 2.0 * signedUnit(random);
        source.driftScale = 0.5 + unit(random);
        for (size_t dimension = 0; dimension < 3; ++dimension) {
            source.driftVelocity[dimension] = 0.2 * signedUnit(random);
        }
        if (index % 31 == 0) {
            source.driftVelocity[0] = 3.0 * signedUnit(random);
        }
        sources.push_back(source);
    }
    SourceForest forest(8);
    forest.replaceActive(0, sources);
    for (int query = 0; query < 200; ++query) {
        std::array<double, 3> const sink{unit(random), unit(random), unit(random)};
        double const support = 0.01 + 0.07 * unit(random);
        double const time = unit(random);
        int const maximumRung = query % 5;
        auto const candidates = forest.query(pointBox(sink), support, time, maximumRung);
        std::unordered_set<uint64_t> ids;
        for (auto const& candidate : candidates) {
            require(candidate.rung <= maximumRung, "query excludes active cohorts");
            require(ids.insert(candidate.id).second, "source occurs only once in forest query");
        }
        for (auto const& source : sources) {
            if (source.rung > maximumRung) {
                continue;
            }
            auto const predicted = sph_hierarchical::predictSource(source, time);
            double distanceSquared = 0.0;
            for (size_t dimension = 0; dimension < 3; ++dimension) {
                double separation = sink[dimension] - predicted.position[dimension];
                separation -= std::round(separation);
                distanceSquared += separation * separation;
            }
            double const radius = std::max(support, predicted.supportRadius);
            if (distanceSquared <= radius * radius) {
                require(ids.contains(source.id), "forest omitted a brute-force periodic neighbor");
            }
        }
    }
}

void testPermutationStableQueryOrder() {
    // Repeated coordinates exercise both splitting-plane ties and coincident
    // sources. A checkpoint restores an unrelated gas-slot permutation.
    std::vector<SourceState> sources;
    for (uint64_t index = 0; index < 257; ++index) {
        auto source = makeSource(1000 + 17 * index, int(index % 4),
            {0.1 + 0.2 * (index % 4), 0.1 + 0.2 * ((index / 4) % 4),
                0.1 + 0.2 * ((index / 16) % 4)});
        source.driftVelocity = {0.02 * (index % 3), -0.01 * (index % 5), 0.0};
        source.supportRadius = 0.03 + 0.01 * (index % 7);
        sources.push_back(source);
    }
    auto const ids = [](std::vector<SourceState> const& rows) {
        std::vector<uint64_t> result;
        for (auto const& row : rows) {
            result.push_back(row.id);
        }
        return result;
    };
    SourceBox const entireDomain{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}};
    SourceBox const localBox = pointBox({0.32, 0.28, 0.3});
    constexpr double kTime = 0.5;
    constexpr double kSupport = 0.12;
    SourceForest baseline(8);
    baseline.replaceActive(0, sources);
    auto const expectedAll = ids(baseline.query(entireDomain, 0.0, kTime));
    auto const expectedLocal = ids(baseline.query(localBox, kSupport, kTime, 2));
    std::unordered_set<uint64_t> const localIds(expectedLocal.begin(), expectedLocal.end());
    for (auto const& source : sources) {
        if (source.rung > 2) {
            continue;
        }
        auto const predicted = sph_hierarchical::predictSource(source, kTime);
        double squaredDistance = 0.0;
        for (size_t dimension = 0; dimension < 3; ++dimension) {
            double separation = predicted.position[dimension] - localBox.lower[dimension];
            separation -= std::round(separation);
            squaredDistance += separation * separation;
        }
        double const radius = std::max(kSupport, predicted.supportRadius);
        if (squaredDistance <= radius * radius) {
            require(localIds.contains(source.id), "ordered query omitted a brute-force neighbor");
        }
    }
    // Canonical source-ID order is independent of topology, rung changes and
    // input ordering, including after a checkpoint rebuild.
    auto canonical = sources;
    std::sort(canonical.begin(), canonical.end(),
        [](auto const& first, auto const& second) { return first.id < second.id; });
    auto const expectedSingleLeaf = ids(canonical);
    std::mt19937_64 random(790301);
    for (int permutation = 0; permutation < 32; ++permutation) {
        std::shuffle(sources.begin(), sources.end(), random);
        SourceForest restored(8);
        restored.replaceActive(0, sources);
        restored.resetCounters();
        require(ids(restored.query(entireDomain, 0.0, kTime)) == expectedAll,
            "input permutation changed source tree traversal order");
        require(ids(restored.query(localBox, kSupport, kTime, 2)) == expectedLocal,
            "input permutation changed local source candidate order");
        require(
            restored.counters().sourceRecordsSorted == 0, "source query sorted retained records");
        SourceForest singleLeaf(1024);
        singleLeaf.replaceActive(0, sources);
        require(ids(singleLeaf.query(entireDomain, 0.0, kTime)) == expectedSingleLeaf,
            "leaf query did not use canonical source identity order");
    }
}

void testReplacementCostAndImmutability() {
    std::vector<SourceState> sources;
    for (uint64_t index = 0; index < 512; ++index) {
        sources.push_back(makeSource(index, 1, {0.1 + 0.0001 * index, 0.1, 0.1}));
    }
    sources.push_back(makeSource(901, 4, {0.8, 0.8, 0.8}));
    sources.push_back(makeSource(902, 4, {0.81, 0.8, 0.8}));
    sources.push_back(makeSource(903, 4, {0.82, 0.8, 0.8}));
    SourceForest forest(4);
    forest.replaceActive(0, sources);
    uint64_t const generation = forest.generation(1);
    uint64_t const fingerprint = forest.fingerprint(1);
    forest.resetCounters();
    forest.replaceActive(3,
        {makeSource(901, 3, {0.8, 0.8, 0.8}), makeSource(902, 3, {0.81, 0.8, 0.8}),
            makeSource(903, 5, {0.82, 0.8, 0.8})});
    require(forest.generation(1) == generation, "inactive cohort generation changed");
    require(forest.fingerprint(1) == fingerprint, "inactive cohort snapshot changed");
    require(forest.sourceCount(1) == 512, "inactive source population changed");
    require(forest.sourceCount(4) == 0, "obsolete active cohort retained");
    require(
        forest.counters().sourceRecordsSorted == 0, "same-membership update sorted source records");
    auto const candidates = forest.query(pointBox({0.8, 0.8, 0.8}), 0.04, 0.5);
    require(candidates.size() == 3, "localized query returned distant slow source leaves");
    require(
        forest.counters().sourceRecordsPredicted == 3, "localized query predicted slow sources");
    require(forest.counters().sourceRecordsQueried == 3, "localized query scanned slow sources");
    require(forest.generation(1) == generation && forest.fingerprint(1) == fingerprint,
        "query mutated retained cohort");
    require(forest.query(pointBox({0.8, 0.8, 0.8}), 0.04, 0.5, 2).empty(),
        "source-rung upper bound did not exclude active sources");
    forest.replaceActive(3, {});
    require(forest.sourceCount(3) == 0 && forest.sourceCount(5) == 0,
        "empty replacement did not retire active cohorts");
    require(forest.generation(1) == generation, "empty replacement changed slower generation");
}

void testPredictionReuseAndParticleFiltering() {
    auto near = makeSource(1, 1, {0.99, 0.5, 0.5});
    near.internalEnergyRate = 0.5;
    auto far = makeSource(2, 1, {0.4, 0.5, 0.5});
    auto extended = makeSource(3, 1, {0.18, 0.5, 0.5});
    extended.supportRadius = 0.2;
    SourceForest forest(16);
    forest.replaceActive(0, {near, far, extended});
    auto const fingerprint = forest.fingerprint(1);
    auto const first = forest.query(pointBox({0.005, 0.5, 0.5}), 0.015, 0.25);
    require(first.size() == 2 && first[0].id == 1 && first[1].id == 3,
        "particle filtering lost periodic or larger-source support, or retained a distant row");
    auto const predictions = forest.counters().sourceRecordsPredicted;
    auto const repeated = forest.query(pointBox({0.005, 0.5, 0.5}), 0.015, 0.25);
    require(repeated.size() == first.size() &&
            repeated[0].internalEnergy == first[0].internalEnergy &&
            repeated[1].supportRadius == first[1].supportRadius,
        "prediction reuse changed returned state");
    require(forest.counters().sourceRecordsPredicted == predictions,
        "overlapping query recomputed immutable predictions at the same time");
    auto const later = forest.query(pointBox({0.005, 0.5, 0.5}), 0.015, 0.5);
    requireNear(later[0].internalEnergy, 1.25, "new-time query reused stale predicted energy");
    require(forest.counters().sourceRecordsPredicted > predictions,
        "new prediction time did not refresh requested sources");
    require(forest.fingerprint(1) == fingerprint, "derived cache mutated source snapshots");
    near.internalEnergy = 8.0;
    forest.replaceActive(1, {near});
    auto const refreshed = forest.query(pointBox({0.005, 0.5, 0.5}), 0.015, 0.5);
    requireNear(refreshed[0].internalEnergy, 8.25,
        "replacement at an unchanged time retained an obsolete prediction");
    auto boundary = makeSource(4, 1, {0.0250002, 0.5, 0.5});
    forest.replaceActive(1, {near, boundary});
    require(forest.query(pointBox({0.005, 0.5, 0.5}), 0.015, 0.5).size() == 2,
        "particle filter omitted the float-staging boundary margin");
}

void testFailureAndValidity() {
    SourceForest forest(1);
    auto near = makeSource(1, 1, {0.8, 0.8, 0.8});
    auto far = makeSource(2, 0, {0.1, 0.1, 0.1});
    far.validUntil = 0.1;
    forest.replaceActive(0, {near, far});
    require(forest.query(pointBox(near.position), 0.01, 0.5).size() == 1,
        "expired distant cohort interfered with valid query");
    requireThrows([&] { forest.query(pointBox(far.position), 0.01, 0.5); },
        "intersecting expired source was extrapolated");
    requireThrows([&] { forest.query(pointBox(near.position), 0.01, -0.1); },
        "source was extrapolated before epoch");
    near.internalEnergyRate = -2.0;
    requireThrows([&] { sph_hierarchical::predictSource(near, 0.75); },
        "nonpositive predicted energy was silently clipped");
    near.internalEnergyRate = 0.0;
    near.logDensityRate = 1000.0;
    requireThrows(
        [&] { sph_hierarchical::predictSource(near, 1.0); }, "overflowing density was accepted");
    auto invalid = near;
    invalid.maximumSupportRadius = 0.6;
    uint64_t const generation = forest.generation(1);
    requireThrows([&] { forest.replaceActive(1, {invalid}); }, "invalid support bound accepted");
    require(forest.generation(1) == generation, "failed replacement changed forest");
    invalid = near;
    invalid.driftScale = 0.0;
    requireThrows([&] { forest.replaceActive(1, {invalid}); }, "zero drift scale accepted");
    invalid = makeSource(3, 1, {0.5, 0.5, 0.5});
    invalid.physicalDivergence = std::numeric_limits<double>::infinity();
    requireThrows(
        [&] { forest.replaceActive(1, {invalid}); }, "nonfinite physical divergence accepted");
    invalid.physicalDivergence = 0.0;
    invalid.gradHCorrectionRate = std::numeric_limits<double>::quiet_NaN();
    requireThrows([&] { forest.replaceActive(1, {invalid}); }, "nonfinite grad-H slope accepted");
    invalid.gradHCorrection = std::numeric_limits<double>::max();
    invalid.gradHCorrectionRate = std::numeric_limits<double>::max();
    requireThrows([&] { sph_hierarchical::predictSource(invalid, 1.0); },
        "overflowing grad-H prediction accepted");
    requireThrows([&] { forest.replaceActive(2, {near}); }, "inactive replacement source accepted");
    requireThrows(
        [&] {
            forest.query(pointBox(near.position), 0.01, std::numeric_limits<double>::quiet_NaN());
        },
        "nonfinite query time accepted");
}

double bruteForceSignalTimestep(SourceState const& sink, std::vector<SourceState> const& sources,
    double time, double courant, double scale) {
    auto const target = sph_hierarchical::predictSource(sink, time);
    auto const sound = [](SourceState const& state) {
        return std::sqrt(state.adiabaticIndex * state.pressureEnergyRatio * state.internalEnergy);
    };
    double shortest = target.supportRadius / sound(target);
    for (auto const& anchor : sources) {
        if (anchor.id == sink.id) {
            continue;
        }
        auto const source = sph_hierarchical::predictSource(anchor, time);
        double squaredDistance = 0.0;
        double radialProduct = 0.0;
        double squaredRelativeVelocity = 0.0;
        for (size_t dimension = 0; dimension < 3; ++dimension) {
            double separation = target.position[dimension] - source.position[dimension];
            separation -= std::round(separation);
            squaredDistance += separation * separation;
            double const relativeVelocity =
                target.synchronizedVelocity[dimension] - source.synchronizedVelocity[dimension];
            radialProduct += separation * relativeVelocity;
            squaredRelativeVelocity += relativeVelocity * relativeVelocity;
        }
        double const distance = std::sqrt(squaredDistance);
        double const signal = sound(target) + sound(source) +
            (distance > 0.0 ? -radialProduct / distance : std::sqrt(squaredRelativeVelocity));
        if (signal > 0.0) {
            shortest = std::min(shortest, (2.0 * target.supportRadius + distance) / signal);
        }
    }
    return courant * scale * shortest;
}

void testSignalEncountersAndUnits() {
    auto sink = makeSource(100, 1, {0.97, 0.4, 0.4});
    sink.internalEnergy = 1.0e-4;
    sink.adiabaticIndex = sink.pressureEnergyRatio = 1.0;
    auto source = sink;
    source.id = 200;
    source.position[0] = 0.03;
    source.rung = 5;
    source.synchronizedVelocity[0] = -5.0;
    SourceForest forest(1);
    forest.replaceActive(0, {sink, source});
    require(forest.query(pointBox(sink.position), sink.supportRadius, 0.0).size() == 1,
        "encounter test source was already a support neighbor");
    double const expected = 0.25 * 0.5 * (2.0 * sink.supportRadius + 0.06) / (0.02 + 5.0);
    requireNear(forest.signalTimestep(sink, 0.0, 0.25, 0.5), expected,
        "nonlocal signal missed periodic approach or used wrong support/scale units");
    source.synchronizedVelocity[0] = 5.0;
    forest.replaceActive(0, {sink, source});
    requireNear(forest.signalTimestep(sink, 0.0, 0.25, 0.5), 0.25 * 0.5 * sink.supportRadius / 0.01,
        "receding signal incorrectly shortened timestep");
    source.position = sink.position;
    source.synchronizedVelocity = {};
    source.internalEnergy = 10000.0;
    forest.replaceActive(0, {sink, source});
    requireNear(forest.signalTimestep(sink, 0.0, 0.25, 0.5),
        0.25 * 0.5 * 2.0 * sink.supportRadius / 100.01,
        "coincident hot neighbor did not give finite thermal constraint");
    source.synchronizedVelocity = {30.0, 40.0, 0.0};
    forest.replaceActive(0, {sink, source});
    requireNear(forest.signalTimestep(sink, 0.0, 0.25, 0.5),
        0.25 * 0.5 * 2.0 * sink.supportRadius / 150.01,
        "coincident streaming neighbor lost its approach-speed constraint");
    source.id = sink.id;
    forest.replaceActive(0, {source});
    requireNear(forest.signalTimestep(sink, 0.0, 0.25, 0.5), 0.25 * 0.5 * sink.supportRadius / 0.01,
        "signal limiter did not exclude its own identity");
}

void testSignalPredictionAndTreeReference() {
    std::mt19937_64 random(1349301);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::uniform_real_distribution<double> signedUnit(-1.0, 1.0);
    std::vector<SourceState> sources;
    for (uint64_t index = 0; index < 257; ++index) {
        auto source = makeSource(index, int(index % 6), {unit(random), unit(random), unit(random)});
        source.internalEnergy = 0.01 + 10.0 * unit(random);
        source.internalEnergyRate = 100.0 * unit(random);
        source.supportRadius = 0.01 + 0.08 * unit(random);
        source.logDensityRate = signedUnit(random);
        for (size_t dimension = 0; dimension < 3; ++dimension) {
            source.synchronizedVelocity[dimension] = 20.0 * signedUnit(random);
            source.hydroAcceleration[dimension] = 100.0 * signedUnit(random);
            source.driftVelocity[dimension] = 2.0 * signedUnit(random);
        }
        sources.push_back(source);
    }
    SourceForest forest(4);
    forest.replaceActive(0, sources);
    for (int query = 0; query < 100; ++query) {
        auto const& sink = sources[(query * 23) % sources.size()];
        double const time = query % 3 == 0 ? 1.0 : unit(random);
        double const scale = 0.1 + unit(random);
        requireNear(forest.signalTimestep(sink, time, 0.25, scale),
            bruteForceSignalTimestep(sink, sources, time, 0.25, scale),
            "tree signal timestep differs from all-particle reference");
    }
    // End-of-interval heating and acceleration must be covered by cached node
    // maxima even if the initial cold, stationary node would have been pruned.
    auto sink = makeSource(900, 2, {0.1, 0.1, 0.1});
    auto evolving = makeSource(901, 0, {0.4, 0.1, 0.1});
    evolving.internalEnergyRate = 1.0e6;
    evolving.hydroAcceleration[0] = -1000.0;
    evolving.driftVelocity[0] = 1.0;
    forest.replaceActive(0, {evolving});
    requireNear(forest.signalTimestep(sink, 1.0, 0.25, 1.0),
        bruteForceSignalTimestep(sink, {evolving}, 1.0, 0.25, 1.0),
        "signal envelope omitted later heating, acceleration or periodic drift");
}

void testSignalPruningAndImmutability() {
    std::vector<SourceState> sources;
    for (uint64_t index = 0; index < 1024; ++index) {
        sources.push_back(makeSource(index, 1, {0.1 + 0.00001 * index, 0.1, 0.1}));
    }
    auto sink = makeSource(2000, 5, {0.8, 0.8, 0.8});
    sink.internalEnergy = 1.0e6;
    sources.push_back(sink);
    SourceForest forest(4);
    forest.replaceActive(0, sources);
    uint64_t const generation = forest.generation(1);
    uint64_t const fingerprint = forest.fingerprint(1);
    forest.resetCounters();
    requireNear(forest.signalTimestep(sink, 0.5, 0.25, 1.0),
        bruteForceSignalTimestep(sink, sources, 0.5, 0.25, 1.0), "pruned signal result changed");
    auto const counters = forest.counters();
    require(counters.signalQueries == 1 && counters.signalNodesPruned > 0,
        "signal query did not record tree pruning");
    require(counters.sourceRecordsSorted == 0 && counters.sourceRecordsQueried <= 4 &&
            counters.sourceRecordsPredicted <= 3,
        "signal query scanned or sorted the inactive population");
    require(forest.generation(1) == generation && forest.fingerprint(1) == fingerprint,
        "signal query mutated an inactive cohort");
    // A shared velocity boost must not affect the exact result or defeat pruning.
    for (auto& source : sources) {
        source.synchronizedVelocity = {1.0e5, -2.0e5, 3.0e5};
    }
    sink = sources.back();
    forest.replaceActive(0, sources);
    forest.resetCounters();
    requireNear(forest.signalTimestep(sink, 0.5, 0.25, 1.0),
        bruteForceSignalTimestep(sink, sources, 0.5, 0.25, 1.0),
        "common velocity boost changed signal timestep");
    require(forest.counters().sourceRecordsQueried <= 4,
        "common velocity boost defeated relative-velocity pruning");
}

void testThermalSourceSignalCeiling() {
    SourceForest forest(1);
    auto sink = makeSource(1, 3, {0.5, 0.5, 0.5});
    auto slow = makeSource(2, 1, {0.51, 0.5, 0.5});
    slow.synchronizedVelocity = {-30.0, 0.0, 0.0};
    forest.replaceActive(0, {sink, slow});
    auto const fingerprint = forest.fingerprint(1);
    auto const generation = forest.generation(1);
    forest.resetCounters();
    forest.setSignalSoundFloor(10.0);
    double const expected = 0.25 *
        std::min(sink.supportRadius / 10.0, (2.0 * sink.supportRadius + 0.01) / (20.0 + 30.0));
    requireNear(forest.signalTimestep(sink, 0.0, 0.25, 1.0), expected,
        "thermal-source ceiling missing from pair signal or pruning");
    require(forest.fingerprint(1) == fingerprint && forest.generation(1) == generation &&
            forest.counters().sourceRecordsSorted == 0,
        "thermal-source ceiling rebuilt a slow cohort");
    auto const sources = forest.query(pointBox(slow.position), 0.02, 0.0);
    for (auto const& source : sources) {
        requireNear(source.internalEnergy, 1.0, "signal ceiling heated physical source state");
    }
    forest.setSignalSoundFloor(0.0);
    requireNear(forest.signalTimestep(sink, 0.0, 0.25, 1.0),
        bruteForceSignalTimestep(sink, {sink, slow}, 0.0, 0.25, 1.0),
        "zero ceiling did not recover adiabatic signal bound");
    requireThrows([&] { forest.setSignalSoundFloor(-1.0); }, "negative signal ceiling accepted");
    requireThrows([&] { forest.setSignalSoundFloor(std::numeric_limits<double>::infinity()); },
        "infinite signal ceiling accepted");
}

void testSignalInvalidInputs() {
    SourceForest forest;
    auto sink = makeSource(1, 1, {0.8, 0.8, 0.8});
    requireThrows([&] { forest.signalTimestep(sink, 0.0, -0.1, 1.0); },
        "negative signal Courant factor accepted");
    requireThrows(
        [&] { forest.signalTimestep(sink, 0.0, 0.1, 0.0); }, "zero signal scale accepted");
    requireThrows(
        [&] { forest.signalTimestep(sink, 0.0, std::numeric_limits<double>::quiet_NaN(), 1.0); },
        "nonfinite signal Courant factor accepted");
    requireThrows([&] { forest.signalTimestep(sink, -0.1, 0.1, 1.0); },
        "signal sink predicted before its epoch");
    auto expired = makeSource(2, 0, {0.1, 0.1, 0.1});
    expired.validUntil = 0.1;
    forest.replaceActive(0, {expired});
    requireThrows([&] { forest.signalTimestep(sink, 0.5, 0.1, 1.0); },
        "expired signal cohort was pruned using stale bounds");
    sink.internalEnergy = -1.0;
    requireThrows([&] { forest.signalTimestep(sink, 0.0, 0.1, 1.0); },
        "negative signal sink energy accepted");
}

}    // namespace detail

int main() {
    try {
        detail::testPrediction();
        detail::testDerivedStatePreservation();
        detail::testPeriodicAndGrowingSupportCoverage();
        detail::testBruteForceCoverage();
        detail::testPermutationStableQueryOrder();
        detail::testReplacementCostAndImmutability();
        detail::testPredictionReuseAndParticleFiltering();
        detail::testFailureAndValidity();
        detail::testSignalEncountersAndUnits();
        detail::testSignalPredictionAndTreeReference();
        detail::testSignalPruningAndImmutability();
        detail::testThermalSourceSignalCeiling();
        detail::testSignalInvalidInputs();
        std::puts("retained SPH source geometry: all tests passed");
        return 0;
    } catch (std::exception const& error) {
        std::fprintf(stderr, "retained SPH source geometry: %s\n", error.what());
        return 1;
    }
}
