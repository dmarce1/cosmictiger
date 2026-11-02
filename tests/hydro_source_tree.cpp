/* CosmicTiger - persistent hydro source geometry regression; BSL-1.0. */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <cosmictiger/hydro/source_tree.hpp>

namespace detail {

using hydro_source::Box;
using hydro_source::Source;
using hydro_source::SourceTree;
using hydro_source::Vector;

void require(bool condition, char const* message) {
    if (!condition) {
        throw std::runtime_error(std::string("Hydro source tree: ") + message);
    }
}

template <class Exception, class Function>
void requireThrows(Function&& function, char const* message) {
    bool caught = false;
    try {
        function();
    } catch (Exception const&) {
        caught = true;
    }
    require(caught, message);
}

Box pointBox(Vector const& point) {
    return {point, point};
}

std::set<uint64_t> ids(std::vector<uint64_t> const& values) {
    std::set<uint64_t> result(values.begin(), values.end());
    require(result.size() == values.size(), "a source was returned more than once");
    return result;
}

// Independent long-double oracle: test the actual trajectory against periodic
// images of an axis-aligned query box, rather than the tree's polynomial bounds.
std::set<uint64_t> brute(std::vector<Source> const& sources, Box const& box, double support,
    double time, int maximumRung = std::numeric_limits<int>::max()) {
    std::set<uint64_t> result;
    for (auto const& source : sources) {
        if (source.rung > maximumRung) {
            continue;
        }
        long double squared = 0.0L;
        long double const age = static_cast<long double>(time) - source.epoch;
        for (size_t dimension = 0; dimension < 3; ++dimension) {
            long double const width = static_cast<long double>(box.upper[dimension]) -
                static_cast<long double>(box.lower[dimension]);
            if (width >= 1.0L) {
                continue;
            }
            long double const position = source.position[dimension] +
                static_cast<long double>(source.velocity[dimension]) * age +
                0.5L * source.acceleration[dimension] * age * age;
            long double const center = box.lower[dimension] + 0.5L * width;
            long double separation = position - center;
            separation -= std::round(separation);
            long double const gap = std::max(0.0L, std::abs(separation) - 0.5L * width);
            squared += gap * gap;
        }
        long double const radius = std::max(support, source.supportUpper);
        if (squared <= radius * radius) {
            result.insert(source.id);
        }
    }
    return result;
}

std::vector<Source> randomSources() {
    std::mt19937_64 random(0x4a928d37);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::vector<Source> result(257);
    for (size_t i = 0; i < result.size(); ++i) {
        auto& source = result[i];
        source.id = 10000 + 13 * i;
        source.epoch = -0.2 * unit(random);
        source.validUntil = 1.25;
        source.supportUpper = 0.01 + 0.07 * unit(random);
        source.signalUpper = 0.1 + unit(random);
        source.rung = i % 5;
        for (size_t dimension = 0; dimension < 3; ++dimension) {
            source.position[dimension] = unit(random);
            source.velocity[dimension] = 8.0 * unit(random) - 4.0;
            source.acceleration[dimension] = 12.0 * unit(random) - 6.0;
            source.signalVelocity[dimension] = 2.0 * unit(random) - 1.0;
            source.signalAcceleration[dimension] = unit(random) - 0.5;
        }
    }
    std::shuffle(result.begin(), result.end(), random);
    return result;
}

void compareQueries(SourceTree const& tree, std::vector<Source> const& sources) {
    std::mt19937_64 random(0x917a33);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    for (double time : {0.0, 0.125, 0.4, 0.75, 1.0}) {
        for (int query = 0; query < 35; ++query) {
            Box box;
            for (size_t dimension = 0; dimension < 3; ++dimension) {
                double const center = unit(random);
                double const halfWidth = 0.025 * unit(random);
                box.lower[dimension] = center - halfWidth;
                box.upper[dimension] = center + halfWidth;
            }
            double const support = 0.015 + 0.06 * unit(random);
            int const maximumRung = query % 6;
            require(ids(tree.query(box, support, time, maximumRung)) ==
                    brute(sources, box, support, time, maximumRung),
                "time-dependent query disagrees with brute force");
        }
    }
}

void randomMotionAndUpdates() {
    auto sources = randomSources();
    SourceTree tree(8);
    tree.reset(sources);
    compareQueries(tree, sources);
    auto const untouched = tree.source(sources.back().id);
    std::array<Source, 3> changed{sources[1], sources[14], sources[27]};
    for (auto& source : changed) {
        source.position[0] += 0.78;
        source.acceleration[1] *= -2.0;
        source.supportUpper *= 0.125;
        source.signalUpper *= 0.5;
        source.rung = 1;
        *std::find_if(sources.begin(), sources.end(),
            [&](Source const& original) { return original.id == source.id; }) = source;
    }
    tree.resetCounters();
    tree.update(changed);
    auto const work = tree.counters();
    require(work.sourcesSorted == 0 && work.topologyRebuilds == 0,
        "updating existing sources rebuilt or sorted topology");
    require(work.sourcesUpdated == changed.size() && work.nodesRefitted <= 3 * 10,
        "refit work was not proportional to changed ancestor paths");
    require(tree.source(untouched.id).position == untouched.position &&
            tree.source(untouched.id).velocity == untouched.velocity &&
            tree.source(untouched.id).epoch == untouched.epoch &&
            tree.source(untouched.id).supportUpper == untouched.supportUpper,
        "an untouched source anchor changed");
    compareQueries(tree, sources);
}

void translatedClump() {
    std::vector<Source> sources(128);
    Vector const bulk{17.25, -3.5, 4.0};
    for (size_t i = 0; i < sources.size(); ++i) {
        auto& source = sources[i];
        source.id = i;
        source.epoch = -0.001 * double(i % 7);
        source.validUntil = 1.0;
        source.supportUpper = 0.0005;
        source.velocity = bulk;
        for (size_t dimension = 0; dimension < 3; ++dimension) {
            source.position[dimension] =
                0.1 + 0.001 * double((i >> (2 * dimension)) & 3) + bulk[dimension] * source.epoch;
        }
    }
    SourceTree tree(8);
    tree.reset(sources);
    hydro_source::Bounds root;
    tree.visit(
        0.4,
        [&](hydro_source::Bounds const& bounds) {
            root = bounds;
            return false;
        },
        [](Source const&, Vector const&, Vector const&) {});
    for (size_t dimension = 0; dimension < 3; ++dimension) {
        require(root.position.upper[dimension] - root.position.lower[dimension] < 0.003000001,
            "coherent bulk motion expanded a node into a swept path");
    }
    tree.resetCounters();
    require(tree.query(pointBox({0.4, 0.4, 0.4}), 0.005, 0.4).empty(),
        "translated clump appeared in an empty region");
    require(tree.counters().nodesVisited == 1 && tree.counters().sourcesVisited == 0,
        "a distant clump query inspected inactive source records");
    Vector position{};
    for (size_t dimension = 0; dimension < 3; ++dimension) {
        position[dimension] = 0.101 + 0.4 * bulk[dimension];
    }
    require(ids(tree.query(pointBox(position), 0.01, 0.4)).size() == sources.size(),
        "translated clump lost sources across periodic images");
    // Refresh only half the sources with wrapped anchors. Their retained
    // geometry must stay in the same image as their still-sleeping neighbours.
    std::vector<Source> refreshed;
    for (size_t i = 0; i < sources.size(); i += 2) {
        auto source = sources[i];
        source.position = tree.position(source.id, 0.4);
        source.epoch = 0.4;
        refreshed.push_back(source);
    }
    tree.resetCounters();
    tree.update(refreshed);
    require(tree.source(refreshed.front().id).position == refreshed.front().position,
        "periodic image continuity changed the supplied source anchor");
    tree.visit(
        0.8,
        [&](hydro_source::Bounds const& bounds) {
            for (size_t dimension = 0; dimension < 3; ++dimension) {
                require(bounds.position.upper[dimension] - bounds.position.lower[dimension] <
                        0.003000001,
                    "partial wrapped-anchor updates separated a coherent clump by a period");
            }
            return false;
        },
        [](Source const&, Vector const&, Vector const&) {});
    require(tree.counters().sourcesSorted == 0 && tree.counters().topologyRebuilds == 0,
        "periodic anchor refresh rebuilt topology");
}

void completeUpdateRebasesTime() {
    std::vector<Source> sources(64);
    for (size_t i = 0; i < sources.size(); ++i) {
        auto& source = sources[i];
        source.id = i;
        source.position = {0.2 + 0.001 * double(i) / sources.size(), 0.3, 0.4};
        source.velocity = {10.0, 0.0, 0.0};
        source.validUntil = 1000001.0;
        source.supportUpper = 0.002;
    }
    SourceTree tree(4);
    tree.reset(sources);
    for (size_t i = 0; i < sources.size(); ++i) {
        sources[i].epoch = 1000000.0;
        sources[i].velocity[0] += 0.1 * double(i) / sources.size();
    }
    tree.resetCounters();
    tree.update(sources);
    tree.visit(
        1000000.0,
        [](hydro_source::Bounds const& bounds) {
            require(bounds.position.upper[0] - bounds.position.lower[0] < 0.001001,
                "complete update retained cancellation from an obsolete reference epoch");
            return false;
        },
        [](Source const&, Vector const&, Vector const&) {});
    auto const counters = tree.counters();
    require(counters.nodesRefitted == 31 && counters.sourcesSorted == 0 &&
            counters.topologyRebuilds == 0 && counters.sourcesUpdated == sources.size(),
        "complete update did not refit each existing node exactly once");
    require(tree.query(pointBox({0.2005, 0.3, 0.4}), 0.002, 1000000.0).size() == sources.size(),
        "reference-time rebase changed physical periodic positions");
}

void periodicSeamsAndReversal() {
    Source seam;
    seam.id = 1;
    seam.position = {0.99, 0.01, 0.5};
    seam.velocity = {0.08, -0.08, 0.0};
    seam.validUntil = 1.0;
    seam.supportUpper = 0.02;
    Source reversal;
    reversal.id = 2;
    reversal.position = {0.3, 0.5, 0.5};
    reversal.velocity = {1.0, 0.0, 0.0};
    reversal.acceleration = {-2.0, 0.0, 0.0};
    reversal.validUntil = 1.0;
    reversal.supportUpper = 0.005;
    SourceTree tree(1);
    tree.reset({seam, reversal});
    require(ids(tree.query({{-0.02, -0.02, 0.49}, {0.02, 0.02, 0.51}}, 0.01, 0.25)) ==
            std::set<uint64_t>{1},
        "periodic seam query missed a source");
    require(ids(tree.query(pointBox({0.55, 0.5, 0.5}), 0.01, 0.5)) == std::set<uint64_t>{2},
        "acceleration reversal was reduced to an endpoint hull");
    require(tree.query(pointBox({0.55, 0.5, 0.5}), 0.01, 0.0).empty() &&
            tree.query(pointBox({0.55, 0.5, 0.5}), 0.01, 1.0).empty(),
        "reversal fixture does not distinguish intermediate from endpoint positions");
    require(ids(tree.query({{-2.0, -2.0, -2.0}, {2.0, 2.0, 2.0}}, 0.0, 0.5)).size() == 2,
        "query wider than a period lost a source");
}

void signalMotionAndShrinkingSupport() {
    Source source;
    source.id = 99;
    source.position = {0.5, 0.5, 0.5};
    source.velocity = {0.25, 0.0, 0.0};
    source.acceleration = {-0.5, 0.0, 0.0};
    source.signalVelocity = {-7.0, 2.0, 1.0};
    source.signalAcceleration = {2.0, -4.0, 3.0};
    source.epoch = -0.25;
    source.validUntil = 1.0;
    source.supportUpper = 0.3;
    source.signalUpper = 4.5;
    source.rung = 3;
    SourceTree tree;
    tree.reset({source});
    size_t visited = 0;
    tree.visit(
        0.25,
        [&](hydro_source::Bounds const& bounds) {
            require(bounds.maxSupport == 0.3 && bounds.maxSignal == 4.5 && bounds.minRung == 3 &&
                    bounds.maxRung == 3 && bounds.latestEpoch == -0.25 &&
                    bounds.earliestValidUntil == 1.0,
                "source metadata was not propagated");
            for (size_t dimension = 0; dimension < 3; ++dimension) {
                double const fluid =
                    source.signalVelocity[dimension] + 0.5 * source.signalAcceleration[dimension];
                double const motion =
                    source.velocity[dimension] + 0.5 * source.acceleration[dimension];
                require(bounds.velocity.lower[dimension] <= fluid &&
                        fluid <= bounds.velocity.upper[dimension] &&
                        bounds.motionVelocity.lower[dimension] <= motion &&
                        motion <= bounds.motionVelocity.upper[dimension],
                    "fluid signal bounds used generator velocity");
            }
            return true;
        },
        [&](Source const& record, Vector const&, Vector const& velocity) {
            require(record.id == source.id && std::abs(velocity[0]) < 1.0e-15,
                "visitor did not receive the source's motion velocity");
            ++visited;
        });
    require(visited == 1, "signal traversal did not visit its source exactly once");
    Box const query = pointBox({0.75, 0.5, 0.5});
    require(!tree.query(query, 0.0, 0.0).empty(), "reverse-support fixture did not overlap");
    source.supportUpper = 0.01;
    source.signalUpper = 0.125;
    tree.update(std::span<Source const>(&source, 1));
    require(tree.query(query, 0.0, 0.0).empty(), "shrinking support retained an obsolete overlap");
    tree.visit(
        0.0,
        [](hydro_source::Bounds const& bounds) {
            require(bounds.maxSupport == 0.01 && bounds.maxSignal == 0.125,
                "refit only grew metadata and failed to shrink it");
            return false;
        },
        [](Source const&, Vector const&, Vector const&) {});
}

void membershipAndValidity() {
    Source source;
    source.id = 1;
    source.position = {0.25, 0.25, 0.25};
    source.validUntil = 0.5;
    source.supportUpper = 0.01;
    SourceTree tree(1);
    tree.reset({source});
    require(tree.query(pointBox({0.8, 0.8, 0.8}), 0.01, 0.75).empty(),
        "a distant expired anchor was needlessly visited");
    requireThrows<std::out_of_range>([&] { tree.query(pointBox(source.position), 0.01, 0.75); },
        "a touched expired anchor was accepted");
    requireThrows<std::out_of_range>(
        [&] { tree.position(source.id, -0.1); }, "position prediction preceded its anchor");
    Source added = source;
    added.id = 2;
    added.position = {0.75, 0.75, 0.75};
    tree.resetCounters();
    tree.upsert(std::span<Source const>(&added, 1));
    require(tree.size() == 2 && tree.counters().topologyRebuilds == 1 &&
            tree.counters().sourcesSorted == 2,
        "insertion fallback was not exposed by rebuild counters");
    tree.resetCounters();
    require(tree.erase(source.id) && !tree.contains(source.id) && tree.contains(added.id),
        "removal left stale source ownership");
    require(tree.counters().topologyRebuilds == 1 &&
            tree.query(pointBox(source.position), 0.01, 0.25).empty(),
        "removed source remained queryable");
    require(!tree.erase(source.id), "removing a missing source reported success");
    requireThrows<std::invalid_argument>([&] { tree.update(std::span<Source const>(&source, 1)); },
        "update accepted a stale source ID");
    requireThrows<std::invalid_argument>(
        [&] { tree.reset({added, added}); }, "duplicate membership was accepted");
    require(
        tree.size() == 1 && tree.contains(added.id), "failed reset corrupted the existing tree");
    Source replacement = added;
    replacement.id = 3;
    uint64_t const removedId = added.id;
    tree.resetCounters();
    tree.replace(
        std::span<Source const>(&replacement, 1), std::span<uint64_t const>(&removedId, 1));
    require(tree.size() == 1 && tree.contains(replacement.id) && !tree.contains(removedId) &&
            tree.counters().topologyRebuilds == 1 && tree.counters().sourcesSorted == 1,
        "combined membership replacement rebuilt more than once or retained a removed ID");
    replacement.supportUpper = 0.05;
    uint64_t const replacedId = replacement.id;
    tree.resetCounters();
    tree.replace(
        std::span<Source const>(&replacement, 1), std::span<uint64_t const>(&replacedId, 1));
    require(tree.contains(replacedId) && tree.source(replacedId).supportUpper == 0.05 &&
            tree.counters().topologyRebuilds == 0,
        "upsert did not take precedence over removal of the same ID");
    added.supportUpper = std::numeric_limits<double>::quiet_NaN();
    requireThrows<std::invalid_argument>([&] { tree.upsert(std::span<Source const>(&added, 1)); },
        "nonfinite metadata was accepted");
    tree.clear();
    require(tree.size() == 0 && tree.query(pointBox({0.0, 0.0, 0.0}), 0.0, 0.0).empty(),
        "empty tree remained queryable");
}

}    // namespace detail

int main() {
    try {
        detail::randomMotionAndUpdates();
        detail::translatedClump();
        detail::completeUpdateRebasesTime();
        detail::periodicSeamsAndReversal();
        detail::signalMotionAndShrinkingSupport();
        detail::membershipAndValidity();
        std::cout << "persistent hydro source tree: all tests passed\n";
        return 0;
    } catch (std::exception const& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
