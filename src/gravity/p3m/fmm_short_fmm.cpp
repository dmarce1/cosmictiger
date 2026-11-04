/* CosmicTiger - P3M screened short-range FMM (FMM-PM); BSL-1.0. */

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>

#include <cosmictiger/core/options.hpp>
#include <cosmictiger/gravity/fmm_order.hpp>
#include <cosmictiger/gravity/kernels/cartesian_operators.hpp>
#include <cosmictiger/gravity/p3m/fmm/kernels.hpp>
#include <cosmictiger/gravity/p3m/fmm/short_fmm.hpp>
#include <cosmictiger/math/kernel.hpp>

#include "state.hpp"

namespace cosmictiger::gravity::p3m::fmm {

namespace {

    template <int order>
    using Multipole = cartesian::Multipole<double, order>;
    template <int order>
    using Expansion = cartesian::Expansion<double, order>;
    template <int order>
    using Green = cartesian::Green<double, order>;

    template <int order>
    Multipole<order> loadMultipole(ShortTree const& tree, int node) {
        Multipole<order> result;
        double const* source = tree.nodeMoments(node);
        for (int i = 0; i < Multipole<order>::kCount_; ++i) {
            result[i] = source[i];
        }
        return result;
    }

    template <int order>
    void storeMultipole(ShortTree& tree, int node, Multipole<order> const& value) {
        double* target = tree.moments.data() + size_t(node) * tree.momentCount;
        for (int i = 0; i < Multipole<order>::kCount_; ++i) {
            target[i] = value[i];
        }
    }

    // Center of mass and the radius enclosing every particle of the range.
    void nodeGeometry(ShortTree& tree, int node) {
        auto& entry = tree.nodes[node];
        Vector3<double> center{};
        double mass = 0;
        for (int i = entry.begin; i < entry.end; ++i) {
            auto const& particle = tree.particles[i];
            for (int axis = 0; axis < kDimensions; ++axis) {
                center[axis] += particle.mass * particle.position[axis];
            }
            mass += particle.mass;
        }
        if (mass > 0) {
            for (int axis = 0; axis < kDimensions; ++axis) {
                center[axis] /= mass;
            }
        } else if (entry.end > entry.begin) {
            for (int axis = 0; axis < kDimensions; ++axis) {
                center[axis] = tree.particles[entry.begin].position[axis];
            }
        }
        double radiusSquared = 0;
        for (int i = entry.begin; i < entry.end; ++i) {
            auto const& particle = tree.particles[i];
            double distanceSquared = 0;
            for (int axis = 0; axis < kDimensions; ++axis) {
                double const d = particle.position[axis] - center[axis];
                distanceSquared += d * d;
            }
            radiusSquared = std::max(radiusSquared, distanceSquared);
        }
        entry.center = center;
        entry.radius = std::sqrt(radiusSquared);
    }

    template <int order>
    void buildNode(ShortTree& tree, int node, int bucketSize, FlopCount& flops) {
        nodeGeometry(tree, node);
        auto const range = std::make_pair(tree.nodes[node].begin, tree.nodes[node].end);
        flops += FlopCount(range.second - range.first) * 14;
        if (range.second - range.first <= bucketSize) {
            tree.nodes[node].leaf = true;
            Multipole<order> moments;
            moments = 0.0;
            for (int i = range.first; i < range.second; ++i) {
                auto const& particle = tree.particles[i];
                Vector3<double> offset;
                for (int axis = 0; axis < kDimensions; ++axis) {
                    offset[axis] = particle.position[axis] - tree.nodes[node].center[axis];
                }
                flops += 3 + cartesian::P2M(moments, particle.mass, offset);
            }
            storeMultipole<order>(tree, node, moments);
            return;
        }
        // Split the longest axis of the particle bounding box at its midpoint,
        // falling back to the median when every particle lies on one side.
        Vector3<double> lower{}, upper{};
        for (int axis = 0; axis < kDimensions; ++axis) {
            lower[axis] = std::numeric_limits<double>::max();
            upper[axis] = -std::numeric_limits<double>::max();
        }
        for (int i = range.first; i < range.second; ++i) {
            for (int axis = 0; axis < kDimensions; ++axis) {
                lower[axis] = std::min(lower[axis], tree.particles[i].position[axis]);
                upper[axis] = std::max(upper[axis], tree.particles[i].position[axis]);
            }
        }
        int axis = 0;
        for (int candidate = 1; candidate < kDimensions; ++candidate) {
            if (upper[candidate] - lower[candidate] > upper[axis] - lower[axis]) {
                axis = candidate;
            }
        }
        double const split = 0.5 * (lower[axis] + upper[axis]);
        auto const first = tree.particles.begin() + range.first;
        auto const last = tree.particles.begin() + range.second;
        auto middle = std::partition(first, last,
            [&](ShortParticle const& particle) { return particle.position[axis] < split; });
        if (middle == first || middle == last) {
            middle = first + (range.second - range.first) / 2;
            std::nth_element(
                first, middle, last, [&](ShortParticle const& a, ShortParticle const& b) {
                    return a.position[axis] < b.position[axis];
                });
        }
        int const mid = int(middle - tree.particles.begin());
        tree.nodes[node].leaf = false;
        int const left = int(tree.nodes.size());
        tree.nodes.push_back({});
        tree.nodes.push_back({});
        tree.nodes[node].left = left;
        tree.nodes[node].right = left + 1;
        tree.nodes[left].begin = range.first;
        tree.nodes[left].end = mid;
        tree.nodes[left + 1].begin = mid;
        tree.nodes[left + 1].end = range.second;
        tree.moments.resize(tree.nodes.size() * size_t(tree.momentCount), 0.0);
        buildNode<order>(tree, left, bucketSize, flops);
        buildNode<order>(tree, left + 1, bucketSize, flops);
        Multipole<order> moments;
        moments = 0.0;
        for (int child : {left, left + 1}) {
            auto const childMoments = loadMultipole<order>(tree, child);
            Vector3<double> displacement;
            for (int d = 0; d < kDimensions; ++d) {
                displacement[d] = tree.nodes[child].center[d] - tree.nodes[node].center[d];
            }
            flops += 3 + cartesian::M2M(moments, childMoments, displacement);
        }
        storeMultipole<order>(tree, node, moments);
    }

    struct SourceItem {
        ShortTree const* tree;
        int node;
        Vector3<double> shift;
    };

    template <int order>
    struct Walker {
        ShortTree const& sinks;
        ShortWalkParameters const& parameters;
        std::vector<std::array<double, 4>>& fields;
        ShortWalkStatistics statistics{};
        GaussianScreening<double> screening{parameters.pair.alpha};
        FlopCount flops = 0;

        void visit(int sinkNode, Expansion<order> const& parent,
            Vector3<double> const& parentCenter, bool hasParent,
            std::vector<SourceItem> checklist) {
            auto const& sink = sinks.nodes[sinkNode];
            Expansion<order> local;
            local = 0.0;
            if (hasParent) {
                Vector3<double> displacement;
                for (int axis = 0; axis < kDimensions; ++axis) {
                    displacement[axis] = sink.center[axis] - parentCenter[axis];
                }
                flops += 3 + cartesian::L2L(local, parent, displacement, parameters.doPotential);
            }
            std::vector<SourceItem> nextlist;
            std::vector<SourceItem> pairs;
            double const cutoff = parameters.cutoffRadius;
            double const support = parameters.pair.smoothingLength;
            double const theta = parameters.openingAngle;
            do {
                nextlist.clear();
                for (auto const& item : checklist) {
                    auto const& source = item.tree->nodes[item.node];
                    Vector3<double> displacement;
                    double distanceSquared = 0;
                    for (int axis = 0; axis < kDimensions; ++axis) {
                        displacement[axis] =
                            sink.center[axis] - (source.center[axis] + item.shift[axis]);
                        distanceSquared += displacement[axis] * displacement[axis];
                    }
                    double const distance = std::sqrt(distanceSquared);
                    double const radii = sink.radius + source.radius;
                    flops += 12;
                    if (distance - radii >= cutoff) {
                        ++statistics.cutoffRejections;
                        continue;
                    }
                    bool const insideCutoff = distance + radii < cutoff;
                    bool const outsideSupport = distance - radii > support;
                    bool const resolved = distance > 0 &&
                        radii < theta * screenedAcceptanceLength(distance, parameters.splitRadius);
                    if (insideCutoff && outsideSupport && resolved) {
                        Green<order> green;
                        p3m::fmm::evaluateGreen(green, displacement, screening);
                        auto const moments = loadMultipole<order>(*item.tree, item.node);
                        flops += 40 + cartesian::M2L(local, moments, green, parameters.doPotential);
                        ++statistics.acceptedMultipoles;
                    } else if (sink.leaf && source.leaf) {
                        pairs.push_back(item);
                    } else if (source.leaf) {
                        nextlist.push_back(item);
                    } else if (sink.leaf || source.radius > sink.radius) {
                        nextlist.push_back({item.tree, source.left, item.shift});
                        nextlist.push_back({item.tree, source.right, item.shift});
                        ++statistics.sourceOpenings;
                    } else {
                        nextlist.push_back(item);
                    }
                }
                std::swap(checklist, nextlist);
            } while (!checklist.empty() && sink.leaf);
            if (sink.leaf) {
                auto const& kernel = selectedGravityKernel();
                for (int i = sink.begin; i < sink.end; ++i) {
                    auto const& particle = sinks.particles[i];
                    auto& field = fields[i];
                    Vector3<double> displacement;
                    for (int axis = 0; axis < kDimensions; ++axis) {
                        displacement[axis] = particle.position[axis] - sink.center[axis];
                    }
                    gravity::Field<double> value;
                    for (int component = 0; component < 4; ++component) {
                        value[component] = 0.0;
                    }
                    flops += 3 + cartesian::L2P(value, local, displacement, parameters.doPotential);
                    field[0] += value.potential();
                    field[1] += value.accelerationX();
                    field[2] += value.accelerationY();
                    field[3] += value.accelerationZ();
                    for (auto const& item : pairs) {
                        auto const& source = item.tree->nodes[item.node];
                        for (int j = source.begin; j < source.end; ++j) {
                            auto const& other = item.tree->particles[j];
                            if (other.rank == particle.rank && other.index == particle.index) {
                                continue;
                            }
                            flops += 3 +
                                accumulateShortPair(
                                    particle.position[0] - (other.position[0] + item.shift[0]),
                                    particle.position[1] - (other.position[1] + item.shift[1]),
                                    particle.position[2] - (other.position[2] + item.shift[2]),
                                    other.mass, parameters.pair, parameters.doPotential, kernel,
                                    field);
                            ++statistics.directPairs;
                        }
                    }
                }
            } else {
                visit(sink.left, local, sink.center, true, checklist);
                visit(sink.right, local, sink.center, true, std::move(checklist));
            }
        }
    };

}    // namespace

std::shared_ptr<ShortTree> buildShortTree(
    std::vector<ShortParticle> particles, ShortWalkParameters const& parameters) {
    auto tree = std::make_shared<ShortTree>();
    tree->order = fmmOrder();
    tree->particles = std::move(particles);
    FlopCount flops = 0;
    fmmDispatch(tree->order, [&](auto orderTag) {
        constexpr int kOrder = decltype(orderTag)::value;
        tree->momentCount = Multipole<kOrder>::kCount_;
        tree->nodes.push_back({});
        tree->nodes[0].begin = 0;
        tree->nodes[0].end = int(tree->particles.size());
        tree->moments.assign(size_t(tree->momentCount), 0.0);
        if (!tree->particles.empty()) {
            buildNode<kOrder>(*tree, 0, std::max(1, parameters.bucketSize), flops);
        } else {
            tree->nodes[0].leaf = true;
        }
    });
    addCpuFlops(flops, FlopPrecision::Double);
    return tree;
}

ShortWalkStatistics walkShortTree(ShortTree const& sinks,
    std::vector<std::pair<ShortTree const*, Vector3<double>>> const& sources,
    ShortWalkParameters const& parameters, std::vector<std::array<double, 4>>& fields) {
    fields.assign(sinks.particles.size(), {});
    ShortWalkStatistics statistics;
    if (sinks.particles.empty()) {
        return statistics;
    }
    fmmDispatch(sinks.order, [&](auto orderTag) {
        constexpr int kOrder = decltype(orderTag)::value;
        Walker<kOrder> walker{sinks, parameters, fields};
        std::vector<SourceItem> checklist;
        for (auto const& [tree, shift] : sources) {
            if (tree != nullptr && !tree->particles.empty()) {
                checklist.push_back({tree, 0, shift});
            }
        }
        Expansion<kOrder> root;
        root = 0.0;
        walker.visit(0, root, {}, false, std::move(checklist));
        statistics = walker.statistics;
        addCpuFlops(walker.flops, FlopPrecision::Double);
    });
    return statistics;
}

}    // namespace cosmictiger::gravity::p3m::fmm

namespace p3m::detail {

using namespace cosmictiger::gravity::p3m;

void evaluateTrees() {
    auto const& gravity = getOptions().gravity;
    int const n = state.meshSize;
    int const width = gravity.pmmmNearWidth;
    fmm::ShortWalkParameters parameters;
    parameters.pair = makeShortPairParameters(
        gravity.p3mSplitRadius, gravity.p3mCutoffRadius, gravity.smoothingLength);
    parameters.splitRadius = gravity.p3mSplitRadius;
    parameters.cutoffRadius = gravity.p3mCutoffRadius;
    parameters.openingAngle = gravity.openingAngle;
    parameters.bucketSize = getOptions().compute.bucketSize;
    parameters.doPotential = state.request.doPotential;
    parallelFor(state.sourceCells.size(), "gravity.p3m.buildSourceTrees", [&](size_t index) {
        auto& source = state.sourceCells[index];
        source.tree = fmm::buildShortTree(source.particles, parameters);
        std::vector<ShortParticle>().swap(source.particles);
    });
    uint64_t treeNodes = 0;
    for (auto const& source : state.sourceCells) {
        treeNodes += source.tree->nodes.size();
    }
    state.diagnostics.sourceTreeNodes = treeNodes;
    parallelFor(state.cells.size(), "gravity.p3m.walkSinkCells", [&](size_t cellIndex) {
        auto const& cell = state.cells[cellIndex];
        bool hasActive = false;
        for (ParticleInteger i = cell.begin; i < cell.end && !hasActive; ++i) {
            hasActive = activeSink(i);
        }
        if (!hasActive) {
            return;
        }
        auto const center = coordinates(cell.key);
        std::vector<std::pair<fmm::ShortTree const*, Vector3<double>>> sources;
        for (int x = -width; x <= width; ++x) {
            for (int y = -width; y <= width; ++y) {
                for (int z = -width; z <= width; ++z) {
                    int const raw[3] = {center[0] + x, center[1] + y, center[2] + z};
                    Vector3<double> shift;
                    uint64_t key = 0;
                    for (int axis = 0; axis < kDimensions; ++axis) {
                        int const wrapped = periodicIndex(raw[axis]);
                        shift[axis] = double(raw[axis] - wrapped) / n;
                        key = key * n + wrapped;
                    }
                    auto const found = state.sourceByKey.find(key);
                    if (found != state.sourceByKey.end()) {
                        sources.emplace_back(state.sourceCells[found->second].tree.get(), shift);
                    }
                }
            }
        }
        auto const& sinkTree = *state.sourceCells[state.sourceByKey.at(cell.key)].tree;
        std::vector<std::array<double, 4>> fields;
        auto const statistics = fmm::walkShortTree(sinkTree, sources, parameters, fields);
        KickReturn result;
        FlopCount flops = 0;
        FlopCounts kickFlops;
        uint64_t evaluated = 0;
        for (size_t i = 0; i < sinkTree.particles.size(); ++i) {
            auto const index = ParticleInteger(sinkTree.particles[i].index);
            if (!activeSink(index)) {
                continue;
            }
            consumeField(index, fields[i], result, flops, kickFlops);
            ++evaluated;
        }
        addCpuFlops(flops + kickFlops.doublePrecision, FlopPrecision::Double);
        addCpuFlops(kickFlops.singlePrecision, FlopPrecision::Single);
        std::lock_guard<MutexType> lock(state.mutex);
        state.kickResult += result;
        state.diagnostics.evaluatedParticles += evaluated;
        state.diagnostics.directPairs += statistics.directPairs;
        state.diagnostics.acceptedMultipoles += statistics.acceptedMultipoles;
        state.diagnostics.cutoffRejections += statistics.cutoffRejections;
        ++state.diagnostics.batches;
    });
}

}    // namespace p3m::detail
