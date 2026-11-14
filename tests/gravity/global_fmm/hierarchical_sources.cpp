/* CosmicTiger - hierarchical geometric source selection regression; BSL-1.0. */

#include <algorithm>
#include <cmath>
#include <exception>
#include <format>
#include <stdexcept>

#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/gravity/direct_reference.hpp>
#include <cosmictiger/gravity/global_fmm/distributed_tree.hpp>
#include <cosmictiger/gravity/global_fmm/execution.hpp>
#include <cosmictiger/math/kernel.hpp>
#include <cosmictiger/particles/particles.hpp>

namespace detail {

constexpr int kHierarchicalSourceParticles = 4;
constexpr float kInactiveForceSentinel = -12345.0f;
constexpr float kHierarchicalSourceTheta = .125f;

Vector3<Position> hierarchicalSourcePosition(int index) {
    Vector3<Position> result;
    for (int axis = 0; axis < kDimensions; ++axis) {
        int const value = ((index + 3) * (37 + 14 * axis) + 11 * index * index) % 251;
        result[axis] = double(value) / 256.0;
    }
    return result;
}

bool hierarchicalSourceActive(int index, bool emptyLast) {
    int const owner = index / kHierarchicalSourceParticles;
    return index % 2 != 0 && !(emptyLast && owner == hpxSize() - 1);
}

}    // namespace detail

void initializeHierarchicalSourceTest(bool emptyLast) {
    particlesResize(detail::kHierarchicalSourceParticles);
    for (int index = 0; index < detail::kHierarchicalSourceParticles; ++index) {
        int const globalIndex = hpxRank() * detail::kHierarchicalSourceParticles + index;
        auto const position = detail::hierarchicalSourcePosition(globalIndex);
        particlesRung(index) = detail::hierarchicalSourceActive(globalIndex, emptyLast) ? 1 : 0;
        for (int axis = 0; axis < kDimensions; ++axis) {
            particlesPos(axis, index) = position[axis];
            particlesVel(axis, index) = 0;
            particlesGravityAcceleration(axis, index) = detail::kInactiveForceSentinel;
        }
    }
}
HPX_PLAIN_ACTION(initializeHierarchicalSourceTest, InitializeHierarchicalSourceTestAction);

void verifyHierarchicalRangeTest(bool fullRange, bool emptyLast) {
    ParticleInteger const expectedFirst = fullRange ? 0 :
        emptyLast && hpxRank() == hpxSize() - 1     ? detail::kHierarchicalSourceParticles :
                                                      detail::kHierarchicalSourceParticles / 2;
    auto const range = particlesCurrentRange();
    if (range != Pair<ParticleInteger>{expectedFirst, detail::kHierarchicalSourceParticles}) {
        throw std::runtime_error("Geometric build changed a hierarchical range boundary");
    }
    if (!fullRange) {
        for (ParticleInteger index = 0; index < particlesOwnedSize(); ++index) {
            if ((index >= range.first) != (particlesRung(index) == 1)) {
                throw std::runtime_error("Geometric build changed hierarchical suffix membership");
            }
        }
    }
}
HPX_PLAIN_ACTION(verifyHierarchicalRangeTest, VerifyHierarchicalRangeTestAction);

Array<double, 3> verifyHierarchicalSourceTest(bool allSources, bool emptyLast) {
    Array<double, 3> norms{};
    auto const& options = getOptions();
    for (ParticleInteger index = 0; index < particlesOwnedSize(); ++index) {
        if (particlesRung(index) == 0) {
            for (int axis = 0; axis < kDimensions; ++axis) {
                if (particlesGravityAcceleration(axis, index) != detail::kInactiveForceSentinel ||
                    particlesVel(axis, index) != 0) {
                    throw std::runtime_error("Geometric rung solve modified an inactive sink");
                }
            }
            continue;
        }
        Vector3<double> reference{};
        for (int source = 0; source < hpxSize() * detail::kHierarchicalSourceParticles; ++source) {
            if (!allSources && !detail::hierarchicalSourceActive(source, emptyLast)) {
                continue;
            }
            auto const position = detail::hierarchicalSourcePosition(source);
            Vector3<double> displacement;
            for (int axis = 0; axis < kDimensions; ++axis) {
                displacement[axis] = doubleDistance(particlesPos(axis, index), position[axis]);
            }
            reference += detail::directGravityReference(
                displacement, selectedGravityKernel<double>(), options.gravity.smoothingLength);
        }
        for (int axis = 0; axis < kDimensions; ++axis) {
            double const value = particlesGravityAcceleration(axis, index);
            double const velocity = particlesVel(axis, index);
            if (!std::isfinite(value) || !std::isfinite(velocity)) {
                throw std::runtime_error("Geometric rung solve produced nonfinite acceleration");
            }
            norms[0] += (value - reference[axis]) * (value - reference[axis]);
            double const velocityError = -4.0 * velocity - reference[axis];
            norms[2] += velocityError * velocityError;
            norms[1] += reference[axis] * reference[axis];
        }
    }
    return norms;
}
HPX_PLAIN_ACTION(verifyHierarchicalSourceTest, VerifyHierarchicalSourceTestAction);

void geometricHierarchicalSourcesTest() {
    auto const savedOptions = getOptions();
    auto options = savedOptions;
    options.hydro.enabled = false;
    options.analysis.saveForce = true;
    options.compute.bucketSize = 8;
    options.gravity.smoothingLength = .005;
    setOptions(options);
    std::exception_ptr failure;
    try {
        for (bool emptyLast : {false, true}) {
            for (bool allSources : {false, true}) {
                auto initialized = detail::geometricLaunchAndDrain<void>(
                    hpxLocalities().size(), [emptyLast](size_t rank) {
                        return hpx::async<InitializeHierarchicalSourceTestAction>(
                            hpxLocalities()[rank], emptyLast);
                    });
                for (auto& future : initialized) {
                    future.get();
                }
                particlesSortByRung(0);
                particlesPushRungs();
                particlesSortByRung(1);
                if (allSources) {
                    bool rejected = false;
                    try {
                        treeCreateGeometric(
                            TreeCreateParameters(1, detail::kHierarchicalSourceTheta));
                    } catch (std::exception const&) {
                        rejected = true;
                    }
                    if (!rejected) {
                        throw std::runtime_error(
                            "A full-source build crossed an active rung boundary");
                    }
                    // A full-source solve filters sinks by rung without first
                    // partitioning its source range into a suffix.
                    particlesPopRungs();
                }
                auto const selection = allSources ? GeometricSourceSelection::AllOwned :
                                                    GeometricSourceSelection::CurrentRungRange;
                treeCreateGeometric(
                    TreeCreateParameters(1, detail::kHierarchicalSourceTheta), selection);
                for (auto const& root : treeGeometricRoots()) {
                    int const expected = allSources ? detail::kHierarchicalSourceParticles :
                        emptyLast && root.root.proc == hpxSize() - 1 ? 0 :
                                                                       2;
                    if (root.sourceCount != expected || root.multi[0] != expected) {
                        throw std::runtime_error("Geometric tree contains the wrong rung sources");
                    }
                }
                KickParameters parameters;
                parameters.htime = true;
                parameters.theta = detail::kHierarchicalSourceTheta;
                parameters.minRung = 1;
                parameters.t0 = 1;
                parameters.gm = 1;
                parameters.eta = options.gravity.timestepEta;
                parameters.gravitySmoothingLength = options.gravity.smoothingLength;
                parameters.saveForce = true;
                parameters.firstCall = false;
                parameters.gpu = options.gravity.cuda;
                parameters.nodeLoad = 1;
                parameters.ascending = true;
                parameters.descending = false;
                parameters.top = false;
                parameters.doPhi = true;
                kickSourceForest(parameters);
                treeRetireGeometric();
                particlesCacheFree();
                Array<double, 3> norms{};
                for (auto const& locality : hpxLocalities()) {
                    hpx::async<VerifyHierarchicalRangeTestAction>(locality, allSources, emptyLast)
                        .get();
                    auto const part = hpx::async<VerifyHierarchicalSourceTestAction>(
                        locality, allSources, emptyLast)
                                          .get();
                    norms[0] += part[0];
                    norms[1] += part[1];
                    norms[2] += part[2];
                }
                if (!allSources) {
                    particlesPopRungs();
                    for (auto const& locality : hpxLocalities()) {
                        hpx::async<VerifyHierarchicalRangeTestAction>(locality, true, emptyLast)
                            .get();
                    }
                }
                if (!(std::sqrt(std::max(norms[0], norms[2])) <=
                        1.0e-4 * std::sqrt(norms[1]) + 1.0e-5)) {
                    throw std::runtime_error(std::format(
                        "Geometric rung sources disagree with direct gravity: allSources={} "
                        "emptyLast={} forceError={} scaledVelocityError={} referenceNorm={}",
                        allSources, emptyLast, std::sqrt(norms[0]), std::sqrt(norms[2]),
                        std::sqrt(norms[1])));
                }
            }
        }
    } catch (...) {
        failure = std::current_exception();
    }
    if (failure) {
        treeRetireGeometric();
        particlesCacheFree();
        setOptions(savedOptions);
        std::rethrow_exception(failure);
    }
    setOptions(savedOptions);
}
