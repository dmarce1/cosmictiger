/* CosmicTiger - global FMM periodic force regression; BSL-1.0. */

// Check legacy/geometric periodic forces against direct sums and exercise source exchange.

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <format>
#include <iostream>
#include <limits>
#include <stdexcept>

#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/gravity/direct_reference.hpp>
#include <cosmictiger/gravity/global_fmm/distributed_tree.hpp>
#include <cosmictiger/gravity/global_fmm/execution.hpp>
#include <cosmictiger/gravity/global_fmm/measurement.hpp>
#include <cosmictiger/gravity/global_fmm/source_forest.hpp>
#include <cosmictiger/gravity/gravity.hpp>
#include <cosmictiger/gravity/tree.hpp>
#include <cosmictiger/gravity/tree_build.hpp>
#include <cosmictiger/math/kernel.hpp>
#include <cosmictiger/particles/domain.hpp>
#include <cosmictiger/particles/particles.hpp>

namespace detail {
constexpr int kForceParticles = 96;
// Eight symmetric corner shells plus the center keep distant depth-14
// quadrants internal at the tighter opening angle used by this fixture.
constexpr int kTerminalClusterParticles = 65;
constexpr float kTerminalFixtureTheta = .25f;

int terminalFixtureCount(int rank) {
    return kTerminalClusterParticles * (rank == 1 ? 2 : 1);
}

Vector3<Position> terminalFixturePosition(int rank, int index) {
    int const cluster = rank == 0 ? 1 : rank == 1 ? index / kTerminalClusterParticles : rank % 8;
    int const point = index % kTerminalClusterParticles;
    Vector3<Position> position;
    for (int axis = 0; axis < kDimensions; ++axis) {
        double const center = .15625 + .375 * ((cluster >> axis) & 1);
        double const offset = point == 0 ?
            0 :
            (1 + (point - 1) / 8) * (((point - 1) >> axis) & 1 ? 1.0 : -1.0) / 1024.0;
        position[axis] = center + offset;
    }
    return position;
}

Vector3<Position> forcePosition(int index, bool imageBoundary) {
    if (imageBoundary) {
        if (index == 0) {
            return {.235, .5, .5};
        }
        // From the isolated target at x=.235, the two source sheets select
        // opposite images: x=.72 lies within half a box, x=.74 beyond it.
        // Their shared cell center lies near x=.73, so accepting the smooth
        // correction there while independently wrapping Newtonian pairs gives
        // a large error. The isolated target keeps that error visible despite
        // the much stronger internal forces within each source sheet.
        int const source = index - 1;
        int const point = source / 2;
        return {.72 + .02 * (source % 2), .5 + .003 * (point % 7 - 3), .5 + .003 * (point / 7 - 3)};
    }
    // Deterministic, uneven points include both sides of each periodic seam.
    Vector3<Position> result;
    for (int dim = 0; dim < kDimensions; ++dim) {
        unsigned const seed = unsigned(index + 1) * unsigned(173 + dim * 52);
        result[dim] = double((seed * seed + 31 * seed + 7) % 4093) / 4096.0;
    }
    return result;
}
}    // namespace detail

void initializeGlobalForceTest(bool imageBoundary, bool emptyLast) {
    int const owners = emptyLast ? hpxSize() - 1 : hpxSize();
    int const begin = hpxRank() == owners ? detail::kForceParticles :
                                            detail::kForceParticles * hpxRank() / owners;
    int const end = hpxRank() == owners ? detail::kForceParticles :
                                          detail::kForceParticles * (hpxRank() + 1) / owners;
    particlesResize(end - begin);
    for (int i = begin; i < end; ++i) {
        auto const position = detail::forcePosition(i, imageBoundary);
        for (int dim = 0; dim < kDimensions; ++dim) {
            particlesPos(dim, i - begin) = position[dim];
            particlesVel(dim, i - begin) = 0.0f;
        }
        particlesRung(i - begin) = 0;
    }
}
HPX_PLAIN_ACTION(initializeGlobalForceTest, InitializeGlobalForceTestAction);

Array<double, 3> verifyGlobalForceTest(bool imageBoundary) {
    auto const& opts = getOptions();
    Array<double, 3> sums{};
    for (ParticleInteger i = 0; i < particlesOwnedSize(); ++i) {
        Vector3<double> reference{};
        for (int source = 0; source < detail::kForceParticles; ++source) {
            auto const position = detail::forcePosition(source, imageBoundary);
            Vector3<double> displacement;
            for (int dim = 0; dim < kDimensions; ++dim) {
                displacement[dim] = doubleDistance(particlesPos(dim, i), position[dim]);
            }
            reference += detail::directGravityReference(
                displacement, selectedGravityKernel<double>(), opts.gravity.smoothingLength);
        }
        double error = 0.0;
        for (int dim = 0; dim < kDimensions; ++dim) {
            double const value = particlesGravityAcceleration(dim, i);
            if (!std::isfinite(value)) {
                throw std::runtime_error("Global FMM produced a nonfinite force");
            }
            error += (value - reference[dim]) * (value - reference[dim]);
        }
        sums[0] += error;
        sums[1] += reference.dot(reference);
        sums[2] = std::max(sums[2], std::sqrt(error / reference.dot(reference)));
        if (!std::isfinite(particlesPot(i))) {
            throw std::runtime_error("Global FMM produced a nonfinite potential");
        }
    }
    return sums;
}
HPX_PLAIN_ACTION(verifyGlobalForceTest, VerifyGlobalForceTestAction);

Vector<double> captureGlobalForceTest() {
    Vector<double> values;
    values.reserve(4 * particlesOwnedSize());
    for (ParticleInteger index = 0; index < particlesOwnedSize(); ++index) {
        for (int axis = 0; axis < kDimensions; ++axis) {
            values.push_back(particlesGravityAcceleration(axis, index));
        }
        values.push_back(particlesPot(index));
    }
    return values;
}
HPX_PLAIN_ACTION(captureGlobalForceTest, CaptureGlobalForceTestAction);

Vector<Vector<double>> captureGlobalForces() {
    Vector<Vector<double>> values;
    for (auto const& locality : hpxLocalities()) {
        values.push_back(hpx::async<CaptureGlobalForceTestAction>(locality).get());
    }
    return values;
}

void compareGlobalForces(
    Vector<Vector<double>> const& reference, Vector<Vector<double>> const& imported) {
    if (reference.size() != imported.size()) {
        throw std::runtime_error("Geometric reference locality count changed");
    }
    for (size_t rank = 0; rank < reference.size(); ++rank) {
        if (reference[rank].size() != imported[rank].size()) {
            throw std::runtime_error("Geometric source walk changed owned sink count");
        }
        for (size_t index = 0; index < reference[rank].size(); ++index) {
            double const bound = 2.0e-5 * std::max(1.0, std::abs(reference[rank][index]));
            if (!(std::abs(reference[rank][index] - imported[rank][index]) <= bound)) {
                throw std::runtime_error(std::format(
                    "Geometric import differs from complete-source reference at rank {} value {} "
                    "(particle {} field {}): reference={} imported={} difference={} bound={}",
                    rank, index, index / (kDimensions + 1), index % (kDimensions + 1),
                    reference[rank][index], imported[rank][index],
                    imported[rank][index] - reference[rank][index], bound));
            }
        }
    }
}

uint64_t geometricFallbackLocal() {
    return treeSourceForestFallbackCount();
}
HPX_PLAIN_ACTION(geometricFallbackLocal, GeometricFallbackLocalAction);

void initializeTerminalPayloadTest() {
    int const count = detail::terminalFixtureCount(hpxRank());
    particlesResize(count);
    for (int index = 0; index < count; ++index) {
        auto const position = detail::terminalFixturePosition(hpxRank(), index);
        for (int axis = 0; axis < kDimensions; ++axis) {
            particlesPos(axis, index) = position[axis];
            particlesVel(axis, index) = 0;
        }
        particlesRung(index) = 0;
    }
}
HPX_PLAIN_ACTION(initializeTerminalPayloadTest, InitializeTerminalPayloadTestAction);

Array<double, 3> verifyTerminalPayloadForce() {
    Array<double, 3> sums{};
    for (ParticleInteger index = 0; index < particlesOwnedSize(); ++index) {
        bool center = false;
        for (int source = 0; source < detail::terminalFixtureCount(hpxRank());
            source += detail::kTerminalClusterParticles) {
            auto const position = detail::terminalFixturePosition(hpxRank(), source);
            bool matches = true;
            for (int axis = 0; axis < kDimensions; ++axis) {
                matches = matches && particlesPos(axis, index) == position[axis];
            }
            center = center || matches;
        }
        if (!center) {
            continue;
        }
        // Symmetric local shells cancel at their centers, exposing a lost
        // distant multipole even when the other sinks have strong local forces.
        Vector3<double> reference{};
        for (int rank = 0; rank < hpxSize(); ++rank) {
            for (int source = 0; source < detail::terminalFixtureCount(rank); ++source) {
                auto const position = detail::terminalFixturePosition(rank, source);
                Vector3<double> displacement;
                for (int axis = 0; axis < kDimensions; ++axis) {
                    displacement[axis] = doubleDistance(particlesPos(axis, index), position[axis]);
                }
                reference += detail::directGravityReference(displacement,
                    selectedGravityKernel<double>(), getOptions().gravity.smoothingLength);
            }
        }
        for (int axis = 0; axis < kDimensions; ++axis) {
            double const error = particlesGravityAcceleration(axis, index) - reference[axis];
            sums[0] += error * error;
            sums[1] += reference[axis] * reference[axis];
        }
        ++sums[2];
    }
    return sums;
}
HPX_PLAIN_ACTION(verifyTerminalPayloadForce, VerifyTerminalPayloadForceAction);

void geometricTerminalPayloadTest() {
    if (hpxSize() < 2) {
        return;
    }
    auto const savedOptions = getOptions();
    auto options = savedOptions;
    options.compute.bucketSize = 8;
    options.initial.numberOfParticles = detail::kTerminalClusterParticles * (hpxSize() + 1);
    options.gravity.smoothingLength = .02;
    // Hand the whole owned tree to CUDA before a CPU walk can consume terminals.
    options.gravity.cudaKickHandoffBlocks = 1;
    options.gravity.cudaKickFineHandoffBlocks = 0;
    setOptions(options);
    std::exception_ptr failure;
    try {
        auto initialized =
            detail::geometricLaunchAndDrain<void>(hpxLocalities().size(), [](size_t rank) {
                return hpx::async<InitializeTerminalPayloadTestAction>(hpxLocalities()[rank]);
            });
        for (auto& future : initialized) {
            future.get();
        }
        // Preserve the deliberate owner geometry: rank zero is close to only
        // the second of rank one's two internal clusters.
        particlesSortByRung(0);
        fmmMeasurementBegin();
        treeCreateGeometric(TreeCreateParameters(0, detail::kTerminalFixtureTheta));
        Vector<TreeId> pending{treeGeometricRoots()[1].root};
        ParticleInteger firstParticle = std::numeric_limits<ParticleInteger>::max();
        ParticleInteger firstTerminalEnd = std::numeric_limits<ParticleInteger>::max();
        while (!pending.empty()) {
            auto const id = pending.back();
            pending.pop_back();
            auto const* node = treeSourceForestImportedNode(id);
            if (node == nullptr) {
                throw std::runtime_error("Terminal payload fixture lost an imported node");
            }
            if (!node->leaf) {
                for (auto const child : node->children) {
                    pending.push_back(child);
                }
                continue;
            }
            if (node->numberOfParticles() == 0) {
                continue;
            }
            Vector<Position> x(node->numberOfParticles());
            Vector<Position> y(node->numberOfParticles());
            Vector<Position> z(node->numberOfParticles());
            bool const readable =
                treeSourceForestReadParticleRange(*node, x.data(), y.data(), z.data(), nullptr, 0);
            if (node->sourceKind == kTreeSourceMultipoleTerminal) {
                if (readable) {
                    throw std::runtime_error("Multipole terminal retained a particle payload");
                }
                firstTerminalEnd = std::min(firstTerminalEnd, node->particleRange.second);
            } else {
                if (node->sourceKind != kTreeSourceParticleLeaf || !readable) {
                    throw std::runtime_error("Fine source leaf lost its particle payload");
                }
                firstParticle = std::min(firstParticle, node->particleRange.first);
            }
        }
        if (firstParticle == std::numeric_limits<ParticleInteger>::max() ||
            firstTerminalEnd > firstParticle) {
            throw std::runtime_error(
                "Terminal fixture requires an omitted range before fine leaves");
        }

        KickParameters parameters;
        parameters.htime = true;
        parameters.theta = detail::kTerminalFixtureTheta;
        parameters.minRung = 0;
        parameters.t0 = 1;
        parameters.gm = 1;
        parameters.eta = options.gravity.timestepEta;
        parameters.gravitySmoothingLength = options.gravity.smoothingLength;
        parameters.saveForce = true;
        parameters.firstCall = true;
        parameters.gpu = options.gravity.cuda;
        parameters.nodeLoad = 1;
        parameters.ascending = parameters.descending = false;
        parameters.top = true;
        parameters.doPhi = true;
        auto const directErrors = [] {
            Array<double, 3> sums{};
            for (auto const& locality : hpxLocalities()) {
                auto const part = hpx::async<VerifyTerminalPayloadForceAction>(locality).get();
                for (size_t index = 0; index < sums.size(); ++index) {
                    sums[index] += part[index];
                }
            }
            return sums;
        };
        fmmMeasurementPrepareWalk();
        kickSourceForest(parameters, true);
        auto const reference = captureGlobalForces();
        auto const referenceErrors = directErrors();
        // Clear complete-source caches so they cannot conceal missing imports.
        fmmMeasurementPrepareWalk();
        if (options.gravity.cuda) {
            setGravityCounterUse(true);
            resetGravityCounters();
        }
        kickSourceForest(parameters);
        auto const imported = captureGlobalForces();
        auto const sums = directErrors();
        std::cout << std::format(
                         "Terminal center direct reference: complete error={} imported error={} "
                         "reference={} centers={} ({} localities)\n",
                         std::sqrt(referenceErrors[0]), std::sqrt(sums[0]), std::sqrt(sums[1]),
                         sums[2], hpxSize())
                  << std::flush;
        compareGlobalForces(reference, imported);
        auto const measurements = fmmMeasurementRead();
        for (auto const& rank : measurements) {
            if (rank.fallbackRequests != 0 ||
                hpx::async<GeometricFallbackLocalAction>(hpxLocalities()[rank.rank]).get() != 0) {
                throw std::runtime_error("Payload-free source walk required a staging fallback");
            }
            if (options.gravity.cuda &&
                !(rank.seconds[size_t(FmmMeasurementPhase::GpuStaging)] > 0)) {
                throw std::runtime_error(
                    "Terminal payload fixture did not reach CUDA on every rank");
            }
            for (size_t owner = 0; owner < rank.peers.size(); ++owner) {
                if (int(owner) == rank.rank) {
                    continue;
                }
                auto const& peer = rank.peers[owner];
                if (peer.coveredSources != uint64_t(detail::terminalFixtureCount(owner)) ||
                    peer.importedParticles + peer.coarseSources != peer.coveredSources ||
                    peer.demandSentMessages != 0 || peer.demandReceivedMessages != 0) {
                    throw std::runtime_error(
                        "Terminal source cover used extra particles or demand reads");
                }
            }
        }
        auto const& mixed = measurements[0].peers[1];
        if (mixed.coarseSources == 0 || mixed.importedParticles == 0) {
            throw std::runtime_error(
                "Terminal payload fixture did not mix coarse and fine sources");
        }
        double close = 0;
        double direct = 0;
        getGravityCounters(close, direct);
        if (options.gravity.cuda && !(close + direct > 0)) {
            throw std::runtime_error("Terminal payload fixture did not execute CUDA interactions");
        }
        // Match the configurable-order convergence bound used by the main
        // force fixture, retaining a floor for cancellation in these centers.
        double const bound = std::max(2.0e-4, std::pow(double(parameters.theta), kFmmOrder - 1));
        if (referenceErrors[2] != hpxSize() + 1 || sums[2] != hpxSize() + 1 ||
            !(std::sqrt(std::max(referenceErrors[0], sums[0])) <=
                bound * std::sqrt(sums[1]) + 1.0e-5)) {
            throw std::runtime_error(std::format("Terminal payload force differs from direct "
                                                 "gravity: complete error={} imported error={} "
                                                 "reference={} centers={}",
                std::sqrt(referenceErrors[0]), std::sqrt(sums[0]), std::sqrt(sums[1]), sums[2]));
        }
        std::cout
            << std::format(
                   "Payload-free terminal {} regression passed: {} coarse and {} particle sources; "
                   "GPU close/direct={}/{} ({} localities)\n",
                   options.gravity.cuda ? "CUDA" : "CPU", mixed.coarseSources,
                   mixed.importedParticles, close, direct, hpxSize())
            << std::flush;
    } catch (...) {
        failure = std::current_exception();
    }
    setGravityCounterUse(false);
    fmmMeasurementEnd();
    treeRetireGeometric(true);
    particlesCacheFree();
    setOptions(savedOptions);
    if (failure) {
        std::rethrow_exception(failure);
    }
}

void geometricHierarchicalSourcesTest();
void geometricPendingKickTest();
void geometricStagingCapacityTest();
void geometricExchangeGuardsTest();

void fmmWalkTest() {
    auto options = getOptions();
    options.hydro.enabled = false;
    options.analysis.saveForce = true;
    options.analysis.groups = options.analysis.tracers = false;
    options.compute.bucketSize = 8;
    options.initial.numberOfParticles = detail::kForceParticles;
    options.gravity.smoothingLength = .005;
    setOptions(options);
    bool passed = true;
    for (bool imageBoundary : {false, true}) {
        Vector<hpx::future<void>> initialization;
        for (auto const& locality : hpxLocalities()) {
            initialization.push_back(
                hpx::async<InitializeGlobalForceTestAction>(locality, imageBoundary, false));
        }
        for (auto& future : initialization) {
            future.get();
        }
        particlesSortByRung(0);
        domainsRebound();
        domainsBegin(0);
        domainsEnd();
        std::array<double, 2> previousRms{
            std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
        for (float const theta : {.5f, .25f}) {
            for (bool const geometric : {false, true}) {
                auto const built = geometric ? treeCreateGeometric(TreeCreateParameters(0, theta)) :
                                               treeCreate(TreeCreateParameters(0, theta));
                KickParameters parameters;
                parameters.htime = true;
                parameters.theta = theta;
                parameters.minRung = 0;
                parameters.t0 = 1.0f;
                parameters.gm = 1.0f;
                parameters.eta = options.gravity.timestepEta;
                parameters.gravitySmoothingLength = options.gravity.smoothingLength;
                parameters.saveForce = true;
                parameters.firstCall = true;
                parameters.gpu = options.gravity.cuda;
                parameters.nodeLoad = 1.0f;
                parameters.ascending = parameters.descending = false;
                parameters.top = true;
                parameters.doPhi = true;
                ExpansionBuffer local;
                local.zero();
                Vector3<Position> center{};
                Vector<TreeId> checks{built.id};
                if (geometric) {
                    kickSourceForest(parameters, true);
                    auto const reference = captureGlobalForces();
                    kickSourceForest(parameters);
                    compareGlobalForces(reference, captureGlobalForces());
                    for (auto const& locality : hpxLocalities()) {
                        if (hpx::async<GeometricFallbackLocalAction>(locality).get() != 0) {
                            throw std::runtime_error(
                                "Geometric source export required a reference fallback");
                        }
                    }
                } else {
                    kick(parameters, local, center, built.id, {}, checks, nullptr);
                }
                Array<double, 3> sums{};
                for (auto const& locality : hpxLocalities()) {
                    auto const part =
                        hpx::async<VerifyGlobalForceTestAction>(locality, imageBoundary).get();
                    sums[0] += part[0];
                    sums[1] += part[1];
                    sums[2] = std::max(sums[2], part[2]);
                }
                double const rms = std::sqrt(sums[0] / sums[1]);
                std::cout
                    << std::format(
                           "Global FMM {} {} theta={} relative RMS={} max={} ({} localities)\n",
                           geometric ? "geometric" : "legacy",
                           imageBoundary ? "image boundary" : "uneven points", theta, rms, sums[2],
                           hpxSize())
                    << std::flush;
                // The first omitted multipole scales with theta to the potential degree.
                // Keep a float-roundoff floor and require the tighter walk to converge.
                double const bound = std::max(1.0e-5, std::pow(double(theta), kFmmOrder - 1));
                passed = passed && rms < bound && sums[2] < 10 * bound &&
                    rms < .5 * previousRms[geometric] + 1.0e-5;
                previousRms[geometric] = rms;
                if (geometric) {
                    treeRetireGeometric(true);
                } else {
                    treeDestroy(true);
                }
                particlesCacheFree();
            }
        }
    }
    if (hpxSize() > 1) {
        Vector<hpx::future<void>> initialization;
        for (auto const& locality : hpxLocalities()) {
            initialization.push_back(
                hpx::async<InitializeGlobalForceTestAction>(locality, false, true));
        }
        for (auto& future : initialization) {
            future.get();
        }
        particlesSortByRung(0);
        treeCreateGeometric(TreeCreateParameters(0, .5f));
        auto const& roots = treeGeometricRoots();
        ParticleInteger totalSources = 0;
        for (auto const& root : roots) {
            totalSources += root.sourceCount;
        }
        if (roots.back().sourceCount != 0 || totalSources != detail::kForceParticles) {
            throw std::runtime_error("Empty-rank geometric source descriptors are incorrect");
        }
        KickParameters parameters;
        parameters.htime = true;
        parameters.theta = .5f;
        parameters.minRung = 0;
        parameters.t0 = 1.0f;
        parameters.gm = 1.0f;
        parameters.eta = options.gravity.timestepEta;
        parameters.gravitySmoothingLength = options.gravity.smoothingLength;
        parameters.saveForce = true;
        parameters.firstCall = true;
        parameters.gpu = options.gravity.cuda;
        parameters.nodeLoad = 1.0f;
        parameters.top = true;
        parameters.doPhi = true;
        kickSourceForest(parameters, true);
        auto const reference = captureGlobalForces();
        kickSourceForest(parameters);
        compareGlobalForces(reference, captureGlobalForces());
        for (auto const& locality : hpxLocalities()) {
            if (hpx::async<GeometricFallbackLocalAction>(locality).get() != 0) {
                throw std::runtime_error("Empty-rank source export used a reference fallback");
            }
            auto const part = hpx::async<VerifyGlobalForceTestAction>(locality, false).get();
            double const bound = std::pow(.5, kFmmOrder - 1);
            passed = passed && std::sqrt(part[0] / std::max(part[1], 1.0e-30)) < bound &&
                part[2] < 10 * bound;
        }
        treeRetireGeometric(true);
        particlesCacheFree();
    }
    if (!passed) {
        throw std::runtime_error("Global FMM disagrees with independent periodic direct force");
    }
    geometricTerminalPayloadTest();
    geometricHierarchicalSourcesTest();
    geometricPendingKickTest();
    geometricStagingCapacityTest();
    geometricExchangeGuardsTest();
}
