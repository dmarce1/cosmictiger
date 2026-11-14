/* CosmicTiger - geometric source lifetime and kick preflight regressions; BSL-1.0. */

// Test source payload serialization, frozen-generation checks and exchange failure cleanup.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <exception>
#include <stdexcept>
#include <vector>

#include <hpx/serialization/input_archive.hpp>
#include <hpx/serialization/output_archive.hpp>

#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/gravity/direct_reference.hpp>
#include <cosmictiger/gravity/global_fmm/distributed_tree.hpp>
#include <cosmictiger/gravity/global_fmm/execution.hpp>
#include <cosmictiger/gravity/global_fmm/source_forest.hpp>
#include <cosmictiger/gravity/gravity.hpp>
#include <cosmictiger/gravity/kernels/cartesian_tracefree_operators.hpp>
#include <cosmictiger/particles/particles.hpp>
#if COSMICTIGER_WITH_SPH
#include <cosmictiger/sph/particles.hpp>
#endif

namespace detail {
constexpr ParticleInteger kSourcePayloadParticles = 6;
#if COSMICTIGER_WITH_SPH
std::array<ParticleInteger, kSourcePayloadParticles> payloadGasIndices;
std::array<char, kSourcePayloadParticles> payloadParticleTypes;
ParticleInteger* savedPayloadGasIndices = nullptr;
char* savedPayloadParticleTypes = nullptr;
bool payloadArraysInstalled = false;
#endif

char sourcePayloadType(int mode, ParticleInteger index) {
    return mode == 0 ? kDarkMatterType :
        mode == 1    ? (index % 2 == 0 ? kSphType : kStarType) :
                       char(index % 3);
}

Vector3<double> sourcePayloadForceReference(Vector3<Position> const& sink, int mode) {
    Vector3<double> reference{};
    for (ParticleInteger index = 0; index < kSourcePayloadParticles; ++index) {
        Vector3<double> displacement;
        for (int axis = 0; axis < kDimensions; ++axis) {
            Position const source = double(index + 1 + axis) / 16.0;
            displacement[axis] = doubleDistance(sink[axis], source);
        }
        // Fixture masses are specified independently of the production type lookup.
        double const mass = mode == 0                         ? 1.0 :
            sourcePayloadType(mode, index) == kDarkMatterType ? 2.5 :
                                                                .75;
        reference += directGravityReference(displacement, selectedGravityKernel<double>(),
                         getOptions().gravity.smoothingLength, false) *
            (mass * hpxSize());
    }
    return reference;
}

void requireSourcePayloadForce(
    Vector3<double> const& actual, Vector3<double> const& expected, char const* message) {
    auto const difference = actual - expected;
    double const bound = 3.0e-6 * std::max(1.0, std::sqrt(expected.dot(expected)));
    if (!(difference.dot(difference) <= bound * bound)) {
        throw std::runtime_error(message);
    }
}

void sourcePayloadWireTest() {
    constexpr int kWirePayloadParticles = 4;
    auto const encode = [](SourceExportView const& view) {
        std::vector<char> bytes;
        hpx::serialization::output_archive archive(bytes);
        archive << view;
        return bytes;
    };
    SourceExportView uniform;
    SourceExportParticle particle;
    particle.position = {Position(.125), Position(.25), Position(.375)};
    uniform.appendParticle(particle, kStarType);
    auto const oneParticleBytes = encode(uniform);
    for (int index = 1; index < kWirePayloadParticles; ++index) {
        uniform.appendParticle(particle, kSphType);
    }
    auto const uniformBytes = encode(uniform);
    if (!uniform.particleTypes.empty() || uniform.uniformParticleType != kSphType ||
        uniformBytes.size() - oneParticleBytes.size() !=
            (kWirePayloadParticles - 1) * kDimensions * sizeof(Position)) {
        throw std::runtime_error("Uniform source wire payload is not three positions per particle");
    }
    SourceExportView mixed;
    for (int index = 0; index < kWirePayloadParticles; ++index) {
        mixed.appendParticle(particle, index % 2 == 0 ? kDarkMatterType : kStarType);
    }
    auto const mixedBytes = encode(mixed);
    if (mixedBytes.size() != uniformBytes.size() + mixed.particles.size()) {
        throw std::runtime_error("Mixed source wire payload does not add exactly one species byte");
    }
    SourceExportView restored;
    hpx::serialization::input_archive archive(mixedBytes);
    archive >> restored;
    if (restored.particles.size() != mixed.particles.size() ||
        restored.particleTypes != mixed.particleTypes) {
        throw std::runtime_error("Source species payload did not survive serialization");
    }
    for (auto const& restoredParticle : restored.particles) {
        for (int axis = 0; axis < kDimensions; ++axis) {
            if (restoredParticle.position[axis].raw() != particle.position[axis].raw()) {
                throw std::runtime_error("Source positions changed during wire serialization");
            }
        }
    }
}
}    // namespace detail

void sourcePayloadFixtureArrays(bool install) {
#if COSMICTIGER_WITH_SPH
    if (install) {
        particlesResize(detail::kSourcePayloadParticles);
        detail::savedPayloadGasIndices = particlesSph;
        detail::savedPayloadParticleTypes = particlesTy;
        particlesSph = detail::payloadGasIndices.data();
        particlesTy = detail::payloadParticleTypes.data();
        detail::payloadArraysInstalled = true;
    } else if (detail::payloadArraysInstalled) {
        particlesResize(0);
        sphParticlesResize(0, false);
        particlesSph = detail::savedPayloadGasIndices;
        particlesTy = detail::savedPayloadParticleTypes;
        detail::payloadArraysInstalled = false;
    }
#endif
}
HPX_PLAIN_ACTION(sourcePayloadFixtureArrays, SourcePayloadFixtureArraysAction);

void initializeSourcePayloadLocal(int mode) {
    particlesResize(detail::kSourcePayloadParticles);
#if COSMICTIGER_WITH_SPH
    sphParticlesResize(mode == 0 ? 0 : mode == 1 ? 3 : 2, false);
    ParticleInteger gasIndex = 0;
#endif
    for (ParticleInteger index = 0; index < particlesOwnedSize(); ++index) {
        for (int axis = 0; axis < kDimensions; ++axis) {
            particlesPos(axis, index) = double(index + 1 + axis) / 16.0;
        }
        particlesRung(index) = 0;
#if COSMICTIGER_WITH_SPH
        if (mode != 0) {
            particlesType(index) = detail::sourcePayloadType(mode, index);
            particlesCatIndex(index) = kNoParticleIndex;
            if (particlesType(index) == kSphType) {
                particlesCatIndex(index) = gasIndex;
                sphParticlesParticleIndex(gasIndex++) = index;
            }
        }
#endif
    }
}
HPX_PLAIN_ACTION(initializeSourcePayloadLocal, InitializeSourcePayloadLocalAction);

void verifySourcePayloadLocal(Vector<SourceRootDescriptor> roots, int mode) {
    for (ParticleInteger index = 0; index < particlesOwnedSize(); ++index) {
        auto const original = ParticleInteger(particlesPos(kXi, index).toDouble() * 16) - 1;
        if (particlesType(index) != detail::sourcePayloadType(mode, original)) {
            throw std::runtime_error("Source export changed an owned particle species");
        }
    }
    double expectedTotalMass = 0;
    for (ParticleInteger index = 0; index < detail::kSourcePayloadParticles; ++index) {
        expectedTotalMass += mode == 0                                ? 1.0 :
            detail::sourcePayloadType(mode, index) == kDarkMatterType ? 2.5 :
                                                                        .75;
    }
    for (auto const& root : roots) {
        if (root.multi[0] != expectedTotalMass) {
            throw std::runtime_error("Source root mass does not match its particle species");
        }
        if (root.root.proc == hpxRank()) {
            continue;
        }
        auto const* node = treeSourceForestImportedNode(root.root);
        if (node == nullptr) {
            throw std::runtime_error("Source payload regression is missing an imported root");
        }
        std::array<Position, detail::kSourcePayloadParticles + 2> x{}, y{}, z{};
        std::array<char, detail::kSourcePayloadParticles + 2> types{};
        x.front() = x.back() = y.front() = y.back() = z.front() = z.back() = Position::max();
        types.fill(-1);
        if (!treeSourceForestReadParticleRange(
                *node, x.data(), y.data(), z.data(), types.data(), 1)) {
            throw std::runtime_error("Source range headers did not recover all particle indices");
        }
        for (int output = 1; output <= detail::kSourcePayloadParticles; ++output) {
            auto const original = ParticleInteger(x[output].toDouble() * 16) - 1;
            char const expectedType = detail::sourcePayloadType(mode, original) == kDarkMatterType ?
                kDarkMatterType :
                kSphType;
            if (original < 0 || original >= detail::kSourcePayloadParticles ||
                y[output].toDouble() != double(original + 2) / 16 ||
                z[output].toDouble() != double(original + 3) / 16 ||
                types[output] != expectedType) {
                throw std::runtime_error("Imported source positions or mass class changed");
            }
        }
        if (x.front() != Position::max() || x.back() != Position::max() ||
            y.front() != Position::max() || y.back() != Position::max() ||
            z.front() != Position::max() || z.back() != Position::max() || types.front() != -1 ||
            types.back() != -1) {
            throw std::runtime_error("Imported source read exceeded its output range");
        }
    }
    Vector<TreeId> sourceRoots;
    for (auto const& root : roots) {
        sourceRoots.push_back(root.root);
    }
    TreeId const sinkRoot = roots[hpxRank()].root;
    ForceVectors forces(detail::kSourcePayloadParticles);
    for (ParticleInteger index = 0; index < detail::kSourcePayloadParticles; ++index) {
        forces.phi[index] = forces.gx[index] = forces.gy[index] = forces.gz[index] = 0;
    }
    cpuGravityPp(forces, true, sinkRoot, sourceRoots, getOptions().gravity.smoothingLength);
    for (ParticleInteger index = 0; index < detail::kSourcePayloadParticles; ++index) {
        Vector3<Position> sink;
        for (int axis = 0; axis < kDimensions; ++axis) {
            sink[axis] = particlesPos(axis, index);
        }
        detail::requireSourcePayloadForce({forces.gx[index], forces.gy[index], forces.gz[index]},
            detail::sourcePayloadForceReference(sink, mode),
            "Species-derived CPU P2P force disagrees with the direct reference");
    }
    // Evaluate at the expansion center so this tests the actual P2L gather without
    // truncation error. Every fixture source is separated from the center, and the
    // six-source ranges also exercise partially filled SIMD packets.
    namespace cartesian = cosmictiger::gravity::cartesian;
    cartesian::Expansion<float> expansion{};
    cpuGravityCp<kFmmOrder>(expansion, sourceRoots, sinkRoot, true);
    cosmictiger::gravity::Field<float> field{};
    cartesian::L2P(expansion, Vector3<float>{}, true, field);
    detail::requireSourcePayloadForce(
        {kScaleFactor2 * field.accelerationX(), kScaleFactor2 * field.accelerationY(),
            kScaleFactor2 * field.accelerationZ()},
        detail::sourcePayloadForceReference(treeGetNode(sinkRoot)->pos, mode),
        "Species-derived CPU P2L force disagrees with the direct reference");
}
HPX_PLAIN_ACTION(verifySourcePayloadLocal, VerifySourcePayloadLocalAction);

void verifyRejectedSourceMassConfigLocal(Vector<SourceRootDescriptor> roots) {
    bool rejected = false;
    try {
        auto const& root = roots[hpxRank()];
        treeSourceForestValidate(roots, root.theta, root.smoothingLength, root.gpu);
    } catch (std::invalid_argument const&) {
        rejected = true;
    }
    if (!rejected) {
        throw std::runtime_error("Changed source mass configuration passed kick preflight");
    }
}
HPX_PLAIN_ACTION(verifyRejectedSourceMassConfigLocal, VerifyRejectedSourceMassConfigLocalAction);

void geometricSourcePayloadTest() {
    detail::sourcePayloadWireTest();
    auto const original = getOptions();
    auto options = original;
    options.gravity.cuda = false;
    options.hydro.enabled = false;
    options.hydro.method = "sph";
    options.starFormation.enabled = false;
    options.chemistry.enabled = false;
    options.gravity.smoothingLength = .005;
    options.initial.darkMatterMass = 2.5;
    options.initial.gasMass = .75;
    setOptions(options);
    std::exception_ptr failure;
    try {
        for (auto const& locality : hpxLocalities()) {
            hpx::async<SourcePayloadFixtureArraysAction>(locality, true).get();
        }
        int const modes = COSMICTIGER_WITH_SPH ? 3 : 1;
        for (int mode = 0; mode < modes; ++mode) {
            options.hydro.enabled = mode != 0;
            setOptions(options);
            for (auto const& locality : hpxLocalities()) {
                hpx::async<InitializeSourcePayloadLocalAction>(locality, mode).get();
            }
            particlesSortByRung(0);
            treeCreateGeometric(TreeCreateParameters(0, .5f));
            for (auto const& locality : hpxLocalities()) {
                hpx::async<VerifySourcePayloadLocalAction>(locality, treeGeometricRoots(), mode)
                    .get();
            }
            if (mode == 2) {
                constexpr int kMassConfigurationChanges = 3;
                for (int change = 0; change < kMassConfigurationChanges; ++change) {
                    auto changed = options;
                    if (change == 0) {
                        changed.hydro.enabled = false;
                    } else if (change == 1) {
                        changed.initial.darkMatterMass *= 2;
                    } else {
                        changed.initial.gasMass *= 2;
                    }
                    setOptions(changed);
                    for (auto const& locality : hpxLocalities()) {
                        hpx::async<VerifyRejectedSourceMassConfigLocalAction>(
                            locality, treeGeometricRoots())
                            .get();
                    }
                    setOptions(options);
                }
            }
            treeRetireGeometric();
        }
    } catch (...) {
        failure = std::current_exception();
    }
    treeRetireGeometric();
    for (auto const& locality : hpxLocalities()) {
        hpx::async<SourcePayloadFixtureArraysAction>(locality, false).get();
    }
    setOptions(original);
    if (failure) {
        std::rethrow_exception(failure);
    }
}

void initializeExchangeGuardsLocal() {
    particlesResize(4);
    for (ParticleInteger index = 0; index < particlesOwnedSize(); ++index) {
        for (int axis = 0; axis < kDimensions; ++axis) {
            particlesPos(axis, index) =
                (double(index + 1) + .125 * (axis + 1)) / 8.0 + .003 * hpxRank();
            particlesVel(axis, index) = .125f * (axis + 1);
            particlesGravityAcceleration(axis, index) = 100.0f + axis;
        }
        particlesPot(index) = 123.0f;
        particlesRung(index) = 0;
    }
}
HPX_PLAIN_ACTION(initializeExchangeGuardsLocal, InitializeExchangeGuardsLocalAction);

Vector<float> captureExchangeGuardsLocal() {
    Vector<float> values;
    for (ParticleInteger index = 0; index < particlesOwnedSize(); ++index) {
        for (int axis = 0; axis < kDimensions; ++axis) {
            values.push_back(particlesVel(axis, index));
            values.push_back(particlesGravityAcceleration(axis, index));
        }
        values.push_back(particlesPot(index));
    }
    return values;
}
HPX_PLAIN_ACTION(captureExchangeGuardsLocal, CaptureExchangeGuardsLocalAction);

void invalidateExchangeGuardsLocal() {
    treeSourceForestReset();
}
HPX_PLAIN_ACTION(invalidateExchangeGuardsLocal, InvalidateExchangeGuardsLocalAction);

namespace detail {
Vector<Vector<float>> captureExchangeGuards() {
    Vector<Vector<float>> values;
    auto pending = geometricLaunchAndDrain<Vector<float>>(hpxLocalities().size(), [](size_t rank) {
        return hpx::async<CaptureExchangeGuardsLocalAction>(hpxLocalities()[rank]);
    });
    for (auto& future : pending) {
        values.push_back(future.get());
    }
    return values;
}

void requireRejectedGeometricKick(
    KickParameters params, bool reference, Vector<Vector<float>> const& unchanged) {
    bool rejected = false;
    try {
        kickSourceForest(params, reference);
    } catch (...) {
        rejected = true;
    }
    if (!rejected) {
        throw std::runtime_error("Incompatible geometric kick was accepted");
    }
    if (captureExchangeGuards() != unchanged) {
        throw std::runtime_error("Rejected geometric kick modified particle state");
    }
}

void geometricLaunchFailureTest() {
    hpx::promise<void> started;
    auto startedFuture = started.get_future();
    hpx::promise<void> release;
    auto releaseFuture = release.get_future();
    std::atomic<bool> completed{false};
    hpx::shared_future<void> launched;
    bool rejected = false;
    try {
        geometricLaunchAndDrain<void>(3, [&](size_t index) -> hpx::future<void> {
            if (index == 0) {
                launched = hpx::async([&]() {
                    started.set_value();
                    releaseFuture.get();
                    hpx::this_thread::sleep_for(std::chrono::milliseconds(20));
                    completed = true;
                }).share();
                return launched.then([](hpx::shared_future<void> future) { future.get(); });
            }
            startedFuture.get();
            release.set_value();
            throw std::runtime_error("Injected export construction failure");
        });
    } catch (std::runtime_error const&) {
        rejected = true;
    }
    bool const drainedOnLaunchFailure = completed;
    // Even a failing regression must leave its captures alive until the task
    // completes; take the observation before this independent safety drain.
    if (launched.valid()) {
        launched.get();
    }
    if (!rejected || !drainedOnLaunchFailure) {
        throw std::runtime_error("Geometric launch failure did not drain earlier work");
    }

    completed = false;
    Vector<hpx::shared_future<void>> transfers(2);
    Vector<hpx::future<void>> pending;
    std::exception_ptr failure;
    try {
        pending = geometricLaunchAndDrain<void>(transfers.size(), [&](size_t index) {
            transfers[index] = hpx::async([&, index]() {
                if (index == 0) {
                    throw std::runtime_error("Injected export transfer failure");
                }
                hpx::this_thread::sleep_for(std::chrono::milliseconds(20));
                completed = true;
            }).share();
            return transfers[index].then([](hpx::shared_future<void> future) { future.get(); });
        });
    } catch (...) {
        failure = std::current_exception();
    }
    if (failure) {
        // Preserve every task's captures even if dispatch, continuation attachment,
        // or the helper itself fails before returning its future collection.
        for (auto& transfer : transfers) {
            if (transfer.valid()) {
                transfer.wait();
            }
        }
        std::rethrow_exception(failure);
    }
    bool const drainedBeforeGet = completed;
    // Unlike HPX wait_all, future::wait must leave stored errors for get().
    pending.front().wait();
    rejected = false;
    try {
        pending.front().get();
    } catch (...) {
        rejected = true;
    }
    pending.back().get();
    if (!rejected || !drainedBeforeGet) {
        throw std::runtime_error("Geometric failed transfer escaped before other work drained");
    }
}
}    // namespace detail

void geometricExchangeGuardsTest() {
    geometricSourcePayloadTest();
    detail::geometricLaunchFailureTest();
    auto const original = getOptions();
    auto options = original;
    options.hydro.enabled = false;
    options.gravity.cuda = false;
    options.gravity.smoothingLength = .005;
    options.analysis.saveForce = true;
    options.analysis.groups = options.analysis.tracers = false;
    options.compute.bucketSize = 8;
    setOptions(options);
    std::exception_ptr failure;
    try {
        auto initialization =
            detail::geometricLaunchAndDrain<void>(hpxLocalities().size(), [](size_t rank) {
                return hpx::async<InitializeExchangeGuardsLocalAction>(hpxLocalities()[rank]);
            });
        for (auto& future : initialization) {
            future.get();
        }
        particlesSortByRung(0);
        treeCreateGeometric(TreeCreateParameters(0, .5f));
        auto const roots = treeGeometricRoots();
        auto const unchanged = detail::captureExchangeGuards();
        KickParameters params;
        params.theta = .5f;
        params.gravitySmoothingLength = options.gravity.smoothingLength;
        params.gpu = false;
        params.forceOnly = true;
        params.saveForce = true;
        params.gm = 1.0f;
        params.nodeLoad = 1.0f;
        for (bool reference : {false, true}) {
            auto changed = params;
            changed.theta = .01f;
            detail::requireRejectedGeometricKick(changed, reference, unchanged);
            changed = params;
            changed.gravitySmoothingLength *= 2;
            detail::requireRejectedGeometricKick(changed, reference, unchanged);
            changed = params;
            changed.gpu = true;
            detail::requireRejectedGeometricKick(changed, reference, unchanged);
        }

        // A stale remote owner must be found before the still-ready localities
        // start their kicks. The same guard is required by reference traversal.
        hpx::async<InvalidateExchangeGuardsLocalAction>(hpxLocalities().back()).get();
        for (bool reference : {false, true}) {
            detail::requireRejectedGeometricKick(params, reference, unchanged);
        }
        treeRetireGeometric(true);
        treeCreateGeometric(TreeCreateParameters(0, .5f));
        bool rejected = false;
        try {
            treeSourceForestValidate(
                roots, params.theta, params.gravitySmoothingLength, params.gpu);
        } catch (std::runtime_error const&) {
            rejected = true;
        }
        if (!rejected) {
            throw std::runtime_error("A prior geometric generation was accepted after rebuild");
        }
        treeRetireGeometric(true);
    } catch (...) {
        failure = std::current_exception();
    }
    if (failure) {
        treeRetireGeometric(true);
        setOptions(original);
        std::rethrow_exception(failure);
    }
    setOptions(original);
}
