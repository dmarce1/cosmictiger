/* CosmicTiger - pending kick tree-retirement regression; BSL-1.0. */

#include <array>
#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>

#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/gravity/global_fmm/distributed_tree.hpp>
#include <cosmictiger/gravity/kick.hpp>
#include <cosmictiger/particles/particles.hpp>
#if COSMICTIGER_WITH_SPH
#include <cosmictiger/sph/particles.hpp>
#endif

namespace detail {

constexpr ParticleInteger kPendingKickParticles = 6;
std::array<ParticleInteger, kPendingKickParticles> pendingKickInitialOrder;
std::array<ParticleInteger, kPendingKickParticles> pendingKickTreeOrder;

#if COSMICTIGER_WITH_SPH
std::array<ParticleInteger, kPendingKickParticles> pendingKickGasIndices;
std::array<char, kPendingKickParticles> pendingKickParticleTypes;
ParticleInteger* savedPendingKickGasIndices = nullptr;
char* savedPendingKickParticleTypes = nullptr;
bool pendingKickGasArraysInstalled = false;
#endif

ParticleInteger pendingKickParticleCount(bool empty) {
    return empty || (hpxSize() > 1 && hpxRank() == hpxSize() - 1) ? 0 : kPendingKickParticles;
}

float pendingKickValue(ParticleInteger originalIndex, int dimension) {
    return float((originalIndex + 1) * (dimension + 1));
}

ParticleInteger pendingKickOriginalIndex(ParticleInteger index) {
    return ParticleInteger(std::lround(16.0 * particlesPos(0, index).toDouble())) - 1;
}

}    // namespace detail

void initializePendingKickTest(bool gas, bool empty) {
    ParticleInteger const count = detail::pendingKickParticleCount(empty);
    particlesResize(count);
#if COSMICTIGER_WITH_SPH
    sphParticlesResize(gas ? count / 2 : 0, false);
#endif
    for (ParticleInteger index = 0; index < count; ++index) {
        for (int dimension = 0; dimension < kDimensions; ++dimension) {
            particlesPos(dimension, index) = double(index + 1) / 16.0;
            particlesVel(dimension, index) = 0.0f;
        }
        particlesRung(index) = 1;
#if COSMICTIGER_WITH_SPH
        if (gas) {
            particlesType(index) = index % 2 == 0 ? kSphType : kDarkMatterType;
            particlesCatIndex(index) = index % 2 == 0 ? index / 2 : kNoParticleIndex;
            if (index % 2 == 0) {
                sphParticlesParticleIndex(index / 2) = index;
            }
        }
#endif
    }
    if (count != 0) {
        // Deliberately scramble the input so the actual tree build must change its
        // ordering. The gas/DM mixture exposes particle-index misassociation.
        particlesSwap(0, 1);
        particlesSwap(1, 2);
        particlesSwap(4, 5);
    }
    for (ParticleInteger index = 0; index < count; ++index) {
        detail::pendingKickInitialOrder[index] = detail::pendingKickOriginalIndex(index);
    }
}
HPX_PLAIN_ACTION(initializePendingKickTest, InitializePendingKickTestAction);

void fillPendingKickTest(bool gas) {
    bool reordered = false;
    for (ParticleInteger index = 0; index < particlesSize(); ++index) {
        detail::pendingKickTreeOrder[index] = detail::pendingKickOriginalIndex(index);
        reordered = reordered ||
            detail::pendingKickTreeOrder[index] != detail::pendingKickInitialOrder[index];
    }
    if (particlesSize() != 0 && !reordered) {
        throw std::runtime_error("Pending-kick fixture did not exercise tree reordering");
    }
    if (!gas) {
        return;
    }
#if COSMICTIGER_WITH_SPH
    auto* increments = gravityDeferredGasOpeningData();
    for (ParticleInteger gasIndex = 0; gasIndex < sphParticlesSize(); ++gasIndex) {
        ParticleInteger const originalIndex =
            detail::pendingKickOriginalIndex(sphParticlesParticleIndex(gasIndex));
        for (int dimension = 0; dimension < kDimensions; ++dimension) {
            // A zero increment exercises the sealed sparse list as well.
            increments[gasIndex][dimension] =
                originalIndex == 2 ? 0.0f : detail::pendingKickValue(originalIndex, dimension);
        }
    }
#endif
}
HPX_PLAIN_ACTION(fillPendingKickTest, FillPendingKickTestAction);

void verifyPendingKickTreeOrder() {
    for (ParticleInteger index = 0; index < particlesSize(); ++index) {
        if (detail::pendingKickOriginalIndex(index) != detail::pendingKickTreeOrder[index]) {
            throw std::runtime_error("Tree retirement changed the pending kick's particle order");
        }
#if COSMICTIGER_WITH_SPH
        if (getOptions().hydro.enabled && particlesType(index) == kSphType &&
            sphParticlesParticleIndex(particlesCatIndex(index)) != index) {
            throw std::runtime_error("Tree retirement corrupted the pending kick's gas mapping");
        }
#endif
    }
}
HPX_PLAIN_ACTION(verifyPendingKickTreeOrder, VerifyPendingKickTreeOrderAction);

void verifyPendingKickTest(bool gas) {
    for (ParticleInteger index = 0; index < particlesSize(); ++index) {
        ParticleInteger const originalIndex = detail::pendingKickOriginalIndex(index);
        for (int dimension = 0; dimension < kDimensions; ++dimension) {
            float const expected = gas && originalIndex % 2 == 0 && originalIndex != 2 ?
                detail::pendingKickValue(originalIndex, dimension) :
                0.0f;
            if (particlesVel(dimension, index) != expected) {
                throw std::runtime_error(
                    "Particle received another particle's pending kick after tree retirement");
            }
        }
    }
    particlesResize(0);
#if COSMICTIGER_WITH_SPH
    sphParticlesResize(0, false);
#endif
}
HPX_PLAIN_ACTION(verifyPendingKickTest, VerifyPendingKickTestAction);

void pendingKickGasFixtureArrays(bool install) {
#if COSMICTIGER_WITH_SPH
    if (install) {
        // The enclosing force fixture starts with hydro disabled. Ensure base capacity
        // before switching to temporary CPU-only species arrays: particlesResize only
        // allocates optional species storage when the base allocation grows.
        particlesResize(detail::pendingKickParticleCount(false));
        detail::savedPendingKickGasIndices = particlesSph;
        detail::savedPendingKickParticleTypes = particlesTy;
        particlesSph = detail::pendingKickGasIndices.data();
        particlesTy = detail::pendingKickParticleTypes.data();
        detail::pendingKickGasArraysInstalled = true;
    } else if (detail::pendingKickGasArraysInstalled) {
        particlesResize(0);
        sphParticlesResize(0, false);
        particlesSph = detail::savedPendingKickGasIndices;
        particlesTy = detail::savedPendingKickParticleTypes;
        detail::pendingKickGasArraysInstalled = false;
    }
#endif
}
HPX_PLAIN_ACTION(pendingKickGasFixtureArrays, PendingKickGasFixtureArraysAction);

void retirePendingKickTest() {
    treeRetireGeometric(true);
    // Repeated retirement must be harmless even while the kick is still pending.
    treeRetireGeometric(true);
    for (auto const& locality : hpxLocalities()) {
        hpx::async<VerifyPendingKickTreeOrderAction>(locality).get();
    }
}

void geometricPendingKickTest() {
    auto const savedOptions = getOptions();
    auto options = savedOptions;
    options.gravity.cuda = false;
    options.gravity.errorCuda = false;
    options.hydro.cuda = false;
    options.drift.cuda = options.sort.cuda = options.initial.cuda = false;
    options.analysis.powerCuda = options.analysis.groupsCuda = options.analysis.tracersCuda = false;
    options.analysis.sliceCuda = options.analysis.viewsCuda = options.lightcone.cuda = false;
    options.chemistry.cuda = options.starFormation.cuda = false;
    options.hydro.enabled = false;
    options.analysis.saveForce = options.analysis.saveAcceleration =
        options.analysis.savePotential = false;
    options.analysis.groups = options.analysis.tracers = false;
    options.gravity.errorInterval = 0;
    options.gravity.smoothingLength = .005;
    options.compute.bucketSize = 1;
    options.chemistry.enabled = options.starFormation.enabled = false;
    options.run.scenario = "random";
    options.initial.darkMatterMass = options.initial.gasMass = 1.0;
    setOptions(options);
    std::exception_ptr failure;
    try {
        for (bool empty : {false, true}) {
            for (auto const& locality : hpxLocalities()) {
                hpx::async<InitializePendingKickTestAction>(locality, false, empty).get();
            }
            particlesSortByRung(0);
            treeCreateGeometric(TreeCreateParameters(0, .5f));
            for (auto const& locality : hpxLocalities()) {
                hpx::async<FillPendingKickTestAction>(locality, false).get();
            }
            retirePendingKickTest();
            for (auto const& locality : hpxLocalities()) {
                hpx::async<VerifyPendingKickTestAction>(locality, false).get();
            }
        }
#if COSMICTIGER_WITH_SPH
        for (auto const& locality : hpxLocalities()) {
            hpx::async<PendingKickGasFixtureArraysAction>(locality, true).get();
        }
        options.hydro.enabled = true;
        options.hydro.method = "sph";
        setOptions(options);
        for (bool sealed : {false, true}) {
            for (auto const& locality : hpxLocalities()) {
                hpx::async<InitializePendingKickTestAction>(locality, true, false).get();
            }
            particlesSortByRung(0);
            treeCreateGeometric(TreeCreateParameters(0, .5f));
            gravityBeginDeferredGasOpening();
            for (auto const& locality : hpxLocalities()) {
                hpx::async<FillPendingKickTestAction>(locality, true).get();
            }
            if (sealed) {
                gravitySealDeferredGasOpening();
            }
            retirePendingKickTest();
            if (!sealed) {
                gravitySealDeferredGasOpening();
            }
            gravityFinishDeferredGasOpening();
            for (auto const& locality : hpxLocalities()) {
                hpx::async<VerifyPendingKickTestAction>(locality, true).get();
            }
        }
        for (auto const& locality : hpxLocalities()) {
            hpx::async<PendingKickGasFixtureArraysAction>(locality, false).get();
        }
#endif
    } catch (...) {
        failure = std::current_exception();
    }
    if (failure) {
        treeRetireGeometric(true);
        gravityDiscardDeferredGasOpening();
        for (auto const& locality : hpxLocalities()) {
            hpx::async<PendingKickGasFixtureArraysAction>(locality, false).get();
        }
        setOptions(savedOptions);
        std::rethrow_exception(failure);
    }
    setOptions(savedOptions);
    std::cout << "Geometric pending-kick retirement regression passed\n" << std::flush;
}
