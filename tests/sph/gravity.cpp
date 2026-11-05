/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <format>
#include <iostream>
#include <limits>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <string>
#include <thread>

#include <cosmictiger/analysis/bh.hpp>
#include <cosmictiger/core/build_config.hpp>
#include <cosmictiger/core/driver.hpp>
#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/gravity/fmm_order.hpp>
#include <cosmictiger/gravity/global_fmm/distributed_tree.hpp>
#include <cosmictiger/gravity/global_fmm/execution.hpp>
#include <cosmictiger/gravity/global_fmm/particle_exchange.hpp>
#include <cosmictiger/gravity/gravity.hpp>
#include <cosmictiger/gravity/kernels/cartesian_tracefree_operators.hpp>
#include <cosmictiger/gravity/kick.hpp>
#include <cosmictiger/gravity/tree.hpp>
#include <cosmictiger/math/vector3.hpp>
#include <cosmictiger/particles/domain.hpp>
#include <cosmictiger/particles/particles.hpp>
#include <cosmictiger/sph/particles.hpp>
#include <cosmictiger/sph/sph.hpp>
#include <cosmictiger/sph/time_integration.hpp>
#include <cosmictiger/sph/tree.hpp>
#include <cosmictiger/sph/viscosity/cullen_and_dehnen_particle_exchange.hpp>

#ifdef USE_CUDA
void verifySelectedKernelsCuda();
#endif

// Stage-storage lifecycle used by the full-force sensor regression below.
void sphPrepareStageRates(SphRunParameters parameters, bool initialize);
void sphFinishPairStages(bool restore);

namespace detail {
constexpr float kGravityScale = 1.75f;
void requireClose(char const* label, double actual, double expected, double tolerance = 3.0e-5) {
    if (!(std::abs(actual - expected) <= tolerance * std::max(1.0, std::abs(expected)))) {
        throw std::runtime_error(std::string(label) + ": got " + std::to_string(actual) +
            ", expected " + std::to_string(expected));
    }
}

// The monopole is the first coefficient of the runtime order's multipole.
float nodeMass(TreeNode const* node) {
    MultipoleBuffer multipole;
    treeReadMultipole(node->mpos, multipole);
    return multipole[0];
}

void verifyKernelSelections() {
    auto const verify = [](char const* label, auto const& table, std::string const& family,
                            int degree) {
        bool const familyMatches =
            (family == "spline" && table.spec.family == radial_kernel::Family::Spline) ||
            (family == "wendland" && table.spec.family == radial_kernel::Family::Wendland) ||
            (family == "ferrers" && table.spec.family == radial_kernel::Family::Ferrers);
        if (!familyMatches || table.spec.degree != degree) {
            throw std::runtime_error(std::string(label) + " kernel table differs from its options");
        }
    };
    auto const& options = getOptions();
    verify("Gravity", selectedGravityKernel(), options.gravity.kernelFamily,
        options.gravity.kernelDegree);
    verify("Hydro", selectedHydroKernel(), options.hydro.kernelFamily, options.hydro.kernelDegree);
}

void verifySphExceptionMask() {
#if defined(__GLIBC__)
    if (sphTreeLeaflistSize() != 0) {
        throw std::runtime_error("Exception-mask regression requires an empty SPH tree");
    }
    // Exception masks belong to the native worker. Keep this coroutine bound
    // across sphRun's future waits, and restore its environment before returning.
    auto executor = hpx::execution::parallel_executor(hpx::threads::thread_priority::bound);
    hpx::async(executor, []() {
        struct FloatingEnvironmentGuard {
            std::fenv_t saved;
            FloatingEnvironmentGuard() {
                if (std::feholdexcept(&saved) != 0) {
                    throw std::runtime_error("Could not save floating-point environment");
                }
            }
            ~FloatingEnvironmentGuard() {
                std::fesetenv(&saved);
            }
        };
        int exceptionMask;
        {
            FloatingEnvironmentGuard guard;
            auto const worker = std::this_thread::get_id();
            if (fegetexcept() != 0) {
                throw std::runtime_error("Could not mask floating-point exceptions");
            }
            SphRunParameters params;
            params.runType = kSphRunRungs;
            params.phase = 0;
            params.minimumRung = 0;
            params.baseTimestep = 1.0f;
            params.scaleFactor = 1.0f;
            params.expansionRate = 0.0f;
            params.courantFactor = getOptions().hydro.cfl;
            // An empty phase exercises orchestration without executing arithmetic
            // that could trap if sphRun incorrectly changes the caller's mask.
            sphRun(params, false);
            exceptionMask = fegetexcept();
            if (std::this_thread::get_id() != worker) {
                throw std::runtime_error("Exception-mask regression migrated between workers");
            }
        }
        if (exceptionMask != 0) {
            throw std::runtime_error("sphRun changed the caller's floating-point exception mask");
        }
    }).get();
#endif
}

float migrationElectronFraction(int target) {
    float electronFraction{1.0f};
    electronFraction = 0.25f + target / 128.0f;
    return electronFraction;
}

void initializePairDivergence() {
    if (getOptions().hydro.viscosityGradientSwitch()) {
        for (ParticleInteger gas = 0; gas < sphParticlesSize(); ++gas) {
            ALWAYS_ASSERT(sphEncodeDivergence(
                sphScaledDivergence[gas], -1.75f, std::ldexp(1.0, -sphParticlesRung(gas))));
        }
    }
}

void verifyPairDivergence(char const* label) {
    if (getOptions().hydro.viscosityGradientSwitch()) {
        for (ParticleInteger gas = 0; gas < sphParticlesSize(); ++gas) {
            requireClose(label,
                sphDecodeDivergence(
                    sphScaledDivergence[gas], std::ldexp(1.0, -sphParticlesRung(gas))),
                -1.75, 0.0);
        }
    }
}

void initializePair(double separation) {
    particlesResize(2);
    sphParticlesResize(1, false);
    for (ParticleInteger i = 0; i < 2; i++) {
        for (int dim = 0; dim < kDimensions; dim++) {
            particlesPos(dim, i) = dim == kXDimension ? 0.375 - i * separation : 0.375;
            particlesVel(dim, i) = i == 1 ? 0.125 * (dim + 1) : 0.0;
        }
        particlesRung(i) = 1;
        particlesType(i) = i == 0 ? kDarkMatterType : kSphType;
        particlesCatIndex(i) = i == 0 ? kNoParticleIndex : 0;
    }
    sphParticlesParticleIndex(0) = 1;
    sphParticlesSmoothingLength(0) = 0.02;
    sphParticlesEntropies[0] = 4.0;
    sphParticlesSetProposedRung(0, sphMidpointEnabled() ? 4 : 0);
    initializePairDivergence();
}
TreeId findLeaf(TreeId node, ParticleInteger particle) {
    auto const* ptr = treeGetNode(node);
    if (ptr->leaf) {
        return node;
    }
    for (int child = 0; child < kChildCount; child++) {
        auto const next = ptr->children[child];
        auto const range = treeGetNode(next)->particleRange;
        if (particle >= range.first && particle < range.second) {
            return findLeaf(next, particle);
        }
    }
    throw std::runtime_error("Particle missing from gravity tree");
}
void verifyMappings() {
    for (ParticleInteger k = 0; k < sphParticlesSize(); k++) {
        auto const i = sphParticlesParticleIndex(k);
        if (i < 0 || i >= particlesSize() || !particlesIsSph(i) || particlesCatIndex(i) != k) {
            throw std::runtime_error("Gas category mapping was not preserved");
        }
    }
}
KickParameters makeKickParams() {
    KickParameters p;
    p.htime = false;
    p.sign = 1.0;
    p.minLevel = 0;
    p.minRung = 0;
    p.a = 1.0;
    p.t0 = 0.001;
    p.theta = 0.5;
    p.gravitySmoothingLength = getOptions().gravity.smoothingLength;
    p.eta = getOptions().gravity.timestepEta;
    p.gm = kGravityScale;
    p.velInit = false;
    p.saveForce = true;
    p.firstCall = true;
    p.gpu = false;
    p.nodeLoad = 1.0;
    p.ascending = true;
    p.descending = true;
    p.top = true;
    p.doPhi = true;
    return p;
}
void resetKickState() {
    for (ParticleInteger i = 0; i < particlesSize(); i++) {
        particlesRung(i) = 1;
        for (int dim = 0; dim < kDimensions; dim++) {
            particlesVel(dim, i) = particlesIsSph(i) ? 0.125 * (dim + 1) : 0.0;
            particlesGravityAcceleration(dim, i) = 0.0;
        }
        particlesPot(i) = 0.0;
    }
    initializePairDivergence();
}
void verifyKick(
    double inverseCube, double inverseRadius, double separation, KickParameters const& parameters) {
    verifyPairDivergence("gravity rung selection preserves decoded divergence");
    for (ParticleInteger i = 0; i < particlesSize(); i++) {
        bool const gas = particlesIsSph(i);
        double const sourceMass = particlesMass(1 - i);
        double const acceleration =
            kGravityScale * (gas ? 1.0 : -1.0) * sourceMass * separation * inverseCube;
        requireClose(
            "kick acceleration", particlesGravityAcceleration(kXDimension, i), acceleration);
        requireClose("kick potential with self mass removed", particlesPot(i),
            -kGravityScale * sourceMass * inverseRadius, 2e-4);
        requireClose("transverse force", particlesGravityAcceleration(kYDimension, i), 0.0);
        if (gas && sphMidpointEnabled()) {
            requireClose("gas gravity retains the latest hydro rung bound", particlesRung(i), 4);
            double const halfStep = 0.5 * std::ldexp(double(parameters.t0), -particlesRung(i));
            requireClose("midpoint gas receives its gravity opening kick",
                particlesVel(kXDimension, i), 0.125 + halfStep * acceleration);
            requireClose("midpoint gravity leaves transverse velocity unchanged",
                particlesVel(kYDimension, i), 0.25);
            requireClose("midpoint gravity leaves the thermal seed unchanged",
                sphParticlesEntropies[0], 4.0);
        } else if (particlesVel(kXDimension, i) == 0.0) {
            throw std::runtime_error("Ordinary DM kick did not update velocity");
        }
    }
}
void verifySynchronizedDiagnostics(KickReturn const& result) {
    double kinetic = 0.0, xMomentum = 0.0, yMomentum = 0.0, zMomentum = 0.0, momentumNorm = 0.0;
    for (ParticleInteger i = 0; i < particlesSize(); i++) {
        double const mass = particlesMass(i);
        double const vx = particlesVel(kXDimension, i), vy = particlesVel(kYDimension, i),
                     vz = particlesVel(kZDimension, i);
        kinetic += 0.5 * mass * (vx * vx + vy * vy + vz * vz);
        xMomentum += mass * vx;
        yMomentum += mass * vy;
        zMomentum += mass * vz;
        momentumNorm += mass * std::sqrt(vx * vx + vy * vy + vz * vz);
    }
    double const gasMass = particlesMass(sphParticlesParticleIndex(0));
    double const initialSpeed2 = 0.125 * 0.125 * (1.0 + 4.0 + 9.0);
    requireClose("synchronized kinetic diagnostic", kinetic + result.darkMatterKineticCorrection,
        0.5 * gasMass * initialSpeed2);
    requireClose("synchronized x momentum", xMomentum + result.darkMatterXMomentumCorrection,
        gasMass * 0.125);
    requireClose("synchronized y momentum", yMomentum + result.darkMatterYMomentumCorrection,
        gasMass * 0.25);
    requireClose("synchronized z momentum", zMomentum + result.darkMatterZMomentumCorrection,
        gasMass * 0.375);
    requireClose("synchronized momentum norm",
        momentumNorm + result.darkMatterMomentumNormCorrection, gasMass * std::sqrt(initialSpeed2));
}

template <class Function>
double integrateReference(Function function, double begin, double end) {
    constexpr int kIntervals = 1024;
    double const step = (end - begin) / kIntervals;
    // Endpoint values do not affect the integral; sample their interior limits.
    double integral = function(std::nextafter(begin, end)) + function(std::nextafter(end, begin));
    for (int i = 1; i < kIntervals; ++i) {
        integral += (i % 2 == 0 ? 2.0 : 4.0) * function(begin + i * step);
    }
    return integral * step / 3.0;
}

void softenedPairReference(double separation, double& inverseCube, double& inverseRadius) {
    auto const& kernel = selectedGravityKernel();
    double const support = getOptions().gravity.smoothingLength;
    double const smoothingNormalizedRadius = separation / support;
    double const fourPi = 4.0 * std::numbers::pi;
    // Integrate the density independently of the force/potential antiderivative tables.
    double const mass = fourPi *
        integrateReference(
            [&](double radius) { return radius * radius * radial_kernel::value(kernel, radius); },
            0.0, smoothingNormalizedRadius);
    inverseCube = mass / (separation * separation * separation);
    inverseRadius =
        (mass / smoothingNormalizedRadius +
            fourPi *
                integrateReference(
                    [&](double radius) { return radius * radial_kernel::value(kernel, radius); },
                    smoothingNormalizedRadius, 1.0)) /
        support;
}

void verifyBhPotentials(bool useCuda) {
    double const separation = 0.25 * getOptions().gravity.smoothingLength;
    DeviceVector<Vector3<float>> sources(2);
    for (int dim = 0; dim < kDimensions; ++dim) {
        sources[0][dim] = 0.375;
        sources[1][dim] = dim == kXDimension ? 0.375 + separation : 0.375;
    }
    // Match the stored float positions used by the independent halo-potential path.
    double const actualSeparation = double(sources[1][kXDimension]) - sources[0][kXDimension];
    double inverseCube, inverseRadius;
    softenedPairReference(actualSeparation, inverseCube, inverseRadius);
    double const expected = -getOptions().gravity.gm * inverseRadius;
    auto potential = bhEvaluatePotential(sources, useCuda);
    for (int i = 0; i < 2; ++i) {
        requireClose("halo potential uses selected gravity kernel", potential[i], expected);
    }
    DeviceVector<Vector3<float>> points(2);
    for (int dim = 0; dim < kDimensions; ++dim) {
        points[0][dim] = 0.375;
        points[1][dim] = dim == kXDimension ? float(0.375 + 0.5 * actualSeparation) : 0.375;
    }
    potential = bhEvaluatePoints(points, sources, useCuda);
    requireClose("halo point potential excludes coincident source", potential[0], expected);
    double expectedMidpoint = 0.0;
    for (int i = 0; i < 2; ++i) {
        double const pointSeparation =
            std::abs(double(points[1][kXDimension]) - sources[i][kXDimension]);
        softenedPairReference(pointSeparation, inverseCube, inverseRadius);
        expectedMidpoint -= getOptions().gravity.gm * inverseRadius;
    }
    requireClose(
        "halo midpoint potential uses selected gravity kernel", potential[1], expectedMidpoint);
}

void runPairCase(bool useCuda, bool softened, float hydroSupport = 0.02f) {
    double const nominalSeparation = softened ? 0.25 * getOptions().gravity.smoothingLength : 0.25;
    initializePair(nominalSeparation);
    float const gasMass = particlesMass(sphParticlesParticleIndex(0));
    sphParticlesSmoothingLength(0) = hydroSupport;
    auto const built =
        treeCreate(TreeCreateParameters(0, 0.5, getOptions().gravity.smoothingLength));
    verifyMappings();
    requireClose("root mixed mass", nodeMass(treeGetNode(built.id)),
        getOptions().initial.darkMatterMass + gasMass);
    requireClose("gravity tree support is independent of hydro smoothing",
        built.maxGravitySmoothingLength, getOptions().gravity.smoothingLength);
    ParticleInteger const gas = sphParticlesParticleIndex(0);
    ParticleInteger const darkMatterIndex = 1 - gas;
    double const separation = particlesPos(kXDimension, darkMatterIndex).toDouble() -
        particlesPos(kXDimension, gas).toDouble();
    double inverseCube = 1.0 / (separation * separation * separation);
    double inverseRadius = 1.0 / separation;
    if (softened) {
        softenedPairReference(separation, inverseCube, inverseRadius);
    }
    for (ParticleInteger sink = 0; sink < 2; sink++) {
        auto const self = findLeaf(built.id, sink);
        auto const source = findLeaf(built.id, 1 - sink);
        Vector<TreeId> sources{source};
        double const sourceMass = particlesMass(1 - sink);
        double const expected =
            (particlesIsSph(sink) ? 1.0 : -1.0) * sourceMass * separation * inverseCube;
        ForceVectors direct(1);
        cpuGravityPp(direct, true, self, sources, getOptions().gravity.smoothingLength);
        requireClose("CPU particle-particle mass", direct.gx[0], expected);
        requireClose("CPU particle-particle potential", direct.phi[0], -sourceMass * inverseRadius);
        if (!softened) {
            requireClose("leaf monopole mass", nodeMass(treeGetNode(source)), sourceMass);
            ForceVectors far(1);
            fmmDispatch([&](auto order) { cpuGravityPc<order.value>(far, true, self, sources); });
            requireClose("CPU particle-cell mass", far.gx[0], expected);
            for (int path = 0; path < 2; path++) {
                ExpansionBuffer local;
                local.zero();
                Vector3<float> dx;
                for (int dim = 0; dim < kDimensions; dim++) {
                    dx[dim] = distance(particlesPos(dim, sink), treeGetNode(self)->pos[dim]);
                }
                auto const value = fmmDispatch([&](auto order) {
                    auto& expansion = local.expansion<order.value>();
                    if (path == 0) {
                        cpuGravityCp<order.value>(expansion, sources, self, true);
                    } else {
                        cpuGravityCc<order.value>(expansion, sources, self, true);
                    }
                    cosmictiger::gravity::Field<float> field{};
                    cosmictiger::gravity::cartesian::tracefree::L2P(
                        field, expansion, fmmScaleDisplacement(dx));
                    return field;
                });
                requireClose(path == 0 ? "CPU cell-particle mass" : "CPU cell-cell mass",
                    kScaleFactor2 * value.accelerationX(), expected);
            }
        }
    }
    ExpansionBuffer local;
    local.zero();
    Vector3<Position> origin;
    for (int dim = 0; dim < kDimensions; dim++) {
        origin[dim] = 0.0;
    }
    auto params = makeKickParams();
    double const darkMatterAcceleration = kGravityScale * gasMass * separation * inverseCube;
    double const gravityEffectiveRadius = getOptions().gravity.smoothingLength /
        selectedGravityKernel().smoothingLengthPerEffectiveRadius;
    double const darkMatterTimestep =
        params.eta * std::sqrt(gravityEffectiveRadius / darkMatterAcceleration);
    params.t0 = 2.2 * darkMatterTimestep;
    resetKickState();
    auto const cpuResult =
        kick(params, local, origin, built.id, Vector<TreeId>{built.id}, Vector<TreeId>{}, nullptr);
    requireClose("CPU kick total potential", cpuResult.pot,
        -kGravityScale * getOptions().initial.darkMatterMass * gasMass * inverseRadius, 2e-4);
    verifyKick(inverseCube, inverseRadius, separation, params);
    requireClose(
        "CPU gravity rung uses the variance smoothing length", particlesRung(darkMatterIndex), 2);
    verifySynchronizedDiagnostics(cpuResult);
#ifdef USE_CUDA
    if (useCuda) {
        resetKickState();
        KickWorkitem item;
        item.l = local;
        item.pos = origin;
        item.self = built.id;
        item.dchecklist.push_back(built.id);
        auto const cudaResult =
            cudaExecuteKicks(params, particlesX[kXDimension], particlesX[kYDimension],
                particlesX[kZDimension], treeData(), Vector<KickWorkitem>{item});
        requireClose("CUDA kick total potential", cudaResult.pot, cpuResult.pot, 2e-4);
        verifyKick(inverseCube, inverseRadius, separation, params);
        requireClose("CUDA gravity rung uses the variance smoothing length",
            particlesRung(darkMatterIndex), 2);
        verifySynchronizedDiagnostics(cudaResult);
    }
#else
    if (useCuda) {
        throw std::runtime_error("CUDA regression requested from a non-CUDA build");
    }
#endif
    if (sphMidpointEnabled()) {
        auto execute = [&](KickParameters const& parameters, bool gpu) {
#ifdef USE_CUDA
            if (gpu) {
                KickWorkitem item;
                item.l = local;
                item.pos = origin;
                item.self = built.id;
                item.dchecklist.push_back(built.id);
                return cudaExecuteKicks(parameters, particlesX[kXDimension],
                    particlesX[kYDimension], particlesX[kZDimension], treeData(),
                    Vector<KickWorkitem>{item});
            }
#else
            ALWAYS_ASSERT(!gpu);
#endif
            return kick(parameters, local, origin, built.id, Vector<TreeId>{built.id},
                Vector<TreeId>{}, nullptr);
        };
        ParticleInteger const gasIndex = sphParticlesParticleIndex(0);
        double const gasAcceleration =
            kGravityScale * getOptions().initial.darkMatterMass * separation * inverseCube;
        for (int backend = 0; backend <= int(useCuda); ++backend) {
            resetKickState();
            auto deferred = params;
            deferred.deferGasOpening = true;
            gravityBeginGlobalKick();
            auto result = execute(deferred, backend != 0);
            requireClose("deferred gas opening leaves the synchronized velocity",
                particlesVel(kXDimension, gasIndex), 0.125, 1e-7);
            requireClose(
                "deferred gas selection retains the hydro floor", particlesRung(gasIndex), 4);
            verifyPairDivergence("deferred gas selection preserves decoded divergence");
            // Stand in for neighbor closure after the fresh gravity choice.
            sphParticlesSetProposedRung(0, 6);
            requireClose("neighbor closure commits the final rung", sphCommitBalancedRungs(), 6);
            sphParticlesSetProposedRung(0, 4);
            verifyPairDivergence("neighbor closure preserves decoded divergence");
            result += gravityFinishBalancedGasKick(params.t0, params.sign);
            requireClose("balanced opening uses the final gas rung",
                particlesVel(kXDimension, gasIndex),
                0.125 + 0.5 * rungDt[6] * params.t0 * gasAcceleration, 1e-6);
            requireClose("balanced opening publishes the final rung", result.maxRung, 6);
            for (int dimension = 0; dimension < kDimensions; ++dimension) {
                if (gravityGlobalAcceleration(dimension) != nullptr) {
                    throw std::runtime_error("Balanced opening retained its event force buffer");
                }
            }

            if (!usesGlobalIntegration(getOptions().run)) {
                for (bool hierarchical : {false, true}) {
                    resetKickState();
                    auto opening = params;
                    opening.a = 0.375f;
                    opening.firstCall = false;
                    opening.freezeGasRungs = true;
                    opening.deferHydroOpening = true;
                    opening.htime = hierarchical;
                    gravityBeginDeferredGasOpening();
                    execute(opening, backend != 0);
                    double const closingInterval =
                        0.5 * (hierarchical ? 1.0 : rungDt[1]) * params.t0;
                    double const closedVelocity = 0.125 + closingInterval * gasAcceleration;
                    requireClose("gas closing precedes deferred opening",
                        particlesVel(kXDimension, gasIndex), closedVelocity, 1e-6);
                    if (hierarchical) {
                        auto signedCorrection = opening;
                        signedCorrection.top = signedCorrection.ascending = false;
                        signedCorrection.minRung = 1;
                        execute(signedCorrection, backend != 0);
                        requireClose("signed descending correction remains deferred",
                            particlesVel(kXDimension, gasIndex), closedVelocity, 1e-6);
                    }
                    gravitySealDeferredGasOpening();
                    // A hydro impulse between the closing and opening operators.
                    particlesVel(kXDimension, gasIndex) += 0.375f;
                    auto const beforeOpening = particlesVel(kXDimension, gasIndex);
                    auto const opened = gravityFinishDeferredGasOpening();
                    double const expectedOpening = 0.25 * params.t0 * gasAcceleration;
                    requireClose("deferred gas opening follows hydro",
                        particlesVel(kXDimension, gasIndex), beforeOpening + expectedOpening, 1e-6);
                    double const kineticChange = 0.5 * gasMass *
                        (sqr(particlesVel(kXDimension, gasIndex)) - sqr(beforeOpening));
                    requireClose("deferred opening kinetic ledger", opened.dkin, kineticChange);
                    requireClose("deferred opening synchronized correction",
                        opened.darkMatterKineticCorrection, -kineticChange);
                    requireClose("deferred opening released its collection state",
                        gravityGasOpeningDeferred(), false, 0.0);
                    verifyPairDivergence("deferred opening preserves divergence history");
                }
            }

            resetKickState();
            auto frozen = params;
            frozen.freezeGasRungs = true;
            execute(frozen, backend != 0);
            requireClose("traditional gas rung remains frozen between common boundaries",
                particlesRung(gasIndex), 1);
            requireClose("frozen gas leaves dark-matter selection unchanged",
                particlesRung(darkMatterIndex), 2);
            requireClose("frozen traditional opening uses the existing interval",
                particlesVel(kXDimension, gasIndex),
                0.125 + 0.5 * rungDt[1] * params.t0 * gasAcceleration, 1e-6);
            verifyPairDivergence("frozen traditional kick preserves decoded divergence");

            resetKickState();
            sphParticlesSetProposedRung(0, 0);
            auto fullSelection = params;
            fullSelection.htime = true;
            fullSelection.fullGasRungSelection = true;
            execute(fullSelection, backend != 0);
            double const gasTimestep =
                params.eta * std::sqrt(gravityEffectiveRadius / gasAcceleration);
            int const expectedRung = int(std::ceil(std::log2(params.t0 / gasTimestep)));
            if (expectedRung <= 1) {
                throw std::runtime_error("Full-force rung fixture did not require a direct jump");
            }
            requireClose("hierarchical gas selects the full-force rung in the top walk",
                particlesRung(gasIndex), expectedRung);
            verifyPairDivergence("hierarchical full-force choice preserves decoded divergence");
            requireClose("full gas choice leaves hierarchical dark-matter selection unchanged",
                particlesRung(darkMatterIndex), 1);
            sphParticlesSetProposedRung(0, 7);
            auto correction = fullSelection;
            correction.fullGasRungSelection = false;
            correction.freezeGasRungs = true;
            correction.top = correction.ascending = false;
            correction.minRung = 1;
            execute(correction, backend != 0);
            requireClose("hierarchical correction preserves balanced gas rung",
                particlesRung(gasIndex), expectedRung);
            requireClose("hierarchical correction still selects dark-matter rung",
                particlesRung(darkMatterIndex), 2);
            verifyPairDivergence("hierarchical correction preserves decoded divergence");
            sphParticlesSetProposedRung(0, 4);
        }
    }
    requireClose(
        "gravity leaves hydro smoothing unchanged", sphParticlesSmoothingLength(0), hydroSupport);
    treeDestroy(true);
}

void checkpointPair() {
    FILE* file = std::tmpfile();
    if (file == nullptr) {
        throw std::runtime_error("Cannot create temporary checkpoint");
    }
    ParticleInteger const gas = sphParticlesParticleIndex(0);
    auto const position = particlesPos(kXDimension, gas);
    int const rung = particlesRung(gas);
    uint16_t const scaledDivergenceBits =
        getOptions().hydro.viscosityGradientSwitch() ? sphScaledDivergence[0].bits : 0;
    bool const retainedBalsara =
        getOptions().hydro.viscosityGradientSwitch() && getOptions().hydro.balsaraSwitch;
    sphParticlesSetConverged(0, true);
    sphParticlesSetSemiactive(0, false);
    uint8_t const packedFlags = sphParticlesFlags[0].bits;
    if (retainedBalsara) {
        sphParticlesBalsaraFactors[0] = 173;
    }
    particlesSave(file);
    particlesPos(kXDimension, gas) = 0.9;
    particlesRung(gas) = 9;
    sphParticlesSmoothingLength(0) = 0.5;
    sphParticlesEntropies[0] = 99.0;
    sphParticlesSetProposedRung(0, 9);
    sphParticlesSetConverged(0, false);
    sphParticlesSetSemiactive(0, true);
    if (getOptions().hydro.viscosityGradientSwitch()) {
        sphScaledDivergence[0] = {};
    }
    if (retainedBalsara) {
        sphParticlesBalsaraFactors[0] = 0;
    }
    std::rewind(file);
    particlesLoad(file);
    std::fclose(file);
    verifyMappings();
    requireClose("checkpoint particle count", particlesSize(), 2);
    requireClose("checkpoint gas count", sphParticlesSize(), 1);
    requireClose("checkpoint position",
        particlesPos(kXDimension, sphParticlesParticleIndex(0)).toDouble(), position.toDouble());
    requireClose("checkpoint hydro length", sphParticlesSmoothingLength(0), 0.02);
    requireClose("checkpoint internal energy", sphParticlesEntropies[0], 4.0);
    requireClose("checkpoint persistent hydro rung", sphParticlesProposedRung(0), 4);
    requireClose("checkpoint packed working flags and hydro rung", sphParticlesFlags[0].bits,
        packedFlags, 0.0);
    requireClose("checkpoint live gas rung", sphParticlesRung(0), rung, 0.0);
    if (retainedBalsara) {
        requireClose("checkpoint retained Balsara byte", sphParticlesBalsaraFactors[0], 173, 0.0);
    }
    if (getOptions().hydro.viscosityGradientSwitch()) {
        auto const& state = sphScaledDivergence[0];
        requireClose("checkpoint divergence scalar bits", state.bits, scaledDivergenceBits, 0.0);
        verifyPairDivergence("checkpoint preserves decoded divergence");
        requireClose("checkpoint excludes temporary viscosity diagnostics",
            sphParticlesViscosityDiagnostics == nullptr, true);
    }
}
}    // namespace detail

void initializeSphDomainRegression() {
    using namespace detail;
    initializePair(0.25);
    int const target = (hpxRank() + 1) % hpxSize();
    for (ParticleInteger i = 0; i < 2; i++) {
        particlesPos(kXDimension, i) = (target + (i == 0 ? 0.25 : 0.75)) / hpxSize();
        particlesPos(kXDimension, i).raw() += 17;
    }
    sphParticlesEntropies[0] = 20.0 + target;
    sphParticlesSetProposedRung(0, 5 + target % 4);
    sphParticlesSetConverged(0, target % 2 != 0);
    sphParticlesSetSemiactive(0, target % 3 != 0);
    if (getOptions().hydro.viscosityEvolution != "none") {
        sphParticlesBulkViscosityCoefficients[0] = 0.5f + target / 16.0f;
    }
    if (getOptions().hydro.viscosityGradientSwitch()) {
        auto& state = sphScaledDivergence[0];
        state = {};
        ALWAYS_ASSERT(sphEncodeDivergence(state, -2.0f + target / 16.0f, 1.0));
        if (getOptions().hydro.balsaraSwitch) {
            sphParticlesBalsaraFactors[0] = uint8_t(64 + target);
        }
    }
    if (getOptions().chemistry.enabled) {
        sphParticlesElectronFractions[0] = migrationElectronFraction(target);
    }
}
HPX_PLAIN_ACTION(initializeSphDomainRegression, InitializeSphDomainRegressionAction);

void validateSphPositionGather() {
    int const remoteRank = (hpxRank() + 1) % hpxSize();
    int const target = (remoteRank + 1) % hpxSize();
    // Reordered, overlapping, and empty ranges exercise both gather tasks and
    // the full native-width stride between the three coordinate planes.
    Vector<Pair<ParticleInteger>> const ranges{{1, 2}, {0, 2}, {0, 1}, {1, 1}};
    ParticleInteger const expectedParticles[] = {1, 0, 1, 0};
    constexpr size_t kGatherCount = 4;
    auto const positions = particlesGet(remoteRank, ranges).get();
    if (positions.size() != kDimensions * kGatherCount) {
        throw std::runtime_error("Remote position gather returned the wrong coordinate count");
    }
    for (int dim = 0; dim < kDimensions; ++dim) {
        for (size_t i = 0; i < kGatherCount; ++i) {
            Position expected(0.375);
            if (dim == kXDimension) {
                expected = (target + (expectedParticles[i] == 0 ? 0.25 : 0.75)) / hpxSize();
                expected.raw() += 17;
            }
            if (positions[dim * kGatherCount + i].raw() != expected.raw()) {
                throw std::runtime_error(
                    "Remote position gather changed native bits or coordinate-plane ordering");
            }
        }
    }
}
HPX_PLAIN_ACTION(validateSphPositionGather, ValidateSphPositionGatherAction);
void validateSphDomainRegression() {
    using namespace detail;
    verifyMappings();
    requireClose("migrated gas count", sphParticlesSize(), 1);
    auto const gas = sphParticlesParticleIndex(0);
    int const target = int(positionDomainCoordinate(particlesPos(kXDimension, gas)) * hpxSize());
    for (ParticleInteger particle = 0; particle < particlesSize(); ++particle) {
        double const offset = particlesType(particle) == kDarkMatterType ? 0.25 : 0.75;
        Position const expected((target + offset) / hpxSize());
        if (particlesPos(kXDimension, particle).raw() != expected.raw() + 17) {
            throw std::runtime_error("Migration changed native position bits");
        }
    }
    requireClose("migration persistent hydro rung", sphParticlesProposedRung(0), 5 + target % 4);
    requireClose("migration convergence flag", sphParticlesConverged(0), target % 2 != 0, 0.0);
    requireClose("migration semiactive flag", sphParticlesSemiactive(0), target % 3 != 0, 0.0);
    requireClose("migration canonical thermal state", sphParticlesEntropies[0], 20.0 + target);
    if (getOptions().hydro.viscosityEvolution != "none") {
        requireClose("migration viscosity alpha", sphParticlesBulkViscosityCoefficients[0],
            0.5 + target / 16.0);
    }
    if (getOptions().hydro.viscosityGradientSwitch()) {
        auto const& state = sphScaledDivergence[0];
        requireClose(
            "migration divergence sample", sphDecodeDivergence(state, 1.0), -2.0 + target / 16.0);
        requireClose("migration excludes temporary viscosity diagnostics",
            sphParticlesViscosityDiagnostics == nullptr, true);
        if (getOptions().hydro.balsaraSwitch) {
            requireClose(
                "migration retained Balsara byte", sphParticlesBalsaraFactors[0], 64 + target, 0.0);
        }
    }
    if (getOptions().chemistry.enabled) {
        auto const expected = migrationElectronFraction(target);
        auto const& actual = sphParticlesElectronFractions[0];
        if (std::bit_cast<uint32_t>(actual) != std::bit_cast<uint32_t>(expected)) {
            throw std::runtime_error("Migration changed the equilibrium electron-guess bits");
        }
    }
    ParticleGlobalRange remote;
    remote.proc = (hpxRank() + 1) % hpxSize();
    remote.range = {0, 1};
    if (getOptions().hydro.viscosityEvolution != "none") {
        float bulkViscosityCoefficient[3] = {-9.0f, -9.0f, -9.0f};
        sphParticlesGlobalReadDensity(remote, nullptr, 1, nullptr, bulkViscosityCoefficient);
        requireClose(
            "remote viscosity cache", bulkViscosityCoefficient[1], 0.5 + remote.proc / 16.0);
        requireClose("remote viscosity lower padding", bulkViscosityCoefficient[0], -9.0);
        requireClose("remote viscosity upper padding", bulkViscosityCoefficient[2], -9.0);
    }
    if (getOptions().hydro.viscosityGradientSwitch()) {
        float divergence[3] = {-9.0f, -9.0f, -9.0f};
        sphParticlesGlobalReadViscosityDiagnostics(remote, divergence, 1);
        requireClose("remote divergence lower padding", divergence[0], -9.0);
        requireClose("remote divergence upper padding", divergence[2], -9.0);
        double const remoteDivergence = -2.0 + remote.proc / 16.0;
        double const expectedSign =
            remoteDivergence > 0.0 ? 1.0 : (remoteDivergence < 0.0 ? -1.0 : 0.0);
        requireClose("remote prior divergence sign cache", divergence[1], expectedSign);
    }
    float velocity[kDimensions][3]{};
    sphParticlesGlobalReadVelocities(
        remote, velocity[kXDimension], velocity[kYDimension], velocity[kZDimension], 1);
    for (int dimension = 0; dimension < kDimensions; ++dimension) {
        requireClose("remote velocity cache", velocity[dimension][1], 0.125 * (dimension + 1));
    }
    Vector3<double> position;
    for (ParticleInteger i = 0; i < particlesSize(); i++) {
        for (int dim = 0; dim < kDimensions; dim++) {
            position[dim] = particlesPos(dim, i).toDouble();
        }
        if (!domainsFindMyBox().contains(position)) {
            throw std::runtime_error("Migrated particle lies outside its domain");
        }
    }
}
HPX_PLAIN_ACTION(validateSphDomainRegression, ValidateSphDomainRegressionAction);

void validateSphRemoteTreeCache() {
    TreeId remote;
    remote.proc = (hpxRank() + 1) % hpxSize();
    remote.index = 0;
    auto const* node = treeGetNode(remote);
    float const rootMass = detail::nodeMass(node);
    if (!(rootMass > 0.0f && rootMass <= float(hpxSize()))) {
        throw std::runtime_error("Remote tree cache lost the populated root multipole");
    }
}
HPX_PLAIN_ACTION(validateSphRemoteTreeCache, ValidateSphRemoteTreeCacheAction);

// These endpoints deliberately live on different ranks in the MPI fixture.
// The fine endpoint owns the interaction and must return the reciprocal impulse
// to a coarse source that is not a target of this midpoint force pass.
void initializeSphMidpointPairRegression(bool emptyRoot) {
    sphParticlesFree();
    particlesResize(0);
    ParticleInteger const localCount = hpxSize() == 1 ? 2 :
        emptyRoot                                     ? (hpxRank() == hpxSize() - 1 ? 2 : 0) :
                    (hpxRank() == 0 || hpxRank() == hpxSize() - 1 ? 1 : 0);
    sphParticlesResize(localCount);
    for (ParticleInteger gas = 0; gas < localCount; ++gas) {
        int const endpoint = hpxSize() == 1 || emptyRoot ? int(gas) : int(hpxRank() != 0);
        for (int dimension = 0; dimension < kDimensions; ++dimension) {
            double const offset = 0.01 * (dimension + 1);
            double const center =
                emptyRoot && dimension == kXDimension ? (hpxSize() - 0.5) / hpxSize() : 0.5;
            sphParticlesPosition(dimension, gas) = center + (endpoint == 0 ? -offset : offset);
            sphParticlesVelocity(dimension, gas) =
                0.025f * (dimension + 1) + (endpoint == 0 ? 0.1f : -0.1f);
        }
        particlesRung(sphParticlesParticleIndex(gas)) = endpoint == 0 ? 1 : 3;
        sphParticlesSetInternalEnergy(gas, 2.0f + endpoint);
        sphParticlesDensity(gas) = 1.0f;
        sphParticlesSmoothingLength(gas) = 0.25f;
        if (getOptions().hydro.viscosityEvolution != "none") {
            // Convergence need not trigger a positive shock target when pressure
            // is already decelerating the pair. Supply retained viscosity so
            // this fixture's strict dissipative-heating check is well posed.
            sphParticlesBulkViscosityCoefficients[gas] = 1.0f;
        }
    }
}
HPX_PLAIN_ACTION(initializeSphMidpointPairRegression, InitializeSphMidpointPairRegressionAction);

Array<double, 14> readSphMidpointPairRegression() {
    if (sphParticlesViscosityDiagnosticsActive()) {
        throw std::runtime_error("Midpoint pair retained viscosity diagnostics after its event");
    }
    Array<double, 14> result{};
    for (ParticleInteger gas = 0; gas < sphParticlesSize(); ++gas) {
        int const rung = particlesRung(sphParticlesParticleIndex(gas));
        if (rung != 1 && rung != 3) {
            throw std::runtime_error("Midpoint pair changed a live endpoint rung");
        }
        int const offset = rung == 1 ? 0 : 7;
        result[offset] += 1.0;
        for (int dimension = 0; dimension < kDimensions; ++dimension) {
            result[offset + 1 + dimension] += sphParticlesVelocity(dimension, gas);
        }
        result[offset + 4] += sphParticlesEntropy(gas);
        result[offset + 5] += hpxRank();
        result[offset + 6] += sphParticlesSmoothingLength(gas);
    }
    return result;
}
HPX_PLAIN_ACTION(readSphMidpointPairRegression, ReadSphMidpointPairRegressionAction);

void runSphMidpointPairRegression(bool useCuda) {
    auto options = getOptions();
    options.chemistry.enabled = false;
    options.starFormation.enabled = false;
    options.run.scenario = "soundWave";
    options.run.minRung = 0;
    options.hydro.viscosityEvolution = "none";
    options.hydro.bulkViscosityCoefficient = 1.0;
    options.hydro.vonNeumannRichtmyerCoefficient = 2.0;
    options.hydro.balsaraSwitch = false;
    options.hydro.minSmoothingLength = options.hydro.maxSmoothingLength = 0.25;
    options.hydro.damping = options.hydro.centralAcceleration = 0.0;
    options.hydro.yAcceleration = options.hydro.zPlaneAcceleration = 0.0;
    options.initial.numberOfParticles = 2;
    auto const collect = [](bool emptyRoot) {
        Array<double, 14> total{};
        Vector<hpx::future<Array<double, 14>>> pending;
        for (auto const& locality : hpxLocalities()) {
            pending.push_back(hpx::async<ReadSphMidpointPairRegressionAction>(locality));
        }
        for (auto& future : pending) {
            auto const local = future.get();
            for (size_t component = 0; component < total.size(); ++component) {
                total[component] += local[component];
            }
        }
        int const coarseRank = emptyRoot ? hpxSize() - 1 : 0;
        if (total[0] != 1.0 || total[7] != 1.0 || total[5] != coarseRank ||
            total[12] != hpxSize() - 1) {
            throw std::runtime_error("Midpoint pair lost endpoint identity or rank ownership");
        }
        return total;
    };
    auto const requireUnchanged = [](auto const& before, auto const& after) {
        for (int component = 0; component < before.size(); ++component) {
            if (before[component] != after[component]) {
                throw std::runtime_error("An unowned or zero-duration midpoint pair changed state");
            }
        }
    };
    for (int fixture = 0; fixture < (hpxSize() > 1 ? 3 : 1); ++fixture) {
        // The second MPI fixture has no root gas allocation. Diagnostics lifetime
        // must still be collective when the driver and RK stages share its scope.
        // The third fixture exchanges current Cullen-Dehnen divergence signs
        // across ranks, rather than the retained endpoint-history signs.
        bool const emptyRoot = fixture == 1;
        options.hydro.viscosityEvolution = fixture == 1 ? "huEtal" :
            fixture == 2                                ? "cullenAndDehnen" :
                                                          "none";
        options.hydro.balsaraSwitch = false;
        for (int backend = 0; backend <= int(useCuda); ++backend) {
            options.hydro.cuda = backend != 0;
            sphTreeDestroy(true);
            setOptions(options);
            Vector<hpx::future<void>> pending;
            for (auto const& locality : hpxLocalities()) {
                pending.push_back(
                    hpx::async<InitializeSphMidpointPairRegressionAction>(locality, emptyRoot));
            }
            for (auto& future : pending) {
                future.get();
            }
            EnergiesT energies{};
            // Initialize both densities and canonical entropies before measuring.
            sphMidpointStep(-1, 0.0, 0.0, 0.001, 1.0, 0.0, energies);
            auto const initial = collect(emptyRoot);
            sphMidpointStep(3, 0.0, 0.0, 0.001, 1.0, 0.0, energies);
            requireUnchanged(initial, collect(emptyRoot));
            sphMidpointStep(1, 0.00025, 0.0005, 0.001, 1.0, 0.0, energies);
            requireUnchanged(initial, collect(emptyRoot));
            sphMidpointStep(3, 0.0000625, 0.000125, 0.001, 1.0, 0.0, energies);
            auto const kicked = collect(emptyRoot);
            for (int dimension = 0; dimension < kDimensions; ++dimension) {
                double const coarse = kicked[1 + dimension] - initial[1 + dimension];
                double const fine = kicked[8 + dimension] - initial[8 + dimension];
                double const scale = std::abs(initial[1 + dimension]) +
                    std::abs(initial[8 + dimension]) + std::abs(kicked[1 + dimension]) +
                    std::abs(kicked[8 + dimension]);
                double const tolerance = 2.0 * std::numeric_limits<float>::epsilon() * scale;
                if (!(std::abs(coarse) > 100.0 * tolerance &&
                        std::abs(coarse + fine) <= tolerance)) {
                    throw std::runtime_error(
                        "Mixed-rung midpoint pair lost its reciprocal impulse at float precision");
                }
            }
            for (int offset : {0, 7}) {
                if (!(kicked[offset + 4] > initial[offset + 4] &&
                        std::isfinite(kicked[offset + 4]))) {
                    throw std::runtime_error(std::format(
                        "Mixed-rung pair failed to heat endpoint {:d} ({:s}, fixture {:d}): "
                        "entropy {:.9e} -> {:.9e}",
                        offset / 7, backend ? "CUDA" : "CPU", fixture, initial[offset + 4],
                        kicked[offset + 4]));
                }
                detail::requireClose("fixed pair support", kicked[offset + 6], 0.25, 1.e-7);
            }
            std::cout
                << std::format(
                       "Conservative mixed-rung SPH pair passed ({:s}, {:d} localities{:s})\n",
                       backend ? "CUDA" : "CPU", hpxSize(),
                       emptyRoot        ? ", empty root with Hu diagnostics" :
                           fixture == 2 ? ", cross-rank Cullen-Dehnen diagnostics" :
                                          "")
                << std::flush;
        }
    }
    sphTreeDestroy(true);
}

namespace detail {
static Vector<char> gravitySensorSnapshot;

Vector<char> gravitySensorParticleBytes() {
    auto const closeFile = [](FILE* stream) { std::fclose(stream); };
    std::unique_ptr<FILE, decltype(closeFile)> file(std::tmpfile(), closeFile);
    if (!file) {
        throw std::runtime_error("Cannot create sensor-state snapshot");
    }
    particlesSave(file.get());
    for (ParticleInteger particle = 0; particle < particlesOwnedSize(); ++particle) {
        for (int dimension = 0; dimension < kDimensions; ++dimension) {
            std::fwrite(
                &particlesGravityAcceleration(dimension, particle), sizeof(float), 1, file.get());
        }
        std::fwrite(&particlesPot(particle), sizeof(float), 1, file.get());
    }
    long const length = std::ftell(file.get());
    if (length < 0 || std::fseek(file.get(), 0, SEEK_SET) != 0) {
        throw std::runtime_error("Cannot rewind sensor-state snapshot");
    }
    Vector<char> result(static_cast<size_t>(length));
    if (std::fread(result.data(), 1, result.size(), file.get()) != result.size()) {
        throw std::runtime_error("Cannot read sensor-state snapshot");
    }
    return result;
}
}    // namespace detail

void prepareSphGravitySensorRegression() {
    for (ParticleInteger particle = 0; particle < particlesOwnedSize(); ++particle) {
        particlesRung(particle) = particlesType(particle) == kSphType ? 1 : 3;
        if (particlesType(particle) == kDarkMatterType) {
            // Break the alternating periodic lattice's exact zero-force
            // symmetry so an unfilled stage buffer cannot pass the comparison.
            particlesPos(kXDimension, particle) =
                particlesPos(kXDimension, particle).toDouble() + 0.03125;
            particlesPos(kYDimension, particle) =
                particlesPos(kYDimension, particle).toDouble() + 0.015625;
        }
    }
}
HPX_PLAIN_ACTION(prepareSphGravitySensorRegression, PrepareSphGravitySensorRegressionAction);

void captureSphGravitySensorRegression() {
    particlesResolveWithSphParticles();
    detail::gravitySensorSnapshot = detail::gravitySensorParticleBytes();
}
HPX_PLAIN_ACTION(captureSphGravitySensorRegression, CaptureSphGravitySensorRegressionAction);

void validateSphGravitySensorRegression() {
    if (detail::gravitySensorSnapshot != detail::gravitySensorParticleBytes()) {
        throw std::runtime_error("Gravity sensor changed particle/gas state or hierarchy ranges");
    }
    detail::verifyMappings();
    for (ParticleInteger gas = 0; gas < sphParticlesSize(); ++gas) {
        ParticleInteger const particle = sphParticlesParticleIndex(gas);
        double referenceNormSquared = 0.0;
        for (int dimension = 0; dimension < kDimensions; ++dimension) {
            double const reference = particlesGravityAcceleration(dimension, particle);
            referenceNormSquared += reference * reference;
            detail::requireClose("full-source gravity sensor includes inactive gas sinks",
                sphStageAccelerations()[gas][dimension], reference, 5.e-5);
        }
        if (!(referenceNormSquared > 0.01)) {
            throw std::runtime_error("Gravity sensor fixture has a vanishing reference force");
        }
    }
    detail::gravitySensorSnapshot.clear();
}
HPX_PLAIN_ACTION(validateSphGravitySensorRegression, ValidateSphGravitySensorRegressionAction);

void runSphGravitySensorRegression() {
    auto const originalOptions = getOptions();
    auto sensorOptions = originalOptions;
    sensorOptions.gravity.gm = detail::kGravityScale;
    setOptions(sensorOptions);
    Vector<hpx::future<void>> futures;
    for (auto const& locality : hpxLocalities()) {
        futures.push_back(hpx::async<PrepareSphGravitySensorRegressionAction>(locality));
    }
    for (auto& future : futures) {
        future.get();
    }
    // Save a full-source FMM reference without either signed kick. a != 1
    // guards against adding the hydro pressure term's 1/a to canonical gravity.
    // Match the sensor's source bounds so this compares force storage/selection,
    // not the truncation error of two different accepted multipole partitions.
    TreeCreateParameters referenceTree(0, 0.5, 0.0);
    referenceTree.allParticles = true;
    auto const built = treeCreate(referenceTree);
    auto parameters = detail::makeKickParams();
    parameters.gm = getOptions().gravity.gm;
    parameters.a = 0.375f;
    parameters.htime = true;
    parameters.ascending = parameters.descending = false;
    ExpansionBuffer expansion;
    expansion.zero();
    Vector3<Position> origin;
    for (int dimension = 0; dimension < kDimensions; ++dimension) {
        origin[dimension] = 0.0;
    }
    kick(parameters, expansion, origin, built.id, Vector<TreeId>{built.id},
        Vector<TreeId>{built.id}, nullptr);
    treeDestroy();
    // The gas lies before the active DM tail. Its full-force sensor must still
    // include every source and sink, then restore the non-geometric order.
    particlesPushRungs();
    particlesSortByRung(2);
    futures.clear();
    for (auto const& locality : hpxLocalities()) {
        futures.push_back(hpx::async<CaptureSphGravitySensorRegressionAction>(locality));
    }
    for (auto& future : futures) {
        future.get();
    }
    auto const retained = sphStep1(0);
    // Capture after SPH sorting, which may legitimately reorder gas records.
    futures.clear();
    for (auto const& locality : hpxLocalities()) {
        futures.push_back(hpx::async<CaptureSphGravitySensorRegressionAction>(locality));
    }
    for (auto& future : futures) {
        future.get();
    }
    std::array<unsigned char, sizeof(SphTreeNode)> retainedRoot{};
    std::memcpy(retainedRoot.data(), sphTreeGetNode(retained.id), retainedRoot.size());
    SphRunParameters stage;
    stage.runType = kSphRunHydro;
    stage.phase = 0;
    stage.minimumRung = 0;
    stage.baseTimestep = 1.0f;
    stage.scaleFactor = 0.375f;
    stage.expansionRate = 0.0f;
    stage.courantFactor = getOptions().hydro.cfl;
    sphPrepareStageRates(stage, true);
    try {
        gravityRefreshSphAcceleration(0.5);
        futures.clear();
        for (auto const& locality : hpxLocalities()) {
            futures.push_back(hpx::async<ValidateSphGravitySensorRegressionAction>(locality));
        }
        hpx::wait_all(futures.begin(), futures.end());
        for (auto& future : futures) {
            future.get();
        }
        if (std::memcmp(retainedRoot.data(), sphTreeGetNode(retained.id), retainedRoot.size()) !=
            0) {
            throw std::runtime_error("Gravity sensor changed the retained SPH tree");
        }
        sphFinishPairStages(false);
    } catch (...) {
        auto const failure = std::current_exception();
        sphFinishPairStages(false);
        std::rethrow_exception(failure);
    }
    particlesPopRungs();
    sphTreeDestroy(true);
    setOptions(originalOptions);
    std::cout << std::format(
                     "Full gravity sensor and exact state restoration passed ({:d} localities)\n",
                     hpxSize())
              << std::flush;
}

void sphGravityTest() {
#ifdef USE_CUDA
    if (getOptions().gravity.cuda) {
        verifySelectedKernelsCuda();
    }
#endif
    using namespace detail;
    resetGravityCounters();
    double closeCount, directCount;
    getGravityCounters(closeCount, directCount);
    requireClose("empty close-interaction diagnostic", closeCount, 0.0);
    requireClose("empty direct-interaction diagnostic", directCount, 0.0);
    auto opts = getOptions();
    bool const useCuda = opts.gravity.cuda;
    opts.hydro.enabled = true;

    opts.gravity.enabled = true;
    opts.run.hierarchicalGravity = getOptions().run.hierarchicalGravity;
    opts.analysis.saveForce = true;

    opts.analysis.groups = false;
    opts.analysis.tracers = false;
    opts.initial.darkMatterMass = 0.75;
    opts.initial.gasMass = 0.25;
    opts.compute.bucketSize = 1;
    opts.initial.numberOfParticles = 2 * hpxSize();
    opts.gravity.smoothingLength = 0.01;
    setOptions(opts);
    verifyKernelSelections();
    if (hpxSize() == 1) {
        verifySphExceptionMask();
        verifyBhPotentials(false);
#ifdef USE_CUDA
        if (useCuda) {
            verifyBhPotentials(true);
        }
#endif
        for (char const* viscosity : {"cullenAndDehnen", "priceEtAl", "huEtal"}) {
            auto selected = opts;
            selected.hydro.viscosityEvolution = viscosity;
            setOptions(selected);
            runPairCase(useCuda, false);
            checkpointPair();
            sphParticlesFree();
            setOptions(opts);
        }
        runPairCase(useCuda, false);
        runPairCase(useCuda, true, 0.001f);
        runPairCase(useCuda, true, 0.1f);
        runPairCase(useCuda, true);
        checkpointPair();
        sphParticlesResize(2);
        auto const secondGas = sphParticlesParticleIndex(1);
        for (int dim = 0; dim < kDimensions; dim++) {
            particlesPos(dim, secondGas) = 0.75;
            particlesVel(dim, secondGas) = 0.0;
        }
        particlesRung(secondGas) = 1;
        sphCheckpointTest();
    }
    Vector<hpx::future<void>> pending;
    for (auto const& locality : hpxLocalities()) {
        pending.push_back(hpx::async<InitializeSphDomainRegressionAction>(locality));
    }
    for (auto& future : pending) {
        future.get();
    }
    pending.clear();
    for (auto const& locality : hpxLocalities()) {
        pending.push_back(hpx::async<ValidateSphPositionGatherAction>(locality));
    }
    for (auto& future : pending) {
        future.get();
    }
    std::cout << std::format(
                     "Native position gather regression passed ({:d} bits, {:d} localities)\n",
                     kPositionBits, hpxSize())
              << std::flush;
    domainsRebound();
    domainsBegin(0);
    domainsEnd();
    pending.clear();
    for (auto const& locality : hpxLocalities()) {
        pending.push_back(hpx::async<ValidateSphDomainRegressionAction>(locality));
    }
    for (auto& future : pending) {
        future.get();
    }
    if (hpxSize() > 1) {
        treeCreate(TreeCreateParameters(0, 0.5, getOptions().gravity.smoothingLength));
        pending.clear();
        for (auto const& locality : hpxLocalities()) {
            pending.push_back(hpx::async<ValidateSphRemoteTreeCacheAction>(locality));
        }
        for (auto& future : pending) {
            future.get();
        }
        treeDestroy(true);
    }
    runSphGravitySensorRegression();
    runSphMidpointPairRegression(useCuda);
    std::cout << std::format("SPH gravity regression passed ({:s}, {:d} localities)\n",
                     useCuda ? "CPU and CUDA" : "CPU", hpxSize())
              << std::flush;
}
