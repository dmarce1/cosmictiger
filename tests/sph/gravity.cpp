/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numbers>
#include <stdexcept>
#include <string>
#include <thread>

#include <cosmictiger/analysis/bh.hpp>
#include <cosmictiger/core/build_config.hpp>
#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/gravity/dehnen/distributed_tree.hpp>
#include <cosmictiger/gravity/dehnen/execution.hpp>
#include <cosmictiger/gravity/fmm_kernels/cartesian_operators.hpp>
#include <cosmictiger/gravity/fmm_order.hpp>
#include <cosmictiger/gravity/gravity.hpp>
#include <cosmictiger/gravity/kick.hpp>
#include <cosmictiger/gravity/tree.hpp>
#include <cosmictiger/particles/domain.hpp>
#include <cosmictiger/particles/particles.hpp>
#include <cosmictiger/sph/cullen_and_dehnen/particle_exchange.hpp>
#include <cosmictiger/sph/sph.hpp>
#include <cosmictiger/sph/sph_particles.hpp>
#include <cosmictiger/sph/sph_tree.hpp>

#ifdef USE_CUDA
void verifySelectedKernelsCuda();
#endif

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
    auto const verify = [](char const* label, radial_kernel::KernelTable const& table,
                            std::string const& family, int degree) {
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

primordial::State migrationChemistryState(int target) {
    primordial::State state{};
    state.abundances[0] = primordial::SignedAbundance::fromMinority(0.0f, true);
    state.abundances[1] = primordial::SignedAbundance::fromMinority(1.0e-25f, true);
    for (int species = 2; species < primordial::kAbundanceCount; ++species) {
        state.abundances[species].setRaw(0.01f * (species + target));
    }
    state.thermalEnergy = 60.0f + target;
    return state;
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
    sphParticlesSupportRadius(0) = 0.02;
    sphParticlesThermodynamicStates[0].internalEnergy = 4.0;
    sphParticlesPredictedInternalEnergyRate(0) = 0.75;
    sphParticlesInternalEnergyRate(0) = 0.125;
    sphParticlesGravityAcceleration(kXDimension, 0) = 0.0;
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
    p.gravitySupportRadius = getOptions().gravity.supportRadius;
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
}
void verifyKick(double inverseCube, double inverseRadius, double separation) {
    for (ParticleInteger i = 0; i < particlesSize(); i++) {
        bool const gas = particlesIsSph(i);
        double const sourceMass =
            gas ? getOptions().initial.darkMatterMass : getOptions().initial.gasMass;
        double const acceleration =
            kGravityScale * (gas ? 1.0 : -1.0) * sourceMass * separation * inverseCube;
        requireClose(
            "kick acceleration", particlesGravityAcceleration(kXDimension, i), acceleration);
        requireClose("kick potential with self mass removed", particlesPot(i),
            -kGravityScale * sourceMass * inverseRadius, 2e-4);
        requireClose("transverse force", particlesGravityAcceleration(kYDimension, i), 0.0);
        if (gas) {
            requireClose("gas acceleration handoff",
                sphParticlesGravityAcceleration(kXDimension, particlesCatIndex(i)), acceleration);
            for (int dim = 0; dim < kDimensions; dim++) {
                requireClose("gas velocity must not be gravity-kicked twice", particlesVel(dim, i),
                    0.125 * (dim + 1));
            }
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
    double const gasMass = getOptions().initial.gasMass;
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
    double const support = getOptions().gravity.supportRadius;
    double const q = separation / support;
    double const fourPi = 4.0 * std::numbers::pi;
    // Integrate the density independently of the force/potential antiderivative tables.
    double const mass = fourPi *
        integrateReference(
            [&](double radius) { return radius * radius * radial_kernel::value(kernel, radius); },
            0.0, q);
    inverseCube = mass / (separation * separation * separation);
    inverseRadius =
        (mass / q +
            fourPi *
                integrateReference(
                    [&](double radius) { return radius * radial_kernel::value(kernel, radius); }, q,
                    1.0)) /
        support;
}

void verifyBhPotentials(bool useCuda) {
    double const separation = 0.25 * getOptions().gravity.supportRadius;
    DeviceVector<Array<float, kDimensions>> sources(2);
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
    DeviceVector<Array<float, kDimensions>> points(2);
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
    double const nominalSeparation = softened ? 0.25 * getOptions().gravity.supportRadius : 0.25;
    initializePair(nominalSeparation);
    sphParticlesSupportRadius(0) = hydroSupport;
    auto const built = treeCreate(TreeCreateParameters(0, 0.5, getOptions().gravity.supportRadius));
    verifyMappings();
    requireClose("root mixed mass", nodeMass(treeGetNode(built.id)), 1.0);
    requireClose("gravity tree support is independent of hydro smoothing",
        built.maxGravitySupportRadius, getOptions().gravity.supportRadius);
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
        cpuGravityPp(direct, true, self, sources, getOptions().gravity.supportRadius);
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
                Array<float, kDimensions> dx;
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
                    cosmictiger::gravity::cartesian::L2P(
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
    Array<Position, kDimensions> origin;
    for (int dim = 0; dim < kDimensions; dim++) {
        origin[dim] = 0.0;
    }
    auto params = makeKickParams();
    double const darkMatterAcceleration =
        kGravityScale * getOptions().initial.gasMass * separation * inverseCube;
    double const gravityEffectiveRadius = getOptions().gravity.supportRadius /
        selectedGravityKernel().supportRadiusPerEffectiveRadius;
    double const darkMatterTimestep =
        params.eta * std::sqrt(gravityEffectiveRadius / darkMatterAcceleration);
    params.t0 = 2.2 * darkMatterTimestep;
    resetKickState();
    auto const cpuResult =
        kick(params, local, origin, built.id, Vector<TreeId>{built.id}, Vector<TreeId>{}, nullptr);
    requireClose("CPU kick total potential", cpuResult.pot,
        -kGravityScale * getOptions().initial.darkMatterMass * getOptions().initial.gasMass *
            inverseRadius,
        2e-4);
    verifyKick(inverseCube, inverseRadius, separation);
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
        verifyKick(inverseCube, inverseRadius, separation);
        requireClose("CUDA gravity rung uses the variance smoothing length",
            particlesRung(darkMatterIndex), 2);
        verifySynchronizedDiagnostics(cudaResult);
    }
#else
    if (useCuda) {
        throw std::runtime_error("CUDA regression requested from a non-CUDA build");
    }
#endif
    requireClose(
        "gravity leaves hydro smoothing unchanged", sphParticlesSupportRadius(0), hydroSupport);
    treeDestroy(true);
}
void checkpointPair() {
    FILE* file = std::tmpfile();
    if (file == nullptr) {
        throw std::runtime_error("Cannot create temporary checkpoint");
    }
    ParticleInteger const gas = sphParticlesParticleIndex(0);
    auto const position = particlesPos(kXDimension, gas);
    particlesSave(file);
    particlesPos(kXDimension, gas) = 0.9;
    sphParticlesSupportRadius(0) = 0.5;
    sphParticlesThermodynamicStates[0].internalEnergy = 99.0;
    sphParticlesPredictedInternalEnergyRate(0) = 99.0;
    sphParticlesInternalEnergyRate(0) = 99.0;
    std::rewind(file);
    particlesLoad(file);
    std::fclose(file);
    verifyMappings();
    requireClose("checkpoint particle count", particlesSize(), 2);
    requireClose("checkpoint gas count", sphParticlesSize(), 1);
    requireClose("checkpoint position",
        particlesPos(kXDimension, sphParticlesParticleIndex(0)).toDouble(), position.toDouble());
    requireClose("checkpoint hydro length", sphParticlesSupportRadius(0), 0.02);
    requireClose(
        "checkpoint internal energy", sphParticlesThermodynamicStates[0].internalEnergy, 4.0);
    requireClose("checkpoint predicted internal-energy rate",
        sphParticlesPredictedInternalEnergyRate(0), 0.75);
    requireClose("checkpoint thermal stage", sphParticlesInternalEnergyRate(0), 0.125);
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
    sphParticlesPredictedInternalEnergyRate(0) = 10.0 + target;
    sphParticlesInternalEnergyRate(0) = 20.0 + target;
    sphParticlesAccelerationStates[0].acceleration[kXDimension] = 30.0 + target;
    sphParticlesPredictedAccelerations[kYDimension][0] = 40.0 + target;
    sphParticlesGravityAccelerations[kZDimension][0] = 50.0 + target;
    if (getOptions().hydro.viscosityEvolution != "none") {
        sphParticlesViscosityStates[0] = {0.5f + target / 16.0f, -0.0625f + target / 128.0f, 99.0f};
    }
    if (getOptions().hydro.viscosityEvolution == "cullenAndDehnen") {
        sphParticlesCullenDehnenStates[0] = {-1.5f + target / 8.0f, 77.0f, 88.0f, 99.0f};
    }
    if (getOptions().chemistry.enabled) {
        sphParticlesChemistryStates[0] = migrationChemistryState(target);
        sphParticlesChemistryEos[0] = {0.4f + 0.01f * target, 1.4f + 0.01f * target};
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
    requireClose("migration predicted internal-energy rate",
        sphParticlesPredictedInternalEnergyRate(0), 10.0 + target);
    requireClose("migration thermal stage", sphParticlesInternalEnergyRate(0), 20.0 + target);
    requireClose("migration acceleration stage",
        sphParticlesAccelerationStates[0].acceleration[kXDimension], 30.0 + target);
    requireClose("migration predicted acceleration",
        sphParticlesPredictedAccelerations[kYDimension][0], 40.0 + target);
    requireClose(
        "migration gravity", sphParticlesGravityAccelerations[kZDimension][0], 50.0 + target);
    if (getOptions().hydro.viscosityEvolution != "none") {
        requireClose(
            "migration viscosity alpha", sphParticlesViscosityStates[0].alpha, 0.5 + target / 16.0);
        requireClose("migration viscosity rate", sphParticlesViscosityStates[0].rate,
            -0.0625 + target / 128.0);
        requireClose("migration resets predictor scratch",
            sphParticlesViscosityStates[0].predictedAlpha, sphParticlesViscosityStates[0].alpha);
    }
    if (getOptions().hydro.viscosityEvolution == "cullenAndDehnen") {
        auto const& state = sphParticlesCullenDehnenStates[0];
        requireClose(
            "migration preserves physical divergence", state.divergence, -1.5 + target / 8.0);
        requireClose("migration resets pending divergence", state.pendingDivergence, 0.0);
        requireClose("migration resets target alpha", state.targetAlpha, 0.0);
        requireClose("migration resets decay time", state.inverseDecayTime, 0.0);
    }
    if (getOptions().chemistry.enabled) {
        auto const expected = migrationChemistryState(target);
        auto const& actual = sphParticlesChemistryStates[0];
        for (int species = 0; species < primordial::kAbundanceCount; ++species) {
            if (std::bit_cast<uint32_t>(actual.abundances[species].raw()) !=
                std::bit_cast<uint32_t>(expected.abundances[species].raw())) {
                throw std::runtime_error("Migration changed encoded chemical abundance bits");
            }
        }
        requireClose("migration chemistry energy", actual.thermalEnergy, expected.thermalEnergy);
        requireClose(
            "migration pressure ratio", sphParticlesChemistryEos[0].first, 0.4f + 0.01f * target);
        requireClose(
            "migration adiabatic index", sphParticlesChemistryEos[0].second, 1.4f + 0.01f * target);
    }
    ParticleGlobalRange remote;
    remote.proc = (hpxRank() + 1) % hpxSize();
    remote.range = {0, 1};
    if (getOptions().hydro.viscosityEvolution != "none") {
        float alpha[3] = {-9.0f, -9.0f, -9.0f};
        sphParticlesGlobalReadDensity(remote, nullptr, 1, nullptr, alpha);
        requireClose("remote viscosity cache", alpha[1], 0.5 + remote.proc / 16.0);
        requireClose("remote viscosity lower padding", alpha[0], -9.0);
        requireClose("remote viscosity upper padding", alpha[2], -9.0);
    }
    if (getOptions().hydro.viscosityEvolution == "cullenAndDehnen") {
        float acceleration[kDimensions][3]{};
        float divergence[3]{};
        sphParticlesGlobalReadViscosityDiagnostics(remote, acceleration[kXDimension],
            acceleration[kYDimension], acceleration[kZDimension], divergence, 1);
        requireClose(
            "remote prior acceleration cache", acceleration[kXDimension][1], 30.0 + remote.proc);
        requireClose("remote prior divergence cache", divergence[1], -1.5 + remote.proc / 8.0);
    }
    float velocity[kDimensions][3]{};
    sphParticlesGlobalReadVelocities(
        remote, velocity[kXDimension], velocity[kYDimension], velocity[kZDimension], 1);
    for (int dimension = 0; dimension < kDimensions; ++dimension) {
        requireClose("remote velocity cache", velocity[dimension][1], 0.125 * (dimension + 1));
    }
    if (getOptions().chemistry.enabled) {
        Position positionX[3]{}, positionY[3]{}, positionZ[3]{};
        sphParticlesGlobalReadPositions(remote, positionX, positionY, positionZ, 1);
        int const remoteTarget = int(positionX[1].toDouble() * hpxSize());
        float pressureRatio[3]{}, adiabaticIndex[3]{};
        sphParticlesGlobalReadChemistryEos(remote, pressureRatio, adiabaticIndex, 1);
        requireClose("remote pressure ratio cache", pressureRatio[1], 0.4f + 0.01f * remoteTarget);
        requireClose(
            "remote adiabatic index cache", adiabaticIndex[1], 1.4f + 0.01f * remoteTarget);
        requireClose("EOS cache preserves preceding destination", pressureRatio[0], 0.0);
        requireClose("EOS cache preserves following destination", adiabaticIndex[2], 0.0);
    }
    Array<double, kDimensions> position;
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

void sphGravityTest() {
#ifdef USE_CUDA
    if (getOptions().compute.cuda) {
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
    bool const useCuda = opts.compute.cuda;
    opts.hydro.enabled = true;

    opts.gravity.enabled = true;
    opts.run.hierarchicalGravity = integration::kBuildUsesHierarchical;
    opts.analysis.saveForce = true;

    opts.analysis.groups = false;
    opts.analysis.tracers = false;
    opts.initial.darkMatterMass = 0.75;
    opts.initial.gasMass = 0.25;
    opts.compute.bucketSize = 1;
    opts.initial.numberOfParticles = 2 * hpxSize();
    opts.gravity.supportRadius = 0.01;
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
    PRINT("Native position gather regression passed (%i bits, %i localities)\n", kPositionBits,
        hpxSize());
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
        treeCreate(TreeCreateParameters(0, 0.5, getOptions().gravity.supportRadius));
        pending.clear();
        for (auto const& locality : hpxLocalities()) {
            pending.push_back(hpx::async<ValidateSphRemoteTreeCacheAction>(locality));
        }
        for (auto& future : pending) {
            future.get();
        }
        treeDestroy(true);
    }
    PRINT("SPH gravity regression passed (%s, %i localities)\n", useCuda ? "CPU and CUDA" : "CPU",
        hpxSize());
}
