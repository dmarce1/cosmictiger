/* CosmicTiger - distributed PMMM timestep and staged kick regression; BSL-1.0. */

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/core/time.hpp>
#include <cosmictiger/gravity/nitadori/kick.hpp>
#include <cosmictiger/gravity/nitadori/solver.hpp>
#include <cosmictiger/particles/particles.hpp>

namespace detail {

struct MeshTimestepSample {
    Array<float, kDimensions> acceleration{};
    Array<float, kDimensions> velocity{};
    double position{};
    int rung{};
    int rank{};

    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & acceleration & velocity & position & rung & rank;
    }
};

void requireMeshTimestep(bool condition, std::string const& message) {
    if (!condition) {
        throw std::runtime_error("PMMM timestep regression: " + message);
    }
}

void initializeMeshTimestepFixture() {
    // The symmetry center has negligible acceleration. On multiple localities
    // both force-limiting particles belong to rank one, testing the reduction.
    bool const center = hpxRank() == 0;
    bool const endpoints = hpxRank() == (hpxSize() == 1 ? 0 : 1);
    particlesResize(int(center) + 2 * int(endpoints));
    ParticleInteger local = 0;
    for (int point = 0; point < 3; ++point) {
        if ((point == 1 && !center) || (point != 1 && !endpoints)) {
            continue;
        }
        for (int dim = 0; dim < kDimensions; ++dim) {
            particlesPos(dim, local) = dim == kXDimension ? 0.375 + 0.125 * point : 0.5;
            particlesVel(dim, local) = 0.0f;
        }
        particlesRung(local) = 1;
        ++local;
    }
}

Vector<MeshTimestepSample> readMeshTimestepFixture() {
    Vector<MeshTimestepSample> samples(particlesSize());
    for (ParticleInteger i = 0; i < particlesSize(); ++i) {
        auto& sample = samples[i];
        sample.acceleration = pmmmAcceleration(i);
        for (int dim = 0; dim < kDimensions; ++dim) {
            sample.velocity[dim] = particlesVel(dim, i);
        }
        sample.position = particlesPos(kXDimension, i).toDouble();
        sample.rung = particlesRung(i);
        sample.rank = hpxRank();
    }
    return samples;
}

}    // namespace detail

HPX_PLAIN_ACTION(detail::initializeMeshTimestepFixture, InitializeMeshTimestepFixtureAction);
HPX_PLAIN_ACTION(detail::readMeshTimestepFixture, ReadMeshTimestepFixtureAction);

namespace detail {

Vector<MeshTimestepSample> meshTimestepSamples() {
    std::vector<hpx::future<Vector<MeshTimestepSample>>> reads;
    for (auto const& locality : hpxLocalities()) {
        reads.push_back(hpx::async<ReadMeshTimestepFixtureAction>(locality));
    }
    Vector<MeshTimestepSample> samples;
    for (auto& future : reads) {
        auto local = future.get();
        samples.insert(samples.end(), local.begin(), local.end());
    }
    std::sort(samples.begin(), samples.end(),
        [](auto const& left, auto const& right) { return left.position < right.position; });
    requireMeshTimestep(samples.size() == 3, "fixture lost particles");
    return samples;
}

void checkMeshTimestepKick(Vector<MeshTimestepSample> const& before,
    Vector<MeshTimestepSample> const& after, int commonRung, bool opening, bool firstCall) {
    for (size_t i = 0; i < before.size(); ++i) {
        requireMeshTimestep(
            before[i].position == after[i].position, "kick changed particle positions");
        requireMeshTimestep(after[i].rung == (opening ? commonRung : before[i].rung),
            opening ? "finalization did not set a globally common rung" :
                      "preparation changed a rung before all proposals were reduced");
        double const halfStep =
            firstCall ? 0.0 : std::ldexp(0.5, -(opening ? commonRung : before[i].rung));
        for (int dim = 0; dim < kDimensions; ++dim) {
            double const expected =
                std::fma(double(before[i].acceleration[dim]), halfStep, before[i].velocity[dim]);
            requireMeshTimestep(std::isfinite(after[i].velocity[dim]) &&
                    std::abs(after[i].velocity[dim] - expected) <
                        2.0e-6 * std::max(1.0, std::abs(expected)),
                "opening/closing half-kick used the wrong timestep or acceleration");
        }
    }
}

void meshTimestepStep(TimeType clock, int expectedRung, bool firstCall = false) {
    auto const before = meshTimestepSamples();
    FlopCount const beforePrepare = readFlops();
    auto const proposal = particleMeshPrepareKick(1.0, 1.0, 1.0, firstCall, minRung(clock));
    // One radius conversion per locality; each dark-matter particle executes
    // fourteen timestep/rung operations, nine kick operations, and optionally
    // two operations to compute the old half-step. Diagnostics do not count.
    FlopCount const expectedPrepare =
        kCountFlops ? FlopCount(hpxSize()) + before.size() * (firstCall ? 23 : 25) : 0;
    requireMeshTimestep(readFlops() - beforePrepare == expectedPrepare,
        "preparation FLOPs included diagnostics or omitted timestep/kick arithmetic");
    requireMeshTimestep(proposal.maxRung == expectedRung,
        "unexpected global proposal: wanted " + std::to_string(expectedRung) + " got " +
            std::to_string(proposal.maxRung));
    auto const synchronized = meshTimestepSamples();
    checkMeshTimestepKick(before, synchronized, expectedRung, false, firstCall);
    FlopCount const beforeFinish = readFlops();
    particleMeshFinishKick(1.0, proposal.maxRung);
    FlopCount const expectedFinish = kCountFlops ? 3 * FlopCount(hpxSize()) + 6 * before.size() : 0;
    requireMeshTimestep(readFlops() - beforeFinish == expectedFinish,
        "finalization FLOPs omitted the shared half-step or particle kicks");
    checkMeshTimestepKick(synchronized, meshTimestepSamples(), expectedRung, true, false);
}

void checkMeshApplyFieldFlops() {
    struct KickCase {
        bool hierarchical;
        bool ascending;
        bool descending;
        bool firstCall;
        FlopCount count;
    };
    // A radius conversion costs one operation, a timestep proposal fifteen,
    // hierarchical sign/half-step setup three, and each vector half-kick nine.
    // Nonhierarchical kicks compute their two-operation half-steps separately.
    std::array<KickCase, 10> const cases{{
        {false, false, false, true, 27},
        {false, false, false, false, 38},
        {true, false, false, true, 4},
        {true, false, false, false, 4},
        {true, true, false, true, 4},
        {true, true, false, false, 13},
        {true, false, true, true, 28},
        {true, false, true, false, 28},
        {true, true, true, true, 28},
        {true, true, true, false, 37},
    }};
    Array<float, kDimensions> acceleration;
    acceleration[kXDimension] = 0.5f;
    acceleration[kYDimension] = -0.75f;
    acceleration[kZDimension] = 1.0f;
    auto const originalRung = particlesRung(0);
    Array<float, kDimensions> originalVelocity;
    for (int dim = 0; dim < kDimensions; ++dim) {
        originalVelocity[dim] = particlesVel(dim, 0);
    }
    FlopCount const initialGlobal = readFlops();
    // Starting above double's exact integer range also checks that accumulation
    // remains integral and does not truncate counts to 32 bits.
    FlopCount const initialBatch = (FlopCount(1) << 54) + 7;
    FlopCount batched = initialBatch;
    FlopCount expectedBatch = initialBatch;
    KickParameters parameters;
    parameters.a = 1.0f;
    parameters.t0 = 1.0f;
    parameters.maxDt = 1.0f;
    parameters.eta = 0.1f;
    parameters.gravitySupportRadius = getOptions().gravity.supportRadius;
    parameters.saveForce = false;
    parameters.minRung = 1;
    parameters.fixedRung = 2;
    for (auto const& sample : cases) {
        parameters.htime = sample.hierarchical;
        parameters.ascending = sample.ascending;
        parameters.descending = sample.descending;
        parameters.firstCall = sample.firstCall;
        for (bool top : {false, true}) {
            parameters.top = top;
            for (bool doPhi : {false, true}) {
                parameters.doPhi = doPhi;
                particlesRung(0) = 2;
                particleMeshApplyField(0, acceleration, -2.0f, parameters, &batched);
                expectedBatch += kCountFlops ? sample.count : 0;
                requireMeshTimestep(batched == expectedBatch,
                    "field kick FLOPs do not match the executed ladder branches");
            }
        }
    }
    requireMeshTimestep(
        readFlops() == initialGlobal, "batched field kick submitted its FLOPs more than once");
    parameters.minRung = 3;
    particlesRung(0) = 2;
    particleMeshApplyField(0, acceleration, -2.0f, parameters, &batched);
    requireMeshTimestep(batched == expectedBatch, "inactive field kick counted arithmetic");
    parameters.minRung = 1;
    parameters.htime = false;
    parameters.firstCall = false;
    particleMeshApplyField(0, acceleration, -2.0f, parameters);
    requireMeshTimestep(readFlops() - initialGlobal == (kCountFlops ? 38 : 0),
        "standalone field kick did not submit its own FLOPs");
    particlesRung(0) = originalRung;
    for (int dim = 0; dim < kDimensions; ++dim) {
        particlesVel(dim, 0) = originalVelocity[dim];
    }
}

}    // namespace detail

void particleMeshTimestepTest() {
    auto options = getOptions();
    options.gravity.method = gravity::kBuildGravityMethod;
    options.gravity.meshSize = 8;
    options.gravity.pmmmNearWidth = 2;
    options.gravity.pmmmNearMethod = gravity::kBuildPmmmNearMethod;
    options.gravity.gm = 1.0;
    options.compute.cuda = false;
    options.hydro.enabled = false;
    options.chemistry.enabled = false;
    options.analysis.saveForce = false;
    options.analysis.groups = false;
    options.analysis.tracers = false;
    options.initial.createGlass = false;
    options.initial.darkMatterMass = 1.0;
    options.initial.numberOfParticles = 3;
    setOptions(options);
    std::vector<hpx::future<void>> initialization;
    for (auto const& locality : hpxLocalities()) {
        initialization.push_back(hpx::async<InitializeMeshTimestepFixtureAction>(locality));
    }
    for (auto& future : initialization) {
        future.get();
    }
    pmmmSolve(true);
    auto const samples = detail::meshTimestepSamples();
    double maximumAcceleration = 0.0;
    double rootAcceleration = 0.0;
    for (auto const& sample : samples) {
        double magnitude = 0.0;
        for (float component : sample.acceleration) {
            magnitude += double(component) * component;
        }
        magnitude = std::sqrt(magnitude);
        maximumAcceleration = std::max(maximumAcceleration, magnitude);
        if (sample.rank == 0) {
            rootAcceleration = std::max(rootAcceleration, magnitude);
        }
    }
    detail::requireMeshTimestep(maximumAcceleration > 1.0e-3,
        "fixture does not provide a measurable gravitational timestep bound");
    detail::requireMeshTimestep(hpxSize() == 1 || rootAcceleration < maximumAcceleration * 0.01,
        "remote locality must provide the finest timestep proposal");
    // A bound of 1/12 lies safely inside rung four, away from roundoff at a
    // dyadic boundary. Reducing gm by 256 then makes gravity permit rung one.
    options.gravity.timestepEta = std::sqrt(maximumAcceleration / pmmmEffectiveRadius()) / 12.0;
    setOptions(options);
    detail::meshTimestepStep(0, 4, true);
    TimeType clock = inc(0, 4);
    options.gravity.gm /= 256.0;
    setOptions(options);
    pmmmSolve(true);
    detail::requireMeshTimestep(minRung(clock) == 4, "unexpected initial integer clock floor");
    detail::meshTimestepStep(clock, 4);
    clock = inc(clock, 4);
    detail::requireMeshTimestep(minRung(clock) == 3, "missing coarsening synchronization point");
    detail::meshTimestepStep(clock, 3);
    clock = inc(clock, 3);
    detail::meshTimestepStep(clock, 2);
    clock = inc(clock, 2);
    options.gravity.gm *= 256.0;
    setOptions(options);
    pmmmSolve(true);
    detail::meshTimestepStep(clock, 4);
    options.gravity.gm /= 256.0;
    setOptions(options);
    pmmmSolve(true);
    // Even at a fully synchronized clock, the old rung limits step growth to
    // a factor of two; it must not jump straight from rung four to rung one.
    detail::meshTimestepStep(0, 3);
    detail::checkMeshApplyFieldFlops();
    pmmmDestroy();
    PRINT("PMMM common-rung refinement, synchronization, coarsening and staged kicks "
          "passed on %i locality/localities\n",
        hpxSize());
}
