/* CosmicTiger - distributed PMMM timestep and staged kick regression; BSL-1.0. */

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/core/time.hpp>
#include <cosmictiger/gravity/pmmm_kick.hpp>
#include <cosmictiger/gravity/softening.hpp>
#include <cosmictiger/gravity/solver.hpp>
#include <cosmictiger/math/vector3.hpp>
#include <cosmictiger/particles/domain.hpp>
#include <cosmictiger/particles/particles.hpp>

namespace detail {

struct MeshTimestepSample {
    Vector3<float> acceleration{};
    Vector3<float> velocity{};
    double mass{};
    double position{};
    int rung{};
    int rank{};

    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & acceleration & velocity & mass & position & rung & rank;
    }
};

void requireMeshTimestep(bool condition, std::string const& message) {
    if (!condition) {
        throw std::runtime_error("PMMM timestep regression: " + message);
    }
}

void initializeMeshTimestepFixture() {
    // The symmetry center has negligible acceleration. Its two endpoints sit one
    // cell away through the periodic seam, so the coarse-aligned domain cuts put
    // the center on a low rank and both force-limiting particles on another
    // locality, testing the global timestep reduction. Every particle starts on
    // the root and is migrated to its spatial owner before the first solve.
    constexpr double kFixtureX[3] = {0.9375, 0.0625, 0.1875};
    particlesResize(hpxRank() == 0 ? 3 : 0);
    for (ParticleInteger local = 0; local < particlesSize(); ++local) {
        for (int dim = 0; dim < kDimensions; ++dim) {
            particlesPos(dim, local) = dim == kXi ? kFixtureX[local] : 0.5;
            particlesVel(dim, local) = 0.0f;
        }
        particlesRung(local) = 1;
    }
}

// PMMM near fields read each cell's particles from its spatial owner.
void distributeMeshTimestepFixture() {
    auto const saved = getOptions();
    auto options = saved;
    options.gravity.enabled = false;
    options.run.hierarchicalGravity = options.gravity.enabled;
    setOptions(options);
    particlesSortByRung(0);
    domainsRebound();
    domainsBegin(0);
    domainsEnd();
    setOptions(saved);
}

Vector<MeshTimestepSample> readMeshTimestepFixture() {
    Vector<MeshTimestepSample> samples(particlesSize());
    for (ParticleInteger i = 0; i < particlesSize(); ++i) {
        auto& sample = samples[i];
        sample.acceleration = pmmmAcceleration(i);
        for (int dim = 0; dim < kDimensions; ++dim) {
            sample.velocity[dim] = particlesVel(dim, i);
        }
        sample.position = particlesPos(kXi, i).toDouble();
        sample.mass = particlesMass(i);
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
    // two operations to compute the old half-step. Six norm promotions and nine
    // kick promotions/narrowings are charged to double precision. Diagnostics do not count.
    FlopCount const expectedPrepare =
        kCountFlops ? FlopCount(hpxSize()) + before.size() * (firstCall ? 38 : 40) : 0;
    requireMeshTimestep(readFlops() - beforePrepare == expectedPrepare,
        "preparation FLOPs included diagnostics or omitted timestep/kick arithmetic");
    requireMeshTimestep(proposal.maxRung == expectedRung,
        "unexpected global proposal: wanted " + std::to_string(expectedRung) + " got " +
            std::to_string(proposal.maxRung));
    auto const synchronized = meshTimestepSamples();
    checkMeshTimestepKick(before, synchronized, expectedRung, false, firstCall);
    FlopCount const beforeFinish = readFlops();
    double const driftKinetic = particleMeshFinishKick(1.0, proposal.maxRung);
    FlopCount const expectedFinish =
        kCountFlops ? 3 * FlopCount(hpxSize()) + 15 * before.size() : 0;
    requireMeshTimestep(readFlops() - beforeFinish == expectedFinish,
        "finalization FLOPs omitted the shared half-step or particle kicks");
    auto const after = meshTimestepSamples();
    checkMeshTimestepKick(synchronized, after, expectedRung, true, false);
    // Compare the distributed result to an independent sum of the stored float
    // momenta after the opening kick, including particles on remote localities.
    double expectedKinetic = 0.0;
    for (auto const& sample : after) {
        double velocitySquared = 0.0;
        for (float velocity : sample.velocity) {
            velocitySquared += double(velocity) * velocity;
        }
        expectedKinetic += 0.5 * sample.mass * velocitySquared;
    }
    requireMeshTimestep(std::isfinite(driftKinetic) && expectedKinetic > 0.0 &&
            std::abs(driftKinetic - expectedKinetic) <=
                16.0 * std::numeric_limits<double>::epsilon() * expectedKinetic,
        "finalization did not return the global mass-weighted drift kinetic energy");
    if (firstCall) {
        requireMeshTimestep(proposal.kin == 0.0 && driftKinetic > proposal.kin,
            "kinetic regression must distinguish synchronized energy from drift energy");
    }
}

void checkMeshApplyFieldFlops() {
    struct KickCase {
        bool hierarchical;
        bool ascending;
        bool descending;
        bool firstCall;
        FlopCount count;
    };
    // The radius expression costs one double operation and two precision
    // conversions; the table below retains arithmetic-only totals so its
    // two additional conversion operations are checked separately.
    // A timestep proposal costs fifteen operations,
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
    Vector3<float> acceleration;
    acceleration[kXi] = 0.5f;
    acceleration[kYi] = -0.75f;
    acceleration[kZi] = 1.0f;
    auto const originalRung = particlesRung(0);
    Vector3<float> originalVelocity;
    for (int dim = 0; dim < kDimensions; ++dim) {
        originalVelocity[dim] = particlesVel(dim, 0);
    }
    FlopCount const initialGlobal = readFlops();
    FlopCounts const initialPrecisions = readFlopCounts();
    // Starting above double's exact integer range also checks that accumulation
    // remains integral and does not truncate counts to 32 bits.
    FlopCount const initialBatch = (FlopCount(1) << 54) + 7;
    FlopCount batched = initialBatch;
    FlopCount expectedBatch = initialBatch;
    FlopCounts batchPrecisions;
    FlopCounts expectedPrecisions;
    KickParameters parameters;
    parameters.a = 1.0f;
    parameters.t0 = 1.0f;
    parameters.maxDt = 1.0f;
    parameters.eta = 0.1f;
    parameters.gravitySmoothingLength = getOptions().gravity.smoothingLength;
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
                particleMeshApplyField(
                    0, acceleration, -2.0f, parameters, &batched, &batchPrecisions);
                expectedBatch += kCountFlops ? sample.count + 2 : 0;
                expectedPrecisions.singlePrecision += kCountFlops ? sample.count - 1 : 0;
                expectedPrecisions.doublePrecision += kCountFlops ? 3 : 0;
                requireMeshTimestep(batched == expectedBatch,
                    "field kick FLOPs do not match the executed ladder branches");
                requireMeshTimestep(
                    batchPrecisions.singlePrecision == expectedPrecisions.singlePrecision &&
                        batchPrecisions.doublePrecision == expectedPrecisions.doublePrecision &&
                        batchPrecisions.unclassified == 0 &&
                        batchPrecisions.total() == batched - initialBatch,
                    "field kick precision counts misplaced mixed-radius conversions");
            }
        }
    }
    requireMeshTimestep(
        readFlops() == initialGlobal, "batched field kick submitted its FLOPs more than once");
    parameters.minRung = 3;
    particlesRung(0) = 2;
    particleMeshApplyField(0, acceleration, -2.0f, parameters, &batched, &batchPrecisions);
    requireMeshTimestep(batched == expectedBatch, "inactive field kick counted arithmetic");
    requireMeshTimestep(batchPrecisions.total() == batched - initialBatch,
        "inactive field kick changed its precision counts");
    parameters.minRung = 1;
    parameters.htime = false;
    parameters.firstCall = false;
    particleMeshApplyField(0, acceleration, -2.0f, parameters);
    requireMeshTimestep(readFlops() - initialGlobal == (kCountFlops ? 40 : 0),
        "standalone field kick did not submit its own FLOPs");
    auto const standalonePrecisions = readFlopCounts();
    requireMeshTimestep(standalonePrecisions.singlePrecision - initialPrecisions.singlePrecision ==
                (kCountFlops ? 37 : 0) &&
            standalonePrecisions.doublePrecision - initialPrecisions.doublePrecision ==
                (kCountFlops ? 3 : 0) &&
            standalonePrecisions.unclassified == initialPrecisions.unclassified,
        "standalone field kick lost its arithmetic precision split");
    particlesRung(0) = originalRung;
    for (int dim = 0; dim < kDimensions; ++dim) {
        particlesVel(dim, 0) = originalVelocity[dim];
    }
}

void checkMeshGlobalTimestepFlops() {
    for (bool firstCall : {true, false}) {
        auto const before = meshTimestepSamples();
        auto const beforePrepare = readFlopCounts();
        particleMeshPrepareKick(1.0, 1.0, 1.0, firstCall, 0, firstCall ? 0.0 : 0.125);
        auto const afterPrepare = readFlopCounts();
        // Twelve acceleration-norm operations/conversions, five timestep
        // operations, one explicit half-step multiply, and eighteen kick
        // operations/conversions. The explicit path does not propose a rung.
        FlopCount const expectedPrepare =
            kCountFlops ? FlopCount(hpxSize()) + 36 * before.size() : 0;
        requireMeshTimestep(
            afterPrepare.doublePrecision - beforePrepare.doublePrecision == expectedPrepare &&
                afterPrepare.singlePrecision == beforePrepare.singlePrecision &&
                afterPrepare.unclassified == beforePrepare.unclassified,
            "explicit preparation counted skipped rung arithmetic or missed conversions");
        auto const beforeFinish = readFlopCounts();
        particleMeshFinishGlobalKick(0.25);
        auto const afterFinish = readFlopCounts();
        // Two shared multiplications and fifteen particle kick
        // operations/conversions; there is no ldexp on this path.
        FlopCount const expectedFinish =
            kCountFlops ? 2 * FlopCount(hpxSize()) + 15 * before.size() : 0;
        requireMeshTimestep(
            afterFinish.doublePrecision - beforeFinish.doublePrecision == expectedFinish &&
                afterFinish.singlePrecision == beforeFinish.singlePrecision &&
                afterFinish.unclassified == beforeFinish.unclassified,
            "explicit finalization counted ldexp or missed kick conversions");
    }
}

}    // namespace detail

void particleMeshTimestepTest() {
    auto options = getOptions();
    options.gravity.meshSize = 8;
    options.gravity.pmmmNearWidth = 2;

    options.gravity.gm = 1.0;
    options.gravity.cuda = false;
    options.hydro.enabled = false;
    options.chemistry.enabled = false;
    options.analysis.saveForce = false;
    options.analysis.groups = false;
    options.analysis.tracers = false;
    options.run.scenario = "cosmology";
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
    detail::distributeMeshTimestepFixture();
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
    options.gravity.timestepEta =
        std::sqrt(maximumAcceleration /
            gravitySofteningEffectiveRadius(getOptions().gravity.smoothingLength)) /
        12.0;
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
    detail::checkMeshGlobalTimestepFlops();
    pmmmDestroy();
    std::cout << std::format("PMMM common-rung refinement, synchronization, coarsening and staged "
                             "kicks passed on {:d} locality/localities\n",
                     hpxSize())
              << std::flush;
}
