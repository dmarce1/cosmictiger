/* CosmicTiger - P3M split closure, short solvers and periodic references; BSL-1.0. */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/gravity/direct_reference.hpp>
#include <cosmictiger/gravity/kernels/gaussian_screening.hpp>
#if COSMICTIGER_ENABLE_P3M_FMM
#include <cosmictiger/gravity/p3m/fmm/short_fmm.hpp>
#endif
#include <cosmictiger/gravity/p3m/short_pair.hpp>
#include <cosmictiger/gravity/p3m/solver.hpp>
#include <cosmictiger/gravity/pm/geometry.hpp>
#include <cosmictiger/gravity/pm/solver.hpp>
#include <cosmictiger/math/kernel.hpp>
#include <cosmictiger/particles/domain.hpp>
#include <cosmictiger/particles/particles.hpp>

namespace detail {

using cosmictiger::gravity::evaluateGaussianRadial;
using cosmictiger::gravity::GaussianComponent;
using cosmictiger::gravity::GaussianScreening;
using cosmictiger::gravity::p3m::accumulateShortPair;
using cosmictiger::gravity::p3m::makeShortPairParameters;

void requireP3m(bool condition, std::string const& message) {
    if (!condition) {
        throw std::runtime_error("P3M regression: " + message);
    }
}

// Short plus long must reproduce the softened pair law inside the support and
// the Newtonian law outside it, including the finite coincident limit.
void checkSplitClosure() {
    double const support = 0.02;
    double const splitRadius = 0.0125;
    auto const parameters = makeShortPairParameters(splitRadius, 10.0, support);
    GaussianScreening<double> const screening(0.5 / splitRadius);
    auto const& kernel = selectedGravityKernel();
    std::mt19937_64 generator(7);
    std::uniform_real_distribution<double> unit(-1.0, 1.0);
    for (int sample = 0; sample < 400; ++sample) {
        double const radius = sample < 200 ? 0.0001 + 0.0199 * sample / 200.0 :
                                             0.02 + 0.3 * unit(generator) * unit(generator);
        Vector3<double> direction{unit(generator), unit(generator), unit(generator)};
        double norm = 0;
        for (int axis = 0; axis < kDimensions; ++axis) {
            norm += direction[axis] * direction[axis];
        }
        norm = std::sqrt(norm);
        if (norm == 0) {
            continue;
        }
        Vector3<double> d;
        for (int axis = 0; axis < kDimensions; ++axis) {
            d[axis] = direction[axis] / norm * std::abs(radius);
        }
        double const r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        double const r = std::sqrt(r2);
        double const mass = 0.7;
        std::array<double, 4> shortField{};
        double curvature = 0.0;
        accumulateShortPair(
            d[0], d[1], d[2], mass, parameters, true, kernel, shortField, &curvature);
        auto const radial = evaluateGaussianRadial<GaussianComponent::Long, 1>(r2, screening);
        double const longFactor =
            -mass * radial.template coefficient<1, 0>() * radial.inverseRadius();
        double const longPotential = mass * radial.template coefficient<0, 0>();
        double expectedFactor;
        double expectedPotential;
        if (r < support) {
            double const q = r / support;
            expectedFactor =
                -mass * radial_kernel::forceFactor(kernel, q) / (support * support * support);
            expectedPotential = -mass * radial_kernel::potential(kernel, q) / support;
        } else {
            expectedFactor = -mass / (r2 * r);
            expectedPotential = -mass / r;
        }
        for (int axis = 0; axis < kDimensions; ++axis) {
            double const total = shortField[axis + 1] + longFactor * d[axis];
            double const expected = expectedFactor * d[axis];
            requireP3m(
                std::abs(total - expected) <= 1.0e-10 * std::abs(expectedFactor * r) + 1.0e-14,
                "short plus long force does not close on the softened pair law");
        }
        requireP3m(std::abs(shortField[0] + longPotential - expectedPotential) <=
                1.0e-10 * std::abs(expectedPotential),
            "short plus long potential does not close on the softened pair law");
        // The timestep coefficient is the full softened pair's m F(r/H)/H^3.
        requireP3m(std::abs(curvature + expectedFactor) <= 1.0e-12 * std::abs(expectedFactor),
            "short pair curvature is not the softened pair coefficient");
    }
    // Coincident distinct particles: finite potential, zero force.
    std::array<double, 4> coincident{};
    accumulateShortPair(0.0, 0.0, 0.0, 1.0, parameters, true, kernel, coincident);
    double const expectedCoincident = -radial_kernel::potential(kernel, 0.0) / support +
        2.0 * (0.5 / splitRadius) * std::numbers::inv_sqrtpi;
    requireP3m(coincident[1] == 0.0 && coincident[2] == 0.0 && coincident[3] == 0.0 &&
            std::abs(coincident[0] - expectedCoincident) < 1.0e-12 * std::abs(expectedCoincident),
        "coincident pair did not use the softened-minus-long interior");
    // Beyond the cutoff nothing is accumulated.
    auto const truncated = makeShortPairParameters(splitRadius, 0.1, support);
    std::array<double, 4> beyond{};
    double beyondCurvature = 0.0;
    accumulateShortPair(0.2, 0.0, 0.0, 1.0, truncated, true, kernel, beyond, &beyondCurvature);
    requireP3m(beyond[0] == 0.0 && beyond[1] == 0.0 && beyondCurvature == 0.0,
        "pair beyond the cutoff contributed");
}

#if COSMICTIGER_ENABLE_P3M_FMM
// The screened tree walk against direct pair sums: a dense source ball seen
// from sinks at several distances inside the cutoff forces multipole acceptance.
void checkScreenedWalk() {
    using namespace cosmictiger::gravity::p3m;
    fmm::ShortWalkParameters parameters;
    double const splitRadius = 0.04;
    parameters.pair = makeShortPairParameters(splitRadius, 0.18, 0.004);
    parameters.splitRadius = splitRadius;
    parameters.cutoffRadius = 0.18;
    parameters.openingAngle = 0.5;
    parameters.bucketSize = 4;
    parameters.doPotential = true;
    std::mt19937_64 generator(99);
    std::normal_distribution<double> normal(0.0, 1.0);
    std::uniform_real_distribution<double> unit(-1.0, 1.0);
    std::vector<ShortParticle> sources;
    for (int i = 0; i < 256; ++i) {
        ShortParticle particle;
        for (int dim = 0; dim < kDimensions; ++dim) {
            particle.position[dim] = 0.5 + 0.004 * normal(generator);
        }
        particle.mass = 0.5 + 0.5 * std::abs(unit(generator));
        particle.rank = 1;
        particle.index = i;
        sources.push_back(particle);
    }
    std::vector<ShortParticle> sinks;
    for (int i = 0; i < 48; ++i) {
        ShortParticle particle;
        double const distance = 0.012 + 0.15 * (i / 48.0);
        Vector3<double> direction{unit(generator), unit(generator), unit(generator)};
        double norm = std::sqrt(direction[0] * direction[0] + direction[1] * direction[1] +
            direction[2] * direction[2]);
        for (int dim = 0; dim < kDimensions; ++dim) {
            particle.position[dim] = 0.5 + distance * direction[dim] / norm;
        }
        particle.mass = 1.0;
        particle.rank = 0;
        particle.index = i;
        sinks.push_back(particle);
    }
    auto const sourceTree = fmm::buildShortTree(sources, parameters);
    auto const sinkTree = fmm::buildShortTree(sinks, parameters);
    auto const& kernel = selectedGravityKernel();
    std::vector<std::array<double, 4>> direct(sinkTree->particles.size());
    for (size_t i = 0; i < sinkTree->particles.size(); ++i) {
        for (auto const& source : sources) {
            accumulateShortPair(
                sinkTree->particles[i], source, parameters.pair, true, kernel, direct[i]);
        }
    }
    double previousError = 0.0;
    for (double theta : {1.0, 0.7, 0.5, 0.35, 0.25}) {
        parameters.openingAngle = theta;
        std::vector<std::array<double, 4>> fields;
        auto const statistics = fmm::walkShortTree(
            *sinkTree, {{sourceTree.get(), Vector3<double>{}}}, parameters, fields);
        double maximumForceError = 0.0;
        double maximumPotentialError = 0.0;
        for (size_t i = 0; i < sinkTree->particles.size(); ++i) {
            double const scale = std::sqrt(direct[i][1] * direct[i][1] +
                direct[i][2] * direct[i][2] + direct[i][3] * direct[i][3]);
            for (int component = 1; component < 4; ++component) {
                maximumForceError = std::max(maximumForceError,
                    std::abs(fields[i][component] - direct[i][component]) /
                        std::max(scale, 1.0e-30));
            }
            maximumPotentialError = std::max(maximumPotentialError,
                std::abs(fields[i][0] - direct[i][0]) / std::max(std::abs(direct[i][0]), 1.0e-30));
        }
        PRINT("Screened walk theta=%.2f: %llu multipoles, %llu pairs, maximum relative force "
              "error %e, potential error %e\n",
            theta, static_cast<unsigned long long>(statistics.acceptedMultipoles),
            static_cast<unsigned long long>(statistics.directPairs), maximumForceError,
            maximumPotentialError);
        // The acceptance scales as theta^order for the screened kernel: about a
        // percent at theta=0.5 and below 1e-4 at theta=0.35 for degree five.
        if (theta == 0.5) {
            requireP3m(statistics.acceptedMultipoles > 0, "walk fixture accepted no multipoles");
            requireP3m(maximumForceError < 5.0e-2, "screened tree walk error at theta 0.5");
            previousError = maximumForceError;
        }
        if (theta == 0.35) {
            requireP3m(maximumForceError < 1.0e-3 && maximumForceError < previousError,
                "screened tree walk does not converge toward the direct pair sum");
            previousError = maximumForceError;
        }
        if (theta == 0.25) {
            requireP3m(maximumForceError < 1.0e-4 && maximumForceError < previousError,
                "screened tree walk does not converge toward the direct pair sum");
        }
    }
}

#endif

struct P3mSample {
    Vector3<double> position{};
    Vector3<float> longAcceleration{};
    Vector3<float> shortAcceleration{};
    float longPotential{};
    float shortPotential{};
    float mass{};
    int id{};

    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & position & longAcceleration & shortAcceleration & longPotential & shortPotential &
            mass & id;
    }
};

// Fixtures: 0 random uniform, 1 a close softened pair, 2 a clustered set whose
// dense cores make the screened tree walk accept multipole interactions.
int p3mFixtureCount(int fixture) {
    return fixture == 0 ? 96 : (fixture == 1 ? 2 : 4096);
}

void initializeP3mFixture(int fixture) {
    int const total = p3mFixtureCount(fixture);
    std::mt19937_64 generator(1234 + fixture);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::normal_distribution<double> normal(0.0, 1.0);
    std::vector<Vector3<double>> positions(total);
    for (int i = 0; i < total; ++i) {
        for (int dim = 0; dim < kDimensions; ++dim) {
            positions[i][dim] = unit(generator);
        }
        if (fixture == 1 && i == 1) {
            // A close pair deep inside the softening support.
            for (int dim = 0; dim < kDimensions; ++dim) {
                positions[i][dim] = positions[0][dim] + 0.0004 * (dim + 1);
            }
        }
        if (fixture == 2 && i % 4 != 0) {
            // Three quarters of the particles form two Gaussian clumps.
            double const center = i % 2 == 0 ? 0.3 : 0.7;
            for (int dim = 0; dim < kDimensions; ++dim) {
                double value = center + 0.015 * normal(generator);
                value -= std::floor(value);
                positions[i][dim] = value;
            }
        }
    }
    int count = 0;
    for (int i = 0; i < total; ++i) {
        count += i % hpxSize() == hpxRank();
    }
    particlesResize(count);
    int local = 0;
    for (int i = 0; i < total; ++i) {
        if (i % hpxSize() != hpxRank()) {
            continue;
        }
        for (int dim = 0; dim < kDimensions; ++dim) {
            particlesPos(dim, local) = positions[i][dim];
            particlesVel(dim, local) = dim == kXDimension ? float(i + 1) : 0.0f;
        }
        particlesRung(local) = 0;
        ++local;
    }
}

Vector<P3mSample> readP3mFixture() {
    Vector<P3mSample> samples(particlesSize());
    for (ParticleInteger i = 0; i < particlesSize(); ++i) {
        auto& sample = samples[i];
        for (int dim = 0; dim < kDimensions; ++dim) {
            sample.position[dim] = particlesPos(dim, i).toDouble();
            sample.longAcceleration[dim] = particlesLongAcceleration(dim, i);
        }
        sample.shortAcceleration = p3mAcceleration(i);
        sample.shortPotential = p3mPotential(i);
        sample.longPotential = pmPotential(i);
        sample.mass = particlesMass(i);
        sample.id = int(particlesVel(kXDimension, i)) - 1;
    }
    return samples;
}

}    // namespace detail

HPX_PLAIN_ACTION(detail::initializeP3mFixture, InitializeP3mFixtureAction);
HPX_PLAIN_ACTION(detail::readP3mFixture, ReadP3mFixtureAction);

namespace detail {

Vector<P3mSample> solveFixture(int fixture) {
    std::vector<hpx::future<void>> initialization;
    for (auto const& locality : hpxLocalities()) {
        initialization.push_back(hpx::async<InitializeP3mFixtureAction>(locality, fixture));
    }
    hpx::wait_all(initialization);
    for (auto& future : initialization) {
        future.get();
    }
    domainsRebound();
    domainsBegin(0);
    domainsEnd();
    PmSolveRequest request;
    request.doPotential = true;
    request.storeFields = true;
    request.applyToParticles = true;
    pmSolve(request);
    p3mSolve(true);
    Vector<P3mSample> samples;
    std::vector<hpx::future<Vector<P3mSample>>> reads;
    for (auto const& locality : hpxLocalities()) {
        reads.push_back(hpx::async<ReadP3mFixtureAction>(locality));
    }
    for (auto& future : reads) {
        auto local = future.get();
        samples.insert(samples.end(), local.begin(), local.end());
    }
    std::sort(
        samples.begin(), samples.end(), [](auto const& a, auto const& b) { return a.id < b.id; });
    return samples;
}

// Direct periodic softened reference: the production Ewald-based oracle,
// evaluated for every sink of index divisible by the stride.
std::vector<Vector3<double>> directReference(Vector<P3mSample> const& samples, size_t stride) {
    auto const& kernel = selectedGravityKernel();
    double const support = getOptions().gravity.supportRadius;
    double const gm = getOptions().gravity.gm;
    std::vector<Vector3<double>> result(samples.size());
    for (size_t i = 0; i < samples.size(); i += stride) {
        Vector3<double> total{};
        for (size_t j = 0; j < samples.size(); ++j) {
            if (i == j) {
                continue;
            }
            Vector3<double> displacement;
            for (int dim = 0; dim < kDimensions; ++dim) {
                double d = samples[i].position[dim] - samples[j].position[dim];
                d -= std::round(d);
                displacement[dim] = d;
            }
            auto const unit = directGravityReference(displacement, kernel, support, true);
            for (int dim = 0; dim < kDimensions; ++dim) {
                total[dim] += gm * samples[j].mass * unit[dim];
            }
        }
        result[i] = total;
    }
    return result;
}

double compareWithReference(
    Vector<P3mSample> const& samples, char const* label, size_t stride = 1) {
    auto const reference = directReference(samples, stride);
    double scale = 1.0e-30;
    for (size_t i = 0; i < samples.size(); i += stride) {
        for (int dim = 0; dim < kDimensions; ++dim) {
            scale = std::max(scale, std::abs(reference[i][dim]));
        }
    }
    double maximumError = 0.0;
    double rms = 0.0;
    size_t compared = 0;
    double momentumNorm = 1.0e-30;
    Vector3<double> momentum{};
    for (size_t i = 0; i < samples.size(); ++i) {
        for (int dim = 0; dim < kDimensions; ++dim) {
            double const total =
                double(samples[i].longAcceleration[dim]) + samples[i].shortAcceleration[dim];
            momentum[dim] += samples[i].mass * total;
            momentumNorm += std::abs(samples[i].mass * total);
            if (i % stride != 0) {
                continue;
            }
            double const error = std::abs(total - reference[i][dim]);
            maximumError = std::max(maximumError, error / scale);
            rms += error * error;
        }
        compared += i % stride == 0;
    }
    rms = std::sqrt(rms / (3.0 * compared)) / scale;
    PRINT("%s: %zu particles (%zu compared), maximum relative force error %e, rms %e, "
          "momentum/sum|m a| (%e %e %e)\n",
        label, samples.size(), compared, maximumError, rms, momentum[0] / momentumNorm,
        momentum[1] / momentumNorm, momentum[2] / momentumNorm);
    for (int dim = 0; dim < kDimensions; ++dim) {
        requireP3m(std::abs(momentum[dim]) < 1.0e-4 * momentumNorm, "total P3M force is not zero");
    }
    return maximumError;
}

}    // namespace detail

void p3mTest() {
    detail::checkSplitClosure();
#if COSMICTIGER_ENABLE_P3M_FMM
    detail::checkScreenedWalk();
#endif
    auto options = getOptions();
    options.gravity.method = "p3m";
    options.gravity.gm = 0.125;
    options.gravity.meshSize = 32;
    options.gravity.cloud = "pcs";
    options.gravity.p3mSplitCells = 1.25;
    options.gravity.p3mCutoff = 4.5;
    options.gravity.pmmmNearWidth = 2;
    options.gravity.supportRadius = 0.02;
    options.gravity.openingAngle = 0.35;
    options.analysis.saveForce = false;
    options.analysis.tracers = false;
    options.analysis.groups = false;
    options.initial.darkMatterMass = 1.0;
    options.hydro.enabled = false;
    options.gravity.p3mSplitRadius = pm::splitRadius(options.gravity);
    options.gravity.p3mCutoffRadius = pm::cutoffRadius(options.gravity);
    options.gravity.p3mMeshSize = pm::shortMeshSize(options.gravity);
    options.gravity.pmmmCoarseMeshSize = pmmm::detail::resolvedCoarseMeshSize(
        options.gravity.p3mMeshSize, 0, options.gravity.pmmmNearWidth);
    std::vector<std::string> nearMethods;
    if (gravity::kBuildHasP3mPp) {
        nearMethods.push_back("pp");
    }
    if (gravity::kBuildHasP3mFmm) {
        nearMethods.push_back("fmm");
    }
    std::vector<Vector<detail::P3mSample>> results;
    std::vector<Vector<detail::P3mSample>> clustered;
    for (auto const& near : nearMethods) {
        options.gravity.pmmmNearMethod = near;
        options.compute.bucketSize = 32;
        pmDestroy();
        p3mDestroy();
        setOptions(options);
        char const* label = near == "pp" ? "PP-PM" : "FMM-PM";
        // Random periodic distribution against the direct softened Ewald reference.
        auto const samples = detail::solveFixture(0);
        double const error = detail::compareWithReference(samples, label);
        detail::requireP3m(error < 2.0e-2, "P3M total force differs from the periodic reference");
        auto const diagnostics = p3mDiagnostics();
        detail::requireP3m(diagnostics.evaluatedParticles == samples.size(),
            "the short solver did not evaluate every particle");
        results.push_back(samples);
        // A close pair inside the softening support: the softened law must hold.
        auto const pair = detail::solveFixture(1);
        double const pairError =
            detail::compareWithReference(pair, near == "pp" ? "PP-PM pair" : "FMM-PM pair");
        detail::requireP3m(pairError < 2.0e-2, "softened close pair differs from the reference");
        // Clustered distribution with small buckets: the screened walk must
        // accept multipoles yet stay within the reference tolerance on a sample.
        options.compute.bucketSize = 4;
        setOptions(options);
        auto const dense = detail::solveFixture(2);
        double const denseError = detail::compareWithReference(
            dense, near == "pp" ? "PP-PM clustered" : "FMM-PM clustered", 64);
        detail::requireP3m(
            denseError < 2.0e-2, "clustered P3M total force differs from the periodic reference");
        auto const denseDiagnostics = p3mDiagnostics();
        PRINT("%s clustered: %llu direct pairs, %llu accepted multipoles, %llu cutoff rejections, "
              "%llu of %llu batches on CUDA\n",
            label, static_cast<unsigned long long>(denseDiagnostics.directPairs),
            static_cast<unsigned long long>(denseDiagnostics.acceptedMultipoles),
            static_cast<unsigned long long>(denseDiagnostics.cutoffRejections),
            static_cast<unsigned long long>(denseDiagnostics.cudaBatches),
            static_cast<unsigned long long>(denseDiagnostics.batches));
        if (near == "pp" && getOptions().gravity.cuda) {
            detail::requireP3m(denseDiagnostics.cudaBatches > 0,
                "CUDA was requested but no PP-PM batch ran on the device");
        }
        if (near == "fmm") {
            detail::requireP3m(denseDiagnostics.acceptedMultipoles > 0,
                "the screened tree walk accepted no multipole interaction");
        }
        clustered.push_back(dense);
    }
    auto const compareShort = [](Vector<detail::P3mSample> const& first,
                                  Vector<detail::P3mSample> const& second, char const* label) {
        double scale = 1.0e-30;
        for (auto const& sample : first) {
            for (int dim = 0; dim < kDimensions; ++dim) {
                scale = std::max(scale,
                    std::abs(double(sample.longAcceleration[dim]) + sample.shortAcceleration[dim]));
            }
        }
        double maximumError = 0.0;
        double rms = 0.0;
        for (size_t i = 0; i < first.size(); ++i) {
            for (int dim = 0; dim < kDimensions; ++dim) {
                double const error = std::abs(
                    double(first[i].shortAcceleration[dim]) - second[i].shortAcceleration[dim]);
                maximumError = std::max(maximumError, error / scale);
                rms += error * error;
            }
            detail::requireP3m(first[i].longAcceleration[0] == second[i].longAcceleration[0],
                "PP-PM and FMM-PM received different mesh forces");
        }
        rms = std::sqrt(rms / (3.0 * first.size())) / scale;
        PRINT("PP-PM versus FMM-PM short force (%s): maximum relative difference %e, rms %e\n",
            label, maximumError, rms);
        return maximumError;
    };
    if (results.size() == 2) {
        detail::requireP3m(compareShort(results[0], results[1], "uniform") < 5.0e-3,
            "PP-PM and FMM-PM short forces disagree");
        detail::requireP3m(compareShort(clustered[0], clustered[1], "clustered") < 1.0e-2,
            "PP-PM and FMM-PM clustered short forces disagree");
    }
    pmDestroy();
    p3mDestroy();
    PRINT("P3M split closure, short solver and periodic reference checks passed\n");
}
