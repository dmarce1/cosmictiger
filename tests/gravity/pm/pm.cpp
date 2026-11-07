/* CosmicTiger - scalar PM regression against a direct Fourier sum; BSL-1.0. */

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <format>
#include <iostream>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/gravity/pm/cloud.hpp>
#include <cosmictiger/gravity/pm/geometry.hpp>
#include <cosmictiger/gravity/pm/interval.hpp>
#include <cosmictiger/gravity/pm/solver.hpp>
#include <cosmictiger/particles/particles.hpp>
#if COSMICTIGER_WITH_SPH
#include <cosmictiger/sph/particles.hpp>
#endif

namespace detail {

struct PmSample {
    Vector3<double> position{};
    Vector3<float> acceleration{};
    Vector3<float> longAcceleration{};
    float potential{};
    float mass{};
    int id{};

    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & position & acceleration & longAcceleration & potential & mass & id;
    }
};

void requirePm(bool condition, std::string const& message) {
    if (!condition) {
        throw std::runtime_error("PM regression: " + message);
    }
}

// Fixtures: 0 seven scattered points, 1 one particle, 2 a uniform lattice,
// 4 the seven points with fixed-point boundary positions, 5 the seven points
// translated by one full cell (translation invariance of the mesh force).
void initializePmFixture(int fixture, int side) {
    std::array<std::array<double, kDimensions>, 7> const points = {{{0.01, 0.234, 0.678},
        {0.99, 0.381, 0.177}, {0.501, 0.001, 0.754}, {0.9999990463256836, 0.999, 0.021},
        {0.25, 0.5, 0.75}, {0.123, 0.456, 0.789}, {0.71, 0.23, 0.49}}};
    int const total = fixture == 0 || fixture == 4 || fixture == 5 ? int(points.size()) :
        fixture == 1                                               ? 1 :
        fixture == 2                                               ? side * side * side :
                                                                     0;
    int count = 0;
    int gasCount = 0;
    for (int i = 0; i < total; ++i) {
        if (i % hpxSize() == hpxRank()) {
            ++count;
            gasCount += getOptions().hydro.enabled && fixture != 2 && i % 2 != 0;
        }
    }
    particlesResize(count);
#if COSMICTIGER_WITH_SPH
    if (getOptions().hydro.enabled) {
        sphParticlesResize(gasCount, false);
    }
#endif
    int local = 0;
    int gas = 0;
    for (int i = total - 1; i >= 0; --i) {
        if (i % hpxSize() != hpxRank()) {
            continue;
        }
        for (int dim = 0; dim < kDimensions; ++dim) {
            double coordinate;
            if (fixture == 2) {
                int const divisor = dim == kXi ? side * side : dim == kYi ? side : 1;
                coordinate = double((i / divisor) % side) / side;
            } else {
                coordinate = points[i][dim];
                if (fixture == 5) {
                    coordinate += 1.0 / side;
                    if (coordinate >= 1.0) {
                        coordinate -= 1.0;
                    }
                }
            }
            particlesPos(dim, local) = coordinate;
            if (fixture == 4) {
                unsigned __int128 const period = static_cast<unsigned __int128>(1) << kPositionBits;
                if ((i == 0 && dim == kXi) || (i == 2 && dim == kYi) || (i == 3 && dim == kZi)) {
                    int const numerator = i == 2 ? 2 : 1;
                    auto const firstAtBoundary = (period * numerator + 2) / 3;
                    particlesPos(dim, local)
                        .setInteger(static_cast<PositionInteger>(firstAtBoundary - 1));
                } else if (i == 1 || i == 4) {
                    particlesPos(dim, local)
                        .setInteger(std::numeric_limits<PositionInteger>::max() - (i == 4 ? 1 : 0));
                }
            }
            particlesVel(dim, local) = dim == kXi ? float(i + 1) : 0.0f;
        }
        particlesRung(local) = 2;
#if COSMICTIGER_WITH_SPH
        if (getOptions().hydro.enabled) {
            bool const isGas = fixture != 2 && i % 2 != 0;
            particlesType(local) = isGas ? kSphType : kDarkMatterType;
            particlesCatIndex(local) = isGas ? gas : -1;
            if (isGas) {
                sphParticlesParticleIndex(gas) = local;
                ++gas;
            }
        }
#endif
        ++local;
    }
}

Vector<PmSample> readPmFixture() {
    Vector<PmSample> samples(particlesSize());
    for (ParticleInteger i = 0; i < particlesSize(); ++i) {
        auto& sample = samples[i];
        for (int dim = 0; dim < kDimensions; ++dim) {
            sample.position[dim] = particlesPos(dim, i).toDouble();
            sample.longAcceleration[dim] = particlesLongAcceleration(dim, i);
        }
        sample.acceleration = pmAcceleration(i);
        sample.potential = pmPotential(i);
        sample.mass = particlesMass(i);
        sample.id = int(particlesVel(kXi, i)) - 1;
    }
    return samples;
}

// Independent closed forms at every grid node; the tiny grid permits a direct DFT.
double assignmentWeight(int order, double coordinate, int node, int side) {
    double const gridPosition = coordinate * side;
    if (order == 1) {
        return int(std::floor(gridPosition + 0.5)) % side == node ? 1.0 : 0.0;
    }
    double radius = std::abs(gridPosition - node);
    radius = std::min(radius, side - radius);
    if (order == 2) {
        return std::max(1.0 - radius, 0.0);
    }
    if (order == 3) {
        return radius < 0.5 ? 0.75 - radius * radius :
            radius < 1.5    ? 0.5 * (1.5 - radius) * (1.5 - radius) :
                              0.0;
    }
    return radius < 1.0 ? 2.0 / 3.0 - radius * radius + 0.5 * radius * radius * radius :
        radius < 2.0    ? std::pow(2.0 - radius, 3) / 6.0 :
                          0.0;
}

double verifyPmDirectFourier(
    Vector<PmSample> const& samples, int side, int order, double splitRadius) {
    using Complex = std::complex<double>;
    double constexpr kTwoPi = 2.0 * std::numbers::pi;
    size_t const count = samples.size();
    std::vector<std::array<double, kDimensions>> expectedForce(count);
    std::vector<double> expectedPotential(count);
    std::vector<std::array<std::vector<Complex>, kDimensions>> transformedWeights(count);
    for (size_t particle = 0; particle < count; ++particle) {
        for (int dim = 0; dim < kDimensions; ++dim) {
            auto& modes = transformedWeights[particle][dim];
            modes.resize(side);
            for (int mode = 0; mode < side; ++mode) {
                for (int node = 0; node < side; ++node) {
                    modes[mode] +=
                        assignmentWeight(order, samples[particle].position[dim], node, side) *
                        std::polar(1.0, -kTwoPi * mode * node / side);
                }
            }
        }
    }
    for (int x = 0; x < side; ++x) {
        for (int y = 0; y < side; ++y) {
            for (int z = 0; z < side; ++z) {
                int const indices[kDimensions] = {x, y, z};
                std::array<double, kDimensions> wave{};
                double waveSquared = 0.0;
                double window = 1.0;
                for (int dim = 0; dim < kDimensions; ++dim) {
                    int const mode = indices[dim] <= side / 2 ? indices[dim] : indices[dim] - side;
                    wave[dim] = kTwoPi * mode;
                    waveSquared += wave[dim] * wave[dim];
                    double const argument = std::numbers::pi * mode / side;
                    window *= mode == 0 ? 1.0 : std::pow(std::sin(argument) / argument, order);
                }
                if (waveSquared == 0.0) {
                    continue;
                }
                std::vector<Complex> weights(count);
                Complex source{};
                for (size_t particle = 0; particle < count; ++particle) {
                    weights[particle] = transformedWeights[particle][0][x] *
                        transformedWeights[particle][1][y] * transformedWeights[particle][2][z];
                    source += double(samples[particle].mass) * weights[particle];
                }
                double const green = -4.0 * std::numbers::pi * getOptions().gravity.gm /
                    (waveSquared * window * window) *
                    std::exp(-waveSquared * splitRadius * splitRadius);
                for (size_t particle = 0; particle < count; ++particle) {
                    Complex const potential = green * std::conj(weights[particle]) *
                        (source - double(samples[particle].mass) * weights[particle]);
                    expectedPotential[particle] += potential.real();
                    for (int dim = 0; dim < kDimensions; ++dim) {
                        if (side % 2 != 0 || indices[dim] != side / 2) {
                            expectedForce[particle][dim] += wave[dim] * potential.imag();
                        }
                    }
                }
            }
        }
    }
    double forceScale = 1.0;
    double potentialScale = 1.0;
    for (size_t i = 0; i < count; ++i) {
        potentialScale = std::max(potentialScale, std::abs(expectedPotential[i]));
        for (int dim = 0; dim < kDimensions; ++dim) {
            forceScale = std::max(forceScale, std::abs(expectedForce[i][dim]));
        }
    }
    std::array<double, kDimensions> momentum{};
    double maximumError = 0.0;
    for (size_t i = 0; i < count; ++i) {
        requirePm(std::abs(samples[i].potential - expectedPotential[i]) < 8.0e-5 * potentialScale,
            "self-subtracted potential differs from the direct Fourier sum");
        for (int dim = 0; dim < kDimensions; ++dim) {
            double const error = std::abs(samples[i].acceleration[dim] - expectedForce[i][dim]);
            maximumError = std::max(maximumError, error / forceScale);
            requirePm(error < 8.0e-5 * forceScale,
                "force differs from the direct Fourier sum (windows, Nyquist, screening)");
            requirePm(samples[i].longAcceleration[dim] == samples[i].acceleration[dim],
                "the stored particle long acceleration differs from the solve result");
            momentum[dim] += samples[i].mass * samples[i].acceleration[dim];
        }
    }
    for (int dim = 0; dim < kDimensions; ++dim) {
        requirePm(std::abs(momentum[dim]) < 8.0e-5 * forceScale,
            "unequal-mass particles have a nonzero total force");
    }
    return maximumError;
}

void checkOuterIntervalSelection() {
    double const coarse = 0.25;
    // The acceleration bound wins when it is the smallest; it is not dyadic.
    requirePm(pmSelectOuterInterval(0.07, 1.0, 0.0, coarse, 1.0) == 0.07,
        "acceleration bound did not select the outer interval");
    // Intervals never cross the next coarse boundary and land exactly on it.
    requirePm(pmSelectOuterInterval(0.07, 1.0, 0.2, coarse, 1.0) == 0.25 - 0.2,
        "the outer interval crossed a coarse boundary");
    requirePm(pmSelectOuterInterval(1.0, 1.0, 0.0, coarse, 1.0) == coarse,
        "the outer interval exceeded the coarse interval");
    requirePm(pmSelectOuterInterval(1.0, 0.11, 0.0, coarse, 1.0) == 0.11,
        "the expansion bound was ignored");
    // The end of the run caps the last interval.
    requirePm(pmSelectOuterInterval(1.0, 1.0, 0.9, coarse, 1.0) == 1.0 - 0.9,
        "the end time did not cap the outer interval");
    requirePm(pmSelectOuterInterval(1.0, 1.0, 1.0, coarse, 1.0) == 0.0,
        "an interval was selected at the end time");
}

}    // namespace detail

HPX_PLAIN_ACTION(detail::initializePmFixture, InitializePmFixtureAction);
HPX_PLAIN_ACTION(detail::readPmFixture, ReadPmFixtureAction);

void pmTest() {
    detail::checkOuterIntervalSelection();
    auto options = getOptions();
    options.gravity.gm = 0.125;
    options.analysis.saveForce = false;
    options.analysis.tracers = false;
    options.analysis.groups = false;
    options.initial.darkMatterMass = 0.75;
    options.initial.gasMass = 0.25;
    options.gravity.pmmmNearMethod = "none";
    options.gravity.p3mSplitRadius = 0.0;
    options.gravity.p3mCutoffRadius = 0.0;
    options.gravity.p3mMeshSize = 0;
    std::vector<std::string> methods;
    if (gravity::kBuildHasPm) {
        methods.push_back("pm");
    }
    if (gravity::kBuildHasP3m) {
        methods.push_back("p3m");
    }
    detail::requirePm(!methods.empty(), "no particle-mesh solver is compiled");
    for (auto const& method : methods) {
        for (int side : {5, 6, 24}) {
            for (auto const* cloud : {"ngp", "cic", "tsc", "pcs"}) {
                if (side == 24 && std::string(cloud) != "pcs") {
                    continue;
                }
                pmDestroy();
                options.gravity.method = method;
                options.gravity.meshSize = side;
                options.gravity.cloud = cloud;
                double splitRadius = 0.0;
                if (method == "p3m") {
                    options.gravity.pmmmNearMethod = gravity::kBuildP3mNearMethod;
                    options.gravity.p3mSplitCells = 1.25;
                    options.gravity.p3mCutoff = 4.5;
                    splitRadius = pm::splitRadius(options.gravity);
                    options.gravity.p3mSplitRadius = splitRadius;
                    options.gravity.p3mCutoffRadius = pm::cutoffRadius(options.gravity);
                    options.gravity.p3mMeshSize = pm::shortMeshSize(options.gravity);
                } else {
                    options.gravity.pmmmNearMethod = "none";
                }
                setOptions(options);
                int const order = pm::cloudOrder(options.gravity);
                std::vector<int> const fixtures =
                    side == 24 ? std::vector<int>{0, 4, 5} : std::vector<int>{0, 1, 2};
                Vector<detail::PmSample> reference;
                for (int fixture : fixtures) {
                    std::vector<hpx::future<void>> initialization;
                    for (auto const& locality : hpxLocalities()) {
                        initialization.push_back(
                            hpx::async<InitializePmFixtureAction>(locality, fixture, side));
                    }
                    hpx::wait_all(initialization);
                    for (auto& future : initialization) {
                        future.get();
                    }
                    PmSolveRequest request;
                    request.doPotential = true;
                    request.storeFields = true;
                    request.applyToParticles = true;
                    request.scale = 1.0;
                    request.eta = options.gravity.timestepEta;
                    auto const result = pmSolve(request);
                    Vector<detail::PmSample> samples;
                    std::vector<hpx::future<Vector<detail::PmSample>>> reads;
                    for (auto const& locality : hpxLocalities()) {
                        reads.push_back(hpx::async<ReadPmFixtureAction>(locality));
                    }
                    for (auto& future : reads) {
                        auto local = future.get();
                        samples.insert(samples.end(), local.begin(), local.end());
                    }
                    std::sort(samples.begin(), samples.end(),
                        [](auto const& a, auto const& b) { return a.id < b.id; });
                    double mass = 0.0;
                    double potentialEnergy = 0.0;
                    for (auto const& sample : samples) {
                        mass += sample.mass;
                        potentialEnergy += 0.5 * sample.mass * sample.potential;
                    }
                    detail::requirePm(
                        std::abs(result.sourceMass - mass) < 1.0e-6 * std::max(1.0, mass),
                        "source mass differs from the particle population");
                    detail::requirePm(
                        std::abs(result.depositedMass - mass) < 2.0e-6 * std::max(1.0, mass),
                        "source deposition lost mass");
                    detail::requirePm(std::abs(result.potentialEnergy - potentialEnergy) <
                            1.0e-5 * std::max(1.0, std::abs(potentialEnergy)),
                        "reported long potential energy differs from the particle sum");
                    detail::requirePm(result.minimumTimestep > 0.0 &&
                            std::isfinite(result.minimumTimestep) &&
                            result.minimumTimestep < 1.0e30,
                        "PM timestep bound was not proposed");
                    if (fixture < 2 || fixture == 4) {
                        double const error =
                            detail::verifyPmDirectFourier(samples, side, order, splitRadius);
                        std::cout << std::format("{:s} Fourier oracle n={:d} cloud={:s} rs={:e} "
                                                 "relative force error={:e}\n",
                                         method.c_str(), side, cloud, splitRadius, error)
                                  << std::flush;
                    } else if (fixture == 2) {
                        for (auto const& sample : samples) {
                            for (float acceleration : sample.acceleration) {
                                detail::requirePm(std::abs(acceleration) < 2.0e-4,
                                    "uniform periodic lattice has a nonzero force");
                            }
                        }
                    }
                    if (fixture == 0) {
                        reference = samples;
                    } else if (fixture == 5) {
                        // Translating every particle by one cell translates the
                        // mesh force exactly, up to single-precision storage.
                        double scale = 1.0e-30;
                        for (auto const& sample : reference) {
                            for (float acceleration : sample.acceleration) {
                                scale = std::max(scale, double(std::abs(acceleration)));
                            }
                        }
                        for (size_t i = 0; i < samples.size(); ++i) {
                            for (int dim = 0; dim < kDimensions; ++dim) {
                                detail::requirePm(
                                    std::abs(samples[i].acceleration[dim] -
                                        reference[i].acceleration[dim]) < 4.0e-6 * scale,
                                    "mesh force is not invariant under a whole-cell translation");
                            }
                        }
                    }
                }
            }
        }
    }
    pmDestroy();
    std::cout << "Particle mesh Fourier, screening, mass, self-force, translation and interval "
                 "checks passed\n"
              << std::flush;
}
