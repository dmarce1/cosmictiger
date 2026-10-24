/* CosmicTiger - distributed PM regression against a direct Fourier sum; BSL-1.0. */

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/gravity/particle_mesh.hpp>
#include <cosmictiger/particles/particles.hpp>
#include <cosmictiger/sph/sph_particles.hpp>

namespace detail {

struct MeshSample {
    Array<double, NDIM> position{};
    Array<float, NDIM> acceleration{};
    float potential{};
    float mass{};
    int id{};

    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & position & acceleration & potential & mass & id;
    }
};

void requireMesh(bool condition, std::string const& message) {
    if (!condition) {
        throw std::runtime_error("Particle mesh regression: " + message);
    }
}

void initializeMeshFixture(int fixture, int side) {
    std::array<std::array<double, NDIM>, 7> const points = {{{0.01, 0.234, 0.678},
        {0.99, 0.381, 0.177}, {0.501, 0.001, 0.754}, {0.9999990463256836, 0.999, 0.021},
        {0.25, 0.5, 0.75}, {0.123, 0.456, 0.789}, {0.71, 0.23, 0.49}}};
    int const total = fixture == 0 || fixture == 4 ? int(points.size()) :
        fixture == 1                               ? 1 :
        fixture == 2                               ? side * side * side :
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
    if (getOptions().hydro.enabled) {
        sphParticlesResize(gasCount, false);
    }
    int local = 0;
    int gas = 0;
    // Reverse insertion order so the solve must actually reorder the fixture.
    for (int i = total - 1; i >= 0; --i) {
        if (i % hpxSize() != hpxRank()) {
            continue;
        }
        for (int dim = 0; dim < NDIM; ++dim) {
            double coordinate;
            if (fixture == 2) {
                int const divisor = dim == XDIM ? side * side : dim == YDIM ? side : 1;
                coordinate = double((i / divisor) % side) / side;
            } else {
                coordinate = points[i][dim];
            }
            particlesPos(dim, local) = coordinate;
            if (fixture == 4) {
                // n=24 places tile edges at 1/3 and 2/3. The exact fixed64
                // predecessor can become an integer after conversion to double
                // and multiplication by n, while sorting still sees the prior
                // cell. PCS must keep its stencil inside that cell's tile halo.
                unsigned __int128 const period = static_cast<unsigned __int128>(1) << kPositionBits;
                if ((i == 0 && dim == XDIM) || (i == 2 && dim == YDIM) || (i == 3 && dim == ZDIM)) {
                    int const numerator = i == 2 ? 2 : 1;
                    auto const firstAtBoundary = (period * numerator + 2) / 3;
                    particlesPos(dim, local)
                        .setInteger(static_cast<PositionInteger>(firstAtBoundary - 1));
                } else if (i == 1 || i == 4) {
                    particlesPos(dim, local)
                        .setInteger(std::numeric_limits<PositionInteger>::max() - (i == 4 ? 1 : 0));
                }
            }
            particlesVel(dim, local) = dim == XDIM ? float(i + 1) : 0.0f;
            particlesGravityAcceleration(dim, local) = 0.0f;
        }
        particlesPot(local) = 0.0f;
        particlesRung(local) = 2;
        if (getOptions().hydro.enabled) {
            bool const isGas = fixture != 2 && i % 2 != 0;
            particlesType(local) = isGas ? SPH_TYPE : DARK_MATTER_TYPE;
            particlesCatIndex(local) = isGas ? gas : NO_INDEX;
            if (isGas) {
                sphParticlesParticleIndex(gas) = local;
                sphParticlesSupportRadius(gas) = 0.1f;
                ++gas;
            }
        }
        ++local;
    }
}

Vector<MeshSample> readMeshFixture() {
    Vector<MeshSample> samples(particlesSize());
    for (ParticleInteger i = 0; i < particlesSize(); ++i) {
        auto& sample = samples[i];
        for (int dim = 0; dim < NDIM; ++dim) {
            sample.position[dim] = particlesPos(dim, i).toDouble();
        }
        sample.acceleration = particleMeshAcceleration(i);
        sample.potential = particleMeshPotential(i);
        sample.mass = particlesMass(i);
        sample.id = int(particlesVel(XDIM, i)) - 1;
        if (getOptions().hydro.enabled && particlesType(i) == SPH_TYPE) {
            requireMesh(sphParticlesParticleIndex(particlesCatIndex(i)) == i,
                "gas reverse index was not repaired after sorting");
        }
    }
    return samples;
}

// Independent closed forms, evaluated at every grid node instead of using the
// production stencil generator. The tiny regression grid permits a direct DFT.
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

void verifyDirectFourier(Vector<MeshSample> const& samples, int side, int order) {
    using Complex = std::complex<double>;
    double constexpr kTwoPi = 2.0 * std::numbers::pi;
    size_t const count = samples.size();
    std::vector<std::array<double, NDIM>> expectedForce(count);
    std::vector<double> expectedPotential(count);
    // Separability reduces this oracle to three independent direct 1D sums.
    std::vector<std::array<std::vector<Complex>, NDIM>> transformedWeights(count);
    for (size_t particle = 0; particle < count; ++particle) {
        for (int dim = 0; dim < NDIM; ++dim) {
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
                int const indices[NDIM] = {x, y, z};
                std::array<double, NDIM> wave{};
                double waveSquared = 0.0;
                double window = 1.0;
                for (int dim = 0; dim < NDIM; ++dim) {
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
                    (waveSquared * window * window);
                for (size_t particle = 0; particle < count; ++particle) {
                    Complex const potential = green * std::conj(weights[particle]) *
                        (source - double(samples[particle].mass) * weights[particle]);
                    expectedPotential[particle] += potential.real();
                    for (int dim = 0; dim < NDIM; ++dim) {
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
        for (int dim = 0; dim < NDIM; ++dim) {
            forceScale = std::max(forceScale, std::abs(expectedForce[i][dim]));
        }
    }
    std::array<double, NDIM> momentum{};
    double maximumError = 0.0;
    for (size_t i = 0; i < count; ++i) {
        requireMesh(std::abs(samples[i].potential - expectedPotential[i]) < 8.0e-5 * potentialScale,
            "self-subtracted potential differs from the direct Fourier sum");
        for (int dim = 0; dim < NDIM; ++dim) {
            double const error = std::abs(samples[i].acceleration[dim] - expectedForce[i][dim]);
            maximumError = std::max(maximumError, error / forceScale);
            requireMesh(error < 8.0e-5 * forceScale,
                "force differs from the direct Fourier sum (including both windows/Nyquist)");
            momentum[dim] += samples[i].mass * samples[i].acceleration[dim];
        }
    }
    for (int dim = 0; dim < NDIM; ++dim) {
        requireMesh(std::abs(momentum[dim]) < 8.0e-5 * forceScale,
            "unequal-mass particles have a nonzero total force");
    }
    PRINT("PM Fourier oracle n=%i order=%i relative force error=%e\n", side, order, maximumError);
}

}    // namespace detail

HPX_PLAIN_ACTION(detail::initializeMeshFixture, InitializeMeshFixtureAction);
HPX_PLAIN_ACTION(detail::readMeshFixture, ReadMeshFixtureAction);

void particleMeshTest() {
    auto options = getOptions();
    options.gravity.method = "pm";
    options.gravity.gm = 0.125;
    options.analysis.saveForce = true;
    options.analysis.tracers = false;
    options.analysis.groups = false;
    options.initial.darkMatterMass = 0.75;
    options.initial.gasMass = 0.25;
    options.run.hierarchicalGravity = false;
    for (int side : {5, 6, 24}) {
        for (auto const* cloud : {"ngp", "cic", "tsc", "pcs"}) {
            if (side == 24 && std::string(cloud) != "pcs") {
                continue;
            }
            particleMeshDestroy();
            options.gravity.meshSize = side;
            options.gravity.cloud = cloud;
            setOptions(options);
            int const order = options.gravity.cloud == "ngp" ? 1 :
                options.gravity.cloud == "cic"               ? 2 :
                options.gravity.cloud == "tsc"               ? 3 :
                                                               4;
            int const firstFixture = side == 24 ? 4 : 0;
            int const lastFixture = side == 24 ? 5 : 4;
            for (int fixture = firstFixture; fixture < lastFixture; ++fixture) {
                std::vector<hpx::future<void>> initialization;
                for (auto const& locality : hpxLocalities()) {
                    initialization.push_back(
                        hpx::async<InitializeMeshFixtureAction>(locality, fixture, side));
                }
                for (auto& future : initialization) {
                    future.get();
                }
                particleMeshSolve(true);
                Vector<detail::MeshSample> samples;
                std::vector<hpx::future<Vector<detail::MeshSample>>> reads;
                for (auto const& locality : hpxLocalities()) {
                    reads.push_back(hpx::async<ReadMeshFixtureAction>(locality));
                }
                for (auto& future : reads) {
                    auto local = future.get();
                    samples.insert(samples.end(), local.begin(), local.end());
                }
                std::sort(samples.begin(), samples.end(),
                    [](auto const& a, auto const& b) { return a.id < b.id; });
                auto const diagnostics = particleMeshDiagnostics();
                double mass = 0.0;
                for (auto const& sample : samples) {
                    mass += sample.mass;
                }
                detail::requireMesh(
                    std::abs(diagnostics.sourceMass - mass) < 1.0e-6 * std::max(1.0, mass),
                    "source mass differs from the particle population");
                detail::requireMesh(
                    std::abs(diagnostics.depositedMass - mass) < 2.0e-6 * std::max(1.0, mass),
                    "source deposition lost mass");
                if (fixture < 2 || fixture == 4) {
                    detail::verifyDirectFourier(samples, side, order);
                } else {
                    for (auto const& sample : samples) {
                        for (float acceleration : sample.acceleration) {
                            detail::requireMesh(std::abs(acceleration) < 2.0e-4,
                                "uniform periodic lattice has a nonzero force");
                        }
                    }
                }
            }
        }
    }
    particleMeshDestroy();
    PRINT("Particle mesh distributed Fourier, mass, self-force and symmetry checks passed\n");
}
