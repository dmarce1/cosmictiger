/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

// Opt-in complete periodic gravity-walk benchmark, injected unchanged into
// baseline and Cartesian builds. This is not a CTest or an initial-condition run.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <hpx/runtime_local/runtime_local_fwd.hpp>
#include <hpx/threading_base/thread_helpers.hpp>
#include <hpx/threading_base/thread_pool_base.hpp>

#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/memused.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/gravity/dehnen/distributed_tree.hpp>
#include <cosmictiger/gravity/dehnen/execution.hpp>
#include <cosmictiger/gravity/fmm_order.hpp>
#include <cosmictiger/gravity/fmm_types.hpp>
#include <cosmictiger/gravity/kick.hpp>
#include <cosmictiger/gravity/kick_workspace.hpp>
#include <cosmictiger/gravity/tree.hpp>
#include <cosmictiger/particles/particles.hpp>
#include <tests/gravity/analytic.hpp>

namespace detail {

using std::size_t;

bool cartesianWalkIsFinite(float value) {
    static_assert(sizeof(float) == sizeof(uint32_t) && std::numeric_limits<float>::is_iec559);
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    // Floating-point classification can be optimized away under -ffast-math.
    return (bits & 0x7f800000u) != 0x7f800000u;
}

struct CartesianWalkChecksums {
    uint64_t positions = 0;
    std::array<double, NDIM> acceleration{};
    double accelerationSquared = 0;
    double potential = 0;
};

long cartesianWalkEnvironmentInteger(char const* name, long fallback, long minimum, long maximum) {
    char const* text = std::getenv(name);
    if (text == nullptr) {
        return fallback;
    }
    char* end = nullptr;
    long const value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value < minimum || value > maximum) {
        throw std::invalid_argument(std::string(name) + " must be between " +
            std::to_string(minimum) + " and " + std::to_string(maximum));
    }
    return value;
}

int cartesianWalkSamples() {
    return int(cartesianWalkEnvironmentInteger("COSMICTIGER_CARTESIAN_WALK_SAMPLES", 5, 1, 100));
}

// Tree leaf bucket size; the default matches the original fixed workload.
int cartesianWalkBucketSize() {
    return int(cartesianWalkEnvironmentInteger("COSMICTIGER_CARTESIAN_WALK_BUCKET", 64, 1, 8192));
}

// Particles compared with the direct Ewald reference after the last
// potential-enabled sample; zero disables the comparison, and counts above the
// particle number compare every particle.
int cartesianWalkReferenceSamples() {
    return int(
        cartesianWalkEnvironmentInteger("COSMICTIGER_CARTESIAN_WALK_REFERENCE", 0, 0, 1 << 24));
}

// Particles per dimension of the fixtures: initial.particlesPerDimension (32 in
// the original fixed workload, 32,768 particles).
int cartesianWalkSide() {
    int const side = getOptions().initial.particlesPerDimension;
    if (side < 2) {
        throw std::invalid_argument("Cartesian walk benchmark requires "
                                    "initial.particlesPerDimension of at least 2");
    }
    return side;
}

bool cartesianWalkFixtureSelected(char const* fixture) {
    char const* selection = std::getenv("COSMICTIGER_CARTESIAN_WALK_FIXTURE");
    if (selection == nullptr || std::string(selection) == "all") {
        return true;
    }
    std::string const selected(selection);
    if (selected != "lattice" && selected != "clustered" && selected != "adaptive") {
        throw std::invalid_argument("COSMICTIGER_CARTESIAN_WALK_FIXTURE must "
                                    "be all, lattice, clustered, or adaptive");
    }
    return selected == fixture;
}

double cartesianWalkJitter(size_t particle, int dimension) {
    uint32_t value = uint32_t(particle + 1) * 2654435761u + uint32_t(dimension + 1) * 2246822519u;
    value ^= value >> 13;
    value *= 3266489917u;
    value ^= value >> 16;
    return (double(value & 0xffffu) / 65535. - 0.5) * 0.2;
}

std::vector<Array<Position, NDIM>> cartesianWalkPositions(std::string const& fixture, int side) {
    bool const clustered = fixture == "clustered";
    bool const adaptive = fixture == "adaptive";
    std::array<double, NDIM> const clusterCenter{0.41, 0.47, 0.53};
    size_t const count = size_t(side) * side * side;
    double const halfSide = 0.5 * side;
    std::vector<Array<Position, NDIM>> positions(count);
    for (int z = 0; z < side; ++z) {
        for (int y = 0; y < side; ++y) {
            for (int x = 0; x < side; ++x) {
                size_t const index = (size_t(z) * side + y) * side + x;
                std::array<int, NDIM> const cell{x, y, z};
                for (int dim = 0; dim < NDIM; ++dim) {
                    double const coordinate = cell[dim] + 0.5 + cartesianWalkJitter(index, dim);
                    if (clustered) {
                        int const half = 2 * cell[dim] / side;
                        double const local = (coordinate - half * halfSide) / halfSide - 0.5;
                        positions[index][dim] = 0.25 + 0.5 * half + 0.24 * local;
                    } else {
                        double const basePosition = coordinate / side;
                        // Odd multiplication permutes all indices of a power-of-two
                        // count (it is the original mask for 32,768): exactly three
                        // quarters enter the dense cube without choosing a spatial
                        // slab; other counts place about three quarters there.
                        bool const denseParticle =
                            (uint32_t(index) * 2654435761u) % uint32_t(count) < 3 * count / 4;
                        positions[index][dim] = adaptive && denseParticle ?
                            clusterCenter[dim] + 0.18 * (basePosition - 0.5) :
                            basePosition;
                    }
                }
            }
        }
    }
    return positions;
}

void cartesianWalkReset(std::vector<Array<Position, NDIM>> const& positions) {
    for (size_t index = 0; index < positions.size(); ++index) {
        for (int dim = 0; dim < NDIM; ++dim) {
            particlesPos(dim, index) = positions[index][dim];
            particlesVel(dim, index) = 0.0f;
            particlesGravityAcceleration(dim, index) = std::numeric_limits<float>::quiet_NaN();
        }
        particlesPot(index) = std::numeric_limits<float>::quiet_NaN();
        particlesRung(index) = 1;
    }
    particlesSortByRung(0);
}

KickParameters cartesianWalkParameters(bool doPhi, int minLevel) {
    auto const& options = getOptions();
    KickParameters params;
    params.htime = true;
    params.sign = 1.0f;
    params.minLevel = minLevel;
    params.minRung = 0;
    params.a = 1.0f;
    params.t0 = 1.0f;
    params.theta = options.gravity.openingAngle;
    params.gravitySupportRadius = options.gravity.supportRadius;
    params.eta = options.gravity.timestepEta;
    params.gm = 1.0f;
    params.velInit = false;
    params.saveForce = true;
    params.firstCall = true;
    params.gpu = options.compute.cuda;
    params.nodeLoad = 1.0f;
    params.maxDt = 1.0e30f;
    params.ascending = false;
    params.descending = false;
    params.top = true;
    params.doPhi = doPhi;
    return params;
}

CartesianWalkChecksums cartesianWalkChecksums(bool doPhi) {
    CartesianWalkChecksums result;
    for (ParticleInteger index = 0; index < particlesSize(); ++index) {
        uint64_t positionHash = 1469598103934665603ull;
        double weight = 1;
        for (int dim = 0; dim < NDIM; ++dim) {
            positionHash ^= uint32_t(particlesPos(dim, index).raw());
            positionHash *= 1099511628211ull;
            weight += (2 * dim + 1) * particlesPos(dim, index).toDouble();
        }
        // A commutative sum preserves the fingerprint across tree reordering.
        result.positions += positionHash;
        for (int dim = 0; dim < NDIM; ++dim) {
            float const accelerationValue = particlesGravityAcceleration(dim, index);
            if (!cartesianWalkIsFinite(accelerationValue)) {
                throw std::runtime_error("Cartesian walk left a particle force unset or nonfinite");
            }
            double const acceleration = accelerationValue;
            result.acceleration[dim] += weight * acceleration;
            result.accelerationSquared += acceleration * acceleration;
        }
        if (doPhi) {
            float const potentialValue = particlesPot(index);
            if (!cartesianWalkIsFinite(potentialValue)) {
                throw std::runtime_error("Cartesian walk left a particle "
                                         "potential unset or nonfinite");
            }
            double const potential = potentialValue;
            result.potential += weight * potential;
        }
    }
    return result;
}

void cartesianWalkSaveForces(std::ofstream& stream, char const* fixture) {
    for (ParticleInteger index = 0; index < particlesSize(); ++index) {
        stream << fixture;
        for (int dim = 0; dim < NDIM; ++dim) {
            stream << ',' << uint32_t(particlesPos(dim, index).raw());
        }
        for (int dim = 0; dim < NDIM; ++dim) {
            stream << ',' << particlesPos(dim, index).toDouble();
        }
        for (int dim = 0; dim < NDIM; ++dim) {
            stream << ',' << particlesGravityAcceleration(dim, index);
        }
        stream << ',' << particlesPot(index) << '\n';
    }
    stream.flush();
    if (!stream) {
        throw std::runtime_error("Failed to write cartesian_walk_forces.csv");
    }
}

double cartesianWalkMedian(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

void cartesianWalkLeafHistogram(TreeId root, char const* fixture) {
    std::map<std::pair<int, ParticleInteger>, size_t> histogram;
    std::vector<TreeId> pending{root};
    size_t occupiedParticles = 0;
    while (!pending.empty()) {
        TreeId const id = pending.back();
        pending.pop_back();
        auto const* node = treeGetNode(id);
        ParticleInteger const count = node->numberOfParticles();
        if (node->leaf) {
            if (count > 0) {
                ++histogram[{int(node->depth), count}];
                occupiedParticles += size_t(count);
            }
        } else {
            for (int child = 0; child < NCHILD; ++child) {
                pending.push_back(node->children[child]);
            }
        }
    }
    if (occupiedParticles != size_t(particlesSize())) {
        throw std::runtime_error("Cartesian walk leaf histogram lost particles");
    }
    for (auto const& entry : histogram) {
        PRINT("CARTESIAN_WALK_LEAF_HISTOGRAM,%s,%i,%zu,%zu\n", fixture, entry.first.first,
            size_t(entry.first.second), entry.second);
    }
}

// Compare the saved forces and potentials of a deterministic particle sample
// with the direct periodic (Ewald) sum. The kick used gm = 1, as does the sum.
void cartesianWalkReferenceErrors(char const* fixture, char const* backend, int count,
    std::vector<Array<Position, NDIM>> const& positions, std::ofstream& stream) {
    size_t const sampleCount = std::min(size_t(count), positions.size());
    // Select physical particles before tree ordering. A coprime permutation
    // avoids the coordinate planes selected by a simple lattice-index stride.
    size_t step = 2654435761u % positions.size();
    while (std::gcd(step, positions.size()) != 1) {
        ++step;
    }
    Vector<Position> sinkx;
    Vector<Position> sinky;
    Vector<Position> sinkz;
    std::map<std::array<uint32_t, NDIM>, size_t> selected;
    for (size_t i = 0; i < sampleCount; ++i) {
        auto const& position = positions[(i * step) % positions.size()];
        std::array<uint32_t, NDIM> key;
        for (int dim = 0; dim < NDIM; ++dim) {
            key[dim] = uint32_t(position[dim].raw());
        }
        if (!selected.emplace(key, i).second) {
            throw std::runtime_error("Cartesian reference sample contains duplicate coordinates");
        }
        sinkx.push_back(position[XDIM]);
        sinky.push_back(position[YDIM]);
        sinkz.push_back(position[ZDIM]);
    }
    std::vector<ParticleInteger> indices(sampleCount, particlesSize());
    for (ParticleInteger index = 0; index < particlesSize(); ++index) {
        std::array<uint32_t, NDIM> key;
        for (int dim = 0; dim < NDIM; ++dim) {
            key[dim] = uint32_t(particlesPos(dim, index).raw());
        }
        auto const found = selected.find(key);
        if (found != selected.end()) {
            if (indices[found->second] != particlesSize()) {
                throw std::runtime_error(
                    "Cartesian reference coordinate matches multiple particles");
            }
            indices[found->second] = index;
        }
    }
    if (std::find(indices.begin(), indices.end(), particlesSize()) != indices.end()) {
        throw std::runtime_error("Cartesian reference sample lost a physical particle");
    }
    auto const reference = gravityAnalyticCallKernel(sinkx, sinky, sinkz);
    std::vector<double> forceErrors;
    std::vector<double> potentialErrors;
    double forceDifference = 0;
    double forceNorm = 0;
    double potentialDifference = 0;
    double potentialNorm = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        stream << fixture;
        for (int dim = 0; dim < NDIM; ++dim) {
            stream << ',' << uint32_t(particlesPos(dim, indices[i]).raw());
        }
        for (int dim = 0; dim < NDIM; ++dim) {
            stream << ',' << particlesGravityAcceleration(dim, indices[i]);
        }
        stream << ',' << particlesPot(indices[i]);
        for (int dim = 0; dim < NDIM; ++dim) {
            stream << ',' << reference.second[dim][i];
        }
        stream << ',' << reference.first[i] << '\n';
        double difference = 0;
        double norm = 0;
        for (int dim = 0; dim < NDIM; ++dim) {
            double const expected = reference.second[dim][i];
            double const actual = particlesGravityAcceleration(dim, indices[i]);
            difference += (actual - expected) * (actual - expected);
            norm += expected * expected;
        }
        double const expectedPotential = reference.first[i];
        double const actualPotential = particlesPot(indices[i]);
        forceErrors.push_back(std::sqrt(difference / std::max(norm, 1e-300)));
        potentialErrors.push_back(std::abs(actualPotential - expectedPotential) /
            std::max(std::abs(expectedPotential), 1e-300));
        forceDifference += difference;
        forceNorm += norm;
        potentialDifference +=
            (actualPotential - expectedPotential) * (actualPotential - expectedPotential);
        potentialNorm += expectedPotential * expectedPotential;
    }
    stream.flush();
    if (!stream) {
        throw std::runtime_error("Failed to write cartesian_walk_reference.csv");
    }
    std::sort(forceErrors.begin(), forceErrors.end());
    std::sort(potentialErrors.begin(), potentialErrors.end());
    size_t const p99 = std::min(forceErrors.size() - 1, 99 * forceErrors.size() / 100);
    PRINT("CARTESIAN_WALK_REFERENCE,%s,%s,%zu,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%."
          "9g,%.9g\n",
        fixture, backend, indices.size(), std::sqrt(forceDifference / forceNorm),
        forceErrors[forceErrors.size() / 2], forceErrors[p99], forceErrors.back(),
        std::sqrt(potentialDifference / potentialNorm), potentialErrors[potentialErrors.size() / 2],
        potentialErrors[p99], potentialErrors.back());
}

void cartesianWalkFixture(char const* fixture, bool doPhi, int samples, std::ofstream& forceOutput,
    std::ofstream& referenceOutput) {
    auto const positions = cartesianWalkPositions(fixture, cartesianWalkSide());
    char const* backend = getOptions().compute.cuda ? "cuda" : "cpu";
    std::vector<double> setupTimes, treeTimes, kickTimes, totalTimes;
    uint64_t expectedPositionHash = 0;
    for (int sample = -1; sample < samples; ++sample) {
        bool treeReady = false;
        try {
            auto const setupBegin = std::chrono::steady_clock::now();
            cartesianWalkReset(positions);
            auto const setupEnd = std::chrono::steady_clock::now();
            TreeCreateParameters treeParams(
                0, getOptions().gravity.openingAngle, getOptions().gravity.supportRadius);
            auto const treeBegin = std::chrono::steady_clock::now();
            auto const built = treeCreate(treeParams);
            auto const treeEnd = std::chrono::steady_clock::now();
            treeReady = true;
            if (built.id.proc != 0 || built.id.index != 0) {
                throw std::runtime_error(
                    "Cartesian walk benchmark requires the complete local root");
            }
            // Inspect only the force-only warmup, after tree timing and before kick
            // timing. No histogram traversal or printing occurs in measured samples.
            if (sample == -1 && !doPhi) {
                cartesianWalkLeafHistogram(built.id, fixture);
            }
            ExpansionBuffer local;
            local.zero();
            auto const params = cartesianWalkParameters(doPhi, treeParams.minLevel);
            auto const kickBegin = std::chrono::steady_clock::now();
            auto const result = kick(params, local, built.pos, built.id, Vector<TreeId>{built.id},
                Vector<TreeId>{built.id}, nullptr);
            // Particle completion can precede trailing empty-node descents. Wait
            // for the pool before timing ends and before a later tree reuses nodes.
            hpx::this_thread::get_pool()->wait();
            auto const kickEnd = std::chrono::steady_clock::now();
            auto const checksums = cartesianWalkChecksums(doPhi);
            if (sample == -1) {
                expectedPositionHash = checksums.positions;
            } else if (checksums.positions != expectedPositionHash) {
                throw std::runtime_error("Cartesian walk benchmark changed its particle workload");
            }
            double const setupSeconds =
                std::chrono::duration<double>(setupEnd - setupBegin).count();
            double const treeSeconds = std::chrono::duration<double>(treeEnd - treeBegin).count();
            double const kickSeconds = std::chrono::duration<double>(kickEnd - kickBegin).count();
            if (sample >= 0) {
                setupTimes.push_back(setupSeconds);
                treeTimes.push_back(treeSeconds);
                kickTimes.push_back(kickSeconds);
                totalTimes.push_back(treeSeconds + kickSeconds);
            }
            PRINT("CARTESIAN_WALK_SAMPLE,%s,%s,%i,%i,%.9g,%.9g,%.9g,%llu,%."
                  "17g,%.17g,%.17g,%.17g,%.17g,%."
                  "17g\n",
                fixture, backend, int(doPhi), sample, setupSeconds, treeSeconds, kickSeconds,
                static_cast<unsigned long long>(checksums.positions), checksums.acceleration[0],
                checksums.acceleration[1], checksums.acceleration[2], checksums.accelerationSquared,
                checksums.potential, doPhi ? result.pot : 0.0);
            if (doPhi && sample == samples - 1) {
                if (forceOutput.is_open()) {
                    cartesianWalkSaveForces(forceOutput, fixture);
                }
                int const referenceSamples = cartesianWalkReferenceSamples();
                // The direct Ewald sum runs on the device and reads the device
                // kernel table, which only a CUDA-enabled run uploads.
                if (referenceSamples > 0 && getOptions().compute.cuda) {
                    cartesianWalkReferenceErrors(
                        fixture, backend, referenceSamples, positions, referenceOutput);
                } else if (referenceSamples > 0) {
                    PRINT("CARTESIAN_WALK_REFERENCE_SKIPPED,%s,%s,cpu_only_"
                          "run\n",
                        fixture, backend);
                }
            }
            treeDestroy();
            treeReady = false;
            kickClearBuffers();
        } catch (...) {
            if (treeReady) {
                treeDestroy();
            }
            kickClearBuffers();
            throw;
        }
    }
    PRINT("CARTESIAN_WALK_MEDIAN,%s,%s,%i,%i,%zu,%.9g,%.9g,%.9g,%.9g,%llu\n", fixture, backend,
        int(doPhi), samples, positions.size(), cartesianWalkMedian(setupTimes),
        cartesianWalkMedian(treeTimes), cartesianWalkMedian(kickTimes),
        cartesianWalkMedian(totalTimes), static_cast<unsigned long long>(expectedPositionHash));
}

}    // namespace detail

void cartesianWalkBenchmark() {
    // Any worker-thread count (--hpx:threads) is accepted and recorded in
    // CARTESIAN_WALK_CONFIG; the CPU side of the walk scales with it.
    if (hpxSize() != 1) {
        throw std::invalid_argument("Cartesian complete-walk benchmark requires one HPX locality");
    }
    // The benchmark needs pool quiescence between reconstructed trees. Stop the
    // background memory observer so that it cannot keep the pool non-idle.
    stopMemuseDaemon();
    hpx::this_thread::get_pool()->wait();
    auto const originalOptions = getOptions();
    auto options = originalOptions;
    options.hydro.enabled = false;
    options.gravity.enabled = true;
    options.analysis.saveForce = true;
    options.analysis.groups = false;
    options.analysis.tracers = false;
    options.compute.bucketSize = detail::cartesianWalkBucketSize();
    int const side = detail::cartesianWalkSide();
    options.initial.numberOfParticles = size_t(side) * side * side;
    setOptions(options);
    try {
        int const samples = detail::cartesianWalkSamples();
        std::ofstream forces;
        if (detail::cartesianWalkEnvironmentInteger(
                "COSMICTIGER_CARTESIAN_WALK_EXPORT_FORCES", 1, 0, 1)) {
            forces.open("cartesian_walk_forces.csv");
            if (!forces) {
                throw std::runtime_error("Cannot open cartesian_walk_forces.csv");
            }
            forces << "fixture,x_raw,y_raw,z_raw,x,y,z,gx,gy,gz,phi\n" << std::setprecision(17);
        }
        std::ofstream references;
        if (detail::cartesianWalkReferenceSamples() > 0 && options.compute.cuda) {
            references.open("cartesian_walk_reference.csv");
            if (!references) {
                throw std::runtime_error("Cannot open cartesian_walk_reference.csv");
            }
            references << "fixture,x_raw,y_raw,z_raw,gx,gy,gz,phi,reference_gx,reference_gy,"
                          "reference_gz,reference_phi\n"
                       << std::setprecision(17);
        }
        particlesResize(options.initial.numberOfParticles);
        PRINT("CARTESIAN_WALK_CONFIG,particles=%zu,threads=%i,cuda=%i,order=%"
              "i,theta=%.17g,support=%."
              "17g,bucket=%i,samples=%i,warmup=1,gm=1\n",
            options.initial.numberOfParticles, int(hpx::get_num_worker_threads()),
            int(options.compute.cuda), fmmOrder(), options.gravity.openingAngle,
            options.gravity.supportRadius, options.compute.bucketSize, samples);
        PRINT("CARTESIAN_WALK_SAMPLE_HEADER,fixture,backend,do_phi,sample,"
              "setup_seconds,tree_seconds,"
              "kick_seconds,position_hash,gx_checksum,gy_checksum,gz_checksum,g2_checksum,phi_"
              "checksum,total_potential\n");
        PRINT("CARTESIAN_WALK_MEDIAN_HEADER,fixture,backend,do_phi,samples,"
              "particles,setup_seconds,tree_"
              "seconds,kick_seconds,tree_plus_kick_seconds,position_hash\n");
        PRINT("CARTESIAN_WALK_LEAF_HISTOGRAM_HEADER,fixture,depth,particles_"
              "per_leaf,leaves\n");
        PRINT("CARTESIAN_WALK_REFERENCE_HEADER,fixture,backend,samples,force_"
              "relative_l2,force_relative_"
              "p50,force_relative_p99,force_relative_max,potential_relative_l2,potential_relative_"
              "p50,potential_relative_p99,potential_relative_max\n");
        for (char const* fixture : {"lattice", "clustered", "adaptive"}) {
            if (detail::cartesianWalkFixtureSelected(fixture)) {
                for (bool doPhi : {false, true}) {
                    detail::cartesianWalkFixture(fixture, doPhi, samples, forces, references);
                }
            }
        }
        // Retain the allocated particle arena: particlesDestroy currently
        // nulls its pointers without resetting the capacity used by resize.
        // Every subsequent benchmark iteration explicitly restores all data.
        kickClearBuffers();
    } catch (...) {
        setOptions(originalOptions);
        throw;
    }
    setOptions(originalOptions);
}
