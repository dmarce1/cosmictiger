/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the GNU General Public License, version 2 or later.
 */

// Opt-in complete periodic gravity-walk benchmark, injected unchanged into
// baseline and STF builds. This is not a CTest or an initial-condition run.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <hpx/runtime_local/runtime_local_fwd.hpp>

#include <cosmictiger/hpx.hpp>
#include <cosmictiger/kick.hpp>
#include <cosmictiger/kick_workspace.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/particles.hpp>
#include <cosmictiger/tree.hpp>

namespace detail {

using std::size_t;

constexpr int kStfWalkSide = 32;
constexpr size_t kStfWalkParticles = size_t(kStfWalkSide) * kStfWalkSide * kStfWalkSide;

struct StfWalkChecksums {
    uint64_t positions = 0;
    std::array<double, NDIM> acceleration{};
    double accelerationSquared = 0;
    double potential = 0;
};

int stfWalkSamples() {
    char const* text = std::getenv("COSMICTIGER_STF_WALK_SAMPLES");
    if (text == nullptr) {
        return 5;
    }
    char* end = nullptr;
    long const count = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || count < 1 || count > 100) {
        throw std::invalid_argument("COSMICTIGER_STF_WALK_SAMPLES must be between 1 and 100");
    }
    return int(count);
}

double stfWalkJitter(size_t particle, int dimension) {
    uint32_t value = uint32_t(particle + 1) * 2654435761u + uint32_t(dimension + 1) * 2246822519u;
    value ^= value >> 13;
    value *= 3266489917u;
    value ^= value >> 16;
    return (double(value & 0xffffu) / 65535. - 0.5) * 0.2;
}

std::vector<Array<Fixed32, NDIM>> stfWalkPositions(bool clustered) {
    std::vector<Array<Fixed32, NDIM>> positions(kStfWalkParticles);
    for (int z = 0; z < kStfWalkSide; ++z) {
        for (int y = 0; y < kStfWalkSide; ++y) {
            for (int x = 0; x < kStfWalkSide; ++x) {
                size_t const index = (size_t(z) * kStfWalkSide + y) * kStfWalkSide + x;
                std::array<int, NDIM> const cell{x, y, z};
                for (int dim = 0; dim < NDIM; ++dim) {
                    double const coordinate = cell[dim] + 0.5 + stfWalkJitter(index, dim);
                    if (clustered) {
                        int const half = cell[dim] / (kStfWalkSide / 2);
                        double const local =
                            (coordinate - half * (kStfWalkSide / 2)) / (kStfWalkSide / 2) - 0.5;
                        positions[index][dim] = 0.25 + 0.5 * half + 0.24 * local;
                    } else {
                        positions[index][dim] = coordinate / kStfWalkSide;
                    }
                }
            }
        }
    }
    return positions;
}

void stfWalkReset(std::vector<Array<Fixed32, NDIM>> const& positions) {
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

KickParameters stfWalkParameters(bool doPhi, int minLevel) {
    auto const& options = getOptions();
    KickParameters params;
    params.htime = true;
    params.sign = 1.0f;
    params.minLevel = minLevel;
    params.minRung = 0;
    params.a = 1.0f;
    params.t0 = 1.0f;
    params.theta = options.theta;
    params.gravitySupportRadius = options.gravitySupportRadius;
    params.eta = options.eta;
    params.gm = 1.0f;
    params.velInit = false;
    params.saveForce = true;
    params.firstCall = true;
    params.gpu = options.cuda;
    params.nodeLoad = 1.0f;
    params.maxDt = 1.0e30f;
    params.ascending = false;
    params.descending = false;
    params.top = true;
    params.doPhi = doPhi;
    return params;
}

StfWalkChecksums stfWalkChecksums(bool doPhi) {
    StfWalkChecksums result;
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
            double const acceleration = particlesGravityAcceleration(dim, index);
            if (!std::isfinite(acceleration)) {
                throw std::runtime_error("STF walk left a particle force unset or nonfinite");
            }
            result.acceleration[dim] += weight * acceleration;
            result.accelerationSquared += acceleration * acceleration;
        }
        if (doPhi) {
            double const potential = particlesPot(index);
            if (!std::isfinite(potential)) {
                throw std::runtime_error("STF walk left a particle potential unset or nonfinite");
            }
            result.potential += weight * potential;
        }
    }
    return result;
}

void stfWalkSaveForces(std::ofstream& stream, char const* fixture) {
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
        throw std::runtime_error("Failed to write stf_walk_forces.csv");
    }
}

double stfWalkMedian(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

void stfWalkFixture(bool clustered, bool doPhi, int samples, std::ofstream& forceOutput) {
    auto const positions = stfWalkPositions(clustered);
    char const* fixture = clustered ? "clustered" : "lattice";
    char const* backend = getOptions().cuda ? "cuda" : "cpu";
    std::vector<double> setupTimes, treeTimes, kickTimes, totalTimes;
    uint64_t expectedPositionHash = 0;
    for (int sample = -1; sample < samples; ++sample) {
        bool treeReady = false;
        try {
            auto const setupBegin = std::chrono::steady_clock::now();
            stfWalkReset(positions);
            auto const setupEnd = std::chrono::steady_clock::now();
            TreeCreateParameters treeParams(
                0, getOptions().theta, getOptions().gravitySupportRadius);
            auto const treeBegin = std::chrono::steady_clock::now();
            auto const built = treeCreate(treeParams);
            auto const treeEnd = std::chrono::steady_clock::now();
            treeReady = true;
            if (built.id.proc != 0 || built.id.index != 0) {
                throw std::runtime_error("STF walk benchmark requires the complete local root");
            }
            Expansion<float> local;
            local = 0.0f;
            auto const params = stfWalkParameters(doPhi, treeParams.minLevel);
            auto const kickBegin = std::chrono::steady_clock::now();
            auto const result = kick(params, local, built.pos, built.id, Vector<TreeId>{built.id},
                Vector<TreeId>{built.id}, nullptr);
            auto const kickEnd = std::chrono::steady_clock::now();
            auto const checksums = stfWalkChecksums(doPhi);
            if (sample == -1) {
                expectedPositionHash = checksums.positions;
            } else if (checksums.positions != expectedPositionHash) {
                throw std::runtime_error("STF walk benchmark changed its particle workload");
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
            PRINT("STF_WALK_SAMPLE,%s,%s,%i,%i,%.9g,%.9g,%.9g,%llu,%.17g,%.17g,%.17g,%.17g,%.17g,%."
                  "17g\n",
                fixture, backend, int(doPhi), sample, setupSeconds, treeSeconds, kickSeconds,
                static_cast<unsigned long long>(checksums.positions), checksums.acceleration[0],
                checksums.acceleration[1], checksums.acceleration[2], checksums.accelerationSquared,
                checksums.potential, doPhi ? result.pot : 0.0);
            if (doPhi && sample == samples - 1) {
                stfWalkSaveForces(forceOutput, fixture);
            }
            treeDestroy();
            treeReady = false;
            KickWorkspace::clearBuffers();
        } catch (...) {
            if (treeReady) {
                treeDestroy();
            }
            KickWorkspace::clearBuffers();
            throw;
        }
    }
    PRINT("STF_WALK_MEDIAN,%s,%s,%i,%i,%zu,%.9g,%.9g,%.9g,%.9g,%llu\n", fixture, backend,
        int(doPhi), samples, positions.size(), stfWalkMedian(setupTimes), stfWalkMedian(treeTimes),
        stfWalkMedian(kickTimes), stfWalkMedian(totalTimes),
        static_cast<unsigned long long>(expectedPositionHash));
}

}    // namespace detail

void stfWalkBenchmark() {
    if (hpxSize() != 1 || hpx::get_num_worker_threads() != 2) {
        throw std::invalid_argument(
            "STF complete-walk benchmark requires one HPX locality with two worker threads");
    }
    auto const originalOptions = getOptions();
    auto options = originalOptions;
    options.sph = false;
    options.gravity = true;
    options.saveForce = true;
    options.doGroups = false;
    options.doTracers = false;
    options.bucketSize = 64;
    options.numberOfParticles = detail::kStfWalkParticles;
    setOptions(options);
    try {
        int const samples = detail::stfWalkSamples();
        std::ofstream forces("stf_walk_forces.csv");
        if (!forces) {
            throw std::runtime_error("Cannot open stf_walk_forces.csv");
        }
        forces << "fixture,x_raw,y_raw,z_raw,x,y,z,gx,gy,gz,phi\n" << std::setprecision(17);
        particlesResize(options.numberOfParticles);
        PRINT("STF_WALK_CONFIG,particles=%zu,threads=%i,cuda=%i,theta=%.17g,support=%.17g,bucket=%"
              "i,samples=%i,warmup=1,gm=1\n",
            options.numberOfParticles, int(hpx::get_num_worker_threads()), int(options.cuda),
            options.theta, options.gravitySupportRadius, options.bucketSize, samples);
        PRINT("STF_WALK_SAMPLE_HEADER,fixture,backend,do_phi,sample,setup_seconds,tree_seconds,"
              "kick_seconds,position_hash,gx_checksum,gy_checksum,gz_checksum,g2_checksum,phi_"
              "checksum,total_potential\n");
        PRINT("STF_WALK_MEDIAN_HEADER,fixture,backend,do_phi,samples,particles,setup_seconds,tree_"
              "seconds,kick_seconds,tree_plus_kick_seconds,position_hash\n");
        for (bool clustered : {false, true}) {
            for (bool doPhi : {false, true}) {
                detail::stfWalkFixture(clustered, doPhi, samples, forces);
            }
        }
        // Retain the allocated particle arena: particlesDestroy currently
        // nulls its pointers without resetting the capacity used by resize.
        // Every subsequent benchmark iteration explicitly restores all data.
        KickWorkspace::clearBuffers();
    } catch (...) {
        setOptions(originalOptions);
        throw;
    }
    setOptions(originalOptions);
}
