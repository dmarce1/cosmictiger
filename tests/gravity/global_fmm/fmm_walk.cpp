/* CosmicTiger - global FMM periodic force regression; BSL-1.0. */
#include <algorithm>
#include <cmath>
#include <format>
#include <iostream>
#include <limits>
#include <stdexcept>

#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/gravity/direct_reference.hpp>
#include <cosmictiger/gravity/global_fmm/distributed_tree.hpp>
#include <cosmictiger/gravity/global_fmm/execution.hpp>
#include <cosmictiger/gravity/tree.hpp>
#include <cosmictiger/math/kernel.hpp>
#include <cosmictiger/particles/domain.hpp>
#include <cosmictiger/particles/particles.hpp>

namespace detail {
constexpr int kForceParticles = 96;

Vector3<Position> forcePosition(int index) {
    // Deterministic, uneven points include both sides of each periodic seam.
    Vector3<Position> result;
    for (int dim = 0; dim < kDimensions; ++dim) {
        unsigned const seed = unsigned(index + 1) * unsigned(173 + dim * 52);
        result[dim] = double((seed * seed + 31 * seed + 7) % 4093) / 4096.0;
    }
    return result;
}
}    // namespace detail

void initializeGlobalForceTest() {
    int const begin = detail::kForceParticles * hpxRank() / hpxSize();
    int const end = detail::kForceParticles * (hpxRank() + 1) / hpxSize();
    particlesResize(end - begin);
    for (int i = begin; i < end; ++i) {
        auto const position = detail::forcePosition(i);
        for (int dim = 0; dim < kDimensions; ++dim) {
            particlesPos(dim, i - begin) = position[dim];
            particlesVel(dim, i - begin) = 0.0f;
        }
        particlesRung(i - begin) = 0;
    }
}
HPX_PLAIN_ACTION(initializeGlobalForceTest, InitializeGlobalForceTestAction);

Array<double, 3> verifyGlobalForceTest() {
    Array<double, 3> sums{};
    for (ParticleInteger i = 0; i < particlesOwnedSize(); ++i) {
        Vector3<double> reference{};
        for (int source = 0; source < detail::kForceParticles; ++source) {
            auto const position = detail::forcePosition(source);
            Vector3<double> displacement;
            for (int dim = 0; dim < kDimensions; ++dim) {
                displacement[dim] = doubleDistance(particlesPos(dim, i), position[dim]);
            }
            reference += detail::directGravityReference(displacement,
                selectedGravityKernel<double>(), getOptions().gravity.smoothingLength);
        }
        double error = 0.0;
        for (int dim = 0; dim < kDimensions; ++dim) {
            double const value = particlesGravityAcceleration(dim, i);
            if (!std::isfinite(value)) {
                throw std::runtime_error("Global FMM produced a nonfinite force");
            }
            error += (value - reference[dim]) * (value - reference[dim]);
        }
        sums[0] += error;
        sums[1] += reference.dot(reference);
        sums[2] = std::max(sums[2], std::sqrt(error / reference.dot(reference)));
        if (!std::isfinite(particlesPot(i))) {
            throw std::runtime_error("Global FMM produced a nonfinite potential");
        }
    }
    return sums;
}
HPX_PLAIN_ACTION(verifyGlobalForceTest, VerifyGlobalForceTestAction);

void fmmWalkTest() {
    auto options = getOptions();
    options.hydro.enabled = false;
    options.analysis.saveForce = true;
    options.analysis.groups = options.analysis.tracers = false;
    options.compute.bucketSize = 8;
    options.initial.numberOfParticles = detail::kForceParticles;
    options.gravity.smoothingLength = .005;
    setOptions(options);
    Vector<hpx::future<void>> initialization;
    for (auto const& locality : hpxLocalities()) {
        initialization.push_back(hpx::async<InitializeGlobalForceTestAction>(locality));
    }
    for (auto& future : initialization) {
        future.get();
    }
    particlesSortByRung(0);
    domainsRebound();
    domainsBegin(0);
    domainsEnd();
    bool passed = true;
    double previousRms = std::numeric_limits<double>::infinity();
    for (float const theta : {.5f, .25f}) {
        auto const built = treeCreate(TreeCreateParameters(0, theta));
        KickParameters parameters;
        parameters.htime = true;
        parameters.theta = theta;
        parameters.minRung = 0;
        parameters.t0 = 1.0f;
        parameters.gm = 1.0f;
        parameters.eta = options.gravity.timestepEta;
        parameters.gravitySmoothingLength = options.gravity.smoothingLength;
        parameters.saveForce = true;
        parameters.firstCall = true;
        parameters.gpu = options.gravity.cuda;
        parameters.nodeLoad = 1.0f;
        parameters.ascending = parameters.descending = false;
        parameters.top = true;
        parameters.doPhi = true;
        ExpansionBuffer local;
        local.zero();
        Vector3<Position> center{};
        Vector<TreeId> checks{built.id};
        kick(parameters, local, center, built.id, checks, checks, nullptr);
        Array<double, 3> sums{};
        for (auto const& locality : hpxLocalities()) {
            auto const part = hpx::async<VerifyGlobalForceTestAction>(locality).get();
            sums[0] += part[0];
            sums[1] += part[1];
            sums[2] = std::max(sums[2], part[2]);
        }
        double const rms = std::sqrt(sums[0] / sums[1]);
        std::cout << std::format("Global FMM theta={} relative RMS={} max={} ({} localities)\n",
                         theta, rms, sums[2], hpxSize())
                  << std::flush;
        // The first omitted multipole scales with theta to the potential degree.
        // Keep a float-roundoff floor and require the tighter walk to converge.
        double const bound = std::max(1.0e-5, std::pow(double(theta), kFmmOrder - 1));
        passed = passed && rms < bound && sums[2] < 10 * bound && rms < .5 * previousRms + 1.0e-5;
        previousRms = rms;
        treeDestroy(true);
        particlesCacheFree();
    }
    if (!passed) {
        throw std::runtime_error("Global FMM disagrees with independent periodic direct force");
    }
}
