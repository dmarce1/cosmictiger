/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <stack>
#include <utility>

#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/gravity/fmm_order.hpp>
#include <cosmictiger/gravity/kick_walk.hpp>
#include <cosmictiger/math/vector3.hpp>
#include <cosmictiger/particles/particles.hpp>
#if COSMICTIGER_WITH_SPH
#include <cosmictiger/sph/time_integration.hpp>
#endif

namespace detail {

static thread_local std::stack<KickLists> localLists;
static Array<float*, kDimensions> globalAccelerations{};
static size_t globalAccelerationCount = 0;

KickLists kickAcquireLists() {
    if (localLists.empty()) {
        localLists.push(KickLists());
    }
    KickLists lists = std::move(localLists.top());
    localLists.pop();
    return std::move(lists);
}

void kickReleaseLists(KickLists&& lists) {
    lists.cclist.resize(0);
    lists.cplist.resize(0);
    lists.pclist.resize(0);
    lists.nextlist.resize(0);
    lists.leaflist.resize(0);
    localLists.push(std::move(lists));
}

bool kickSourcesAreSingleRank(Vector<TreeId> const& checklist) {
    for (auto const& source : checklist) {
        if (!treeGetNode(source)->isLocal()) {
            return false;
        }
    }
    return true;
}

}    // namespace detail

KickReturn kickWalkDispatch(KickParameters params, ExpansionBuffer l, Vector3<Position> pos,
    TreeId self, Vector<TreeId> direct, Vector<TreeId> periodic,
    std::shared_ptr<KickWorkspace> workspace, KickContext const& context) {
    return fmmDispatch([&](auto orderTag) {
        return kickWalk<decltype(orderTag)::value>(params, std::move(l), pos, self,
            std::move(direct), std::move(periodic), std::move(workspace), context);
    });
}

HPX_PLAIN_ACTION(gravityBeginGlobalKick, GravityBeginGlobalKickAction);
HPX_PLAIN_ACTION(gravityFinishGlobalKick, GravityFinishGlobalKickAction);
HPX_PLAIN_ACTION(gravityFinishBalancedGasKick, GravityFinishBalancedGasKickAction);

void gravityBeginGlobalKick() {
    Vector<hpx::future<void>> futures;
    for (auto const& child : hpxChildren()) {
        futures.push_back(hpx::async<GravityBeginGlobalKickAction>(child));
    }
    size_t const count = particlesSize();
    if (count != detail::globalAccelerationCount) {
        std::array<std::unique_ptr<float, decltype(&freeManagedMemory)>, kDimensions> replacement{
            {{nullptr, &freeManagedMemory}, {nullptr, &freeManagedMemory},
                {nullptr, &freeManagedMemory}}};
        try {
            for (int dim = 0; dim < kDimensions; ++dim) {
                replacement[dim].reset(count == 0 ?
                        nullptr :
                        static_cast<float*>(allocateManagedMemory(count * sizeof(float))));
            }
        } catch (...) {
            hpx::wait_all(futures.begin(), futures.end());
            throw;
        }
        for (int dim = 0; dim < kDimensions; ++dim) {
            freeManagedMemory(detail::globalAccelerations[dim]);
            detail::globalAccelerations[dim] = replacement[dim].release();
        }
        detail::globalAccelerationCount = count;
    }
    hpx::wait_all(futures.begin(), futures.end());
    for (auto& future : futures) {
        future.get();
    }
}

float* gravityGlobalAcceleration(int dimension) {
    return detail::globalAccelerations[dimension];
}

KickReturn gravityFinishBalancedGasKick(double baseTimestep, float sign) {
    ALWAYS_ASSERT(std::isfinite(baseTimestep) && baseTimestep > 0.0);
    Vector<hpx::future<KickReturn>> futures;
    for (auto const& child : hpxChildren()) {
        futures.push_back(
            hpx::async<GravityFinishBalancedGasKickAction>(child, baseTimestep, sign));
    }
    ALWAYS_ASSERT(detail::globalAccelerationCount == particlesSize());
    KickReturn result;
#if COSMICTIGER_WITH_SPH
    ALWAYS_ASSERT(sphMidpointEnabled());
    FlopCount flops = 0;
    for (ParticleInteger index = 0; index < particlesSize(); ++index) {
        if (particlesType(index) != kSphType) {
            continue;
        }
        int const rung = particlesRung(index);
        ALWAYS_ASSERT(rung >= 0 && rung < kMaxRung);
        result.maxRung = std::max(result.maxRung, rung);
        float const halfTimestep = sign * 0.5f * rungDt[rung] * float(baseTimestep);
        Vector3<float> before;
        Vector3<float> after;
        for (int dimension = 0; dimension < kDimensions; ++dimension) {
            before[dimension] = particlesVel(dimension, index);
            after[dimension] = std::fma(
                detail::globalAccelerations[dimension][index], halfTimestep, before[dimension]);
            particlesVel(dimension, index) = after[dimension];
        }
        flops += 9;
        double const mass = particlesMass(index);
        double const kineticChange = 0.5 * mass * (after.dot(after) - before.dot(before));
        result.dkin += kineticChange;
        result.darkMatterKineticCorrection -= kineticChange;
        result.darkMatterXMomentumCorrection += mass * (before[kXDimension] - after[kXDimension]);
        result.darkMatterYMomentumCorrection += mass * (before[kYDimension] - after[kYDimension]);
        result.darkMatterZMomentumCorrection += mass * (before[kZDimension] - after[kZDimension]);
        result.darkMatterMomentumNormCorrection += mass * (before.norm() - after.norm());
    }
    addCpuFlops(flops, FlopPrecision::Single);
#endif
    for (int dimension = 0; dimension < kDimensions; ++dimension) {
        freeManagedMemory(detail::globalAccelerations[dimension]);
        detail::globalAccelerations[dimension] = nullptr;
    }
    detail::globalAccelerationCount = 0;
    hpx::wait_all(futures.begin(), futures.end());
    for (auto& future : futures) {
        result += future.get();
    }
    return result;
}

double gravityFinishGlobalKick(double timestep) {
    ALWAYS_ASSERT(std::isfinite(timestep) && timestep > 0.0);
    Vector<hpx::future<double>> futures;
    for (auto const& child : hpxChildren()) {
        futures.push_back(hpx::async<GravityFinishGlobalKickAction>(child, timestep));
    }
    ALWAYS_ASSERT(detail::globalAccelerationCount == particlesSize());
    auto const& options = getOptions();
#if COSMICTIGER_WITH_SPH
    bool const midpointHydro = sphMidpointEnabled();
#else
    bool const midpointHydro = false;
#endif
    double const halfStep = (options.run.scenario == "glass" ? -0.5 : 0.5) * timestep;
    double kinetic = 0.0;
    FlopCount flops = 1;
    for (ParticleInteger i = 0; i < particlesSize(); ++i) {
        particlesRung(i) = 0;
        double velocitySquared = 0.0;
        for (int dim = 0; dim < kDimensions; ++dim) {
            if (!options.hydro.enabled || particlesType(i) != kSphType || midpointHydro) {
                particlesVel(dim, i) =
                    std::fma(detail::globalAccelerations[dim][i], halfStep, particlesVel(dim, i));
                flops += 5;    // Double FMA, two float inputs, and float output conversion.
            }
            double const velocity = particlesVel(dim, i);
            velocitySquared += velocity * velocity;
        }
        kinetic += 0.5 * particlesMass(i) * velocitySquared;
    }
    addCpuFlops(flops, FlopPrecision::Double);
    for (int dimension = 0; dimension < kDimensions; ++dimension) {
        freeManagedMemory(detail::globalAccelerations[dimension]);
        detail::globalAccelerations[dimension] = nullptr;
    }
    detail::globalAccelerationCount = 0;
    hpx::wait_all(futures.begin(), futures.end());
    for (auto& future : futures) {
        kinetic += future.get();
    }
    return kinetic;
}
