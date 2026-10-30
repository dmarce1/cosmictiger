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
#include <cosmictiger/particles/particles.hpp>

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

KickReturn kickWalkDispatch(KickParameters params, ExpansionBuffer l,
    Array<Position, kDimensions> pos, TreeId self, Vector<TreeId> direct, Vector<TreeId> periodic,
    std::shared_ptr<KickWorkspace> workspace, KickContext const& context) {
    return fmmDispatch([&](auto orderTag) {
        return kickWalk<decltype(orderTag)::value>(params, std::move(l), pos, self,
            std::move(direct), std::move(periodic), std::move(workspace), context);
    });
}

HPX_PLAIN_ACTION(gravityBeginGlobalKick, GravityBeginGlobalKickAction);
HPX_PLAIN_ACTION(gravityFinishGlobalKick, GravityFinishGlobalKickAction);

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

double gravityFinishGlobalKick(double timestep) {
    ALWAYS_ASSERT(std::isfinite(timestep) && timestep > 0.0);
    Vector<hpx::future<double>> futures;
    for (auto const& child : hpxChildren()) {
        futures.push_back(hpx::async<GravityFinishGlobalKickAction>(child, timestep));
    }
    ALWAYS_ASSERT(detail::globalAccelerationCount == particlesSize());
    auto const& options = getOptions();
    double const halfStep = (options.run.scenario == "glass" ? -0.5 : 0.5) * timestep;
    double kinetic = 0.0;
    FlopCount flops = 1;
    for (ParticleInteger i = 0; i < particlesSize(); ++i) {
        particlesRung(i) = 0;
        double velocitySquared = 0.0;
        for (int dim = 0; dim < kDimensions; ++dim) {
            if (!options.hydro.enabled || particlesType(i) != kSphType) {
                particlesVel(dim, i) =
                    std::fma(detail::globalAccelerations[dim][i], halfStep, particlesVel(dim, i));
                flops += 2;
            }
            double const velocity = particlesVel(dim, i);
            velocitySquared += velocity * velocity;
        }
        kinetic += 0.5 * particlesMass(i) * velocitySquared;
    }
    addCpuFlops(flops);
    hpx::wait_all(futures.begin(), futures.end());
    for (auto& future : futures) {
        kinetic += future.get();
    }
    return kinetic;
}
