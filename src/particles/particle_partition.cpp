/* CosmicTiger - common particle permutation; BSL-1.0. */

#include <cstring>
#include <type_traits>

#include <cosmictiger/particles/particles.hpp>
#include <cosmictiger/sph/sph_particles.hpp>

namespace detail {

template <class Value>
void swapParticleBytes(Value& first, Value& second) {
    static_assert(std::is_trivially_copyable_v<Value>);
    unsigned char temporary[sizeof(Value)];
    std::memcpy(temporary, &first, sizeof(Value));
    std::memcpy(&first, &second, sizeof(Value));
    std::memcpy(&second, temporary, sizeof(Value));
}

}    // namespace detail

void particlesSwap(ParticleInteger first, ParticleInteger second) {
    if (first == second) {
        return;
    }
    CHECK_PART_BOUNDS(first);
    CHECK_PART_BOUNDS(second);
    for (int dim = 0; dim < NDIM; ++dim) {
        detail::swapParticleBytes(particlesX[dim][first], particlesX[dim][second]);
    }
    detail::swapParticleBytes(particlesV[first], particlesV[second]);
    detail::swapParticleBytes(particlesR[first], particlesR[second]);
    auto const& options = getOptions();
    if (saveParticleAcceleration(options.analysis)) {
        for (int dim = 0; dim < NDIM; ++dim) {
            detail::swapParticleBytes(particlesGravityAccelerations[dim][first],
                particlesGravityAccelerations[dim][second]);
        }
    }
    if (saveParticlePotential(options.analysis)) {
        detail::swapParticleBytes(particlesP[first], particlesP[second]);
    }
    if (options.analysis.groups) {
        detail::swapParticleBytes(particlesLgrp[first], particlesLgrp[second]);
    }
    if (particlesGrp != nullptr) {
        auto const firstGroup = particlesGrp[first].load(std::memory_order_relaxed);
        auto const secondGroup = particlesGrp[second].load(std::memory_order_relaxed);
        particlesGrp[first].store(secondGroup, std::memory_order_relaxed);
        particlesGrp[second].store(firstGroup, std::memory_order_relaxed);
    }
    if (options.analysis.tracers) {
        detail::swapParticleBytes(particlesTr[first], particlesTr[second]);
    }
    if (options.hydro.enabled) {
        detail::swapParticleBytes(particlesSph[first], particlesSph[second]);
        detail::swapParticleBytes(particlesTy[first], particlesTy[second]);
    }
}
