/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <cmath>

#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/sph/cullen_and_dehnen/viscosity.hpp>
#include <cosmictiger/sph/sph_particles.hpp>

namespace cullen_and_dehnen {

// NOLINTBEGIN(readability-identifier-naming)
// Cullen & Dehnen (2010), equation (A15): immediate increases, exponential decay.
float evolvedAlpha(float alpha, float targetAlpha, float inverseDecayTime, float timeStep) {
    float const α_i = alpha;
    float const α_loc_i = targetAlpha;
    float const Δt = timeStep;
    return α_i < α_loc_i ? α_loc_i : α_loc_i + (α_i - α_loc_i) * std::exp(-Δt * inverseDecayTime);
}
// NOLINTEND(readability-identifier-naming)

}    // namespace cullen_and_dehnen

HPX_PLAIN_ACTION(sphParticlesUpdateCullenDehnen, SphParticlesUpdateCullenDehnenAction);
void sphParticlesUpdateCullenDehnen(int minimumRung, float timeStepScale, float time) {
    if (getOptions().hydro.viscosityEvolution != "cullenAndDehnen") {
        return;
    }
    Vector<hpx::future<void>> pending;
    for (auto const& child : hpxChildren()) {
        pending.push_back(hpx::async<SphParticlesUpdateCullenDehnenAction>(
            child, minimumRung, timeStepScale, time));
    }
    FlopCounter<> flops;
    for (ParticleInteger index = 0; index < sphParticlesSize(); ++index) {
        int const rung = sphParticlesRung(index);
        bool const active = rung >= minimumRung;
        auto& diagnostic = sphParticlesCullenDehnenStates[index];
        if (active || sphParticlesSemiactive(index)) {
            diagnostic.divergence = diagnostic.pendingDivergence;
        }
        if (active) {
            float const timeStep = time > 0.0f ? timeStepScale / (1 << rung) : 0.0f;
            auto& state = sphParticlesViscosityStates[index];
            state.alpha = cullen_and_dehnen::evolvedAlpha(
                state.alpha, diagnostic.targetAlpha, diagnostic.inverseDecayTime, timeStep);
            flops += 7;
        }
    }
    addCpuFlops(flops);
    for (auto& future : pending) {
        future.get();
    }
}
