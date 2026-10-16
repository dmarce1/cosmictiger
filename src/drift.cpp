/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 This program is free software; you can redistribute it and/or
 modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation; either version 2
 of the License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */

#include <cmath>

#include <cosmictiger/cosmology.hpp>
#include <cosmictiger/drift.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/kernel.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/sph_particles.hpp>

HPX_PLAIN_ACTION(drift, DriftAction);

void drift(
    double scale, double dt, double tau0, double tau1, double tauMax, int rung, double drag) {
    profilerEnter(__FUNCTION__);
    bool const sph = getOptions().sph;
    Vector<hpx::future<void>> futures;
    if (sph && rung < 0) {
        for (auto const& child : hpxChildren()) {
            futures.push_back(
                hpx::async<DriftAction>(child, scale, dt, tau0, tau1, tauMax, rung, drag));
        }
    }
    cudaDrift(rung, scale, dt, tau0, tau1, tauMax, drag);
    particlesMemadviseCpu();
    if (sph) {
        double const adot = getOptions().test.empty() ? scale * cosmosDadt(scale) : 0.0;
        float const hmin = getOptions().hmin;
        float const hmax = getOptions().hmax;
        for (ParticleInteger i = 0; i < sphParticlesSize(); i++) {
            if (rung >= 0 && sphParticlesRung(i) != rung) {
                continue;
            }
            if (sphParticlesIsstar(i)) {
                continue;
            }
            float& h = sphParticlesSmoothLen(i);
            float const divv = sphParticlesDivv(i);
            // Preserve the donor predictor; the next prehydro solve enforces the neighbor
            // constraint.
            double const dloghdt =
                (1.0 / 3.0) * (divv - 3.0 * adot / scale) / (3.0 + dlogsmoothXDlogh(h, hmin, hmax));
            h *= std::exp(dloghdt * dt);
        }
    }
    for (auto& future : futures) {
        future.get();
    }
    profilerExit();
}
