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

HPX_PLAIN_ACTION(drift);

void drift(double scale, double dt, double tau0, double tau1, double tau_max, int rung, double drag) {
	profiler_enter(__FUNCTION__);
	bool const sph = get_options().sph;
	vector<hpx::future<void>> futures;
	if (sph && rung < 0) {
		for (auto const& child : hpx_children()) {
			futures.push_back(hpx::async<drift_action>(child, scale, dt, tau0, tau1, tau_max, rung, drag));
		}
	}
	cuda_drift(rung, scale, dt, tau0, tau1, tau_max, drag);
	particles_memadvise_cpu();
	if (sph) {
		double const adot = get_options().test.empty() ? scale * cosmos_dadt(scale) : 0.0;
		float const hmin = get_options().hmin;
		float const hmax = get_options().hmax;
		for (part_int i = 0; i < sph_particles_size(); i++) {
			if (rung >= 0 && sph_particles_rung(i) != rung) {
				continue;
			}
			if (sph_particles_isstar(i)) {
				continue;
			}
			float& h = sph_particles_smooth_len(i);
			float const divv = sph_particles_divv(i);
			// Preserve the donor predictor; the next prehydro solve enforces the neighbor constraint.
			double const dloghdt = (1.0 / 3.0) * (divv - 3.0 * adot / scale) / (3.0 + dlogsmoothX_dlogh(h, hmin, hmax));
			h *= std::exp(dloghdt * dt);
		}
	}
	for (auto& future : futures) {
		future.get();
	}
	profiler_exit();
}
