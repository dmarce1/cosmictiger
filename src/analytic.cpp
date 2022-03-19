/*
CosmicTiger - A cosmological N-Body code
Copyright (C) 2021  Dominic C. Marcello

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

#include <cosmictiger/analytic.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/particles.hpp>
#include <cosmictiger/timer.hpp>

using return_type = std::pair<vector<double>, array<vector<double>, NDIM>>;

static return_type do_analytic(const vector<fixed32>& sinkx, const vector<fixed32>& sinky, const vector<fixed32>& sinkz);

HPX_PLAIN_ACTION(do_analytic);

using return_type = std::pair<vector<double>, array<vector<double>, NDIM>>;

static return_type do_analytic(const vector<fixed32>& sinkx, const vector<fixed32>& sinky, const vector<fixed32>& sinkz) {
#ifdef USE_CUDA
	vector<hpx::future<return_type>> futs;
	for (auto c : hpx_children()) {
		futs.push_back(hpx::async<do_analytic_action>(HPX_PRIORITY_HI, c, sinkx, sinky, sinkz));
	}

	auto results = gravity_analytic_call_kernel(sinkx, sinky, sinkz);

	for (auto& f : futs) {
		auto other = f.get();
		for (int i = 0; i < sinkx.size(); i++) {
			results.first[i] += other.first[i];
			for (int dim = 0; dim < NDIM; dim++) {
				results.second[dim][i] += other.second[dim][i];
			}
		}
	}
	return results;
#else
	return return_type();
#endif
}

void analytic_compare(int Nsamples) {
#ifdef USE_CUDA
	timer tm;
	tm.start();
	auto samples = particles_sample(Nsamples);
	tm.stop();
	PRINT( "particles_sample: %e\n", tm.read());
	vector<fixed32> sinkx(Nsamples);
	vector<fixed32> sinky(Nsamples);
	vector<fixed32> sinkz(Nsamples);
	for (int i = 0; i < Nsamples; i++) {
		sinkx[i] = samples[i].x[XDIM];
		sinky[i] = samples[i].x[YDIM];
		sinkz[i] = samples[i].x[ZDIM];
	}
	const double gm = get_options().GM;
	auto results = do_analytic(sinkx, sinky, sinkz);
	double lmax_phi = 0.0;
	double lmax_force = 0.0;
	double lerr_phi = 0.0;
	double lerr_force = 0.0;
	double phi_norm = 0.0;
	double force_norm = 0.0;
	for (int i = 0; i < Nsamples; i++) {
		double f1 = 0.0, f2 = 0.0;
		double g1 = 0.0, g2 = 0.0;
		for (int dim = 0; dim < NDIM; dim++) {
			g1 += sqr(samples[i].g[dim]);
			g2 += sqr(gm * results.second[dim][i]);
		}
		g1 = sqrt(g1);
		g2 = sqrt(g2);
		f1 = samples[i].p;
		f2 = gm * results.first[i];
		phi_norm += fabs(f1);
		force_norm += fabs(g1);
		const double gerr = fabs(g1-g2);
		const double ferr = fabs(f1-f2);
		lerr_phi += (ferr);
		lerr_force += gerr;
		lmax_phi = std::max(lmax_phi, ferr);
		lmax_force = std::max(lmax_force, gerr);
		printf("%.10e %.10e %.10e | %.10e %.10e %.10e |%.10e %.10e %.10e \n", sinkx[i].to_float(), sinky[i].to_float(), sinkz[i].to_float(), g1, g2, g2 / g1, f1, f2, f1/f2);
	}
	lerr_force =(lerr_force/force_norm);
	lerr_phi = (lerr_phi/phi_norm);
	PRINT("Force RMS Error     = %e\n", lerr_force);
	PRINT("Force Max Error     = %e\n", lmax_force);
	PRINT("Potential RMS Error = %e\n", lerr_phi);
	PRINT("Potential Max Error = %e\n", lmax_phi);
#else
	PRINT( "analytic compare not available without CUDA\n");
#endif
}
