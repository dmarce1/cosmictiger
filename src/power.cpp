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

#include <cosmictiger/domain.hpp>
#include <cosmictiger/fft.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/particles.hpp>
#include <cosmictiger/power.hpp>
#include <cosmictiger/kernels.hpp>

static void compute_density_folded(int M);

HPX_PLAIN_ACTION (compute_density_folded);

power_spectrum_t power_spectrum_compute(int M) {
	const int N = get_options().Nfour;
	fft3d_init(N, -1.0f);
	compute_density_folded(M);
	fft3d_execute();
	auto power = fft3d_power_spectrum();
	for (int i = 0; i < power.k.size(); i++) {
		power.k[i] *= M;
	}
	fft3d_destroy();
	return power;
}

static void compute_density_folded(int M) {
	vector<float> rho0;
	vector<hpx::future<void>> futs1;
	vector<hpx::future<void>> futs2;
	for (auto c : hpx_children()) {
		futs1.push_back(hpx::async<compute_density_folded_action>(c, M));
	}
	const size_t Ndim = get_options().Nfour;
	const size_t nparts = get_options().nparts;
	const size_t parts_per_rank = nparts / hpx_size();
	const int Np = pow(nparts / parts_per_rank, 1.0 / NDIM) + 0.5;
	for (int j = 0; j < Np; j++) {
		for (int k = 0; k < Np; k++) {
			for (int l = 0; l < Np; l++) {
				range<int64_t> intbox;
				intbox.begin[XDIM] = (size_t) j * Ndim / Np + CLOUD_MIN;
				intbox.begin[YDIM] = (size_t) k * Ndim / Np + CLOUD_MIN;
				intbox.begin[ZDIM] = (size_t) l * Ndim / Np + CLOUD_MIN;
				intbox.end[XDIM] = (size_t)(j + 1) * Ndim / Np + CLOUD_MAX;
				intbox.end[YDIM] = (size_t)(k + 1) * Ndim / Np + CLOUD_MAX;
				intbox.end[ZDIM] = (size_t)(l + 1) * Ndim / Np + CLOUD_MAX;
				auto rho = accumulate_density_cuda(M, Ndim, intbox);
				fft3d_accumulate_real(intbox, rho);
			}
		}
	}
	hpx::wait_all(futs1.begin(), futs1.end());
}

