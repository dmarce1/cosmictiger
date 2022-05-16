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

static void compute_density();

HPX_PLAIN_ACTION (compute_density);

vector<float> power_spectrum_compute() {
	const int N = get_options().parts_dim;
	fft3d_init(N);
	compute_density();
	fft3d_execute();
	auto power = fft3d_power_spectrum();
	fft3d_destroy();
	return power;
}

static void compute_density() {
	vector<float> rho0;
	vector<hpx::future<void>> futs1;
	vector<hpx::future<void>> futs2;
	for (auto c : hpx_children()) {
		futs1.push_back(hpx::async<compute_density_action>(HPX_PRIORITY_HI, c));
	}
	const int N = get_options().parts_dim;
	const auto dblbox = domains_find_my_box();
	range<int64_t> intbox;
	for (int dim = 0; dim < NDIM; dim++) {
		intbox.begin[dim] = dblbox.begin[dim] * N;
		intbox.end[dim] = dblbox.end[dim] * N + 1;
	}
	vector<std::atomic<float>> rho(intbox.volume());
	for (int i = 0; i < rho.size(); i++) {
		rho[i] = 0.0;
	}
	array<int64_t, NDIM> I;
	for (I[0] = intbox.begin[XDIM]; I[0] < intbox.end[XDIM] - 1; I[0]++) {
		for (I[1] = intbox.begin[YDIM]; I[1] < intbox.end[YDIM] - 1; I[1]++) {
			for (I[2] = intbox.begin[ZDIM]; I[2] < intbox.end[ZDIM] - 1; I[2]++) {
				rho[intbox.index(I)] = -1.0;
			}
		}
	}
	const int nthreads = hpx::thread::hardware_concurrency();
	for (int proc = 0; proc < nthreads; proc++) {
		futs2.push_back(hpx::async([proc,nthreads,N,intbox,&rho]() {
			const part_int begin = (size_t) proc * particles_size() / nthreads;
			const part_int end = (size_t) (proc+1) * particles_size() / nthreads;
			const float Ninv = 1.0 / N;
			for( int i = begin; i < end; i++) {
				const double x = particles_pos(XDIM,i).to_double();
				const double y = particles_pos(YDIM,i).to_double();
				const double z = particles_pos(ZDIM,i).to_double();
				const part_int i0 = std::min(std::max((int64_t) (x * N), intbox.begin[XDIM]), intbox.end[XDIM] - 2);
				const part_int j0 = std::min(std::max((int64_t) (y * N), intbox.begin[YDIM]), intbox.end[YDIM] - 2);
				const part_int k0 = std::min(std::max((int64_t) (z * N), intbox.begin[ZDIM]), intbox.end[ZDIM] - 2);
				const part_int i1 = i0 + 1;
				const part_int j1 = j0 + 1;
				const part_int k1 = k0 + 1;
				const float wx1 = x * N - i0;
				const float wy1 = y * N - j0;
				const float wz1 = z * N - k0;
				const float wx0 = 1.0 - wx1;
				const float wy0 = 1.0 - wy1;
				const float wz0 = 1.0 - wz1;
				const float w000 = wx0 * wy0 * wz0;
				const float w001 = wx0 * wy0 * wz1;
				const float w010 = wx0 * wy1 * wz0;
				const float w011 = wx0 * wy1 * wz1;
				const float w100 = wx1 * wy0 * wz0;
				const float w101 = wx1 * wy0 * wz1;
				const float w110 = wx1 * wy1 * wz0;
				const float w111 = wx1 * wy1 * wz1;
				const part_int i000 = intbox.index(i0,j0,k0);
				const part_int i001 = intbox.index(i0,j0,k1);
				const part_int i010 = intbox.index(i0,j1,k0);
				const part_int i011 = intbox.index(i0,j1,k1);
				const part_int i100 = intbox.index(i1,j0,k0);
				const part_int i101 = intbox.index(i1,j0,k1);
				const part_int i110 = intbox.index(i1,j1,k0);
				const part_int i111 = intbox.index(i1,j1,k1);
				atomic_add(rho[i000], w000);
				atomic_add(rho[i001], w001);
				atomic_add(rho[i010], w010);
				atomic_add(rho[i011], w011);
				atomic_add(rho[i100], w100);
				atomic_add(rho[i101], w101);
				atomic_add(rho[i110], w110);
				atomic_add(rho[i111], w111);
			}
		}));
	}
	hpx::wait_all(futs2.begin(), futs2.end());
	rho0.resize(rho.size());
	hpx_copy(PAR_EXECUTION_POLICY, rho.begin(), rho.end(), rho0.begin()).get();
	fft3d_accumulate_real(intbox, rho0);
	hpx::wait_all(futs1.begin(), futs1.end());
}
