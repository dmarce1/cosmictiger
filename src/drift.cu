/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2022  Dominic C. Marcello

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

#include <cosmictiger/defs.hpp>
#include <cosmictiger/drift.hpp>
#include <cosmictiger/fixed.hpp>
#include <cosmictiger/particles.hpp>
#include <cosmictiger/cuda_reduce.hpp>
#include <cosmictiger/device_vector.hpp>
#include <cosmictiger/lightcone.hpp>
#include <cosmictiger/timer.hpp>
#include <cosmictiger/healpix.hpp>

#define BLOCK_SIZE 512

using list_type = device_vector<device_vector<lc_entry>>;

__managed__ list_type* list_ptr;

__global__ void cuda_drift_kernel(fixed32* const __restrict__ x, fixed32* const __restrict__ y, fixed32* const __restrict__ z, array<float, NDIM>* const vels,
		const char* const rungs, part_int count, char rung, double a, double dt, double tau0, double tau1, double tau_max, bool do_lc, int nside, double drag) {
	do_lc = do_lc && (tau_max - tau1 < 1.0f);
	const auto& tid = threadIdx.x;
	const auto& bid = blockIdx.x;
	auto& list = (*list_ptr)[bid];
	const double c0(1.0 / tau_max);
	const double t0(tau0);
	const double ainv = 1.0 / a;
	const auto& nblocks = gridDim.x;
	const part_int begin = (size_t) bid * count / nblocks;
	const part_int end = (size_t)(bid + 1) * count / nblocks;
	const part_int end0 = round_up(end - begin, BLOCK_SIZE) + begin;
	const double c1 = dt / a;
	list.resize(0);
	for (part_int i = begin + tid; i < end0; i += BLOCK_SIZE) {
		int this_rung;
		double x0, y0, z0, x1, y1, z1;
		array<double, NDIM> X0;
		array<double, NDIM> X1;
		float vx, vy, vz;
		float* vel;
		if (i < end) {
			this_rung = rungs[i];
			if (this_rung == rung) {
				vel = vels[i].data();
				x0 = x[i].to_double();
				y0 = y[i].to_double();
				z0 = z[i].to_double();
				x1 = x0 + double(vel[XDIM] * c1);
				y1 = y0 + double(vel[YDIM] * c1);
				z1 = z0 + double(vel[ZDIM] * c1);
				if (drag != 0.0) {
					const double c0 = 1.0 / (1.0 + drag * dt);
					for (int dim = 0; dim < NDIM; dim++) {
						vel[dim] *= c0;
					}
				}
			}
		}
		if (do_lc) {
			for (int xi = -1; xi <= 0; xi++) {
				X0[XDIM] = x0 + (double) xi;
				X1[XDIM] = x1 + (double) xi;
				for (int yi = -1; yi <= 0; yi++) {
					X0[YDIM] = y0 + (double) yi;
					X1[YDIM] = y1 + (double) yi;
					for (int zi = -1; zi <= 0; zi++) {
						bool test = false;
						lc_entry entry;
						if (i < end && this_rung == rung) {
							X0[ZDIM] = z0 + (double) zi;
							X1[ZDIM] = z1 + (double) zi;
							const double dist0 = sqrt(sqr(X0[0], X0[1], X0[2]));
							const double dist1 = sqrt(sqr(X1[0], X1[1], X1[2]));
							const double tau0_ = double(tau0) + dist0;
							const double tau1_ = double(tau1) + dist1;
							const int i0 = (double) (tau0_ * c0);
							const int i1 = (double) (tau1_ * c0);
							if (i0 != i1 && ((dist1 <= 1.0 || dist0 <= 1.0))) {
								vx = vel[XDIM] * ainv;
								vy = vel[YDIM] * ainv;
								vz = vel[ZDIM] * ainv;
								const double& x0 = X0[XDIM];
								const double& y0 = X0[YDIM];
								const double& z0 = X0[ZDIM];
								const double ti = double((double) (i0 + 1)) * double(tau_max);
								const double sqrtauimtau0 = sqr(ti - t0);
								const double tau0mtaui = t0 - ti;
								const double A = 1.0 - sqr(vx, vy, vz);                                                     // 1
								const double B = 2.0 * (tau0mtaui - (vx * x0 + vy * y0 + vz * z0));                                    // 2
								const double C = sqrtauimtau0 - sqr(x0, y0, z0);                                            // 1
								const double t = -(B + sqrt(B * B - 4.0 * A * C)) / (2.0 * A);                // 15
								const double x1 = x0 + vx * t;                                            // 2
								const double y1 = y0 + vy * t;                                            // 2
								const double z1 = z0 + vz * t;                                            // 2
								const double R2 = sqr(x1, y1, z1);
								if (R2 <= 1.0) {
									long ipix;
									double X[NDIM] = { x1, y1, z1 };
									vec2pix_nest(nside, X, &ipix);
									entry.pix = ipix;
									entry.x = x1;
									entry.y = y1;
									entry.z = z1;
									entry.vx = vel[XDIM];
									entry.vy = vel[YDIM];
									entry.vz = vel[ZDIM];
									test = true;
								}
							}
						}
						int index = test;
						int total;
						compute_indices < BLOCK_SIZE > (index, total);
						if (total) {
							const auto start = list.size();
							list.resize(start + total);
							if (test) {
								list[start + index] = entry;
							}
						}
					}
				}
			}
		}
		if (i < end && this_rung == rung) {
			x1 -= double(x1 >= 1.0);
			y1 -= double(y1 >= 1.0);
			z1 -= double(z1 >= 1.0);
			x1 += double(x1 < 0.0);
			y1 += double(y1 < 0.0);
			z1 += double(z1 < 0.0);
			x[i] = x1;
			y[i] = y1;
			z[i] = z1;
		}
	}
}

void cuda_drift(char rung, float a, float dt, float tau0, float tau1, float tau_max, double drag) {
	static bool init = false;
	int nblocks;
	timer tm1;
	tm1.start();
	CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nblocks, (const void*) cuda_drift_kernel, BLOCK_SIZE, 0));
	cudaFuncAttributes attr;
	CUDA_CHECK(cudaFuncGetAttributes(&attr, (const void*) cuda_drift_kernel));
	nblocks *= cuda_smp_count();
	const auto rng = particles_current_range();
	nblocks = std::min(nblocks, std::max((rng.second - rng.first) / BLOCK_SIZE, 1));
	auto cnt = rng.second - rng.first;
	const int list_size = round_up(cnt, BLOCK_SIZE) / BLOCK_SIZE;
	if (!init) {
		list_ptr = (list_type*) cuda_malloc(sizeof(list_type));
		new (list_ptr) list_type;
		list_ptr->resize(nblocks);
		for (int i = 0; i < nblocks; i++) {
			new (list_ptr->data() + i) device_vector<lc_entry>;
		}
		init = false;
	}
	fixed32* x = &particles_pos(XDIM, rng.first);
	fixed32* y = &particles_pos(YDIM, rng.first);
	fixed32* z = &particles_pos(ZDIM, rng.first);
	auto* vels = particles_vel_data() + rng.first;
	const auto* rungs = &particles_rung(rng.first);
	drag = get_options().create_glass ? M_PI : 0.0;
	cuda_drift_kernel<<<nblocks,BLOCK_SIZE>>>(x, y, z, vels, rungs, cnt, rung, a, dt, tau0, tau1, tau_max, get_options().do_lc, get_options().lc_map_size, drag);
	CUDA_CHECK(cudaDeviceSynchronize());
	cnt = 0;
	static size_t total_cnt = 0;
	cnt += lc_add_parts(*list_ptr, a, tau0);
	if (cnt != 0) {
		total_cnt += cnt;
		PRINT("%i flushed now %i flushed total\n", cnt, total_cnt);
	}

	tm1.stop();
}
