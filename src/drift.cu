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

#define BLOCK_SIZE 32
using list_type = device_vector<device_vector<lc_entry>>;

__managed__ list_type* list_ptr;
__managed__ int next_list_entry;

__global__ void cuda_drift_kernel(fixed32* const __restrict__ x, fixed32* const __restrict__ y, fixed32* const __restrict__ z,
		const array<float, NDIM>* const vels, const char* const rungs, part_int count, char rung, float a, float dt, float tau0, float tau1, float tau_max,
		bool do_lc) {
	const auto& tid = threadIdx.x;
	const auto& bid = blockIdx.x;
	auto& list = (*list_ptr)[bid];
	const double one(1.0);
	const double two(2.0);
	const double four(4.0);
	const double zero(0.0);
	const double c0(1.0 / tau_max);
	const double t0(tau0);
	const auto& nblocks = gridDim.x;
	const part_int begin = (size_t) bid * count / nblocks;
	const part_int end = (size_t)(bid + 1) * count / nblocks;
	const part_int end0 = round_up(end - begin, BLOCK_SIZE) + begin;
	const float c1 = dt / a;
	for (part_int i = begin + tid; i < end0; i += BLOCK_SIZE) {
		int this_rung;
		double x0, y0, z0, x1, y1, z1;
		array<double, NDIM> X0;
		array<double, NDIM> X1;
		double vx, vy, vz;
		const float* vel;
		if (i < end) {
			this_rung = rungs[i];
			if (this_rung == rung) {
				vel = vels[i].data();
				x0 = x[i].to_double();
				y0 = y[i].to_double();
				z0 = z[i].to_double();
				x1 = x0 + double((float) (vel[XDIM] * c1));
				y1 = y0 + double((float) (vel[YDIM] * c1));
				z1 = z0 + double((float) (vel[ZDIM] * c1));
			}
		}
		if (do_lc) {
			ALWAYS_ASSERT(false);
			for (int xi = -1; xi <= 0; xi++) {
				if (i < end && this_rung == rung) {
					X0[XDIM] = x0;
					X1[XDIM] = x1;
					if (xi == -1) {
						X0[XDIM] -= one;
						X1[XDIM] -= one;
					}
				}
				for (int yi = -1; yi <= 0; yi++) {
					if (i < end && this_rung == rung) {
						X0[YDIM] = y0;
						X1[YDIM] = y1;
						if (yi == -1) {
							X0[YDIM] -= one;
							X1[YDIM] -= one;
						}
					}
					for (int zi = -1; zi <= 0; zi++) {
						bool test = false;
						lc_entry entry;
						if (i < end && this_rung == rung) {
							X0[ZDIM] = z0;
							X1[ZDIM] = z1;
							if (zi == -1) {
								X0[ZDIM] -= one;
								X1[ZDIM] -= one;
							}
							const double dist0 = sqrt(sqr(X0[0], X0[1], X0[2]));
							const double dist1 = sqrt(sqr(X1[0], X1[1], X1[2]));
							const double tau0_ = double(tau0) + dist0;
							const double tau1_ = double(tau1) + dist1;
							const int i0 = (double) (tau0_ * c0);
							const int i1 = (double) (tau1_ * c0);
							if (dist1 <= one || dist0 <= one) {
								if (i0 != i1) {
									vx = (vel[XDIM]);
									vy = (vel[YDIM]);
									vz = (vel[ZDIM]);
									const double& x0 = X0[XDIM];
									const double& y0 = X0[YDIM];
									const double& z0 = X0[ZDIM];
									const double ti = double((double) (i0 + 1)) * double(tau_max);
									const double sqrtauimtau0 = sqr(ti - t0);
									const double tau0mtaui = t0 - ti;
									const double u2 = sqr(vx, vy, vz);                                    // 5
									const double x2 = sqr(x0, y0, z0);                                       // 5
									const double udotx = vx * x0 + vy * y0 + vz * z0;               // 5
									const double A = one - u2;                                                     // 1
									const double B = two * (tau0mtaui - udotx);                                    // 2
									const double C = sqrtauimtau0 - x2;                                            // 1
									const double t = -(B + sqrt(B * B - four * A * C)) / (two * A);                // 15
									const double x1 = x0 + vx * t;                                            // 2
									const double y1 = y0 + vy * t;                                            // 2
									const double z1 = z0 + vz * t;                                            // 2
									if (sqr(x1, y1, z1) <= one) {                                                 // 6
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
			if (x1 >= one) {
				x1 -= one;
			} else if (x1 < zero) {
				x1 += one;
			}
			if (y1 >= one) {
				y1 -= one;
			} else if (y1 < zero) {
				y1 += one;
			}
			if (z1 >= one) {
				z1 -= one;
			} else if (z1 < zero) {
				z1 += one;
			}
			x[i] = x1;
			y[i] = y1;
			z[i] = z1;
		}
	}
}

static __global__ void list_init(int nvecs) {
	new (list_ptr) list_type;
	list_ptr->resize(nvecs);
	for (int i = 0; i < nvecs; i++) {
		new (list_ptr->data() + i) device_vector<lc_entry>;
	}

}

static __global__ void list_free() {
	const int nvecs = list_ptr->size();
	for (int i = 0; i < nvecs; i++) {
		(list_ptr->data() + i)->~device_vector<lc_entry>();
	}
	list_ptr->~list_type();
}

void cuda_drift(char rung, float a, float dt, float tau0, float tau1, float tau_max) {
	int nblocks;
	timer tm;
	tm.start();
	CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nblocks, (const void*) cuda_drift_kernel, BLOCK_SIZE, 0));
	cudaFuncAttributes attr;
	CUDA_CHECK(cudaFuncGetAttributes(&attr, (const void*) cuda_drift_kernel));
	nblocks *= cuda_smp_count();
	const auto rng = particles_current_range();
	nblocks = std::min(nblocks, std::max((rng.second - rng.first) / BLOCK_SIZE, 1));
	auto cnt = rng.second - rng.first;
	const int list_size = round_up(cnt, BLOCK_SIZE) / BLOCK_SIZE;
	CUDA_CHECK(cudaMallocManaged(&list_ptr, sizeof(list_type)));
	next_list_entry = 0;
	list_init<<<1,1>>>(nblocks);
	CUDA_CHECK(cudaDeviceSynchronize());
	fixed32* x = &particles_pos(XDIM, rng.first);
	fixed32* y = &particles_pos(YDIM, rng.first);
	fixed32* z = &particles_pos(ZDIM, rng.first);
	const auto* vels = particles_vel_data() + rng.first;
	const auto* rungs = &particles_rung(rng.first);
	cuda_drift_kernel<<<nblocks,BLOCK_SIZE>>>(x, y, z, vels, rungs, cnt, rung, a, dt, tau0, tau1, tau_max, get_options().do_lc);
	CUDA_CHECK(cudaDeviceSynchronize());
	cnt = 0;
	for (int li = 0; li < list_ptr->size(); li++) {
		const auto& list = (*list_ptr)[li];
		lc_add_parts(list.data(), list.size());
		const auto sz = list.size();
		cnt += sz;
	}
	if (cnt != 0) {
		PRINT("%i flushed\n", cnt);
	}

	static long long total_count = 0;
	total_count += cnt;
	list_free<<<1,1>>>();
	CUDA_CHECK(cudaDeviceSynchronize());
	CUDA_CHECK(cudaFree(list_ptr));
	tm.stop();
}
