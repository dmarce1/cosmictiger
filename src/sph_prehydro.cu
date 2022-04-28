/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2021  Dominic C. Marcello

 This program is free software; you can redistribute it and/or
 modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation; either version 2
 of the License, or (at your option) any later version.

 This program is distribufted in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */

#include <cosmictiger/sph_cuda.hpp>

struct prehydro1_record1 {
	fixed32 x;
	fixed32 y;
	fixed32 z;
};

struct prehydro1_workspace {
	device_vector<prehydro1_record1> rec1;
};

__global__ void sph_cuda_prehydro1(sph_run_params params, sph_run_cuda_data data, sph_reduction* reduce) {
	const int tid = threadIdx.x;
	const int block_size = blockDim.x;
	__shared__
	int index;
	__shared__ prehydro1_workspace ws;
	__syncthreads();
	if (tid == 0) {
		index = atomicAdd(&reduce->counter, 1);
	}
	__syncthreads();
	new (&ws) prehydro1_workspace();
	array<fixed32, NDIM> x;
	while (index < data.nselfs) {

		int flops = 0;
		__syncthreads();
		ws.rec1.resize(0);
		const sph_tree_node& self = data.trees[data.selfs[index]];
		for (int ni = self.neighbor_range.first; ni < self.neighbor_range.second; ni++) {
			const sph_tree_node& other = data.trees[data.neighbors[ni]];
			const int maxpi = round_up(other.part_range.second - other.part_range.first, block_size) + other.part_range.first;
			for (int pi = other.part_range.first + tid; pi < maxpi; pi += block_size) {
				bool contains = false;
				int j;
				int total;
				if (pi < other.part_range.second) {
					x[XDIM] = data.x[pi];
					x[YDIM] = data.y[pi];
					x[ZDIM] = data.z[pi];
					contains = (self.outer_box.contains(x));
				}
				j = contains;
				compute_indices < PREHYDRO1_BLOCK_SIZE > (j, total);
				const int offset = ws.rec1.size();
				__syncthreads();
				const int next_size = offset + total;
				ws.rec1.resize(next_size);
				if (contains) {
					const int k = offset + j;
					ws.rec1[k].x = x[XDIM];
					ws.rec1[k].y = x[YDIM];
					ws.rec1[k].z = x[ZDIM];
				}
			}
		}
		float hmin_all = 1e+20;
		float hmax_all = 0.0;
		const float w0 = kernelW(0.f);
		__syncthreads();
		for (int i = self.part_range.first; i < self.part_range.second; i++) {
			__syncthreads();
			const int snki = self.sink_part_range.first - self.part_range.first + i;
			const auto rung = data.rungs_snk[data.dm_index_snk[snki]];
			const bool active = rung >= params.min_rung;
			const auto& converged = data.converged_snk[snki];
			const bool use = active && !converged;
			if (use) {
				x[XDIM] = data.x[i];
				x[YDIM] = data.y[i];
				x[ZDIM] = data.z[i];
				float& h = data.rec2_snk[snki].h;
				float h0 = params.hmin;
				const auto test = [h0,x,block_size,tid,&data](float h) {
					float n = 0.f;
					const float hinv = 1.f / h;
					for (int j = tid; j < ws.rec1.size(); j += block_size) {
						const float dx = distance(x[XDIM], ws.rec1[j].x); // 2
						const float dy = distance(x[YDIM], ws.rec1[j].y);// 2
						const float dz = distance(x[ZDIM], ws.rec1[j].z);// 2
						const float r2 = sqr(dx, dy, dz);// 2
						const float r = sqrt(r2);// 4
						const float q = r * hinv;// 1
						if (q < 1.f) {                               // 1
							const float w = kernelW(q);// 4
							n+= w;// 1
						}
					}
					shared_reduce_add<float, PREHYDRO1_BLOCK_SIZE>(n);
					const float h3 = sqr(h)*h;
					n *= (4.0f * float(M_PI) / 3.0f) * (1.f - (h0*sqr(h0))/(h*sqr(h)));
					return data.N - n;
				};
				float hmax = 1.0f;
				for (int dim = 0; dim < NDIM; dim++) {
					hmax = fminf(hmax, fabs(distance(self.outer_box.begin[dim], x[dim])));
					hmax = fminf(hmax, fabs(distance(self.outer_box.end[dim], x[dim])));
				}
				float hmin = h0;
				const float hmax0 = hmax;
				const float hmin0 = hmin;
				float test_max = test(hmax);
				float hmid;
				int iters = 0;
				if (hmax > hmin) {
					do {
						hmid = 0.5f * hmax + 0.5f * hmin;
						const float test_mid = test(hmid);
						if (test_mid * test_max < 0.f) {
							hmin = hmid;
						} else {
							hmax = hmid;
							test_max = test_mid;
						}
						iters++;
					} while (hmax > 1.001f * hmin);
				}
				if (hmax <= hmin || hmin == hmin0 || hmax == hmax0) {
					if (tid == 0) {
						atomicAdd(&reduce->flag, 1);
						h = hmax0;
					}
				} else {
					if (tid == 0) {
						h = hmid;
						hmin_all = fminf(hmin_all, h);
						hmax_all = fmaxf(hmax_all, h);
					}
				}
			}
		}
		shared_reduce_add<int, PREHYDRO1_BLOCK_SIZE>(flops);
		if (tid == 0) {
			atomicMax(&reduce->hmax, hmax_all);
			atomicMin(&reduce->hmin, hmin_all);
			atomicAdd(&reduce->flops, (double) flops);
			index = atomicAdd(&reduce->counter, 1);
		}
		flops = 0;
		__syncthreads();
	}
	(&ws)->~prehydro1_workspace();
}

struct prehydro2_record1 {
	fixed32 x;
	fixed32 y;
	fixed32 z;
	float h;
	char star;
};

struct prehydro2_record2 {
	float vx;
	float vy;
	float vz;
	float entr;
};

struct prehydro2_workspace {
	device_vector<prehydro2_record1> rec1;
	device_vector<prehydro2_record2> rec2;
	device_vector<int> neighbors;
};

__global__ void sph_cuda_prehydro2(sph_run_params params, sph_run_cuda_data data, sph_reduction* reduce) {
	const int tid = threadIdx.x;
	const int block_size = blockDim.x;
	__shared__
	int index;
	__shared__ prehydro2_workspace ws;
	__syncthreads();
	if (tid == 0) {
		index = atomicAdd(&reduce->counter, 1);
	}
	__syncthreads();
	new (&ws) prehydro2_workspace();
	array<fixed32, NDIM> x;
	while (index < data.nselfs) {
		int flops = 0;
		__syncthreads();
		ws.rec1.resize(0);
		ws.rec2.resize(0);
		const sph_tree_node& self = data.trees[data.selfs[index]];
		for (int ni = self.neighbor_range.first; ni < self.neighbor_range.second; ni++) {
			const sph_tree_node& other = data.trees[data.neighbors[ni]];
			const int maxpi = round_up(other.part_range.second - other.part_range.first, block_size) + other.part_range.first;
			for (int pi = other.part_range.first + tid; pi < maxpi; pi += block_size) {
				bool contains = false;
				int j;
				int total;
				if (pi < other.part_range.second) {
					x[XDIM] = data.x[pi];
					x[YDIM] = data.y[pi];
					x[ZDIM] = data.z[pi];
					contains = (self.outer_box.contains(x));
					if (!contains) {
						contains = true;
						const float& h = data.h[pi];
						for (int dim = 0; dim < NDIM; dim++) {
							if (distance(x[dim], self.inner_box.begin[dim]) + 1.01f * h < 0.f) {
								flops += 3;
								contains = false;
								break;
							}
							if (distance(self.inner_box.end[dim], x[dim]) + 1.01f * h < 0.f) {
								flops += 3;
								contains = false;
								break;
							}
						}
					}
				}
				j = contains;
				compute_indices < PREHYDRO2_BLOCK_SIZE > (j, total);
				const int offset = ws.rec1.size();
				__syncthreads();
				const int next_size = offset + total;
				ws.rec1.resize(next_size);
				ws.rec2.resize(next_size);
				if (contains) {
					const int k = offset + j;
					ws.rec1[k].x = x[XDIM];
					ws.rec1[k].y = x[YDIM];
					ws.rec1[k].z = x[ZDIM];
					ws.rec1[k].h = data.h[pi];
					if (params.stars) {
						ws.rec1[k].star = data.stars[pi];
					} else {
						ws.rec1[k].star = false;
					}
					ws.rec2[k].vx = data.vx[pi];
					ws.rec2[k].vy = data.vy[pi];
					ws.rec2[k].vz = data.vz[pi];
				}
			}
		}
		__syncthreads();
		float hmin = 1e+20;
		float hmax = 0.0;
		const float w0 = kernelW(0.f);
		__syncthreads();
		for (int i = self.part_range.first; i < self.part_range.second; i++) {
			__syncthreads();
			const int snki = self.sink_part_range.first - self.part_range.first + i;
			const auto rung = data.rungs_snk[data.dm_index_snk[snki]];
			const bool active = rung >= params.min_rung;
			int semiactive = 0;
			x[XDIM] = data.x[i];
			x[YDIM] = data.y[i];
			x[ZDIM] = data.z[i];
			const fixed32& x_i = x[XDIM];
			const fixed32& y_i = x[YDIM];
			const fixed32& z_i = x[ZDIM];
			const float& h_i = data.rec2_snk[snki].h;
			const float hinv_i = 1.f / h_i; 										// 4
			const float h2_i = sqr(h_i);    										// 1
			if (active) {
				if (tid == 0) {
					data.sa_snk[snki] = true;
				}
			} else {
				const int jmax = round_up(ws.rec1.size(), block_size);
				if (tid == 0) {
					data.sa_snk[snki] = false;
				}
				for (int j = tid; j < jmax; j += block_size) {
					if (j < ws.rec1.size()) {
						const auto& rec1 = ws.rec1[j];
						const auto& x_j = rec1.x;
						const auto& y_j = rec1.y;
						const auto& z_j = rec1.z;
						const auto& h_j = rec1.h;
						const auto h2_j = sqr(h_j);									// 1
						const float x_ij = distance(x_i, x_j);						// 1
						const float y_ij = distance(y_i, y_j);						// 1
						const float z_ij = distance(z_i, z_j);						// 1
						const float r2 = sqr(x_ij, y_ij, z_ij);					// 5
						if (r2 < fmaxf(sqr(h_i), sqr(h_j))) {									// 2
							semiactive++;
						}
						flops += 11;
					}
					shared_reduce_add<int, PREHYDRO2_BLOCK_SIZE>(semiactive);
					if (semiactive) {
						if (tid == 0) {
							data.sa_snk[snki] = true;
						}
						break;
					}
				}
			}
			int box_xceeded = false;
			if (semiactive && !params.vsoft) {
				x[XDIM] = data.x[i];
				x[YDIM] = data.y[i];
				x[ZDIM] = data.z[i];
				float& h = data.rec2_snk[snki].h;
				float h0 = params.hmin;
				const auto test = [h0,x,block_size,tid,&data](float h) {
					float n = 0.f;
					const float hinv = 1.f / h;
					for (int j = tid; j < ws.rec1.size(); j += block_size) {
						const float dx = distance(x[XDIM], ws.rec1[j].x); // 2
						const float dy = distance(x[YDIM], ws.rec1[j].y);// 2
						const float dz = distance(x[ZDIM], ws.rec1[j].z);// 2
						const float r2 = sqr(dx, dy, dz);// 2
						const float r = sqrt(r2);// 4
						const float q = r * hinv;// 1
						if (q < 1.f) {                               // 1
							const float w = kernelW(q);// 4
							n+= w;// 1
						}
					}
					shared_reduce_add<float, PREHYDRO1_BLOCK_SIZE>(n);
					const float h3 = sqr(h)*h;
					n *= (4.0f * float(M_PI) / 3.0f) * (1.f - (h0*sqr(h0))/(h*sqr(h)));
					return data.N - n;
				};
				float hmax = 1.0f;
				for (int dim = 0; dim < NDIM; dim++) {
					hmax = fminf(hmax, fabs(distance(self.outer_box.begin[dim], x[dim])));
					hmax = fminf(hmax, fabs(distance(self.outer_box.end[dim], x[dim])));
				}
				float hmin = h0;
				const float hmax0 = hmax;
				const float hmin0 = hmin;
				float test_max = test(hmax);
				float hmid;
				int iters = 0;
				if (hmax > hmin) {
					do {
						hmid = 0.5f * hmax + 0.5f * hmin;
						const float test_mid = test(hmid);
						if (test_mid * test_max < 0.f) {
							hmin = hmid;
						} else {
							hmax = hmid;
							test_max = test_mid;
						}
						iters++;
					} while (hmax > 1.001f * hmin);
				}
				if (hmax <= hmin || hmin == hmin0 || hmax == hmax0) {
					box_xceeded = true;
					if (tid == 0) {
						atomicAdd(&reduce->flag, 1);
						h = hmax0;
					}
				} else {
					if (tid == 0) {
						h = hmid;
					}
				}
			}
			if (active || (semiactive && !box_xceeded)) {
				__syncthreads();
				const int snki = self.sink_part_range.first - self.part_range.first + i;
				const float& h_i = data.rec2_snk[snki].h;
				const float& vx_i = data.vx[i];
				const float& vy_i = data.vy[i];
				const float& vz_i = data.vz[i];
				const fixed32& x_i = x[XDIM];
				const fixed32& y_i = x[YDIM];
				const fixed32& z_i = x[ZDIM];
				const float hinv_i = 1.f / h_i; 										// 4
				const float h3inv_i = sqr(hinv_i) * hinv_i;
				const float h2_i = sqr(h_i);    										// 1
				float drho_dh;
				drho_dh = 0.f;
				float rhoh30 = (3.0f * data.N) / (4.0f * float(M_PI));   // 5
				float dvx_dx = 0.f;
				float dvx_dy = 0.f;
				float dvx_dz = 0.f;
				float dvy_dx = 0.f;
				float dvy_dy = 0.f;
				float dvy_dz = 0.f;
				float dvz_dx = 0.f;
				float dvz_dy = 0.f;
				float dvz_dz = 0.f;
				float rho_i = 0.f;
				ws.neighbors.resize(0);
				__syncthreads();
				flops += 10;
				const int jmax = round_up(ws.rec1.size(), PREHYDRO2_BLOCK_SIZE);
				for (int j = tid; j < jmax; j += block_size) {
					bool contains = false;
					if (j < ws.rec1.size()) {
						const auto& rec1 = ws.rec1[j];
						const fixed32& x_j = rec1.x;
						const fixed32& y_j = rec1.y;
						const fixed32& z_j = rec1.z;
						const auto star_j = rec1.star;
						const float x_ij = distance(x_i, x_j); // 1
						const float y_ij = distance(y_i, y_j); // 1
						const float z_ij = distance(z_i, z_j); // 1
						const float r2 = sqr(x_ij, y_ij, z_ij);
						const float r = sqrtf(r2);                    // 4
						const float q = r * hinv_i;                    // 1
						if (q < 1.f) {                               // 1
							const float w = kernelW(q);
							const float dwdq = dkernelW_dq(q);
							drho_dh -= 3.f * w + q * dwdq;                      // 2
							if (!star_j) {
								rho_i += data.m * w * h3inv_i;
							}
							contains = true;
							flops += 2;
						}
						flops += 9;
					}
					int k = contains;
					int total;
					compute_indices < PREHYDRO2_BLOCK_SIZE > (k, total);
					const int offset = ws.neighbors.size();
					__syncthreads();
					const int next_size = offset + total;
					ws.neighbors.resize(next_size);
					if (contains) {
						const int l = offset + k;
						ws.neighbors[l] = j;
					}
				}
				shared_reduce_add<float, PREHYDRO2_BLOCK_SIZE>(drho_dh);
				shared_reduce_add<float, PREHYDRO2_BLOCK_SIZE>(rho_i);
				data.rho_snk[snki] = rho_i;
				const float omega_i = 1.f + 0.33333333333f * drho_dh / rhoh30;								// 4
				ALWAYS_ASSERT(omega_i > 0.f);
				__syncthreads();
				const float h4inv_i = h3inv_i * hinv_i;						// 1
				flops += 16;
				for (int j = tid; j < ws.neighbors.size(); j += block_size) {
					const int kk = ws.neighbors[j];
					const auto& rec1 = ws.rec1[kk];
					const auto& rec2 = ws.rec2[kk];
					const fixed32& x_j = rec1.x;
					const fixed32& y_j = rec1.y;
					const fixed32& z_j = rec1.z;
					const float& vx_j = rec2.vx;
					const float& vy_j = rec2.vy;
					const float& vz_j = rec2.vz;
					const float x_ij = distance(x_i, x_j);                  // 1
					const float y_ij = distance(y_i, y_j);                  // 1
					const float z_ij = distance(z_i, z_j);                  // 1
					const float r2 = sqr(x_ij, y_ij, z_ij);                 // 5
					const float r = sqrtf(r2);                               // 4
					const float q = r * hinv_i;                               // 1
					const float vx_ij = vx_i - vx_j + x_ij * params.adot;   // 3
					const float vy_ij = vy_i - vy_j + y_ij * params.adot;   // 3
					const float vz_ij = vz_i - vz_j + z_ij * params.adot;   // 3
					const float rinv = 1.0f / (1.0e-30f + r);               // 5
					const float dwdq = dkernelW_dq(q);
					const float dWdr_i = dwdq * h4inv_i / omega_i;             // 2
					const float dWdr_i_rinv = dWdr_i * rinv;                // 1
					const float dWdr_i_x = dWdr_i_rinv * x_ij;				  // 1
					const float dWdr_i_y = dWdr_i_rinv * y_ij;              // 1
					const float dWdr_i_z = dWdr_i_rinv * z_ij;              // 1
					dvx_dx -= vx_ij * dWdr_i_x; // 2
					dvy_dx -= vy_ij * dWdr_i_x; // 2
					dvz_dx -= vz_ij * dWdr_i_x; // 2
					dvx_dy -= vx_ij * dWdr_i_y; // 2
					dvy_dy -= vy_ij * dWdr_i_y; // 2
					dvz_dy -= vz_ij * dWdr_i_y; // 2
					dvx_dz -= vx_ij * dWdr_i_z; // 2
					dvy_dz -= vy_ij * dWdr_i_z; // 2
					dvz_dz -= vz_ij * dWdr_i_z; // 2
					flops += 68;
				}
				shared_reduce_add<float, PREHYDRO2_BLOCK_SIZE>(dvx_dx);		// 127
				shared_reduce_add<float, PREHYDRO2_BLOCK_SIZE>(dvx_dy);		// 127
				shared_reduce_add<float, PREHYDRO2_BLOCK_SIZE>(dvx_dz);		// 127
				shared_reduce_add<float, PREHYDRO2_BLOCK_SIZE>(dvy_dx);		// 127
				shared_reduce_add<float, PREHYDRO2_BLOCK_SIZE>(dvy_dy);		// 127
				shared_reduce_add<float, PREHYDRO2_BLOCK_SIZE>(dvy_dz);		// 127
				shared_reduce_add<float, PREHYDRO2_BLOCK_SIZE>(dvz_dx);		// 127
				shared_reduce_add<float, PREHYDRO2_BLOCK_SIZE>(dvz_dy);		// 127
				shared_reduce_add<float, PREHYDRO2_BLOCK_SIZE>(dvz_dz);		// 127
				if (tid == 0) {
					flops += 38 + (PREHYDRO2_BLOCK_SIZE - 1) * 11;
					float div_v;
					const float mrhoinv = data.m / rho_i;                       // 4
					dvx_dx *= mrhoinv;                                       // 1
					dvx_dy *= mrhoinv;                                       // 1
					dvx_dz *= mrhoinv;                                       // 1
					dvy_dx *= mrhoinv;                                       // 1
					dvy_dy *= mrhoinv;                                       // 1
					dvy_dz *= mrhoinv;                                       // 1
					dvz_dx *= mrhoinv;                                       // 1
					dvz_dy *= mrhoinv;                                       // 1
					dvz_dz *= mrhoinv;                                       // 1
					div_v = dvx_dx + dvy_dy + dvz_dz;                        // 2
					float shear_xx, shear_xy, shear_xz, shear_yy, shear_yz, shear_zz;
					shear_xx = dvx_dx - (1.f / 3.f) * div_v;                 // 2
					shear_yy = dvy_dy - (1.f / 3.f) * div_v;                 // 2
					shear_zz = dvz_dz - (1.f / 3.f) * div_v;                 // 2
					shear_xy = 0.5f * (dvx_dy + dvy_dx);                     // 2
					shear_xz = 0.5f * (dvx_dz + dvz_dx);                     // 2
					shear_yz = 0.5f * (dvy_dz + dvz_dy);                     // 2
					const float shearv = sqrtf(sqr(shear_xx) + sqr(shear_yy) + sqr(shear_zz) + 2.0f * (sqr(shear_xy) + sqr(shear_xz) + sqr(shear_yz))); // 16
					data.shear_snk[snki] = shearv;
					data.omega_snk[snki] = omega_i;
				}
			}
		}
		shared_reduce_add<int, PREHYDRO2_BLOCK_SIZE>(flops);
		if (tid == 0) {
			atomicMax(&reduce->hmax, hmax);
			atomicMin(&reduce->hmin, hmin);
			atomicAdd(&reduce->flops, (double) flops);
			index = atomicAdd(&reduce->counter, 1);
		}
		flops = 0;
		__syncthreads();
	}
	(&ws)->~prehydro2_workspace();
}

