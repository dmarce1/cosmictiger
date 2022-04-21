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

struct prehydro_record1 {
	fixed32 x;
	fixed32 y;
	fixed32 z;
	float h;
};

struct prehydro_record2 {
	float entr;
	float cfrac;
	float vx;
	float vy;
	float vz;
};

struct prehydro_workspace {
	device_vector<prehydro_record1> x;
	device_vector<prehydro_record2> v;
	device_vector<prehydro_record1> xc;
	device_vector<prehydro_record2> vc;
};

__global__ void sph_cuda_prehydro(sph_run_params params, sph_run_cuda_data data, sph_reduction* reduce) {
	const int tid = threadIdx.x;
	const int block_size = blockDim.x;
	__shared__
	int index;
	__shared__ prehydro_workspace ws;
	__syncthreads();
	if (tid == 0) {
		index = atomicAdd(&reduce->counter, 1);
	}
	__syncthreads();
	new (&ws) prehydro_workspace();
	array<fixed32, NDIM> x;
	while (index < data.nselfs) {

		int flops = 0;
		__syncthreads();
		ws.x.resize(0);
		ws.v.resize(0);
		const sph_tree_node& self = data.trees[data.selfs[index]];
		bool found_self = false;
		for (int ni = self.neighbor_range.first; ni < self.neighbor_range.second; ni++) {
			const sph_tree_node& other = data.trees[data.neighbors[ni]];
			if (data.neighbors[ni] == data.selfs[index]) {
				found_self = true;
			}
			const int maxpi = round_up(other.part_range.second - other.part_range.first, block_size) + other.part_range.first;
			for (int pi = other.part_range.first + tid; pi < maxpi; pi += block_size) {
				bool contains = false;
				int j;
				int total;
				if (pi < other.part_range.second) {
					x[XDIM] = data.x[pi];
					x[YDIM] = data.y[pi];
					x[ZDIM] = data.z[pi];
					const float h_i = data.h[pi];
					if (self.outer_box.contains(x)) {
						contains = true;
					}
					if (!contains) {
						contains = true;
						for (int dim = 0; dim < NDIM; dim++) {
							if (distance(x[dim], self.inner_box.begin[dim]) + h_i < 0.f) {
								contains = false;
								flops += 3;
								break;
							}
							if (distance(self.inner_box.end[dim], x[dim]) + h_i < 0.f) {
								contains = false;
								flops += 3;
								break;
							}
						}
					}
				}
				j = contains;
				compute_indices < PREHYDRO_BLOCK_SIZE > (j, total);
				const int offset = ws.x.size();
				__syncthreads();
				const int next_size = offset + total;
				ws.x.resize(next_size);
				ws.v.resize(next_size);
				if (contains) {
					const int k = offset + j;
					ws.x[k].x = x[XDIM];
					ws.x[k].y = x[YDIM];
					ws.x[k].z = x[ZDIM];
					ws.x[k].h = data.h[pi];
					ws.v[k].vx = data.vx[pi];
					ws.v[k].vy = data.vy[pi];
					ws.v[k].vz = data.vz[pi];
					ws.v[k].entr = data.entr[pi];
					if (params.stars) {
						ws.v[k].cfrac = data.cold_frac[pi];
					} else {
						ws.v[k].cfrac = 0.f;
					}
				}
			}
		}
		ALWAYS_ASSERT(found_self);
		ALWAYS_ASSERT(ws.x.size());
		const float gamma0 = data.def_gamma;
		for (int i = self.part_range.first; i < self.part_range.second; i++) {
			__syncthreads();
			const int snki = self.sink_part_range.first - self.part_range.first + i;
			const bool active = data.rungs_snk[data.dm_index_snk[snki]] >= params.min_rung;
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
				const int jmax = round_up(ws.x.size(), block_size);
				if (tid == 0) {
					data.sa_snk[snki] = false;
				}
				for (int j = tid; j < jmax; j += block_size) {
					if (j < ws.x.size()) {
						const auto x_j = ws.x[j].x;
						const auto y_j = ws.x[j].y;
						const auto z_j = ws.x[j].z;
						const auto h_j = ws.x[j].h;
						const auto h2_j = sqr(h_j);									// 1
						const float x_ij = distance(x_i, x_j);						// 1
						const float y_ij = distance(y_i, y_j);						// 1
						const float z_ij = distance(z_i, z_j);						// 1
						const float r2 = sqr(x_ij, y_ij, z_ij);					// 5
						if (r2 < fmaxf(h2_i, h2_j)) {									// 2
							semiactive++;
						}
						flops += 11;
					}
					shared_reduce_add<int, PREHYDRO_BLOCK_SIZE>(semiactive);
					if (semiactive) {
						if (tid == 0) {
							data.sa_snk[snki] = true;
						}
						break;
					}
				}
			}
			if (active || semiactive) {
				float drho_dh;
				const float vx_i = data.vx[i];
				const float vy_i = data.vy[i];
				const float vz_i = data.vz[i];
				drho_dh = 0.f;
				float rhoh30 = (3.0f * data.N) / (4.0f * float(M_PI));   // 5
				const fixed32& x_i = x[XDIM];
				const fixed32& y_i = x[YDIM];
				const fixed32& z_i = x[ZDIM];
				float dvx_dx = 0.f;
				float dvx_dy = 0.f;
				float dvx_dz = 0.f;
				float dvy_dx = 0.f;
				float dvy_dy = 0.f;
				float dvy_dz = 0.f;
				float dvz_dx = 0.f;
				float dvz_dy = 0.f;
				float dvz_dz = 0.f;
				float pre = 0.f;
				float dpdh = 0.f;
				ws.xc.resize(0);
				ws.vc.resize(0);
				__syncthreads();
				flops += 10;
				const int jmax = round_up(ws.x.size(), PREHYDRO_BLOCK_SIZE);
				for (int j = tid; j < jmax; j += block_size) {
					bool contains = false;
					if (j < ws.x.size()) {
						const fixed32 x_j = ws.x[j].x;
						const fixed32 y_j = ws.x[j].y;
						const fixed32 z_j = ws.x[j].z;
						const float x_ij = distance(x_i, x_j); // 1
						const float y_ij = distance(y_i, y_j); // 1
						const float z_ij = distance(z_i, z_j); // 1
						const float r2 = sqr(x_ij, y_ij, z_ij);
						const float r = sqrtf(r2);                    // 4
						const float q = r * hinv_i;                    // 1
						if (q < 1.f) {                               // 1
							float w;
							const float dwdq = dkernelW_dq(q, &w, &flops);
							drho_dh -= q * dwdq;                      // 2
							contains = true;
							flops += 2;
						}
						flops += 9;
					}
					int k = contains;
					int total;
					compute_indices < PREHYDRO_BLOCK_SIZE > (k, total);
					const int offset = ws.xc.size();
					__syncthreads();
					const int next_size = offset + total;
					ws.xc.resize(next_size);
					ws.vc.resize(next_size);
					if (contains) {
						const int l = offset + k;
						ws.xc[l] = ws.x[j];
						ws.vc[l] = ws.v[j];
					}
				}
				shared_reduce_add<float, PREHYDRO_BLOCK_SIZE>(drho_dh);
				flops += (PREHYDRO_BLOCK_SIZE - 1);
				const float m = data.m;
				drho_dh *= 0.33333333333f / rhoh30;								// 5
				const float fpre = 1.0f / drho_dh;								// 4
				__syncthreads();
				const float h3inv_i = hinv_i * sqr(hinv_i);					// 2
				const float h4inv_i = h3inv_i * hinv_i;						// 1
				flops += 16;
				for (int j = tid; j < ws.xc.size(); j += block_size) {
					const fixed32 x_j = ws.xc[j].x;
					const fixed32 y_j = ws.xc[j].y;
					const fixed32 z_j = ws.xc[j].z;
					const float A_j = ws.vc[j].entr;
					const float fc_j = ws.vc[j].cfrac;
					const float fh_j = 1.f - fc_j;
					const float x_ij = distance(x_i, x_j);                  // 1
					const float y_ij = distance(y_i, y_j);                  // 1
					const float z_ij = distance(z_i, z_j);                  // 1
					const float r2 = sqr(x_ij, y_ij, z_ij);                 // 5
					const float r = sqrtf(r2);                               // 4
					const float q = r * hinv_i;                               // 1
					const float vx_j = ws.vc[j].vx;
					const float vy_j = ws.vc[j].vy;
					const float vz_j = ws.vc[j].vz;
					const float vx_ij = vx_i - vx_j + x_ij * params.adot;   // 3
					const float vy_ij = vy_i - vy_j + y_ij * params.adot;   // 3
					const float vz_ij = vz_i - vz_j + z_ij * params.adot;   // 3
					const float rinv = 1.0f / (1.0e-30f + r);               // 5
					float w;
					const float dwdq = dkernelW_dq(q, &w, &flops);
					const float dWdr_i = fpre * dwdq * h4inv_i;             // 2
					const float A0_j = fh_j * powf(A_j, 1.0f / gamma0);     // 9
					pre = fmaf(m, A0_j * w * h3inv_i, pre);                 // 4
					dpdh -= A0_j * (3.f * w + q * dwdq);                    // 5
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
				shared_reduce_add<float, PREHYDRO_BLOCK_SIZE>(dpdh);       // 127
				shared_reduce_add<float, PREHYDRO_BLOCK_SIZE>(pre);			// 127
				shared_reduce_add<float, PREHYDRO_BLOCK_SIZE>(dvx_dx);		// 127
				shared_reduce_add<float, PREHYDRO_BLOCK_SIZE>(dvx_dy);		// 127
				shared_reduce_add<float, PREHYDRO_BLOCK_SIZE>(dvx_dz);		// 127
				shared_reduce_add<float, PREHYDRO_BLOCK_SIZE>(dvy_dx);		// 127
				shared_reduce_add<float, PREHYDRO_BLOCK_SIZE>(dvy_dy);		// 127
				shared_reduce_add<float, PREHYDRO_BLOCK_SIZE>(dvy_dz);		// 127
				shared_reduce_add<float, PREHYDRO_BLOCK_SIZE>(dvz_dx);		// 127
				shared_reduce_add<float, PREHYDRO_BLOCK_SIZE>(dvz_dy);		// 127
				shared_reduce_add<float, PREHYDRO_BLOCK_SIZE>(dvz_dz);		// 127
				if (tid == 0) {
					flops += 38 + (PREHYDRO_BLOCK_SIZE - 1) * 11;
					float shear_xx, shear_xy, shear_xz, shear_yy, shear_yz, shear_zz;
					float div_v;
					const float c0 = float(3.0f / 4.0f / M_PI * data.N);     // 1
					const float rho_i = c0 * h3inv_i;                        // 1
					const float mrhoinv = 1.f / rho_i;                       // 4
					dvx_dx *= mrhoinv;                                       // 1
					dvx_dy *= mrhoinv;                                       // 1
					dvx_dz *= mrhoinv;                                       // 1
					dvy_dx *= mrhoinv;                                       // 1
					dvy_dy *= mrhoinv;                                       // 1
					dvy_dz *= mrhoinv;                                       // 1
					dvz_dx *= mrhoinv;                                       // 1
					dvz_dy *= mrhoinv;                                       // 1
					dvz_dz *= mrhoinv;                                       // 1
					pre = powf(pre, gamma0);                                 // 4
					dpdh *= 0.33333333333f / rhoh30;                         // 5
					div_v = dvx_dx + dvy_dy + dvz_dz;                        // 2
					shear_xx = dvx_dx - (1.f / 3.f) * div_v;                 // 2
					shear_yy = dvy_dy - (1.f / 3.f) * div_v;                 // 2
					shear_zz = dvz_dz - (1.f / 3.f) * div_v;                 // 2
					shear_xy = 0.5f * (dvx_dy + dvy_dx);                     // 2
					shear_xz = 0.5f * (dvx_dz + dvz_dx);                     // 2
					shear_yz = 0.5f * (dvy_dz + dvz_dy);                     // 2
					const float shearv = sqrtf(sqr(shear_xx) + sqr(shear_yy) + sqr(shear_zz) + 2.0f * (sqr(shear_xy) + sqr(shear_xz) + sqr(shear_yz))); // 16
					data.shear_snk[snki] = shearv;
					data.fpre1_snk[snki] = fpre;
					data.fpre2_snk[snki] = dpdh;
					data.pre_snk[snki] = pre;
				}
			}
		}
		shared_reduce_add<int, PREHYDRO_BLOCK_SIZE>(flops);
		if (tid == 0) {
			atomicAdd(&reduce->flops, (double) flops);
			index = atomicAdd(&reduce->counter, 1);
		}
		flops = 0;
		__syncthreads();
	}
	(&ws)->~prehydro_workspace();
}

