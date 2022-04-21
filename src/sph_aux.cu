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
struct aux_record1 {
	fixed32 x;
	fixed32 y;
	fixed32 z;
};

struct aux_record2 {
	float vx;
	float vy;
	float vz;
};

struct aux_workspace {
	device_vector<aux_record1> x;
	device_vector<aux_record2> v;
	device_vector<aux_record1> xc;
	device_vector<aux_record2> vc;
};

__global__ void sph_cuda_aux(sph_run_params params, sph_run_cuda_data data, sph_reduction* reduce) {
	const int tid = threadIdx.x;
	const int block_size = blockDim.x;
	__shared__
	int index;
	__shared__ aux_workspace ws;
	__syncthreads();
	if (tid == 0) {
		index = atomicAdd(&reduce->counter, 1);
	}
	__syncthreads();
	new (&ws) aux_workspace();
	array<fixed32, NDIM> x;
	char max_rung = 0;
	int flops = 0;
	const float gamma0 = data.def_gamma;
	const float ainv = 1.f / params.a;
	const float invgamma = 1.f / gamma0;
	flops += 8;
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
					if (self.outer_box.contains(x)) {
						contains = true;
					}
				}
				j = contains;
				compute_indices < AUX_BLOCK_SIZE > (j, total);
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
					ws.v[k].vx = data.vx[pi];
					ws.v[k].vy = data.vy[pi];
					ws.v[k].vz = data.vz[pi];
				}
			}
		}
		ALWAYS_ASSERT(found_self);
		ALWAYS_ASSERT(ws.x.size());
		for (int i = self.part_range.first; i < self.part_range.second; i++) {
			__syncthreads();
			const int snki = self.sink_part_range.first - self.part_range.first + i;
			const bool active = data.rungs_snk[data.dm_index_snk[snki]] >= params.min_rung;
			const bool use = active;
			if (use) {
				x[XDIM] = data.x[i];
				x[YDIM] = data.y[i];
				x[ZDIM] = data.z[i];
				float& h = data.rec2_snk[snki].h;
				const float vx_i = data.vx[i];
				const float vy_i = data.vy[i];
				const float vz_i = data.vz[i];
				const float fpre_i = data.fpre1_snk[snki];
				const float hinv = 1.f / h;                               // 4
				const float h2 = sqr(h);                                  // 1
				float rhoh30 = (3.0f * data.N) / (4.0f * float(M_PI));    // 2
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
				ws.xc.resize(0);
				ws.vc.resize(0);
				__syncthreads();
				flops += 7;
				const int jmax = round_up(ws.x.size(), AUX_BLOCK_SIZE);
				for (int j = tid; j < jmax; j += block_size) {
					bool contains = false;
					if (j < ws.x.size()) {
						const fixed32 x_j = ws.x[j].x;
						const fixed32 y_j = ws.x[j].y;
						const fixed32 z_j = ws.x[j].z;
						const float x_ij = distance(x_i, x_j);       // 1
						const float y_ij = distance(y_i, y_j);       // 1
						const float z_ij = distance(z_i, z_j);       // 1
						const float r2 = sqr(x_ij, y_ij, z_ij);      // 5
						const float r = sqrtf(r2);                   // 4
						const float q = r * hinv;                    // 1
						if (q < 1.f) {                               // 1
							contains = true;
						}
						flops += 14;
					}
					int k = contains;
					int total;
					compute_indices < AUX_BLOCK_SIZE > (k, total);
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
				__syncthreads();
				const float hinv_i = 1.0f / h;							 // 4
				const float h3inv_i = hinv_i * sqr(hinv_i);         // 3
				const float h4inv_i = h3inv_i * hinv_i;             // 1
				flops += 8;
				float vsig = 0.0f;
				for (int j = tid; j < ws.xc.size(); j += block_size) {
					const fixed32 x_j = ws.xc[j].x;
					const fixed32 y_j = ws.xc[j].y;
					const fixed32 z_j = ws.xc[j].z;
					const float x_ij = distance(x_i, x_j);           // 1
					const float y_ij = distance(y_i, y_j);           // 1
					const float z_ij = distance(z_i, z_j);           // 1
					const float r2 = sqr(x_ij, y_ij, z_ij);          // 5
					const float r = sqrtf(r2);                       // 4
					const float q = r * hinv;                        // 1
					const float vx_j = ws.vc[j].vx;
					const float vy_j = ws.vc[j].vy;
					const float vz_j = ws.vc[j].vz;
					const float vx_ij = vx_i - vx_j + x_ij * params.adot; // 3
					const float vy_ij = vy_i - vy_j + y_ij * params.adot; // 3
					const float vz_ij = vz_i - vz_j + z_ij * params.adot; // 3
					const float rinv = 1.0f / (1.0e-30f + r);             // 5
					float w;
					const float dwdq = fpre_i * dkernelW_dq(q, &w, &flops);
					const float dWdr_i = dwdq * h4inv_i;                  // 1
					const float w_ij = fminf(x_ij * vx_ij + y_ij * vy_ij + z_ij * vz_ij, 0.f) * rinv; // 7
					vsig = fmaxf(vsig, -w_ij);                            // 2
					const float dWdr_i_rinv = dWdr_i * rinv;					// 1
					const float dWdr_i_x = dWdr_i_rinv * x_ij;                   // 1
					const float dWdr_i_y = dWdr_i_rinv * y_ij;                   // 1
					const float dWdr_i_z = dWdr_i_rinv * z_ij;                   // 1
					dvx_dx -= vx_ij * dWdr_i_x;									// 2
					dvy_dx -= vy_ij * dWdr_i_x;									// 2
					dvz_dx -= vz_ij * dWdr_i_x;									// 2
					dvx_dy -= vx_ij * dWdr_i_y;									// 2
					dvy_dy -= vy_ij * dWdr_i_y;									// 2
					dvz_dy -= vz_ij * dWdr_i_y;									// 2
					dvx_dz -= vx_ij * dWdr_i_z;									// 2
					dvy_dz -= vy_ij * dWdr_i_z;									// 2
					dvz_dz -= vz_ij * dWdr_i_z;									// 2
					flops += 59;
				}
				shared_reduce_add<float, AUX_BLOCK_SIZE>(dvx_dx); // 127
				shared_reduce_add<float, AUX_BLOCK_SIZE>(dvx_dy); // 127
				shared_reduce_add<float, AUX_BLOCK_SIZE>(dvx_dz); // 127
				shared_reduce_add<float, AUX_BLOCK_SIZE>(dvy_dx); // 127
				shared_reduce_add<float, AUX_BLOCK_SIZE>(dvy_dy); // 127
				shared_reduce_add<float, AUX_BLOCK_SIZE>(dvy_dz); // 127
				shared_reduce_add<float, AUX_BLOCK_SIZE>(dvz_dx); // 127
				shared_reduce_add<float, AUX_BLOCK_SIZE>(dvz_dy); // 127
				shared_reduce_add<float, AUX_BLOCK_SIZE>(dvz_dz); // 127
				shared_reduce_max < AUX_BLOCK_SIZE > (vsig);          // 127
				if (tid == 0) {
					float div_v, curl_vx, curl_vy, curl_vz;
					const float c0 = float(3.0f / 4.0f / M_PI) * data.N;    // 1
					const float rho_i = c0 * h3inv_i;                       // 1
					const float mrhoinv = 1.f / rho_i;                      // 4
					dvx_dx *= mrhoinv;                                      // 1
					dvx_dy *= mrhoinv;                                      // 1
					dvx_dz *= mrhoinv;                                      // 1
					dvy_dx *= mrhoinv;                                      // 1
					dvy_dy *= mrhoinv;                                      // 1
					dvy_dz *= mrhoinv;                                      // 1
					dvz_dx *= mrhoinv;                                      // 1
					dvz_dy *= mrhoinv;                                      // 1
					dvz_dz *= mrhoinv;                                      // 1
					div_v = dvx_dx + dvy_dy + dvz_dz;                        // 2
					curl_vx = dvz_dy - dvy_dz;										   // 1
					curl_vy = -dvz_dx + dvx_dz;                              // 2
					curl_vz = dvy_dx - dvx_dy;
					const float h_i = data.rec2_snk[snki].h;
					const float pre_i = data.pre_snk[snki];
					const float A_i = data.rec2_snk[snki].A;
					const float c_i = sqrtf(gamma0 * powf(pre_i, 1.0f - invgamma) * powf(A_i, invgamma));  // 15
					vsig += c_i;																								   // 1
					char& rung = data.rungs_snk[data.dm_index_snk[snki]];
					float dt2 = params.t0 * rung_dt[rung];                                                  // 1
					const float dloghdt = fabsf(div_v - 3.f * params.adot * ainv) * (1.f / 3.f);           // 5
					const float dt_divv = params.cfl / (dloghdt + 1e-30f);                                 // 5
					flops += 108 + 10 * (AUX_BLOCK_SIZE - 1);
					if (dt_divv < dt2) {                                                                    // 1
						dt2 = dt_divv;
						rung = ceilf(log2f(params.t0) - log2f(dt_divv));                                    // 10
						dt2 = params.t0 * rung_dt[rung];                                                     // 1
						flops += 11;
					}
					const float dt1 = params.t0 * rung_dt[data.oldrung_snk[snki]];
					max_rung = max(max_rung, rung);
					const float curlv = sqrtf(sqr(curl_vx, curl_vy, curl_vz));                             // 9
					const float div_v0 = data.divv_snk[snki];
					data.divv_snk[snki] = div_v;
					const float alpha = data.rec1_snk[snki].alpha;
					const float ddivv_dt = (div_v - div_v0) / dt1 - params.adot * ainv * div_v;             // 8
					const float S = sqr(h_i) * fmaxf(0.f, -ddivv_dt) * sqr(params.a);                      // 6
					const float limiter = sqr(div_v) / (sqr(div_v) + sqr(curlv) + 1.0e-4f * sqr(c_i / h_i * ainv)); // 14
					const float alpha_targ = fmaxf(params.alpha1 * S / (S + sqr(c_i)), params.alpha0);     // 8
					const float lambda0 = params.alpha_decay * vsig * hinv_i * ainv;                       // 3
					const float dthydro = params.cfl * params.a * h_i / vsig;                              // 6
					const float lambda1 = 1.f / dthydro;                                                   // 4
					if (alpha < limiter * alpha_targ) {                                                    // 2
						data.rec1_snk[snki].alpha = (limiter * alpha_targ);                                 // 1
						flops++;
					} else {
						data.rec1_snk[snki].alpha = (limiter * (alpha_targ + (alpha - alpha_targ) * expf(-lambda0 * dt1))); // 10
						flops += 10;
					}
				}
			}
		}
		shared_reduce_add<int, AUX_BLOCK_SIZE>(flops);
		if (tid == 0) {
			atomicAdd(&reduce->flops, (double) flops);
			index = atomicAdd(&reduce->counter, 1);
			atomicMax(&reduce->max_rung, (int) max_rung);
		}
		flops = 0;
		__syncthreads();
	}
	(&ws)->~aux_workspace();
}
