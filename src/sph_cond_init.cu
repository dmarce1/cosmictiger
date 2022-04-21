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

struct cond_init_record1 {
	fixed32 x;
	fixed32 y;
	fixed32 z;
	float h;
};
struct cond_init_record2 {
	float entr;
	float cfrac;
};

struct cond_init_workspace {
	device_vector<cond_init_record1> rec1;
	device_vector<cond_init_record2> rec2;
	device_vector<int> neighbors;
};

__global__ void sph_cuda_cond_init(sph_run_params params, sph_run_cuda_data data, sph_reduction* reduce) {
	const int tid = threadIdx.x;
	const int block_size = blockDim.x;
	__shared__
	int index;
	__shared__ cond_init_workspace ws;
	if (tid == 0) {
		index = atomicAdd(&reduce->counter, 1);
	}
	__syncthreads();
	new (&ws) cond_init_workspace();
	int flops = 0;
	array<fixed32, NDIM> x;
	const float gamma0 = data.def_gamma;
	const float code_to_energy = sqr(params.code_to_cm) / sqr(params.code_to_s);									// 5
	const float code_to_density = params.code_to_g / pow(params.code_to_cm, 3.);									// 12
	const float colog0 = log(1.5 * pow(constants::kb, 1.5) * pow(constants::e, -3) * pow(M_PI, -0.5));    // 35
	const float kappa0 = 20.0 * pow(2.0 / M_PI, 1.5) * pow(constants::kb, 2.5) * pow(constants::me, -0.5) * pow(constants::e, -4.0) * params.code_to_s
			* params.code_to_cm / (params.code_to_g * constants::avo);													// 49
	const float propc0 = 0.4 * (gamma0 - 1.0) * sqrtf(2.0 * constants::kb / (M_PI * constants::me)) / constants::c; // 17
	const float cv0 = constants::kb / (gamma0 - 1.0f);																		// 5
	const float invgm1 = 1.f / (gamma0 - 1.0);
	flops += 128;
	while (index < data.nselfs) {
		__syncthreads();
		int flops = 0;
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
				float h;
				int rung;
				if (pi < other.part_range.second) {
					h = data.h[pi];
					rung = data.rungs[pi];
					if (rung >= params.min_rung) {
						x[XDIM] = data.x[pi];
						x[YDIM] = data.y[pi];
						x[ZDIM] = data.z[pi];
						contains = (self.outer_box.contains(x));
					}
				}
				j = contains;
				compute_indices < COND_INIT_BLOCK_SIZE > (j, total);
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
					ws.rec1[k].h = h;
					ws.rec2[k].entr = data.entr[pi];
					if (params.stars) {
						ws.rec2[k].cfrac = data.cold_frac[pi];
					} else {
						ws.rec2[k].cfrac = 0.f;
					}
				}
			}
		}
		const float& m = data.m;
		const float c0 = float(3.0f / (4.0f * M_PI)) * data.N;					// 1
		flops++;
		for (int i = self.part_range.first; i < self.part_range.second; i++) {
			__syncthreads();
			const int snki = self.sink_part_range.first - self.part_range.first + i;
			const bool active = data.rungs[i] >= params.min_rung;
			const bool semiactive = data.sa_snk[i] && !active;
			const float h_i = data.h[i];
			const float A_i = data.entr[i];
			const auto x_i = data.x[i];
			const auto y_i = data.y[i];
			const auto z_i = data.z[i];
			float cfrac_i;
			if (params.stars) {
				cfrac_i = data.cold_frac[i];
			} else {
				cfrac_i = 0.f;
			}
			const float hfrac_i = 1.f - cfrac_i;									// 1
			const float h2_i = sqr(h_i);												// 1
			flops += 2;
			if (semiactive || active) {
				const float hinv_i = 1.f / h_i;													// 4
				const float h3inv_i = (sqr(hinv_i) * hinv_i);								// 2
				const float rho_i = m * c0 * h3inv_i;											// 2
				const float ene_i = A_i * powf(rho_i * hfrac_i, gamma0 - 1.0f);		// 11
				float gradx = 0.0f;
				float grady = 0.0f;
				float gradz = 0.0f;
				const float fpre_i = data.fpre1_snk[snki];
				flops += 11;
				ws.neighbors.resize(0);
				const float jmax = round_up(ws.rec1.size(), block_size);
				for (int j = tid; j < jmax; j += block_size) {
					int k;
					int total;
					bool use = false;
					if (j < ws.rec1.size()) {
						const auto rec1 = ws.rec1[j];
						const fixed32 x_j = rec1.x;
						const fixed32 y_j = rec1.y;
						const fixed32 z_j = rec1.z;
						const float h_j = rec1.h;
						const float x_ij = distance(x_i, x_j);				// 1
						const float y_ij = distance(y_i, y_j);				// 1
						const float z_ij = distance(z_i, z_j);				// 1
						const float r2 = sqr(x_ij, y_ij, z_ij);			// 5
						if (r2 < fmaxf(sqr(h_i), sqr(h_j))) {				// 4
							use = true;
						}
						flops += 12;
					}
					k = use;
					compute_indices < COND_INIT_BLOCK_SIZE > (k, total);
					const int offset = ws.neighbors.size();
					__syncthreads();
					const int next_size = offset + total;
					ws.neighbors.resize(next_size);
					if (use) {
						const int l = offset + k;
						ws.neighbors[l] = j;
					}
				}
				__syncthreads();
				for (int j = tid; j < ws.neighbors.size(); j += COND_INIT_BLOCK_SIZE) {
					const int kk = ws.neighbors[j];
					const auto& rec1 = ws.rec1[kk];
					const auto& rec2 = ws.rec2[kk];
					const float A_j = rec2.entr;
					const float cfrac_j = rec2.cfrac;
					ALWAYS_ASSERT(cfrac_j <= 1.0f);
					const float hfrac_j = 1.f - cfrac_j;
					const fixed32 x_j = rec1.x;
					const fixed32 y_j = rec1.y;
					const fixed32 z_j = rec1.z;
					const float x_ij = distance(x_i, x_j);				// 1
					const float y_ij = distance(y_i, y_j);				// 1
					const float z_ij = distance(z_i, z_j);				// 1
					const float r2 = sqr(x_ij, y_ij, z_ij);			// 5
					flops += 10;
					if (r2 < h2_i && r2 != 0.0f) {						// 2
						const float h_j = rec1.h;
						const float r = sqrtf(r2);							// 4
						const float rinv = 1.f / r;						// 4
						const float q = r * hinv_i;						// 1
						const float hinv_j = 1.f / h_j;					// 4
						const float h3inv_j = sqr(hinv_j) * hinv_j;	// 2
						const float rho_j = m * c0 * h3inv_j;			// 2
						const float ene_j = A_j * powf(rho_j * hfrac_j, gamma0 - 1.0f); // 11
						float w;
						const float dwdq = dkernelW_dq(q, &w, &flops);
						const float dWdr_i = fpre_i * dwdq * h3inv_i * hinv_i; // 3
						const float tmp = dWdr_i * rinv * logf(ene_j / ene_i); // 14
						gradx = fmaf(tmp, x_ij, gradx);					// 2
						grady = fmaf(tmp, y_ij, grady);					// 2
						gradz = fmaf(tmp, z_ij, gradz);					// 2
						flops += 49;
					}
				}
				const float tmp = m / (params.a * rho_i);		// 5
				gradx *= tmp;												// 1
				grady *= tmp;												// 1
				gradz *= tmp;												// 1
				flops += 8;
				shared_reduce_add<float, COND_INIT_BLOCK_SIZE>(gradx); //31
				shared_reduce_add<float, COND_INIT_BLOCK_SIZE>(grady); //31
				shared_reduce_add<float, COND_INIT_BLOCK_SIZE>(gradz); //31
				if (tid == 0) {
					flops += 93;
					const auto& frac_i = data.rec1_snk[snki].frac;
					const float& cfrac_i = data.cold_mass_snk[snki];
					const float& A_i = data.rec2_snk[snki].A;
					const float& H = frac_i[CHEM_H];
					const float& Hp = frac_i[CHEM_HP];
					const float& Hn = frac_i[CHEM_HN];
					const float& H2 = frac_i[CHEM_H2];
					const float& He = frac_i[CHEM_HE];
					const float& Hep = frac_i[CHEM_HEP];
					const float& Hepp = frac_i[CHEM_HEPP];
					const float grad2 = sqr(gradx, grady, gradz);	// 5
					const float gradToT = sqrtf(grad2);					// 4

					const float hfrac_i = 1.f - cfrac_i;				// 1
					const float rho0 = rho_i * hfrac_i / (sqr(params.a) * params.a); // 7
					float n0 = (H + fmaf(2.f, Hp, fmaf(.5f, H2, fmaf(.25f, He, fmaf(.5f, Hep, .75f * Hepp))))); // 10
					const float mmw_i = 1.0f / n0;						// 4
					const float ne_i = fmaxf((Hp - Hn + fmaf(0.25f, Hep, 0.5f * Hepp)) * rho0 * (constants::avo * code_to_density), 1e-30f);						// 8
					const float eint = code_to_energy * A_i * powf(rho0 * hfrac_i, gamma0 - 1.0) * invgm1; // 13
					const float T_i = mmw_i * eint / (cv0 * constants::avo); // 6
					const float colog_i = colog0 + 1.5f * logf(T_i) - 0.5f * logf(ne_i); // 20
					float kappa_i = (gamma0 - 1.f) * kappa0 * powf(T_i, 2.5f) / colog_i; // 15
					const float sigmax_i = propc0 * sqrtf(T_i);      // 5
					const float R = 2.f * mmw_i * kappa_i * gradToT / (rho_i * sigmax_i); // 7
					const float phi = (2.f + 3.f * R) / (2.f + 3.f * R + 3.f * sqr(R)); // 11
					kappa_i *= phi;											 // 1
					ALWAYS_ASSERT(isfinite(kappa_i));
					data.kap_snk[snki] = kappa_i;
					data.entr0_snk[snki] = A_i;
				}
			}

		}
		shared_reduce_add<int, COND_INIT_BLOCK_SIZE>(flops);
		if (tid == 0) {
			atomicAdd(&reduce->flops, (float) flops);
			index = atomicAdd(&reduce->counter, 1);
		}
		flops = 0;
		__syncthreads();
	}
	(&ws)->~cond_init_workspace();

}

