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
};

struct prehydro_record2 {
	float vx;
	float vy;
	float vz;
	float entr;
	float h;
};

struct prehydro_workspace {
	device_vector<prehydro_record1> rec1;
	device_vector<prehydro_record2> rec2;
	device_vector<int> neighbors;
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
	float error;
	const float gamma0 = data.def_gamma;
	const float code_to_energy = sqr(params.code_to_cm) / sqr(params.code_to_s);									// 5
	const float code_to_density = params.code_to_g / pow(params.code_to_cm, 3.);									// 12
	const float colog0 = log(1.5 * pow(constants::kb, 1.5) * pow(constants::e, -3) * pow(M_PI, -0.5));    // 35
	const float kappa0 = 20.0 * pow(2.0 / M_PI, 1.5) * pow(constants::kb, 2.5) * pow(constants::me, -0.5) * pow(constants::e, -4.0) * params.code_to_s
			* params.code_to_cm / (params.code_to_g * constants::avo);													// 49
	const float propc0 = 0.4 * (gamma0 - 1.0) * sqrtf(2.0 * constants::kb / (M_PI * constants::me)) / constants::c; // 17
	const float cv0 = constants::kb / (gamma0 - 1.0f);																		// 5
	const float invgm1 = 1.f / (gamma0 - 1.0);
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
				}
				j = contains;
				compute_indices < PREHYDRO_BLOCK_SIZE > (j, total);
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
					ws.rec2[k].vx = data.vx[pi];
					ws.rec2[k].vy = data.vy[pi];
					ws.rec2[k].vz = data.vz[pi];
					if (params.conduction) {
						ws.rec2[k].entr = data.entr[pi];
						ws.rec2[k].h = data.h[pi];
					}
				}
			}
		}
		float hmin = 1e+20;
		float hmax = 0.0;
		const float w0 = kernelW(0.f);
		__syncthreads();
		for (int i = self.part_range.first; i < self.part_range.second; i++) {
			__syncthreads();
			const int snki = self.sink_part_range.first - self.part_range.first + i;
			const bool active = data.rungs_snk[data.dm_index_snk[snki]] >= params.min_rung;
			const auto& converged = data.converged_snk[snki];
			const bool use = active && !converged;
			if (use) {
				x[XDIM] = data.x[i];
				x[YDIM] = data.y[i];
				x[ZDIM] = data.z[i];
				int box_xceeded = false;
				int iter = 0;
				float& h = data.rec2_snk[snki].h;
				float drho_dh;
				float rhoh3;
				float last_dlogh = 0.0f;
				float w1 = 1.f;
				do {
					const float hinv = 1.f / h; 										// 4
					const float h2 = sqr(h);    										// 1
					drho_dh = 0.f;
					rhoh3 = 0.f;
					float rhoh30 = (3.0f * data.N) / (4.0f * float(M_PI));   // 4
					flops += 9;
					for (int j = tid; j < ws.rec1.size(); j += block_size) {
						const float dx = distance(x[XDIM], ws.rec1[j].x);        // 1
						const float dy = distance(x[YDIM], ws.rec1[j].y);        // 1
						const float dz = distance(x[ZDIM], ws.rec1[j].z);        // 1
						const float r2 = sqr(dx, dy, dz);                     // 5
						const float r = sqrtf(r2);                            // 4
						const float q = r * hinv;                             // 1
						flops += 13;
						if (q < 1.f) {                                        // 1
							float w;
							const float dwdq = dkernelW_dq(q, &w, &flops);
							const float dwdh = -q * dwdq * hinv; 					// 3
							drho_dh -= dwdq;												// 1
							rhoh3 += w;														// 1
							flops += 5;
						}

					}
					shared_reduce_add<float, PREHYDRO_BLOCK_SIZE>(drho_dh);	 // 127
					shared_reduce_add<float, PREHYDRO_BLOCK_SIZE>(rhoh3);	 // 127
					if (tid) {
						flops += 2 * (PREHYDRO_BLOCK_SIZE - 1);
					}
					float dlogh;
					__syncthreads();

					if (rhoh3 <= 1.01f * w0) {											// 2
						if (tid == 0) {
							h *= 1.1f;														// 1
							flops++;
						}
						flops++;
						iter--;
						error = 1.0;
					} else {
						drho_dh *= 0.33333333333f / rhoh30;							// 5
						const float fpre = fminf(fmaxf(1.0f / (drho_dh), 0.5f), 2.0f);							// 6
						dlogh = fminf(fmaxf(powf(rhoh30 / rhoh3, fpre * 0.3333333333333333f) - 1.f, -.1f), .1f); // 9
						error = fabs(1.0f - rhoh3 / rhoh30); // 3
						if (last_dlogh * dlogh < 0.f) { // 2
							w1 *= 0.9f;                       // 1
							flops++;
						} else {
							w1 = fminf(1.f, w1 / 0.9f); // 5
							flops += 5;
						}
						if (tid == 0) {
							h *= (1.f + w1 * dlogh);    // 3
							flops += 3;
						}
						flops += 23;
						last_dlogh = dlogh;
					}
					flops += 2;

					__syncthreads();
					if (tid == 0) {
						if (iter > 100000) {
							PRINT("Solver failed to converge - %i %e %e %e\n", iter, h, dlogh, error);
							if (iter > 100010) {
								__trap();
							}
						}
					}
					for (int dim = 0; dim < NDIM; dim++) {
						if (self.outer_box.end[dim] < range_fixed(x[dim] + fixed32(h)) + range_fixed::min()) {
							box_xceeded = true;
							break;
						}
						if (range_fixed(x[dim]) < self.outer_box.begin[dim] + range_fixed(h) + range_fixed::min()) {
							box_xceeded = true;
							break;
						}
					}
					__syncthreads();
					iter++;
					shared_reduce_add<int, PREHYDRO_BLOCK_SIZE>(box_xceeded);
				} while (error > SPH_SMOOTHLEN_TOLER && !box_xceeded);
				if (box_xceeded) {
					if (tid == 0) {
						atomicAdd(&reduce->flag, 1);
					}
				} else {
					__syncthreads();
					const int snki = self.sink_part_range.first - self.part_range.first + i;
					x[XDIM] = data.x[i];
					x[YDIM] = data.y[i];
					x[ZDIM] = data.z[i];
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
					const float c0 = float(3.0f / 4.0f / M_PI * data.N);     // 1
					const float rho_i = data.m * c0 * h3inv_i;                        // 1
					float A_i, ene_i;
					if (params.conduction) {
						A_i = data.entr[i];
						ene_i = A_i * powf(rho_i, gamma0 - 1.0f);		// 11
					}
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
					float gradx = 0.0f;
					float grady = 0.0f;
					float gradz = 0.0f;
					ws.neighbors.resize(0);
					__syncthreads();
					flops += 10;
					const int jmax = round_up(ws.rec1.size(), PREHYDRO_BLOCK_SIZE);
					for (int j = tid; j < jmax; j += block_size) {
						bool contains = false;
						if (j < ws.rec1.size()) {
							const auto& rec1 = ws.rec1[j];
							const fixed32& x_j = rec1.x;
							const fixed32& y_j = rec1.y;
							const fixed32& z_j = rec1.z;
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
						const int offset = ws.neighbors.size();
						__syncthreads();
						const int next_size = offset + total;
						ws.neighbors.resize(next_size);
						if (contains) {
							const int l = offset + k;
							ws.neighbors[l] = j;
						}
					}
					shared_reduce_add<float, PREHYDRO_BLOCK_SIZE>(drho_dh);
					flops += (PREHYDRO_BLOCK_SIZE - 1);
					const float omega_i = 0.33333333333f * drho_dh / rhoh30;								// 4
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
						float w;
						const float dwdq = dkernelW_dq(q, &w, &flops);
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
						if (params.conduction) {
							const float& h_j = rec2.h;
							const float& A_j = rec2.entr;
							const float hinv_j = 1.f / h_j;					// 4
							const float h3inv_j = sqr(hinv_j) * hinv_j;	// 2
							const float rho_j = data.m * c0 * h3inv_j;			// 2
							const float ene_j = A_j * powf(rho_j, gamma0 - 1.0f); // 11
							const float tmp = dWdr_i * rinv * (ene_j - ene_i); // 14
							gradx = fmaf(tmp, x_ij, gradx);					// 2
							grady = fmaf(tmp, y_ij, grady);					// 2
							gradz = fmaf(tmp, z_ij, gradz);					// 2
						}

						flops += 68;
					}
					shared_reduce_add<float, PREHYDRO_BLOCK_SIZE>(dvx_dx);		// 127
					shared_reduce_add<float, PREHYDRO_BLOCK_SIZE>(dvx_dy);		// 127
					shared_reduce_add<float, PREHYDRO_BLOCK_SIZE>(dvx_dz);		// 127
					shared_reduce_add<float, PREHYDRO_BLOCK_SIZE>(dvy_dx);		// 127
					shared_reduce_add<float, PREHYDRO_BLOCK_SIZE>(dvy_dy);		// 127
					shared_reduce_add<float, PREHYDRO_BLOCK_SIZE>(dvy_dz);		// 127
					shared_reduce_add<float, PREHYDRO_BLOCK_SIZE>(dvz_dx);		// 127
					shared_reduce_add<float, PREHYDRO_BLOCK_SIZE>(dvz_dy);		// 127
					shared_reduce_add<float, PREHYDRO_BLOCK_SIZE>(dvz_dz);		// 127
					if (params.conduction) {
						const float tmp = data.m / (params.a * rho_i * ene_i);		// 5
						gradx *= tmp;												// 1
						grady *= tmp;												// 1
						gradz *= tmp;												// 1
						flops += 8;
						shared_reduce_add<float, PREHYDRO_BLOCK_SIZE>(gradx); //31
						shared_reduce_add<float, PREHYDRO_BLOCK_SIZE>(grady); //31
						shared_reduce_add<float, PREHYDRO_BLOCK_SIZE>(gradz); //31
					}
					if (tid == 0) {
						flops += 38 + (PREHYDRO_BLOCK_SIZE - 1) * 11;
						float shear_xx, shear_xy, shear_xz, shear_yy, shear_yz, shear_zz;
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
						shear_xx = dvx_dx - (1.f / 3.f) * div_v;                 // 2
						shear_yy = dvy_dy - (1.f / 3.f) * div_v;                 // 2
						shear_zz = dvz_dz - (1.f / 3.f) * div_v;                 // 2
						shear_xy = 0.5f * (dvx_dy + dvy_dx);                     // 2
						shear_xz = 0.5f * (dvx_dz + dvz_dx);                     // 2
						shear_yz = 0.5f * (dvy_dz + dvz_dy);                     // 2
						const float shearv = sqrtf(sqr(shear_xx) + sqr(shear_yy) + sqr(shear_zz) + 2.0f * (sqr(shear_xy) + sqr(shear_xz) + sqr(shear_yz))); // 16
						data.shear_snk[snki] = shearv;
						data.omega_snk[snki] = omega_i;
						if (params.conduction) {
							const auto& frac_i = data.rec1_snk[snki].frac;
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
							const float rho0 = rho_i / (sqr(params.a) * params.a); // 7
							float n0 = (H + fmaf(2.f, Hp, fmaf(.5f, H2, fmaf(.25f, He, fmaf(.5f, Hep, .75f * Hepp))))); // 10
							const float mmw_i = 1.0f / n0;						// 4
							const float ne_i = fmaxf((Hp - Hn + fmaf(0.25f, Hep, 0.5f * Hepp)) * rho0 * (constants::avo * code_to_density), 1e-30f) * hfrac_i;		// 8
							const float eint = code_to_energy * A_i * powf(rho0, gamma0 - 1.0) * invgm1 / hfrac_i; // 13
							const float T_i = mmw_i * eint / (cv0 * constants::avo); // 6
							const float colog_i = colog0 + 1.5f * logf(T_i) - 0.5f * logf(ne_i); // 20
							float kappa_i = (gamma0 - 1.f) * kappa0 * powf(T_i, 2.5f) / colog_i; // 15
							const float sigmax_i = propc0 * sqrtf(T_i);      // 5
							const float R = 2.f * mmw_i * kappa_i * gradToT / (rho_i * sigmax_i); // 7
							const float phi = (2.f + 3.f * R) / (2.f + 3.f * R + 3.f * sqr(R)); // 11
							kappa_i *= phi;											 // 1
							ALWAYS_ASSERT(isfinite(kappa_i));
							data.kap_snk[snki] = kappa_i;
						}
					}
				}

				hmin = fminf(hmin, h);
				hmax = fmaxf(hmax, h);
			}
		}
		shared_reduce_add<int, PREHYDRO_BLOCK_SIZE>(flops);
		if (tid == 0) {
			atomicMax(&reduce->hmax, hmax);
			atomicMin(&reduce->hmin, hmin);
			atomicAdd(&reduce->flops, (double) flops);
			index = atomicAdd(&reduce->counter, 1);
		}
		flops = 0;
		__syncthreads();
	}
	(&ws)->~prehydro_workspace();
}

