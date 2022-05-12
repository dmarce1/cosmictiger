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
struct hydro_record1 {
	fixed32 x;
	fixed32 y;
	fixed32 z;
	float h;
};

struct hydro_record2 {
	array<float, NCHEMFRACS> chem;
	float vx;
	float vy;
	float vz;
#ifdef ENTROPY
	float entr;
#else
	float eint;
#endif
	float alpha;
	float omega;
	float omegaP;
	float pre;
	float shearv;
	float cold_frac;
	float kappa;
	float rho;
	char star;
};

struct hydro_workspace {
	device_vector<hydro_record1> rec1;
	device_vector<hydro_record2> rec2;
	device_vector<int> neighbors;
};

#ifdef ENTROPY
__device__
inline float pressure(float A, float rho, float gamma) {
	return A * powf(rho, gamma);
}
#else
__device__
inline float pressure(float eint, float rho, float gamma) {
	return (gamma - 1.0f) * eint * rho;
}
#endif

__global__ void sph_cuda_hydro(sph_run_params params, sph_run_cuda_data data, sph_reduction* reduce) {
	const int tid = threadIdx.x;
	const int block_size = blockDim.x;
	__shared__
	int index;
	__shared__ hydro_workspace ws;
	new (&ws) hydro_workspace();
	if (tid == 0) {
		index = atomicAdd(&reduce->counter, 1);
	}
	__syncthreads();
	array<fixed32, NDIM> x;
	float total_vsig_max = 0.;
	int max_rung_hydro = 0;
	int max_rung_grav = 0;
	int max_rung = 0;
	int flops = 0;
	const float ainv = 1.0f / params.a;
	const float gamma0 = data.def_gamma;
	const float invgamma0 = 1.f / gamma0;
	const float log2fparamst0 = log2f(params.t0);
	float visc = 0.f;
	flops += 16;
	while (index < data.nselfs) {
		__syncthreads();
		const sph_tree_node& self = data.trees[data.selfs[index]];
		ws.rec1.resize(0);
		ws.rec2.resize(0);
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
					if (self.outer_box.contains(x)) {
						contains = true;
					}
					if (!contains) {
						contains = true;
						const float& h = data.h[pi];
						for (int dim = 0; dim < NDIM; dim++) {
							if (distance(x[dim], self.inner_box.begin[dim]) + h < 0.f) {
								flops += 3;
								contains = false;
								break;
							}
							if (distance(self.inner_box.end[dim], x[dim]) + h < 0.f) {
								flops += 3;
								contains = false;
								break;
							}
						}
					}
				}
				j = contains;
				compute_indices < HYDRO_BLOCK_SIZE > (j, total);
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
					ws.rec2[k].vx = data.vx[pi];
					ws.rec2[k].vy = data.vy[pi];
					ws.rec2[k].vz = data.vz[pi];
#ifdef ENTROPY
					ws.rec2[k].entr = data.entr[pi];
#else
					ws.rec2[k].eint = data.eint[pi];
#endif
					ws.rec2[k].alpha = data.alpha[pi];
					ws.rec2[k].omega = data.omega[pi];
					ws.rec2[k].omegaP = data.omegaP[pi];
					ws.rec2[k].rho = data.rho[pi];
					ws.rec2[k].pre = data.pre[pi];
					if (params.diffusion && params.phase == 1) {
						ws.rec2[k].shearv = data.shearv[pi];
					}
					if (params.stars) {
						ws.rec2[k].cold_frac = data.cold_frac[pi];
						ws.rec2[k].star = data.stars[pi];
					} else {
						ws.rec2[k].cold_frac = 0.f;
						ws.rec2[k].star = false;
					}
					if (params.diffusion & params.phase == 1) {
						if (data.chemistry) {
							ws.rec2[k].chem = data.chem[pi];
						}
					}
				}
			}
		}
		for (int i = self.part_range.first; i < self.part_range.second; i++) {
			__syncthreads();
			const int snki = self.sink_part_range.first - self.part_range.first + i;
			int rung_i = data.sph_rungs_snk[snki];
			bool use = rung_i >= params.min_rung;
			const float& m = data.m;
			const float minv = 1.f / m;
			if (use) {
				float difco_i;
				float cfrac_i;
				const auto& x_i = data.x[i];
				const auto& y_i = data.y[i];
				const auto& z_i = data.z[i];
				const float& omega_i = data.omega[i];
				const float& omegaP_i = data.omegaP[i];
				const auto& vx_i = data.vx[i];
				const auto& vy_i = data.vy[i];
				const auto& vz_i = data.vz[i];
				const float& h_i = data.h[i];
#ifdef ENTROPY
				const float& A_i = data.entr[i];
#else
				const float& eint_i = data.eint[i];
#endif
				const float rho_i = data.rho[i];										// 2
				const float alpha_i = data.alpha[i];
				const float hinv_i = 1.f / h_i;												// 4
				const float h3inv_i = (sqr(hinv_i) * hinv_i);							// 3
				bool star_i;
				if (params.stars) {
					cfrac_i = data.cold_frac[i];
					star_i = data.stars[i];
				} else {
					cfrac_i = 0.f;
					star_i = false;
				}
				const float hfrac_i = 1.f - cfrac_i;
				const float rhoinv_i = 1.f / rho_i;					// 4
				array<float, NCHEMFRACS> frac_i;
				if ((params.diffusion && params.phase == 1) && data.chemistry) {
					frac_i = data.chem[i];
				}
				if (params.diffusion && params.phase == 1) {
					const float shearv_i = data.shearv[i];
					difco_i = SPH_DIFFUSION_C * sqr(h_i) * shearv_i;					// 3
				}
#ifdef HOPKINS
				const float& pre_i = data.pre[i];
#else
#ifdef ENTROPY
				const float pre_i = pressure(A_i, rho_i, gamma0);
#else
				const float pre_i = pressure(eint_i, rho_i, gamma0);
#endif
#endif
#ifdef ENTROPY
				const float eint = A_i * powf(rho_i, gamma0 - 1.f) / (gamma0 - 1.f);
				const float de_dt0 = A_i / eint;	// 12
				const float c_i = sqrtf(gamma0 * powf(A_i, 1.0f / gamma0) * powf(pre_i, 1.f- 1.f/gamma0) / hfrac_i);// 22
#else
				const float de_dt0 = 1.f;	// 12
				const float c_i = sqrtf(gamma0 * (gamma0 - 1.f) * eint_i);	// 22
#endif
				flops += 55;
				float ax = 0.f;
				float ay = 0.f;
				float az = 0.f;
				float de_dt1 = 0.f;
				float de_dt2 = 0.f;
				float dcm_dt = 0.f;
				array<float, NCHEMFRACS> dfrac_dt;
				if (data.chemistry) {
					for (int fi = 0; fi < NCHEMFRACS; fi++) {
						dfrac_dt[fi] = 0.f;
					}
				}
				float dtinv_cfl = 0.f;
				float dtinv_visc = 0.f;
//				float one = 0.0f;
				const float& adot = params.adot;
				float Dd = 0.f;
				ws.neighbors.resize(0);
				const float jmax = round_up(ws.rec1.size(), block_size);
				for (int j = tid; j < jmax; j += block_size) {
					int k;
					int total;
					bool use = false;
					if (j < ws.rec1.size()) {
						const auto& rec1 = ws.rec1[j];
						const fixed32& x_j = rec1.x;
						const fixed32& y_j = rec1.y;
						const fixed32& z_j = rec1.z;
						const float& h_j = rec1.h;
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
					compute_indices < HYDRO_BLOCK_SIZE > (k, total);
					const int offset = ws.neighbors.size();
					__syncthreads();
					const int next_size = offset + total;
					ws.neighbors.resize(next_size);
					if (use) {
						const int l = offset + k;
						ws.neighbors[l] = j;
					}
				}
				float dvx_dx = 0.f;
				float dvx_dy = 0.f;
				float dvx_dz = 0.f;
				float dvy_dx = 0.f;
				float dvy_dy = 0.f;
				float dvy_dz = 0.f;
				float dvz_dx = 0.f;
				float dvz_dy = 0.f;
				float dvz_dz = 0.f;
				float vsig = 0.f;
				__syncthreads();
				if (!star_i) {
					for (int j = tid; j < ws.neighbors.size(); j += block_size) {



						const int kk = ws.neighbors[j];
						const auto& rec1 = ws.rec1[kk];
						const auto& rec2 = ws.rec2[kk];
						const auto& star_j = rec2.star;
						if (!star_j) {
							const float& h_j = rec1.h;
							const fixed32& x_j = rec1.x;
							const fixed32& y_j = rec1.y;
							const fixed32& z_j = rec1.z;
							const float& cfrac_j = rec2.cold_frac;
							const float& shearv_j = rec2.shearv;
							const auto& frac_j = rec2.chem;
							const float& vx_j = rec2.vx;
							const float& vy_j = rec2.vy;
							const float& vz_j = rec2.vz;
							const float& omega_j = rec2.omega;
							const float& omegaP_j = rec2.omegaP;
							const float& alpha_j = rec2.alpha;
							const float& rho_j = rec2.rho;												// 2
#ifdef ENTROPY
							const float& A_j = rec2.entr;
#else
							const float& eint_j = rec2.eint;
#endif
#ifdef HOPKINS
							const float& pre_j = rec2.pre;
#ifdef ENTROPY
							const float f_ij = (1.f - smoothX(h_j, params.hmin, params.hmax) * powf(A_j, -1.0f / gamma0) / omegaP_i);
							const float f_ji = (1.f - smoothX(h_i, params.hmin, params.hmax) * powf(A_i, -1.0f / gamma0) / omegaP_j);
#else
							const float f_ij = (1.f - smoothX(h_j, params.hmin, params.hmax) / ((gamma0 - 1.0f) * eint_j * omegaP_i));
							const float f_ji = (1.f - smoothX(h_i, params.hmin, params.hmax) / ((gamma0 - 1.0f) * eint_i * omegaP_j));
#endif
#else
#ifdef ENTROPY
							const float pre_j = pressure(A_j, rho_j, gamma0);
#else
							const float pre_j = pressure(eint_j, rho_j, gamma0);
#endif
#endif
							const float hfrac_j = 1.f - cfrac_j;
							const float hinv_j = 1.f / h_j;															// 4
							const float h3inv_j = sqr(hinv_j) * hinv_j;										// 3
							const float x_ij = distance(x_i, x_j);													// 1
							const float y_ij = distance(y_i, y_j);													// 1
							const float z_ij = distance(z_i, z_j);													// 1
							const float r2 = sqr(x_ij, y_ij, z_ij);												// 5
							const float r = sqrtf(r2);																   // 4
							const float q_i = r * hinv_i;																// 1
							const float q_j = r * hinv_j;																// 1
#ifdef ENTROPY
									const float c_j = sqrtf(gamma0 * powf(A_j, 1.0f / gamma0) * powf(pre_j, 1.f - 1.f/gamma0)); //23
#else
							const float c_j = sqrtf(gamma0 * (gamma0 - 1.f) * eint_j); //23
#endif
							const float vx0_ij = vx_i - vx_j;									// 3
							const float vy0_ij = vy_i - vy_j;									// 3
							const float vz0_ij = vz_i - vz_j;									// 3
							const float vx_ij = vx0_ij + x_ij * adot;									// 3
							const float vy_ij = vy0_ij + y_ij * adot;									// 3
							const float vz_ij = vz0_ij + z_ij * adot;									// 3
							const float rinv = 1.0f / (r > 0.f ? r : 1e37f);								// 5
							const float vdotx_ij = fmaf(x_ij, vx_ij, fmaf(y_ij, vy_ij, z_ij * vz_ij));								// 5
							const float h_ij = 0.5f * (h_i + h_j);												// 2
							const float w_ij = fminf(vdotx_ij * rinv, 0.f);									// 2
							const float mu_ij = w_ij * h_ij / sqrtf(r2 + ETA * h_ij);			// 12
							const float rho_ij = 0.5f * (rho_i + rho_j);										// 2
							const float c_ij = 0.5f * (c_i + c_j);												// 2
							const float hfrac_ij = 2.0f * hfrac_i * hfrac_j / (hfrac_i + hfrac_j + 1e-36f);
							const float alpha_ij = 0.5f * (alpha_i + alpha_j);
							const float vsig_ij = alpha_ij * (c_ij - params.beta * mu_ij); // 4
							const float dWdr_i = dkernelW_dq(q_i) * hinv_i * h3inv_i;   // 2
							const float dWdr_j = dkernelW_dq(q_j) * hinv_j * h3inv_j;   // 2
							//				ALWAYS_ASSERT(omega_i > 0.f);
							if (omega_j <= 0.0) {
								PRINT("%i %e %e %e\n", data.selfs[index], omega_j, q_i, q_j);
							}
							ALWAYS_ASSERT(omega_j > 0.f);
							const float dWdr_ij = 0.5f * (dWdr_i + dWdr_j);     // 4
							const float dWdr_i_rinv = dWdr_i * rinv;						// 2
							const float dWdr_j_rinv = dWdr_j * rinv;						// 2
							const float dWdr_ij_rinv = dWdr_ij * rinv;										// 1
							const float dWdr_x_ij = x_ij * dWdr_ij_rinv;										// 1
							const float dWdr_y_ij = y_ij * dWdr_ij_rinv;										// 1
							const float dWdr_z_ij = z_ij * dWdr_ij_rinv;										// 1
							const float dWdr_x_i = dWdr_i_rinv * x_ij;								// 1
							const float dWdr_y_i = dWdr_i_rinv * y_ij;								// 1
							const float dWdr_z_i = dWdr_i_rinv * z_ij;								// 1
							const float dWdr_x_j = dWdr_j_rinv * x_ij;								// 1
							const float dWdr_y_j = dWdr_j_rinv * y_ij;								// 1
							const float dWdr_z_j = dWdr_j_rinv * z_ij;								// 1
							const float mainv = m * ainv;															// 1
#ifdef HOPKINS
#ifdef ENTROPY
							const float dp_i = f_ij * mainv * powf(A_i * A_j, 1.f / gamma0) * powf(pre_i, 1.f - 2.f / gamma0);
							const float dp_j = f_ji * mainv * powf(A_j * A_i, 1.f / gamma0) * powf(pre_j, 1.f - 2.f / gamma0);
#else
							const float dp_i = f_ij * mainv * sqr(gamma0 - 1.f) * eint_i * eint_j / pre_i;
							const float dp_j = f_ji * mainv * sqr(gamma0 - 1.f) * eint_i * eint_j / pre_j;
#endif
#else
							const float dp_i = mainv * pre_i / sqr(rho_i) / omega_i;
							const float dp_j = mainv * pre_j / sqr(rho_j) / omega_j;
#endif
							const float pi_ij = -mainv * w_ij * h_ij * rinv * vsig_ij / rho_ij;     // 9
							ax -= dp_i * dWdr_x_i + dp_j * dWdr_x_j;											// 4
							ay -= dp_i * dWdr_y_i + dp_j * dWdr_y_j;											// 4
							az -= dp_i * dWdr_z_i + dp_j * dWdr_z_j;											// 4
							ax -= pi_ij * dWdr_x_ij;																// 2
							ay -= pi_ij * dWdr_y_ij;																// 2
							az -= pi_ij * dWdr_z_ij;																// 2
							const float vdW_ij = fmaf(vx0_ij, dWdr_x_ij, fmaf(vy0_ij, dWdr_y_ij, vz0_ij * dWdr_z_ij));														// 5
							const float dt = rung_dt[rung_i] * 0.5f * params.t0;
							//			const float adW_ij = fmaf(adot * x_ij, dWdr_x_ij, fmaf(adot * y_ij, dWdr_y_ij, adot * z_ij * dWdr_z_ij));
							de_dt2 = fmaf(de_dt0, 0.5f * pi_ij * (vdW_ij - adot), de_dt2);                     // 4
							visc = fmaf(0.5f, -pi_ij * adot * dt, visc);                     // 4
#ifndef ENTROPY
							const float v0dW_i = fmaf(vx0_ij, dWdr_x_i, fmaf(vy0_ij, dWdr_y_i, vz0_ij * dWdr_z_i));
#ifdef HOPKINS// 5
							de_dt2 = fmaf(mainv * f_ij * v0dW_i, sqr(gamma0 - 1.f) * eint_i * eint_j / pre_i, de_dt2);													// 4
#else
									de_dt2 = fmaf(mainv* v0dW_i, (gamma0 - 1.f) * eint_i * rhoinv_i / omega_i, de_dt2);
#endif
#endif
							dvx_dx -= m * rhoinv_i * vx_ij * dWdr_x_i / omega_i;									// 2
							dvy_dy -= m * rhoinv_i * vy_ij * dWdr_y_i / omega_i;									// 2
							dvz_dz -= m * rhoinv_i * vz_ij * dWdr_z_i / omega_i;									// 2
							vsig = fmaxf(vsig, -w_ij);												// 1
							if (params.phase == 0) {
								dvy_dx -= m * rhoinv_i * vy_ij * dWdr_x_i / omega_i;									// 2
								dvz_dx -= m * rhoinv_i * vz_ij * dWdr_x_i / omega_i;									// 2
								dvx_dy -= m * rhoinv_i * vx_ij * dWdr_y_i / omega_i;									// 2
								dvz_dy -= m * rhoinv_i * vz_ij * dWdr_y_i / omega_i;									// 2
								dvx_dz -= m * rhoinv_i * vx_ij * dWdr_z_i / omega_i;									// 2
								dvy_dz -= m * rhoinv_i * vy_ij * dWdr_z_i / omega_i;									// 2
							} else if (params.phase == 1) {
								const float h = fminf(h_i, h_j * (1 << MAX_RUNG_DIF));                  // 3
								dtinv_cfl = fmaxf(dtinv_cfl, c_ij / h);												// 1
								dtinv_visc = fmaxf(dtinv_visc, 0.6f * vsig_ij / h);
								if (params.diffusion) {
									flops += 29;
									float dif_max = 0.f;
									const float difco_j = SPH_DIFFUSION_C * sqr(h_j) * shearv_j;			// 3
									const float difco_ij = 0.5f * (difco_i + difco_j);							// 2
									const float D_ij = -2.f * m / rho_ij * difco_ij * dWdr_ij * rinv;		// 7
									ALWAYS_ASSERT(D_ij >= 0.f);
#ifdef ENTROPY
									const float A0_j = A_j * powf(rho_j * rhoinv_i / hfrac_i * hfrac_j, gamma0 - 1.f);
									dif_max = fabs(A0_j - A_i) / fmaxf(A0_j, A_i);
									de_dt1 -= D_ij * (A_i - A0_j); // 16;
#else
									dif_max = fabs(eint_j - eint_i) / fmaxf(eint_j, eint_i);
									de_dt1 -= D_ij * (eint_i - eint_j); // 16;
#endif
									if (params.stars) {
										dcm_dt -= D_ij * (cfrac_i - cfrac_j);										// 3
										dif_max = fmaxf(dif_max, fabs(cfrac_j - cfrac_i) / fmaxf(cfrac_j, cfrac_i));
										flops += 3;
									}
									if (data.chemistry) {
										for (int fi = 0; fi < NCHEMFRACS; fi++) {
											ALWAYS_ASSERT(frac_j[fi] >= 0.f);
											dfrac_dt[fi] -= D_ij * (frac_i[fi] - frac_j[fi]);										// 5
											dif_max = fmaxf(dif_max, fabs(frac_i[fi] - frac_j[fi]) / fmaxf(frac_i[fi], frac_j[fi]));
											flops += 5;
										}
									}
									Dd += D_ij * dif_max;																				// 1
								}
							}
						}
					}
				}
				if (params.phase == 0) {
					shared_reduce_add<float, HYDRO_BLOCK_SIZE>(dvx_dy); // 127
					shared_reduce_add<float, HYDRO_BLOCK_SIZE>(dvx_dz); // 127
					shared_reduce_add<float, HYDRO_BLOCK_SIZE>(dvy_dz); // 127
					shared_reduce_add<float, HYDRO_BLOCK_SIZE>(dvz_dx); // 127
					shared_reduce_add<float, HYDRO_BLOCK_SIZE>(dvz_dy); // 127
					shared_reduce_add<float, HYDRO_BLOCK_SIZE>(dvy_dx); // 127
				} else if (params.phase == 1) {
					shared_reduce_max < HYDRO_BLOCK_SIZE > (dtinv_cfl);    // 31
					shared_reduce_max < HYDRO_BLOCK_SIZE > (dtinv_visc);    // 31
					shared_reduce_add<float, HYDRO_BLOCK_SIZE>(de_dt1); // 31
				}
				shared_reduce_max < HYDRO_BLOCK_SIZE > (vsig);    // 31
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(dvx_dx); // 127
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(dvy_dy); // 127
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(dvz_dz); // 127
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(de_dt2); // 31
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(ax);    // 31
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(ay);    // 31
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(az);    // 31
				if (params.diffusion && params.phase == 1) {
					shared_reduce_add<float, HYDRO_BLOCK_SIZE>(Dd);
					if (params.stars) {
						shared_reduce_add<float, HYDRO_BLOCK_SIZE>(dcm_dt); //31
					}
					if (data.chemistry) {
						for (int fi = 0; fi < NCHEMFRACS; fi++) {
							shared_reduce_add<float, HYDRO_BLOCK_SIZE>(dfrac_dt[fi]); // 31
						}
					}
				}
				visc *= data.m;
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(visc);
				if (tid == 0) {
					atomicAdd(&reduce->visc, visc);
					data.rec6_snk[snki].dvel[XDIM] = ax;
					data.rec6_snk[snki].dvel[YDIM] = ay;
					data.rec6_snk[snki].dvel[ZDIM] = az;
#ifdef ENTROPY
					data.dentr1_snk[snki] = de_dt1;
					data.dentr2_snk[snki] = de_dt2;
#else
					data.deint1_snk[snki] = de_dt1;
					data.deint2_snk[snki] = de_dt2;
#endif
					if (params.stars) {
						data.rec5_snk[snki].dfcold = dcm_dt;
					}
					if (data.chemistry) {
						data.rec5_snk[snki].dfrac = dfrac_dt;
					}
					if (params.phase == 0 && params.tau > 0.f) {
						float div_v, curl_vx, curl_vy, curl_vz;
						div_v = dvx_dx + dvy_dy + dvz_dz;                        // 2
						curl_vx = dvz_dy - dvy_dz;										   // 1
						curl_vy = -dvz_dx + dvx_dz;                              // 2
						curl_vz = dvy_dx - dvx_dy;
						char& rung = data.sph_rungs_snk[snki];
						const float dt1 = params.t0 * rung_dt[rung];
						const float curlv = sqrtf(sqr(curl_vx, curl_vy, curl_vz));                             // 9
						const float div_v0 = data.divv_snk[snki];
						float& alpha = data.rec1_snk[snki].alpha;
						const float ddivv_dt = (div_v - div_v0) / dt1 - params.adot * ainv * div_v;             // 8
						const float limiter = sqr(div_v) / (sqr(div_v) + sqr(curlv) + 1.0e-4f * sqr(c_i / h_i * ainv)); // 14
						const float S = limiter * sqr(h_i) * fmaxf(0.f, -ddivv_dt) * sqr(params.a);                      // 6
						vsig += c_i;
						const float alpha_targ = params.alpha0 + (params.alpha1 - params.alpha0) * S / (S + sqr(c_i));     // 8
						const float lambda0 = params.alpha_decay * vsig * hinv_i * ainv;                       // 3
						const float lambda1 = vsig * hinv_i * ainv / params.cfl;                       // 3
						if (alpha < alpha_targ) {                                                    // 2
							alpha = (alpha + lambda1 * dt1 * alpha_targ) / (1.f + lambda1 * dt1);
							flops++;
						} else {
							alpha = (alpha + lambda0 * dt1 * alpha_targ) / (1.f + lambda0 * dt1);
							flops += 10;
						}
					} else if (params.phase == 1) {
						float div_v;
						div_v = dvx_dx + dvy_dy + dvz_dz;                        // 2
						data.divv_snk[snki] = div_v;
						float dtinv_hydro = 0.f;
						const float h_i = data.rec2_snk[snki].h;
						const float dloghdt = fabsf(div_v - 3.f * params.adot * ainv) / (3.f + dlogsmoothX_dlogh(h_i, params.hmin, params.hmax));           // 5
						const float dtinv_divv = params.a * (dloghdt);
#ifndef ENTROPY
						const float dtinv_eint = (1.01f * params.cfl) * params.a * fmaxf(-de_dt2 - de_dt1, 0.f) / (eint_i + 1e-37f);           // 5
#endif
						float dtinv_diff = 0.0f;
						if (params.diffusion) {
							dtinv_diff = params.a * Dd * (1.0f / 3.0f);
						}
						const float a2 = sqr(ax, ay, az);											// 5
						float dtinv_acc = sqrtf(sqrtf(a2) * hinv_i) * params.a * params.cfl / (data.eta * sqrtf(params.a));	// 11
						dtinv_hydro = fmaxf(dtinv_hydro, dtinv_divv);							// 1
#ifndef ENTROPY
						dtinv_hydro = fmaxf(dtinv_hydro, dtinv_eint);							// 1
#endif
						dtinv_hydro = fmaxf(dtinv_hydro, dtinv_cfl + dtinv_visc);							// 1
						dtinv_hydro = fmaxf(dtinv_hydro, dtinv_diff);							// 1
						dtinv_hydro = fmaxf(dtinv_hydro, dtinv_acc);							// 1
						float dthydro = params.cfl * params.a / dtinv_hydro;	// 6
						dthydro = fminf(dthydro, params.max_dt);									// 1
						total_vsig_max = fmaxf(total_vsig_max, dtinv_hydro * h_i);			// 2
						char& rung = data.sph_rungs_snk[snki];
						const float last_dt = rung_dt[rung] * params.t0;						// 1
						const int rung_hydro = ceilf(log2fparamst0 - log2f(dthydro));     // 10
						max_rung_hydro = max(max_rung_hydro, rung_hydro);
						if (data.gravity) {
							const char& rung_grav = data.part_rungs_snk[data.dm_index_snk[snki]];
							max_rung_grav = rung_grav;
							rung = max(max((int) max(rung_hydro, rung_grav), max(params.min_rung, (int) rung - 1)), 1);
							ALWAYS_ASSERT(rung >= rung_grav);
						} else {
							rung = max(max((int) (rung_hydro), max(params.min_rung, (int) rung - 1)), 1);
						}
						max_rung = max(max_rung, rung);
						atomicMax(&reduce->dtinv_cfl, dtinv_cfl);
						atomicMax(&reduce->dtinv_visc, dtinv_visc);
						atomicMax(&reduce->dtinv_diff, dtinv_diff);
						atomicMax(&reduce->dtinv_acc, dtinv_acc);
						atomicMax(&reduce->dtinv_divv, dtinv_divv);
						if (rung < 0 || rung > 10) {
							if (tid == 0) {
								PRINT("Rung out of range %i %e %e %e %e |%e %e %e %e %e\n", rung_hydro, vsig, cfrac_i, h_i, c_i, dtinv_cfl, dtinv_visc, dtinv_diff,
										dtinv_acc, dtinv_divv);
								//		__trap();
							}
						}
					}
				}
				visc = 0.f;
			}
		}
		shared_reduce_add<int, HYDRO_BLOCK_SIZE>(flops);
		if (tid == 0) {
			atomicAdd(&reduce->flops, (float) flops);
			index = atomicAdd(&reduce->counter, 1);
		}
		flops = 0;
		__syncthreads();
	}
	if (tid == 0) {
		atomicMax(&reduce->vsig_max, total_vsig_max);
		atomicMax(&reduce->max_rung, max_rung);
		atomicMax(&reduce->max_rung_hydro, max_rung_hydro);
		atomicMax(&reduce->max_rung_grav, max_rung_grav);
	}
	(&ws)->~hydro_workspace();
}
