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
	float entr;
	float alpha;
	float omega;
	float shearv;
	float cold_frac;
	float kappa;
};

struct hydro_workspace {
	device_vector<hydro_record1> rec1;
	device_vector<hydro_record2> rec2;
	device_vector<int> neighbors;
};

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
					ws.rec2[k].entr = data.entr[pi];
					ws.rec2[k].alpha = data.alpha[pi];
					ws.rec2[k].omega = data.omega[pi];
					if (params.diffusion) {
						ws.rec2[k].shearv = data.shearv[pi];
					}
#ifndef IMPLICIT_CONDUCTION
					if (params.conduction) {
						ws.rec2[k].kappa = data.kappa[pi];
					}
#endif
					if (params.stars) {
						ws.rec2[k].cold_frac = data.cold_frac[pi];
					} else {
						ws.rec2[k].cold_frac = 0.f;
					}
					if (params.diffusion) {
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
			int rung_i = data.rungs_snk[data.dm_index_snk[snki]];
			bool use = rung_i >= params.min_rung;
			const float& m = data.m;
			const float minv = 1.f / m;
			const float c0 = float(3.0f / 4.0f / M_PI * data.N);
			const float c0inv = 1.0f / c0;
			if (use) {
				float difco_i;
				float cfrac_i;
				const auto& x_i = data.x[i];
				const auto& y_i = data.y[i];
				const auto& z_i = data.z[i];
				const float& omega_i = data.omega[i];
				const auto& vx_i = data.vx[i];
				const auto& vy_i = data.vy[i];
				const auto& vz_i = data.vz[i];
				const float& h_i = data.h[i];
				const float& A_i = data.entr[i];
				const float& alpha_i = fmaxf(data.alpha[i], params.alpha0);
				const float hinv_i = 1.f / h_i;												// 4
				const float h3inv_i = (sqr(hinv_i) * hinv_i);							// 3
				const float rho_i = m * c0 * h3inv_i;										// 2
				const float rhoinv_i = minv * c0inv * sqr(h_i) * h_i;					// 4
				const float pre_i = A_i * powf(rho_i, gamma0);
				if (params.stars) {
					cfrac_i = data.cold_frac[i];
				} else {
					cfrac_i = 0.f;
				}
				array<float, NCHEMFRACS> frac_i;
				if ((params.diffusion) && data.chemistry) {
					frac_i = data.chem[i];
				}
				if (params.diffusion) {
					const float shearv_i = data.shearv[i];
					difco_i = SPH_DIFFUSION_C * sqr(h_i) * shearv_i;					// 3
				}
				const float de_dt0 = (gamma0 - 1.f) * powf(rho_i, 1.f - gamma0);	// 12
				const float c_i = sqrtf(gamma0 * pre_i * rhoinv_i); // 22
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
#ifndef IMPLICIT_CONDUCTION
				float Dc = 0.f;
#endif
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
				__syncthreads();
				for (int j = tid; j < ws.neighbors.size(); j += block_size) {
					const int kk = ws.neighbors[j];
					const auto& rec1 = ws.rec1[kk];
					const auto& rec2 = ws.rec2[kk];
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
					const float& A_j = rec2.entr;
					const float& alpha_j = fmaxf(rec2.alpha, params.alpha0);
					const float hinv_j = 1.f / h_j;															// 4
					const float h3inv_j = sqr(hinv_j) * hinv_j;										// 3
					const float x_ij = distance(x_i, x_j);													// 1
					const float y_ij = distance(y_i, y_j);													// 1
					const float z_ij = distance(z_i, z_j);													// 1
					const float r2 = sqr(x_ij, y_ij, z_ij);												// 5
					const float r = sqrtf(r2);																   // 4
					const float q_i = r * hinv_i;																// 1
					const float q_j = r * hinv_j;																// 1
					const float rho_j = m * c0 * h3inv_j;												// 2
					const float pre_j = A_j * powf(rho_j, gamma0);
					const float c_j = sqrtf(gamma0 * pre_j / rho_j); //23
					const float vx_ij = vx_i - vx_j + x_ij * adot;									// 3
					const float vy_ij = vy_i - vy_j + y_ij * adot;									// 3
					const float vz_ij = vz_i - vz_j + z_ij * adot;									// 3
					const float rinv = 1.0f / (r > 0.f ? r : 1e37f);								// 5
					const float vdotx_ij = fmaf(x_ij, vx_ij, fmaf(y_ij, vy_ij, z_ij * vz_ij));								// 5
					const float h_ij = 0.5f * (h_i + h_j);												// 2
					const float w_ij = fminf(vdotx_ij * rinv, 0.f);									// 2
					const float mu_ij = fminf(vdotx_ij, 0.f) * h_ij * rinv * rinv;			// 12
					const float rho_ij = 0.5f * (rho_i + rho_j);										// 2
					const float c_ij = 0.5f * (c_i + c_j);												// 2
					const float alpha_ij = 0.5f * (alpha_i + alpha_j);								// 2
					const float vsig_ij = alpha_ij * (c_ij - params.beta * w_ij); // 4
					float w;
					const float dWdr_i = dkernelW_dq(q_i, &w, &flops) * hinv_i * h3inv_i / omega_i;   // 2
					const float dWdr_j = dkernelW_dq(q_j, &w, &flops) * hinv_j * h3inv_j / omega_j;   // 2
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
					const float dp_i = mainv * pre_i / sqr(rho_i);
					const float dp_j = mainv * pre_j / sqr(rho_j);
					const float pi_ij = -mainv * mu_ij * vsig_ij / rho_ij;     // 9
					ax -= dp_i * dWdr_x_i + dp_j * dWdr_x_j;											// 4
					ay -= dp_i * dWdr_y_i + dp_j * dWdr_y_j;											// 4
					az -= dp_i * dWdr_z_i + dp_j * dWdr_z_j;											// 4
					ax -= pi_ij * dWdr_x_ij;																// 2
					ay -= pi_ij * dWdr_y_ij;																// 2
					az -= pi_ij * dWdr_z_ij;																// 2
					const float vdW_ij = fmaf(vx_ij, dWdr_x_ij, fmaf(vy_ij, dWdr_y_ij, vz_ij * dWdr_z_ij));																// 5
					de_dt2 = fmaf(de_dt0, 0.5f * pi_ij * vdW_ij, de_dt2);                     // 4
					const float h = fminf(h_i, h_j * (1 << MAX_RUNG_DIF));                  // 3
					dtinv_cfl = fmaxf(dtinv_cfl, c_ij / h);												// 1
					dtinv_visc = fmaxf(dtinv_visc, 0.6f * vsig_ij / h);												// 1
					flops += 206;
					if (params.diffusion) {
						flops += 29;
						float dif_max = 0.f;
						const float difco_j = SPH_DIFFUSION_C * sqr(h_j) * shearv_j;			// 3
						const float difco_ij = 0.5f * (difco_i + difco_j);							// 2
						const float D_ij = -2.f * m / rho_ij * difco_ij * dWdr_ij * rinv;		// 7
						ALWAYS_ASSERT(D_ij >= 0.f);
						const float A0_j = A_j * powf(rho_j * rhoinv_i, gamma0 - 1.f);
						dif_max = fabs(A0_j - A_i) / fmaxf(A0_j, A_i);
						de_dt1 -= D_ij * (A_i - A0_j); // 16;
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
#ifndef IMPLICIT_CONDUCTION
					if (params.conduction) {
						const float& kappa_i = data.kappa[i];
						const float& kappa_j = rec2.kappa;
						const float kappa_ij = 2.f * kappa_i * kappa_j / (kappa_i + kappa_j + 1.0e-35f); // 8
						const float D_ij = -2.f * sqr(params.a) * m * kappa_ij * dWdr_ij / (rho_i * rho_j) * r / (r2 + ETA * h_i * h_j);// 10
						ALWAYS_ASSERT(D_ij >= 0.f);
						const float A0_j = A_j * powf(rho_j * rhoinv_i, gamma0 - 1.f);
						Dc += D_ij * fabs((A_i - A0_j)) / fmaxf(A_i, A0_j);// 1
						de_dt1 -= D_ij * (A_i - A0_j);// 16;
					}
#endif
				}
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(de_dt1); // 31
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(de_dt2); // 31
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(ax);    // 31
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(ay);    // 31
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(az);    // 31
//				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(one);
				shared_reduce_max < HYDRO_BLOCK_SIZE > (dtinv_cfl);    // 31
				shared_reduce_max < HYDRO_BLOCK_SIZE > (dtinv_visc);    // 31
				if (tid == 0) {
					flops += 5 * (HYDRO_BLOCK_SIZE - 1);
				}
				if (params.diffusion) {
					shared_reduce_add<float, HYDRO_BLOCK_SIZE>(Dd);
					if (params.stars) {
						shared_reduce_add<float, HYDRO_BLOCK_SIZE>(dcm_dt); //31
						if (tid == 0) {
							flops += (HYDRO_BLOCK_SIZE - 1);
						}
					}
					if (data.chemistry) {
						for (int fi = 0; fi < NCHEMFRACS; fi++) {
							shared_reduce_add<float, HYDRO_BLOCK_SIZE>(dfrac_dt[fi]); // 31
							if (tid == 0) {
								flops += (HYDRO_BLOCK_SIZE - 1);
							}
						}
					}
				}
#ifndef IMPLICIT_CONDUCTION
				if (params.conduction) {
					shared_reduce_add<float, HYDRO_BLOCK_SIZE>(Dc);
				}
#endif
				/*				if (fabs(1. - one) > 1.0e-4 && tid == 0) {
				 PRINT("one is off %e %i\n", one, data.converged_snk[snki]);
				 __trap();
				 }*/
				if (tid == 0) {
					float gx_i;
					float gy_i;
					float gz_i;
					if (data.gravity) {
						gx_i = data.gx_snk[snki];
						gy_i = data.gy_snk[snki];
						gz_i = data.gz_snk[snki];
					} else {
						gx_i = 0.f;
						gy_i = 0.f;
						gz_i = 0.f;
					}
					ax += gx_i;																			// 1
					ay += gy_i;																			// 1
					az += gz_i;																			// 1
					data.rec6_snk[snki].dvel[XDIM] = ax;
					data.rec6_snk[snki].dvel[YDIM] = ay;
					data.rec6_snk[snki].dvel[ZDIM] = az;
					data.dentr1_snk[snki] = de_dt1;
					data.dentr2_snk[snki] = de_dt2;
					if (params.stars) {
						data.rec5_snk[snki].dfcold = dcm_dt;
					}
					if (data.chemistry) {
						data.rec5_snk[snki].dfrac = dfrac_dt;
					}
//					const float div_v = data.rec3_snk[snki].divv;
//					const float dtinv_divv = params.a * fabsf(div_v - 3.f * params.adot * ainv) * (1.f / 3.f);
					float dtinv_hydro = 0.f;
//					dtinv_hydro1 = fmaxf(dtinv_hydro1, dtinv_divv);
					float dtinv_diff = 0.0f, dtinv_cond = 0.f;
					if (params.diffusion) {
						dtinv_diff = params.a * Dd * (1.0f / 3.0f);
					}
#ifndef IMPLICIT_CONDUCTION
					if (params.conduction) {
						dtinv_cond = params.a * Dc * (1.0f / 3.0f);
					}
#endif
					const float a2 = sqr(ax, ay, az);											// 5
					float dtinv_acc = sqrtf(sqrtf(a2) * hinv_i) * params.a * params.cfl / data.eta * sqrtf(params.a);	// 11
					dtinv_hydro = fmaxf(dtinv_hydro, dtinv_cfl + dtinv_visc);							// 1
					dtinv_hydro = fmaxf(dtinv_hydro, dtinv_diff + dtinv_cond);							// 1
					dtinv_hydro = fmaxf(dtinv_hydro, dtinv_acc);							// 1
					float dthydro = params.cfl * params.a / dtinv_hydro;	// 6
					dthydro = fminf(data.eta * sqrtf(params.a) / (dtinv_acc + 1e-30f), dthydro);	// 11
					const float g2 = sqr(gx_i, gy_i, gz_i);									// 5
					const float dtinv_grav = sqrtf(sqrtf(g2));								// 8
					float dtgrav = data.eta * sqrtf(params.a * data.gsoft) / (dtinv_grav + 1e-30f); // 11
					dthydro = fminf(dthydro, params.max_dt);									// 1
					dtgrav = fminf(dtgrav, params.max_dt);										// 1
					total_vsig_max = fmaxf(total_vsig_max, dtinv_hydro * h_i);			// 2
					char& rung = data.rungs_snk[data.dm_index_snk[snki]];
					const float last_dt = rung_dt[rung] * params.t0;						// 1
					int orung;
					if (params.phase == 0) {
						data.oldrung_snk[snki] = rung;
						orung = rung;
					} else {
						orung = data.oldrung_snk[snki];
					}
					const int rung_hydro = ceilf(log2fparamst0 - log2f(dthydro));     // 10
					const int rung_grav = ceilf(log2fparamst0 - log2f(dtgrav));       // 10
					max_rung_hydro = max(max_rung_hydro, rung_hydro);
					max_rung_grav = max(max_rung_grav, rung_grav);
					rung = max(max((int) max(rung_hydro, rung_grav), max(params.min_rung, (int) orung - 1)), 1);
					max_rung = max(max_rung, rung);
					flops += 87;
					if (rung < 0 || rung >= MAX_RUNG) {
						if (tid == 0) {
							PRINT("Rung out of range \n");
							__trap();
						}
					}
					atomicMax(&reduce->dtinv_cfl, dtinv_cfl);
					atomicMax(&reduce->dtinv_visc, dtinv_visc);
					atomicMax(&reduce->dtinv_diff, dtinv_diff);
					atomicMax(&reduce->dtinv_acc, dtinv_acc);
					atomicMax(&reduce->dtinv_cond, dtinv_cond);
				}
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
