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

#include <cosmictiger/ewald_indices.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/safe_io.hpp>

#include <cosmictiger/fmm_kernels.hpp>

#include <algorithm>
ewald_constants ec;
__managed__ ewald_constants ec_dev;

void ewald_compute(float& pot, float& fx, float& fy, float& fz, float dx0, float dx1, float dx2) {
#ifndef TREEPM
	const float cons1 = float(4.0f / sqrtf(M_PI));
	fx = 0.0;
	fy = 0.0;
	fz = 0.0;
	pot = 0.0;
	const auto r2 = sqr(dx0, dx1, dx2);  // 5

	if (r2 > 0.f) {
		const float dx = dx0;
		const float dy = dx1;
		const float dz = dx2;
		const float r2 = sqr(dx, dy, dz);
		const float r = sqrt(r2);
		const float rinv = 1.f / r;
		const float r2inv = rinv * rinv;
		const float r3inv = r2inv * rinv;
		float exp0 = exp(-4.0f * r2);
		float erf0 = erf(2.0f * r);
		const float expfactor = cons1 * r * exp0;
		const float d0 = erf0 * rinv;
		const float d1 = (expfactor - erf0) * r3inv;
		pot += d0;
		fx -= dx * d1;
		fy -= dy * d1;
		fz -= dz * d1;
		for (int xi = -3; xi <= +3; xi++) {
			for (int yi = -3; yi <= +3; yi++) {
				for (int zi = -3; zi <= +3; zi++) {
					const bool center = sqr(xi, yi, zi) == 0;
					if (center) {
						continue;
					}
					const float dx = dx0 - xi;
					const float dy = dx1 - yi;
					const float dz = dx2 - zi;
					const float r2 = sqr(dx, dy, dz);
					if (r2 < 2.6f * 2.6f) {
						const float r = sqrt(r2);
						const float rinv = 1.f / r;
						const float r2inv = rinv * rinv;
						const float r3inv = r2inv * rinv;
						float exp0 = exp(-4.0f * r2);
						float erfc0 = erfc(2.0f * r);
						const float expfactor = cons1 * r * exp0;
						const float d0 = -erfc0 * rinv;
						const float d1 = (expfactor + erfc0) * r3inv;
						pot += d0;
						fx -= dx * d1;
						fy -= dy * d1;
						fz -= dz * d1;
					}
				}
			}
		}
		pot += float(M_PI / 4.f);
		for (int xi = -2; xi <= +2; xi++) {
			for (int yi = -2; yi <= +2; yi++) {
				for (int zi = -2; zi <= +2; zi++) {
					const float hx = xi;
					const float hy = yi;
					const float hz = zi;
					const float h2 = sqr(hx, hy, hz);
					if (h2 > 0.0f && h2 <= 8) {
						const float hdotx = dx0 * hx + dx1 * hy + dx2 * hz;
						const float omega = float(2.0 * M_PI) * hdotx;
						float c, s;
						sincosf(omega, &s, &c);
						const float c0 = -1.0f / h2 * expf(float(-M_PI * M_PI * 0.25f) * h2) * float(1.f / M_PI);
						const float c1 = -s * 2.0 * M_PI * c0;
						pot += c0 * c;
						fx -= c1 * hx;
						fy -= c1 * hy;
						fz -= c1 * hz;
					}
				}
			}
		}
	} else {
		pot += 2.837291f;
	}
#endif
}

void ewald_const::init_gpu() {
#ifndef TREEPM
	double dx = 0.5 / (EWALD_TABLE_SIZE - 3.0);
	for (int i0 = 0; i0 < EWALD_TABLE_SIZE; i0++) {
		for (int j0 = 0; j0 < EWALD_TABLE_SIZE; j0++) {
			for (int k0 = 0; k0 < EWALD_TABLE_SIZE; k0++) {
				const double dx0 = i0 * dx - dx;
				const double dx1 = j0 * dx - dx;
				const double dx2 = k0 * dx - dx;
				float pot;
				float fx;
				float fy;
				float fz;

				ewald_compute(pot, fx, fy, fz, dx0, dx1, dx2);
				ec.table[NDIM][i0][j0][k0] = pot;
				ec.table[XDIM][i0][j0][k0] = fx;
				ec.table[YDIM][i0][j0][k0] = fy;
				ec.table[ZDIM][i0][j0][k0] = fz;
			}
		}
	}
	/*double err = 0.0;
	 int cnt = 0;
	 for (double x = 0.0; x < 0.5; x += 0.1) {
	 for (double y = 0.0; y < 0.5; y += 0.1) {
	 for (double z = 0.0; z < 0.5; z += 0.1) {
	 float pot;
	 float fx;
	 float fy;
	 float fz;
	 ewald_compute(pot, fx, fy, fz, x, y, z);
	 ewald_const tbl;
	 double this_err = pot - tbl.table_interp(NDIM,x,y,z);
	 err += sqr(this_err);
	 cnt ++;
	 }
	 }
	 }
	 err /= cnt;
	 err = sqrt(err);
	 PRINT( "Table error = %e\n", err);
	 */
	int n2max = 12;
	int nmax = std::sqrt(n2max) + 1;
	array<ewald_type, NDIM> this_h;
	int count = 0;
	for (int i = -nmax; i <= nmax; i++) {
		for (int j = -nmax; j <= nmax; j++) {
			for (int k = -nmax; k <= nmax; k++) {
				const int i2 = i * i + j * j + k * k;
				const double x = std::max(abs(i) - 0.5, 0.0);
				const double y = std::max(abs(j) - 0.5, 0.0);
				const double z = std::max(abs(k) - 0.5, 0.0);
				const double d = sqrt(sqr(x, y, z));
				if (d < EWALD_REAL_CUTOFF && i2 > 0) {
					this_h[0] = i;
					this_h[1] = j;
					this_h[2] = k;
					ec.real_indices[count++] = this_h;
				}
			}
		}
	}
	PRINT("count = %i %i\n", count, NREAL);
	const auto sort_func = [](const array<ewald_type,NDIM>& a, const array<ewald_type,NDIM>& b) {
		const auto a2 = sqr(a[0],a[1],a[2]);
		const auto b2 = sqr(b[0],b[1],b[2]);
		return a2 > b2;
	};
	std::sort(ec.real_indices.begin(), ec.real_indices.end(), sort_func);
	n2max = 8;
	nmax = std::sqrt(n2max) + 1;
	count = 0;
	for (int i = -nmax; i <= nmax; i++) {
		for (int j = -nmax; j <= nmax; j++) {
			for (int k = -nmax; k <= nmax; k++) {
				if (i * i + j * j + k * k <= n2max) {
					this_h[0] = i;
					this_h[1] = j;
					this_h[2] = k;
					const auto hdot = sqr(this_h[0]) + sqr(this_h[1]) + sqr(this_h[2]);
					if (hdot > 0) {
						ec.four_indices[count++] = this_h;
					}
				}
			}
		}
	}
	std::sort(ec.four_indices.begin(), ec.four_indices.end(), sort_func);
	count = 0;
	for (int i = 0; i < NFOUR; i++) {
		array<ewald_type, NDIM> h = ec.four_indices[i];
		auto D0 = vector_to_sym_tensor<ewald_type, LORDER>(h);
		const ewald_type h2 = sqr(h[0]) + sqr(h[1]) + sqr(h[2]);                     // 5 OP
		const ewald_type c0 = -1.0 / h2 * exp(-M_PI * M_PI * h2 / 4.0) / M_PI;
		array<int, NDIM> n;
		const int signs[4] = { 1, -1, -1, 1 };
		for (n[0] = 0; n[0] < LORDER; n[0]++) {
			for (n[1] = 0; n[1] < LORDER - n[0]; n[1]++) {
				for (n[2] = 0; n[2] < LORDER - n[0] - n[1]; n[2]++) {
					const int n0 = n[0] + n[1] + n[2];
					D0(n) *= (signs[n0 % 4] * pow(2.0 * M_PI * SCALE_FACTOR_INV1, n0) * c0) * SCALE_FACTOR_INV1;
				}
			}
		}
		ec.four_expanse[count] = D0.detraceD();
		count++;
	}
	tensor_sym<ewald_type, LORDER> D;
	for (int n = 0; n < (LORDER + 2) * (LORDER + 1) * LORDER / 6; n++) {
		D[n] = 0.0;
	}
	constexpr double alpha = 2.0;
	for (int n = 0; n < LORDER; n += 2) {
		for (int m = 0; m < LORDER - n; m += 2) {
			for (int l = 0; l < LORDER - n - m; l += 2) {
				D(n, m, l) = pow(SCALE_FACTOR_INV1, n + m + l + 1) * pow(-2.0, (n + m + l) / 2 + 1) / ((n + m + l + 1.0) * sqrt(M_PI)) * pow(alpha, n + m + l + 1)
						* double_factorial(n + m + l - 1);
			}
		}
	}
	ec.D0 = D;
	cuda_set_device();
	ec_dev = ec;
#endif
}

CUDA_EXPORT int ewald_const::nfour() {
	return NFOUR;
}

CUDA_EXPORT int ewald_const::nreal() {
	return NREAL;
}

CUDA_EXPORT const tensor_sym<ewald_type, LORDER> ewald_const::D0() {
#ifdef __CUDA_ARCH__
	return ec_dev.D0;
#else
	return ec.D0;
#endif
}

CUDA_EXPORT const array<ewald_type, NDIM>& ewald_const::real_index(int i) {
#ifdef __CUDA_ARCH__
	return ec_dev.real_indices[i];
#else
	return ec.real_indices[i];
#endif
}

CUDA_EXPORT const array<ewald_type, NDIM>& ewald_const::four_index(int i) {
#ifdef __CUDA_ARCH__
	return ec_dev.four_indices[i];
#else
	return ec.four_indices[i];
#endif
}

CUDA_EXPORT void ewald_const::table_interp(float& pot, float& fx, float& fy, float& fz, float x, float y, float z, bool do_pot) {
#ifdef __CUDA_ARCH__
	auto& table = ec_dev.table;
#else
	auto& table = ec.table;
#endif
	const float dxinv = 2.0 * (EWALD_TABLE_SIZE - 3.f);
	const float dx = 1.f / dxinv;
	array<float, NDIM> xi;
	array<float, NDIM> xi2;
	array<float, NDIM> xi3;
	xi[XDIM] = x * dxinv + 1.0;
	xi[YDIM] = y * dxinv + 1.0;
	xi[ZDIM] = z * dxinv + 1.0;
	const auto weight = [](int i, float t, float t2, float t3) {
		if( i == 0 ) {
			return -0.5f * t + t2 - 0.5f * t3;
		} else if( i == 1 ) {
			return 1.f - 2.5f * t2 + 1.5f * t3;
		} else if( i == 2 ) {
			return 0.5f * t + 2.f * t2 - 1.5f * t3;
		} else {
			return -0.5f * t2 + 0.5f * t3;
		}
	};
	array<int, NDIM> i0, i1;
	for (int dim = 0; dim < NDIM; dim++) {
		i0[dim] = xi[dim] - 1;
		if (i0[dim] > EWALD_TABLE_SIZE - 3) {
			i0[dim] = EWALD_TABLE_SIZE - 3;
		}
		xi[dim] -= i0[dim] + 1;
		xi2[dim] = sqr(xi[dim]);
		xi3[dim] = xi[dim] * xi2[dim];
	}
	fx = fy = fz = pot = 0.0f;
	for (i1[XDIM] = 0; i1[XDIM] < 4; i1[XDIM]++) {
		for (i1[YDIM] = 0; i1[YDIM] < 4; i1[YDIM]++) {
			for (i1[ZDIM] = 0; i1[ZDIM] < 4; i1[ZDIM]++) {
				float wt = 1.f;
				array<int, NDIM> i2;
				for (int dim = 0; dim < NDIM; dim++) {
					wt *= weight(i1[dim], xi[dim], xi2[dim], xi3[dim]);
				}
				for (int dim = 0; dim < NDIM; dim++) {
					i2[dim] = i0[dim] + i1[dim];
				}
				//		PRINT( "%i %i %i %e\n", i2[XDIM], i2[YDIM], i2[ZDIM], table[interp_dim][i2[XDIM]][i2[YDIM]][i2[ZDIM]] );
				fx += wt * table[XDIM][i2[XDIM]][i2[YDIM]][i2[ZDIM]];
				fy += wt * table[YDIM][i2[XDIM]][i2[YDIM]][i2[ZDIM]];
				fz += wt * table[ZDIM][i2[XDIM]][i2[YDIM]][i2[ZDIM]];
				if (do_pot) {
					pot += wt * table[NDIM][i2[XDIM]][i2[YDIM]][i2[ZDIM]];
				}
			}
		}
	}
}

CUDA_EXPORT const tensor_trless_sym<ewald_type, LORDER>& ewald_const::four_expansion(int i) {
#ifdef __CUDA_ARCH__
	return ec_dev.four_expanse[i];
#else
	return ec.four_expanse[i];
#endif
}
