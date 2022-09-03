/*
 * treepm_kernels.hpp
 *
 *  Created on: Sep 3, 2022
 *      Author: dmarce1
 */

#ifndef TREEPM_KERNELS_HPP_
#define TREEPM_KERNELS_HPP_

#pragma once

#include <cosmictiger/defs.hpp>

CUDA_EXPORT inline array<float, LORDER> green_kernel(float r, float rsinv, float rsinv2) {
	array<float, LORDER> d;
	float q = r * rsinv;
	float q2 = sqr(q);
	float q4 = sqr(q2);
	float q6 = q2 * q4;
	float rinv = 1.f / r;
	float rinv2 = sqr(rinv);
	float rinv3 = rinv * rinv2;
	float rinv4 = sqr(rinv2);
	if (q2 < 1.f) {
		d[0] = (35.f / 16.f - 35.f / 16.f * q2 + 21.f / 16.f * q4 - 5.f / 16.f * q6) * rsinv;
		d[1] = (-35.f / 8.f + 21.f / 4.f * q2 - 15.f / 8.f * q4) * q * rsinv2;
		d[2] = (21.0f / 2.0f - 15.0f * q2 / 2.f) * q2 * rsinv2 * rsinv;
		d[3] = -15.f * q2 * q * sqr(rsinv2);
		d[0] -= rinv;
		d[1] += rinv2;
		d[2] -= 3.f * rinv3;
		d[3] += 15.f * rinv4;
	} else {
		d[0] = d[1] = d[2] = d[3] = 0.f;
	}
	return d;
}

CUDA_EXPORT inline void green_direct(float& phi, float& f, float r, float r2, float rinv, float rsinv, float rsinv2) {
	const float q = r * rsinv;
	float q2 = sqr(q);
	if (q2 < 1.f) {
		float q4 = sqr(q2);
		float q6 = q2 * q4;
		phi = rinv;
		f = rinv * sqr(rinv);
		phi -= (35.f / 16.f - 35.f / 16.f * q2 + 21.f / 16.f * q4 - 5.f / 16.f * q6) * rsinv;
		f -= (35.0f / 8.0f - 21.0f / 4.0f * q2 + 15.0f / 8.0f * q4) * rsinv * rsinv2;
	} else {
		f = phi = 0.f;
	}
}

CUDA_EXPORT inline float green_filter(float k) {
	if (k > 0.02) {
		return 105 * (k * (-15 + sqr(k)) * cos(k) + 3 * (5 - 2 * sqr(k)) * sin(k)) / (sqr(sqr(k)) * sqr(k) * k);
	} else {
		return 1.0 - 1.0 / 18.0 * sqr(k) + 1.0 / 792.0 * sqr(sqr(k));
	}
}

CUDA_EXPORT inline float green_phi0(float nparts, float rs) {
	return 4.0 * M_PI * sqr(rs) * (nparts - 1) / 18.0 + 35.0 / 16.0 / rs;
}

/*

 CUDA_EXPORT inline array<float, LORDER> green_kernel(float r, float rsinv, float rsinv2) {
 array<float, LORDER> d;
 const float nr = float(-0.5) * r * rsinv2;
 const float rinv = 1.f / r;
 float exp0 = expf(-float(0.25) * rsinv2 * sqr(r));
 float erfc0 = erfcf(float(0.5) * rsinv * r);
 const float expfactor = float(1.0 / 1.77245385e+00) * rsinv * exp0;
 float e0 = expfactor * rinv;
 d[0] = -erfc0 * rinv;
 for (int l = 1; l < LORDER; l++) {
 d[l] = fmaf(-float(2 * l - 1) * d[l - 1], rinv, e0);
 e0 *= nr;
 }
 return d;
 }

 CUDA_EXPORT inline void green_direct(float& phi, float& f, float r, float r2, float rinv, float rsinv, float rsinv2) {
 const float erfc0 = erfcf(0.5f * r * rsinv);
 const float exp0 = expf(-0.25f * r2 * rsinv2);
 const float cons = .5641895835f;
 phi = erfc0 * rinv;
 f = (erfc0 + cons * r * rsinv * exp0) * sqr(rinv) * rinv;
 }


 CUDA_EXPORT inline float green_filter(float krs) {
 return exp(-sqr(krs));
 }

 CUDA_EXPORT inline float green_phi0(float nparts, float rs) {
 return 4.0 * M_PI * sqr(rs) * (nparts -1)+ 1.0 / sqrt(M_PI) / rs;
 }


 */
#endif /* TREEPM_KERNELS_HPP_ */
