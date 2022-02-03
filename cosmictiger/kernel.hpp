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

#pragma once

#include <cosmictiger/cuda.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/options.hpp>

#define KERNEL_CUBIC_SPLINE 0
#define KERNEL_QUARTIC_SPLINE 1
#define KERNEL_QUINTIC_SPLINE 2
#define KERNEL_WENDLAND_C2 3
#define KERNEL_WENDLAND_C4 4
#define KERNEL_WENDLAND_C6 5

void kernel_set_type(int type);
void kernel_output();

#ifdef __CUDACC__
#ifdef __KERNEL_CU__
extern __managed__ int kernel_type;
#else
__managed__ int kernel_type;
#endif
#endif

template<class T>
CUDA_EXPORT
inline T kernelW(T q) {
	T w1, w2, sw, res;
#ifdef __CUDACC__
	switch(kernel_type) {
#else
	static const int kernel_type = get_options().kernel;
	switch (kernel_type) {
#endif
	case KERNEL_CUBIC_SPLINE: {
		const T c0 = T(8.0f / float(M_PI));
		w1 = T(6);
		w1 = fmaf(q, w1, -T(6));
		w1 *= q;
		w1 = fmaf(q, w1, T(1));
		w2 = -T(2);
		w2 = fmaf(q, w2, T(6));
		w2 = fmaf(q, w2, -T(6));
		w2 = fmaf(q, w2, T(2));
		sw = q < T(0.5);
		res = c0 * (sw * w1 + (T(1) - sw) * w2);
	}
		break;
	};
	return res;
}

template<class T>
CUDA_EXPORT
inline T dkernelW_dq(T q) {
	T w1, w2, sw, res;
#ifdef __CUDACC__
	switch(kernel_type) {
#else
	static const int kernel_type = get_options().kernel;
	switch (kernel_type) {
#endif
	case KERNEL_CUBIC_SPLINE: {
		const T c0 = T(8.0f / float(M_PI));
		w1 = T(18);
		w1 = fmaf(q, w1, -T(12));
		w1 *= q;
		w2 = -T(6);
		w2 = fmaf(q, w2, T(12));
		w2 = fmaf(q, w2, -T(6));
		sw = q < T(0.5);
		res = c0 * (sw * w1 + (T(1) - sw) * w2);
	}
		break;
	};
	return res;
}

template<class T>
CUDA_EXPORT
inline T kernelFqinv(T q) {
	T w1, w2, sw, res;
#ifdef __CUDACC__
	switch(kernel_type) {
#else
	static const int kernel_type = get_options().kernel;
	switch (kernel_type) {
#endif
	case KERNEL_CUBIC_SPLINE: {
		w1 = T(32);
		w1 = fmaf(q, w1, -T(192.0 / 5.0));
		w1 *= q;
		w1 = fmaf(q, w1, T(32.0 / 3.0));
		w2 = -T(32.0 / 3.0);
		w2 = fmaf(q, w2, T(192 / 5.0));
		w2 = fmaf(q, w2, -T(48.0));
		w2 = fmaf(q, w2, T(64.0 / 3.0));
		w2 -= T(1.0 / 15.0) / (q * sqr(q));
		sw = q < T(0.5);
		res = (sw * w1 + (T(1) - sw) * w2);
	}
		break;
	};
	return res;
}

template<class T>
CUDA_EXPORT
inline T kernelPot(T q) {
	T w1, w2, sw, res;
#ifdef __CUDACC__
	switch(kernel_type) {
#else
	static const int kernel_type = get_options().kernel;
	switch (kernel_type) {
#endif
	case KERNEL_CUBIC_SPLINE: {
		w1 = -T(32.0 / 5.0);
		w1 = fmaf(q, w1, T(48.0 / 5.0));
		w1 *= q;
		w1 = fmaf(q, w1, T(-16.0 / 3.0));
		w1 *= q;
		w1 = fmaf(q, w1, T(14.0 / 5.0));
		w2 = T(32.0 / 15.0);
		w2 = fmaf(q, w2, T(-48.0 / 5.0));
		w2 = fmaf(q, w2, T(16.0));
		w2 = fmaf(q, w2, T(-32.0 / 3.0));
		w2 *= q;
		w2 = fmaf(q, w2, T(16.0 / 5.0));
		w2 -= T(1.0 / 15.0) / q;
		sw = q < T(0.5);
		res = (sw * w1 + (T(1) - sw) * w2);
	}
		break;
	};
	return res;
}
