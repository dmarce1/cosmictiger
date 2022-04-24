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

void kernel_set_type(int type);
void kernel_output();
void kernel_adjust_options(options& opts);

template<class T>
CUDA_EXPORT
inline T kernelW(T q) {
	T w1, w2, w3, res, sw1, sw2, sw3, mq;
	const T c0 = T(15625.0f / M_PI / 512.0f);
	w1 = T(6);
	w1 *= q;
	w1 = fmaf(q, w1, -T(12.0f / 5.0f));
	w1 *= q;
	w1 = fmaf(q, w1, T(46.0f / 125.0f));
	w2 = -T(4);
	w2 = fmaf(q, w2, T(8));
	w2 = fmaf(q, w2, -T(24.0f / 5.f));
	w2 = fmaf(q, w2, T(8. / 25.));
	w2 = fmaf(q, w2, T(44. / 125.));
	mq = T(1) - q;
	w3 = sqr(sqr(mq));
	sw1 = q < T(0.2f);
	sw3 = q > T(0.6f);
	sw2 = (T(1) - sw1) * (T(1) - sw3);
	res = c0 * (sw1 * w1 + sw2 * w2 + sw3 * w3);
	res *= q < T(1.f);
	return res;
}

template<class T>
CUDA_EXPORT
inline T dkernelW_dq(T q) {
	T w1, w2, w3, res, sw1, sw2, sw3, mq;
	const T c0 = T(15625.0f / M_PI / 512.0f);
	w1 = T(24);
	w1 *= q;
	w1 = fmaf(q, w1, -T(24.0f / 5.0f));
	w1 *= q;
	w2 = -T(16);
	w2 = fmaf(q, w2, T(24));
	w2 = fmaf(q, w2, -T(48.0f / 5.f));
	w2 = fmaf(q, w2, T(8. / 25.));
	mq = T(1.) - q;
	w3 = -T(4) * sqr(mq) * mq;
	sw1 = q < T(0.2f);
	sw3 = q > T(0.6f);
	sw2 = (T(1) - sw1) * (T(1) - sw3);
	res = c0 * (sw1 * w1 + sw2 * w2 + sw3 * w3);
	res *= q < T(1.f);
	return res;
}

template<class T>
CUDA_EXPORT
inline T kernelFqinv(T q) {
	T w1, w2, res, q3inv, w3, sw1, sw2, sw3;
	w1 = T(46875.0f / 448.0f);
	w1 *= q;
	w1 = fmaf(q, w1, -T(1875.f / 32.f));
	w1 *= q;
	w1 = fmaf(q, w1, T(2875.f / 192.f));
	w2 = -T(15625.0f / 224.0f);
	w2 = fmaf(q, w2, T(15625.f / 96.f));
	w2 = fmaf(q, w2, -T(1875.0f / 16.f));
	w2 = fmaf(q, w2, T(625.f / 64.f));
	w2 = fmaf(q, w2, T(1375.f / 96.f));
	q3inv = T(1) / (q + T(1e-10f));
	q3inv = sqr(q3inv) * q3inv;
	w2 += T(1.0f / 6720.0f) * q3inv;
	w3 = T(15625.0f / 896.0f);
	w3 = fmaf(q, w3, -T(15625.0f / 192.0f));
	w3 = fmaf(q, w3, T(9375.0f / 64.0f));
	w3 = fmaf(q, w3, -T(15625.0f / 128.0f));
	w3 = fmaf(q, w3, T(15625.0f / 384.0f));
	w3 -= T(437.0f / 2688.0f) * q3inv;
	sw1 = q < T(0.2f);
	sw3 = q > T(0.6f);
	sw2 = (T(1) - sw1) * (T(1) - sw3);
	res = (sw1 * w1 + sw2 * w2 + sw3 * w3);
	sw1 = q < T(1);
	sw2 = T(1) - sw1;
	res = sw1 * res + sw2 / (sqr(q) * q + T(1e-30));
	return res;
}

template<class T>
CUDA_EXPORT
inline T kernelPot(T q) {
	T w1, w2, res, q1inv, w3, sw1, sw2, sw3;
	w1 = T(-15625.0f / 896.0f);
	w1 *= q;
	w1 = fmaf(q, w1, T(1875.f / 128.f));
	w1 *= q;
	w1 = fmaf(q, w1, -T(2875.f / 384.f));
	w1 *= q;
	w1 = fmaf(q, w1, T(1199.f / 384.f));
	w2 = T(15625.0f / 1344.0f);
	w2 = fmaf(q, w2, T(-3125.f / 96.f));
	w2 = fmaf(q, w2, T(1875.0f / 64.f));
	w2 = fmaf(q, w2, -T(625.f / 192.f));
	w2 = fmaf(q, w2, -T(1375.f / 192.f));
	w2 *= q;
	w2 = fmaf(q, w2, T(599.f / 192.f));
	q1inv = T(1) / (q + T(1e-15f));
	w2 += T(1.0f / 6720.0f) * q1inv;
	w3 = -T(15625.0f / 5376.0f);
	w3 = fmaf(q, w3, T(3125.0f / 192.0f));
	w3 = fmaf(q, w3, -T(9375.0f / 256.0f));
	w3 = fmaf(q, w3, T(15625.0f / 384.0f));
	w3 = fmaf(q, w3, -T(15625.0f / 768.0f));
	w3 *= q;
	w3 = fmaf(q, w3, T(3125.0f / 768.0f));
	w3 -= T(437.0f / 2688.0f) * q1inv;
	sw1 = q < T(0.2f);
	sw3 = q > T(0.6f);
	sw2 = (T(1) - sw1) * (T(1) - sw3);
	res = (q > T(0)) * (sw1 * w1 + sw2 * w2 + sw3 * w3);
	sw1 = q < T(1);
	sw2 = T(1) - sw1;
	res = sw1 * res + sw2 / (q + T(1e-30f));
	return res;
}
