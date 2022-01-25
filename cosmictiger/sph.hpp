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

#ifndef SPH_HPP_
#define SPH_HPP_

#define SPH_KERNEL_ORDER 5

#include <cosmictiger/defs.hpp>

template<class T, int N>
T ipow(T x) {
	if (N == 0) {
		return T(1);
	} else if (N == 1) {
		return x;
	} else if (N == 2) {
		return sqr(x);
	} else {
		constexpr int M = N / 2;
		constexpr int L = N - M;
		return ipow<T, L>(x) * ipow<T, M>(x);
	}
}

template<class T>
inline T sph_W(T r, T hinv, T h3inv) {
	static constexpr T n = SPH_KERNEL_ORDER;
	static const T c0 = pow(M_PI, -1.5) / tgamma(T(2.5) + n) / tgamma(T(1) + n);
	const T q = r * hinv;
	const T tmp = T(1) - sqr(q);
	return c0 * ipow<T, SPH_KERNEL_ORDER>(tmp) * h3inv;
}

template<class T>
inline T sph_dWdr(T r, T hinv, T h3inv) {
	static constexpr T n = SPH_KERNEL_ORDER;
	static const T c0 = T(2) * pow(M_PI, -1.5) / tgamma(T(2.5) + n) / tgamma(n);
	const T q = r * hinv;
	const T tmp = T(1) - sqr(q);
	return -c0 * ipow<T, SPH_KERNEL_ORDER - 1>(tmp) * h3inv * hinv * q;
}

template<class T>
inline T sph_dWdh(T r, T hinv, T h3inv) {
	static constexpr T n = SPH_KERNEL_ORDER;
	static const T c0 = pow(M_PI, -1.5) / tgamma(T(2.5) + n) / tgamma(T(1) + n);
	static const T c1 = T(3) + T(2) * n;
	const T q = r * hinv;
	const T tmp = T(1) - sqr(q);
	return c0 * ipow<T, SPH_KERNEL_ORDER - 1>(tmp) * h3inv * hinv * (c1 * q - T(2));
}

template<class T>
inline T sph_den(T hinv3) {
	static const T c0 = T(3.0 / 4.0 / M_PI * SPH_NEIGHBOR_COUNT);
	return c0 * hinv3;
}

#endif /* SPH_HPP_ */
