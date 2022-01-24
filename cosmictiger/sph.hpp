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

#include <cosmictiger/defs.hpp>

template<class T>
inline T sph_W(T r, T hinv, T h3inv) {
	static const T _8piinv = T(8) / T(M_PI);
	const T C = h3inv * _8piinv;
	T q = r * hinv;
	if (q < T(1)) {
		return C * (T(1) - T(6) * sqr(q) * (T(1) - q));
	} else if (q < T(2)) {
		const T tmp = T(1) - q;
		return T(2) * tmp * sqr(tmp) * C;
	}
}

template<class T>
inline T sph_dWdr(T r, T hinv, T h3inv) {
	static const T _8piinv = T(48) / T(M_PI);
	const T C = hinv * h3inv * _8piinv;
	T q = r * hinv;
	if (q < T(0.5)) {
		return C * q * (T(3) * q - T(2));
	} else if (q < T(1)) {
		const T tmp = T(1) - q;
		return -sqr(tmp) * C;
	}
}

template<class T>
inline T sph_dWdh(T r, T hinv, T h3inv) {
	static const T _8piinv = T(24) / T(M_PI);
	const T C = h3inv * hinv * _8piinv;
	T q = r * hinv;
	if (q < T(0.5)) {
		return -C * (T(1) - T(10) * sqr(q) * (T(1) - T(6.0 / 5.0) * q)) * hinv;
	} else if (q < T(1)) {
		const T tmp = T(1) - q;
		return -sqr(T(1) - q) * (T(1) - T(2) * q) * C * hinv * T(2);
	}
}

template<class T>
inline T sph_den(T hinv3) {
	static const T c0 = T(3.0 / 4.0 / M_PI * SPH_NEIGHBOR_COUNT);
	return c0 * hinv3;
}

#endif /* SPH_HPP_ */
