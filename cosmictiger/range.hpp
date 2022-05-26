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

#ifndef RANGE_HPP_
#define RANGE_HPP_

#include <cosmictiger/cuda.hpp>
#include <cosmictiger/defs.hpp>
#include <cosmictiger/containers.hpp>
#include <cosmictiger/options.hpp>

template<class T>
inline array<T, NDIM> shift_up(array<T, NDIM> i) {
	array<T, NDIM> j;
	j[2] = i[0];
	j[0] = i[1];
	j[1] = i[2];
	return j;
}

template<class T>
inline array<T, NDIM> shift_down(array<T, NDIM> i) {
	array<T, NDIM> j;
	j[1] = i[0];
	j[2] = i[1];
	j[0] = i[2];
	return j;
}

template<class T, int N = NDIM>
struct range {
	array<T, N> begin;
	array<T, N> end;

	CUDA_EXPORT
	inline range<T, N> intersection(const range<T, N>& other) const {
		range<T, N> I;
		for (int dim = 0; dim < N; dim++) {
#ifdef __CUDA_ARCH__
			I.begin[dim] = max(begin[dim], other.begin[dim]);
			I.end[dim] = min(end[dim], other.end[dim]);
#else
			I.begin[dim] = std::max(begin[dim], other.begin[dim]);
			I.end[dim] = std::min(end[dim], other.end[dim]);
#endif
		}
		return I;
	}
	CUDA_EXPORT
	inline range periodic_intersection(const range& other) const {
		range I;
#ifdef __CUDA_ARCH__
		for (int dim = 0; dim < N; dim++) {
			I.begin[dim] = max(begin[dim], other.begin[dim]);
			I.end[dim] = min(end[dim], other.end[dim]);
			if (I.end[dim] <= I.begin[dim]) {
				I.begin[dim] = max(begin[dim] + T(1), other.begin[dim]);
				I.end[dim] = min(end[dim] + T(1), other.end[dim]);
				if (I.end[dim] <= I.begin[dim]) {
					I.begin[dim] = max(begin[dim] - T(1), other.begin[dim]);
					I.end[dim] = min(end[dim] - T(1), other.end[dim]);
				}
			}
		}
#else
		for (int dim = 0; dim < N; dim++) {
			I.begin[dim] = std::max(begin[dim], other.begin[dim]);
			I.end[dim] = std::min(end[dim], other.end[dim]);
			if (I.end[dim] <= I.begin[dim]) {
				I.begin[dim] = std::max(begin[dim] + T(1), other.begin[dim]);
				I.end[dim] = std::min(end[dim] + T(1), other.end[dim]);
				if (I.end[dim] <= I.begin[dim]) {
					I.begin[dim] = std::max(begin[dim] - T(1), other.begin[dim]);
					I.end[dim] = std::min(end[dim] - T(1), other.end[dim]);
				}
			}
		}
#endif
		return I;
	}
	CUDA_EXPORT
	inline bool periodic_intersects(const range& other) const {
		range I;
#ifdef __CUDA_ARCH__
		for (int dim = 0; dim < N; dim++) {
			I.begin[dim] = max(begin[dim], other.begin[dim]);
			I.end[dim] = min(end[dim], other.end[dim]);
			if (I.end[dim] <= I.begin[dim]) {
				I.begin[dim] = max(begin[dim] + T(1), other.begin[dim]);
				I.end[dim] = min(end[dim] + T(1), other.end[dim]);
				if (I.end[dim] <= I.begin[dim]) {
					I.begin[dim] = max(begin[dim] - T(1), other.begin[dim]);
					I.end[dim] = min(end[dim] - T(1), other.end[dim]);
				}
			}
		}
#else
		for (int dim = 0; dim < N; dim++) {
			I.begin[dim] = std::max(begin[dim], other.begin[dim]);
			I.end[dim] = std::min(end[dim], other.end[dim]);
			if (I.end[dim] <= I.begin[dim]) {
				I.begin[dim] = std::max(begin[dim] + T(1), other.begin[dim]);
				I.end[dim] = std::min(end[dim] + T(1), other.end[dim]);
				if (I.end[dim] <= I.begin[dim]) {
					I.begin[dim] = std::max(begin[dim] - T(1), other.begin[dim]);
					I.end[dim] = std::min(end[dim] - T(1), other.end[dim]);
				}
			}
		}
#endif
		for (int dim = 0; dim < NDIM; dim++) {
			if (I.end[dim] <= I.begin[dim]) {
				return false;
			}
		}
		return true;
	}

	inline bool empty() const {
		return volume() == T(0);
	}

	range() = default;
	range(const range&) = default;
	range(range&&) = default;
	range& operator=(const range&) = default;
	range& operator=(range&&) = default;

	inline range shift(const array<T, N>& s) const {
		range r = *this;
		for (int dim = 0; dim < N; dim++) {
			r.begin[dim] += s[dim];
			r.end[dim] += s[dim];
		}
		return r;
	}

	range(const T& sz) {
		for (int dim = 0; dim < N; dim++) {
			begin[dim] = T(0);
			end[dim] = sz;
		}
	}

	CUDA_EXPORT
	inline bool contains(const range<T, N>& box) const {
		bool rc = true;
		for (int dim = 0; dim < N; dim++) {
			if (begin[dim] > box.begin[dim]) {
				rc = false;
				break;
			}
			if (end[dim] < box.end[dim]) {
				rc = false;
				break;
			}
		}
		return rc;
	}

	CUDA_EXPORT
	inline bool contains(const array<T, N>& p) const {
		for (int dim = 0; dim < N; dim++) {
			if (p[dim] < begin[dim] || p[dim] >= end[dim]) {
				return false;
			}
		}
		return true;
	}

	inline bool periodic_contains(const array<T, N>& p) const {
		for (int dim = 0; dim < N; dim++) {
			if (p[dim] < begin[dim] || p[dim] >= end[dim]) {
				if (p[dim] + T(1) < begin[dim] || p[dim] + T(1) >= end[dim]) {
					if (p[dim] - T(1) < begin[dim] || p[dim] - T(1) >= end[dim]) {
						return false;
					}
				}
			}
		}
		return true;
	}

	inline std::string to_string() const {
		std::string str;
		for (int dim = 0; dim < N; dim++) {
			str += std::to_string(dim) + ":(";
			str += std::to_string(begin[dim]) + ",";
			str += std::to_string(end[dim]) + ") ";
		}
		return str;
	}

	__device__ inline int longest_dim() const {
		int max_dim;
		T max_span = T(-1);
		for (int dim = 0; dim < N; dim++) {
			const T span = end[dim] - begin[dim];
			if (span > max_span) {
				max_span = span;
				max_dim = dim;
			}
		}
		return max_dim;
	}

	inline int shortest_dim() const {
		int min_dim;
		T min_span = T(1);
		for (int dim = 0; dim < N; dim++) {
			const T span = end[dim] - begin[dim];
			if (span <= min_span) {
				min_span = span;
				min_dim = dim;
			}
		}
		return min_dim;
	}

	inline std::pair<range<T, N>, range<T, N>> split() const {
		auto left = *this;
		auto right = *this;
		int max_dim;
		T max_span = 0;
		for (int dim = 0; dim < N; dim++) {
			const T span = end[dim] - begin[dim];
			if (span > max_span) {
				max_span = span;
				max_dim = dim;
			}
		}
		const T mid = (end[max_dim] + begin[max_dim]) / T(2);
		left.end[max_dim] = right.begin[max_dim] = mid;
		return std::make_pair(left, right);
	}

	range<T, N> shift_up() const {
		range<T, N> rc;
		rc.begin = ::shift_up(begin);
		rc.end = ::shift_up(end);
		return rc;
	}

	range<T, N> shift_down() const {
		range<T, N> rc;
		rc.begin = ::shift_down(begin);
		rc.end = ::shift_down(end);
		return rc;
	}

	CUDA_EXPORT
	inline T index(T xi, T yi, T zi) const {
		const auto spanz = end[2] - begin[2];
		const auto spany = end[1] - begin[1];
		return spanz * (spany * (xi - begin[0]) + (yi - begin[1])) + (zi - begin[2]);
	}

	CUDA_EXPORT
	inline T index(const array<T, N> & i) const {
		return index(i.data());
	}

	CUDA_EXPORT
	inline T index(const T * i) const {
		const auto spanz = end[2] - begin[2];
		const auto spany = end[1] - begin[1];
		return spanz * (spany * (i[0] - begin[0]) + (i[1] - begin[1])) + (i[2] - begin[2]);
	}

	inline range<T, N> transpose(int dim1, int dim2) const {
		auto rc = *this;
		std::swap(rc.begin[dim1], rc.begin[dim2]);
		std::swap(rc.end[dim1], rc.end[dim2]);
		return rc;
	}

	CUDA_EXPORT
	inline T volume() const {
		T vol = T(1);
		for (int dim = 0; dim < N; dim++) {
			const T span = end[dim] - begin[dim];
			if (span < T(0)) {
				return T(0);
			}
			vol *= span;
		}
		return vol;
	}

	template<class A>
	void serialize(A&& arc, unsigned) {
		for (int dim = 0; dim < N; dim++) {
			arc & begin[dim];
			arc & end[dim];
		}
	}

	CUDA_EXPORT
	inline range<T, N> pad(T dx = T(1)) const {
		range<T, N> r;
		for (int dim = 0; dim < N; dim++) {
			r.begin[dim] = begin[dim] - dx;
			r.end[dim] = end[dim] + dx;
		}
		return r;
	}

};

template<class T, int N = NDIM>
inline range<T, N> unit_box() {
	range<T, N> r;
	for (int dim = 0; dim < N; dim++) {
		r.begin[dim] = T(0);
		r.end[dim] = T(1);
	}
	return r;
}

#include <cosmictiger/fixed.hpp>

inline range<fixed32> fixed32_unit_box() {
	range<fixed32> r;
	for (int dim = 0; dim < NDIM; dim++) {
		r.begin[dim] = 0.0;
		r.end[dim] = fixed32::max();
	}
	return r;
}

inline range<fixed32> rngdbl2rngfixed32(const range<double>& other) {
	range<fixed32> rc;
	for (int dim = 0; dim < NDIM; dim++) {
		rc.begin[dim] = other.begin[dim];
		rc.end[dim] = other.end[dim];
	}
	return rc;
}

using range_fixed = fixed<int,29>;

struct fixed32_range: public range<range_fixed> {
	fixed32_range() {
	}
	CUDA_EXPORT
	fixed32_range& operator=(const fixed32_range& other) {
		begin = other.begin;
		end = other.end;
		return *this;
	}
	CUDA_EXPORT
	fixed32_range& operator=(fixed32_range&& other) {
		begin = other.begin;
		end = other.end;
		return *this;
	}
	CUDA_EXPORT
	fixed32_range(const fixed32_range& other) {
		begin = other.begin;
		end = other.end;
	}
	CUDA_EXPORT
	fixed32_range(fixed32_range&& other) {
		begin = other.begin;
		end = other.end;
	}
	CUDA_EXPORT
	bool contains(const array<fixed32, NDIM>& pt) const {
		for (int i = -1; i <= 1; i++) {
			for (int j = -1; j <= 1; j++) {
				for (int k = -1; k <= 1; k++) {
					bool contains = true;
					array<double, NDIM> I;
					I[0] = i;
					I[1] = j;
					I[2] = k;
					for (int dim = 0; dim < NDIM; dim++) {
						if (range_fixed(pt[dim]) + range_fixed::min() < begin[dim] + range_fixed(I[dim])) {
							contains = false;
							break;
						}
						if (range_fixed(pt[dim]) > end[dim] + range_fixed(I[dim]) + range_fixed::min()) {
							contains = false;
							break;
						}
					}
					if (contains == true) {
						return true;
					}
				}
			}
		}
		return false;
	}
	/*	void accumulate(const array<fixed32, NDIM>& pt, float h = float(0)) {
	 if (!valid) {
	 for (int dim = 0; dim < NDIM; dim++) {
	 begin[dim] = pt[dim].to_double() - h;
	 end[dim] = pt[dim].to_double() + h;
	 }
	 valid = true;
	 } else {
	 for (int dim = 0; dim < NDIM; dim++) {
	 begin[dim] = std::min(begin[dim], pt[dim].to_double() - h);
	 end[dim] = std::max(end[dim], pt[dim].to_double() + h);
	 }
	 }
	 }*/
	/*	void accumulate(const fixed32_range& other) {
	 if (valid && other.valid) {
	 for (int dim = 0; dim < NDIM; dim++) {
	 begin[dim] = std::min(begin[dim], other.begin[dim]);
	 end[dim] = std::max(end[dim], other.end[dim]);
	 }
	 } else if (!valid && other.valid) {
	 *this = other;
	 }
	 }*/
	template<class A>
	void serialize(A&& arc, unsigned i) {
		range<range_fixed>::serialize(arc, i);
	}
};

CUDA_EXPORT
inline float distance(range_fixed a, fixed32 b) {
	float f = a.to_double() - b.to_double();
	while (f > 0.5) {
		f -= 1.0;
	}
	while (f < -0.5) {
		f += 1.0;
	}
	return f;
}

CUDA_EXPORT
inline float distance(fixed32 b, range_fixed a) {
	return -distance(a, b);

}
#endif /* RANGE_HPP_ */
