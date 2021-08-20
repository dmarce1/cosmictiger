/*
 * range.hpp
 *
 *  Created on: Jun 23, 2021
 *      Author: dmarce1
 */

#ifndef RANGE_HPP_
#define RANGE_HPP_

#include <cosmictiger/cuda.hpp>
#include <cosmictiger/defs.hpp>
#include <cosmictiger/containers.hpp>

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

	inline range<T,N> intersection(const range<T,N>& other) const {
		range<T,N> I;
		for (int dim = 0; dim < N; dim++) {
			I.begin[dim] = std::max(begin[dim], other.begin[dim]);
			I.end[dim] = std::min(end[dim], other.end[dim]);
		}
		return I;
	}

	inline range periodic_intersection(const range& other) const {
		range I;
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
		return I;
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

	inline bool contains(const range<T,N>& box) const {
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

	inline bool contains(const array<T, N>& p) const {
		for (int dim = 0; dim < N; dim++) {
			if (p[dim] < begin[dim] || p[dim] >= end[dim]) {
				return false;
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

	inline int longest_dim() const {
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

	inline std::pair<range<T,N>, range<T,N>> split() const {
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

	range<T,N> shift_up() const {
		range<T,N> rc;
		rc.begin = ::shift_up(begin);
		rc.end = ::shift_up(end);
		return rc;
	}

	range<T,N> shift_down() const {
		range<T,N> rc;
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

	inline range<T,N> transpose(int dim1, int dim2) const {
		auto rc = *this;
		std::swap(rc.begin[dim1], rc.begin[dim2]);
		std::swap(rc.end[dim1], rc.end[dim2]);
		return rc;
	}

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

	inline range<T,N> pad(T dx = T(1)) const {
		range<T,N> r;
		for (int dim = 0; dim < N; dim++) {
			r.begin[dim] = begin[dim] - dx;
			r.end[dim] = end[dim] + dx;
		}
		return r;
	}

};

template<class T, int N = NDIM>
inline range<T,N> unit_box() {
	range<T,N> r;
	for (int dim = 0; dim < N; dim++) {
		r.begin[dim] = T(0);
		r.end[dim] = T(1);
	}
	return r;
}

#endif /* RANGE_HPP_ */
