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

#include <cosmictiger/containers.hpp>
#include <cosmictiger/cuda.hpp>
#include <cosmictiger/defs.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/simd.hpp>

template <class T>
inline Array<T, NDIM> shiftUp(Array<T, NDIM> i) {
    Array<T, NDIM> j;
    j[2] = i[0];
    j[0] = i[1];
    j[1] = i[2];
    return j;
}

template <class T>
inline Array<T, NDIM> shiftDown(Array<T, NDIM> i) {
    Array<T, NDIM> j;
    j[1] = i[0];
    j[2] = i[1];
    j[0] = i[2];
    return j;
}

template <class T, int n = NDIM>
struct Range {
    Array<T, n> begin;
    Array<T, n> end;

    CUDA_EXPORT
    inline Range<T, n> intersection(Range<T, n> const& other) const {
        Range<T, n> iValue;
        for (int dim = 0; dim < n; dim++) {
#ifdef __CUDA_ARCH__
            iValue.begin[dim] = max(begin[dim], other.begin[dim]);
            iValue.end[dim] = min(end[dim], other.end[dim]);
#else
            iValue.begin[dim] = std::max(begin[dim], other.begin[dim]);
            iValue.end[dim] = std::min(end[dim], other.end[dim]);
#endif
        }
        return iValue;
    }
    CUDA_EXPORT
    inline Range periodicIntersection(Range const& other) const {
        Range iValue;
#ifdef __CUDA_ARCH__
        for (int dim = 0; dim < n; dim++) {
            iValue.begin[dim] = max(begin[dim], other.begin[dim]);
            iValue.end[dim] = min(end[dim], other.end[dim]);
            if (iValue.end[dim] <= iValue.begin[dim]) {
                iValue.begin[dim] = max(begin[dim] + T(1), other.begin[dim]);
                iValue.end[dim] = min(end[dim] + T(1), other.end[dim]);
                if (iValue.end[dim] <= iValue.begin[dim]) {
                    iValue.begin[dim] = max(begin[dim] - T(1), other.begin[dim]);
                    iValue.end[dim] = min(end[dim] - T(1), other.end[dim]);
                }
            }
        }
#else
        for (int dim = 0; dim < n; dim++) {
            iValue.begin[dim] = std::max(begin[dim], other.begin[dim]);
            iValue.end[dim] = std::min(end[dim], other.end[dim]);
            if (iValue.end[dim] <= iValue.begin[dim]) {
                iValue.begin[dim] = std::max(begin[dim] + T(1), other.begin[dim]);
                iValue.end[dim] = std::min(end[dim] + T(1), other.end[dim]);
                if (iValue.end[dim] <= iValue.begin[dim]) {
                    iValue.begin[dim] = std::max(begin[dim] - T(1), other.begin[dim]);
                    iValue.end[dim] = std::min(end[dim] - T(1), other.end[dim]);
                }
            }
        }
#endif
        return iValue;
    }
    CUDA_EXPORT
    inline bool periodicIntersects(Range const& other) const {
        Range iValue;
#ifdef __CUDA_ARCH__
        for (int dim = 0; dim < n; dim++) {
            iValue.begin[dim] = max(begin[dim], other.begin[dim]);
            iValue.end[dim] = min(end[dim], other.end[dim]);
            if (iValue.end[dim] <= iValue.begin[dim]) {
                iValue.begin[dim] = max(begin[dim] + T(1), other.begin[dim]);
                iValue.end[dim] = min(end[dim] + T(1), other.end[dim]);
                if (iValue.end[dim] <= iValue.begin[dim]) {
                    iValue.begin[dim] = max(begin[dim] - T(1), other.begin[dim]);
                    iValue.end[dim] = min(end[dim] - T(1), other.end[dim]);
                }
            }
        }
#else
        for (int dim = 0; dim < n; dim++) {
            iValue.begin[dim] = std::max(begin[dim], other.begin[dim]);
            iValue.end[dim] = std::min(end[dim], other.end[dim]);
            if (iValue.end[dim] <= iValue.begin[dim]) {
                iValue.begin[dim] = std::max(begin[dim] + T(1), other.begin[dim]);
                iValue.end[dim] = std::min(end[dim] + T(1), other.end[dim]);
                if (iValue.end[dim] <= iValue.begin[dim]) {
                    iValue.begin[dim] = std::max(begin[dim] - T(1), other.begin[dim]);
                    iValue.end[dim] = std::min(end[dim] - T(1), other.end[dim]);
                }
            }
        }
#endif
        for (int dim = 0; dim < n; dim++) {
            if (iValue.end[dim] <= iValue.begin[dim]) {
                return false;
            }
        }
        return true;
    }

    inline bool empty() const {
        return volume() == T(0);
    }

    Range() = default;
    Range(Range const&) = default;
    Range(Range&&) = default;
    Range& operator=(Range const&) = default;
    Range& operator=(Range&&) = default;

    inline Range shift(Array<T, n> const& s) const {
        Range r = *this;
        for (int dim = 0; dim < n; dim++) {
            r.begin[dim] += s[dim];
            r.end[dim] += s[dim];
        }
        return r;
    }

    Range(T const& sz) {
        for (int dim = 0; dim < n; dim++) {
            begin[dim] = T(0);
            end[dim] = sz;
        }
    }

    CUDA_EXPORT
    inline bool contains(Range<T, n> const& box) const {
        bool rc = true;
        for (int dim = 0; dim < n; dim++) {
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
    inline bool contains(Array<T, n> const& p) const {
        for (int dim = 0; dim < n; dim++) {
            if (p[dim] < begin[dim] || p[dim] >= end[dim]) {
                return false;
            }
        }
        return true;
    }

    inline bool periodicContains(Array<T, n> const& p) const {
        for (int dim = 0; dim < n; dim++) {
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

    inline std::string toString() const {
        std::string str;
        for (int dim = 0; dim < n; dim++) {
            str += std::to_string(dim) + ":(";
            str += std::to_string(begin[dim]) + ",";
            str += std::to_string(end[dim]) + ") ";
        }
        return str;
    }

    inline int longestDim() const {
        int maxDim;
        T maxSpan = T(-1);
        for (int dim = 0; dim < n; dim++) {
            T const span = end[dim] - begin[dim];
            if (span > maxSpan) {
                maxSpan = span;
                maxDim = dim;
            }
        }
        return maxDim;
    }

    inline int shortestDim() const {
        int minDim;
        T minSpan = T(1);
        for (int dim = 0; dim < n; dim++) {
            T const span = end[dim] - begin[dim];
            if (span <= minSpan) {
                minSpan = span;
                minDim = dim;
            }
        }
        return minDim;
    }

    inline std::pair<Range<T, n>, Range<T, n>> split() const {
        auto left = *this;
        auto right = *this;
        int maxDim;
        T maxSpan = 0;
        for (int dim = 0; dim < n; dim++) {
            T const span = end[dim] - begin[dim];
            if (span > maxSpan) {
                maxSpan = span;
                maxDim = dim;
            }
        }
        T const mid = (end[maxDim] + begin[maxDim]) / T(2);
        left.end[maxDim] = right.begin[maxDim] = mid;
        return std::make_pair(left, right);
    }

    Range<T, n> shiftUp() const {
        Range<T, n> rc;
        rc.begin = ::shiftUp(begin);
        rc.end = ::shiftUp(end);
        return rc;
    }

    Range<T, n> shiftDown() const {
        Range<T, n> rc;
        rc.begin = ::shiftDown(begin);
        rc.end = ::shiftDown(end);
        return rc;
    }

    CUDA_EXPORT
    inline T index(T xi, T yi, T zi) const {
        auto const spanz = end[2] - begin[2];
        auto const spany = end[1] - begin[1];
        return spanz * (spany * (xi - begin[0]) + (yi - begin[1])) + (zi - begin[2]);
    }

    CUDA_EXPORT
    inline T index(Array<T, n> const& i) const {
        return index(i.data());
    }

    CUDA_EXPORT
    inline T index(T const* i) const {
        auto const spanz = end[2] - begin[2];
        auto const spany = end[1] - begin[1];
        return spanz * (spany * (i[0] - begin[0]) + (i[1] - begin[1])) + (i[2] - begin[2]);
    }

    inline Range<T, n> transpose(int dim1, int dim2) const {
        auto rc = *this;
        std::swap(rc.begin[dim1], rc.begin[dim2]);
        std::swap(rc.end[dim1], rc.end[dim2]);
        return rc;
    }

    CUDA_EXPORT
    inline T volume() const {
        T vol = T(1);
        for (int dim = 0; dim < n; dim++) {
            T const span = end[dim] - begin[dim];
            if (span < T(0)) {
                return T(0);
            }
            vol *= span;
        }
        return vol;
    }

    template <class A>
    void serialize(A&& arc, unsigned) {
        for (int dim = 0; dim < n; dim++) {
            arc& begin[dim];
            arc& end[dim];
        }
    }

    CUDA_EXPORT
    inline Range<T, n> pad(T dx = T(1)) const {
        Range<T, n> r;
        for (int dim = 0; dim < n; dim++) {
            r.begin[dim] = begin[dim] - dx;
            r.end[dim] = end[dim] + dx;
        }
        return r;
    }
};

template <class T, int n = NDIM>
inline Range<T, n> unitBox() {
    Range<T, n> r;
    for (int dim = 0; dim < n; dim++) {
        r.begin[dim] = T(0);
        r.end[dim] = T(1);
    }
    return r;
}

#include <cosmictiger/fixed.hpp>

inline Range<Fixed32> fixed32UnitBox() {
    Range<Fixed32> r;
    for (int dim = 0; dim < NDIM; dim++) {
        r.begin[dim] = 0.0;
        r.end[dim] = Fixed32::max();
    }
    return r;
}

inline Range<Fixed32> rngdbl2rngfixed32(Range<double> const& other) {
    Range<Fixed32> rc;
    for (int dim = 0; dim < NDIM; dim++) {
        rc.begin[dim] = other.begin[dim];
        rc.end[dim] = other.end[dim];
    }
    return rc;
}

using RangeFixed = Fixed<int, 29>;

struct Fixed32Range : public Range<RangeFixed> {
    Fixed32Range() {}
    CUDA_EXPORT
    Fixed32Range& operator=(Fixed32Range const& other) {
        begin = other.begin;
        end = other.end;
        return *this;
    }
    CUDA_EXPORT
    Fixed32Range& operator=(Fixed32Range&& other) {
        begin = other.begin;
        end = other.end;
        return *this;
    }
    CUDA_EXPORT
    Fixed32Range(Fixed32Range const& other) {
        begin = other.begin;
        end = other.end;
    }
    CUDA_EXPORT
    Fixed32Range(Fixed32Range&& other) {
        begin = other.begin;
        end = other.end;
    }
    CUDA_EXPORT
    bool contains(Array<Fixed32, NDIM> const& pt) const {
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                for (int k = -1; k <= 1; k++) {
                    bool contains = true;
                    Array<double, NDIM> iValue;
                    iValue[0] = i;
                    iValue[1] = j;
                    iValue[2] = k;
                    for (int dim = 0; dim < NDIM; dim++) {
                        if (RangeFixed(pt[dim]) + RangeFixed::min() <
                            begin[dim] + RangeFixed(iValue[dim])) {
                            contains = false;
                            break;
                        }
                        if (RangeFixed(pt[dim]) >
                            end[dim] + RangeFixed(iValue[dim]) + RangeFixed::min()) {
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
    template <class A>
    void serialize(A&& arc, unsigned i) {
        Range<RangeFixed>::serialize(arc, i);
    }
};

CUDA_EXPORT
inline float distance(RangeFixed a, Fixed32 b) {
    float f = a.toDouble() - b.toDouble();
    f -= double(f > 0.5);
    f += double(f < -0.5);
    return f;
}

CUDA_EXPORT
inline float distance(Fixed32 b, RangeFixed a) {
    return -distance(a, b);
}

CUDA_EXPORT inline bool boxIntersectsSphere(
    Fixed32Range const& box, Array<Fixed32, NDIM> const& x, float r) {
    float d2 = 0.0f;
    for (int dim = 0; dim < NDIM; dim++) {
        float const d2begin = sqr(distance(x[dim], box.begin[dim]));
        float const d2end = sqr(distance(x[dim], box.end[dim]));
        d2 += ((RangeFixed(x[dim]) < box.begin[dim]) + (RangeFixed(x[dim]) > box.end[dim])) *
            fminf(d2begin, d2end);
    }
    return d2 < sqr(r);
}

#ifndef __CUDACC__
inline SimdFloat boxIntersectsSphere(
    Range<SimdDouble8> const& box, Array<SimdInt, NDIM> const& x, SimdFloat r) {
    SimdFloat d2 = 0.0f;
    for (int dim = 0; dim < NDIM; dim++) {
        SimdDouble8 xValue = SimdDouble8(x[dim]) * SimdDouble8(Fixed32::kToFloatFactor_);
        // Positions are unsigned coordinates in [0, 1), not signed displacements.
        xValue = xValue + (xValue < SimdDouble8(0.0));
        SimdDouble8 d2begin = xValue - box.begin[dim];
        SimdDouble8 d2beginp1 = d2begin + SimdDouble8(1.0);
        SimdDouble8 d2beginm1 = d2begin + SimdDouble8(-1.0);
        d2begin = sqr(d2begin);
        d2beginp1 = sqr(d2beginp1);
        d2beginm1 = sqr(d2beginm1);
        d2begin = min(d2begin, min(d2beginp1, d2beginm1));
        SimdDouble8 d2end = xValue - box.end[dim];
        SimdDouble8 d2endp1 = d2end + SimdDouble8(1.0);
        SimdDouble8 d2endm1 = d2end + SimdDouble8(-1.0);
        d2end = sqr(d2end);
        d2endp1 = sqr(d2endp1);
        d2endm1 = sqr(d2endm1);
        d2end = min(d2end, min(d2endp1, d2endm1));
        d2 +=
            SimdFloat(((xValue < box.begin[dim]) + (xValue > box.end[dim])) * min(d2begin, d2end));
    }
    return d2 < sqr(r);
}
#endif
#endif /* RANGE_HPP_ */
