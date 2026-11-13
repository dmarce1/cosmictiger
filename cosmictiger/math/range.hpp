/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2021  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#ifndef RANGE_HPP_
#define RANGE_HPP_

#include <type_traits>

#include <cosmictiger/core/containers.hpp>
#include <cosmictiger/core/defs.hpp>
#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/cuda/cuda.hpp>
#include <cosmictiger/math/math.hpp>
#include <cosmictiger/math/position.hpp>
#include <cosmictiger/math/simd_math.hpp>
#include <cosmictiger/math/vector3.hpp>

/// Cyclically permute three coordinates from (x,y,z) to (y,z,x).
template <class T>
inline Array<T, kDimensions> shiftUp(Array<T, kDimensions> i) {
    Array<T, kDimensions> j;
    j[2] = i[0];
    j[0] = i[1];
    j[1] = i[2];
    return j;
}

/// Cyclically permute three coordinates from (x,y,z) to (z,x,y).
template <class T>
inline Array<T, kDimensions> shiftDown(Array<T, kDimensions> i) {
    Array<T, kDimensions> j;
    j[1] = i[0];
    j[2] = i[1];
    j[0] = i[2];
    return j;
}

template <class T, int n = kDimensions>
struct Range {
    Array<T, n> begin;
    Array<T, n> end;

    /// Return coordinatewise overlap bounds; disjoint boxes produce an empty range.
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
    /// Find overlap with the current box or its adjacent unit-period images along each
    /// axis.
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
    /// Test for positive-volume overlap through adjacent unit-period images, optionally
    /// counting shift operations.
    CUDA_EXPORT
    inline bool periodicIntersects(Range const& other, FlopCounter<>* flops = nullptr) const {
        Range iValue;
#ifdef __CUDA_ARCH__
        for (int dim = 0; dim < n; dim++) {
            iValue.begin[dim] = max(begin[dim], other.begin[dim]);
            iValue.end[dim] = min(end[dim], other.end[dim]);
            if (iValue.end[dim] <= iValue.begin[dim]) {
                iValue.begin[dim] = max(begin[dim] + T(1), other.begin[dim]);
                iValue.end[dim] = min(end[dim] + T(1), other.end[dim]);
                if constexpr (!std::is_integral_v<T>) {
                    // Floating shifts or the two integer-to-fixed scaling products.
                    if (flops) {
                        *flops += 2;
                    }
                }
                if (iValue.end[dim] <= iValue.begin[dim]) {
                    iValue.begin[dim] = max(begin[dim] - T(1), other.begin[dim]);
                    iValue.end[dim] = min(end[dim] - T(1), other.end[dim]);
                    if constexpr (!std::is_integral_v<T>) {
                        // Floating shifts or the two integer-to-fixed scaling products.
                        if (flops) {
                            *flops += 2;
                        }
                    }
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
                if constexpr (!std::is_integral_v<T>) {
                    // Floating shifts or the two integer-to-fixed scaling products.
                    if (flops) {
                        *flops += 2;
                    }
                }
                if (iValue.end[dim] <= iValue.begin[dim]) {
                    iValue.begin[dim] = std::max(begin[dim] - T(1), other.begin[dim]);
                    iValue.end[dim] = std::min(end[dim] - T(1), other.end[dim]);
                    if constexpr (!std::is_integral_v<T>) {
                        // Floating shifts or the two integer-to-fixed scaling products.
                        if (flops) {
                            *flops += 2;
                        }
                    }
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

    /// Test whether the range has zero volume, including inverted bounds.
    inline bool empty() const {
        return volume() == T(0);
    }

    Range() = default;
    Range(Range const&) = default;
    Range(Range&&) = default;
    Range& operator=(Range const&) = default;
    Range& operator=(Range&&) = default;

    /// Return a copy translated by the supplied coordinate displacement.
    inline Range shift(Array<T, n> const& s) const {
        Range r = *this;
        for (int dim = 0; dim < n; dim++) {
            r.begin[dim] += s[dim];
            r.end[dim] += s[dim];
        }
        return r;
    }

    /// Return a copy translated by the supplied coordinate displacement.
    inline Range shift(Vector3<T> const& s) const
        requires(n == 3)
    {
        Array<T, n> components;
        for (int dim = 0; dim < n; ++dim) {
            components[dim] = s[dim];
        }
        return shift(components);
    }

    /// Construct an origin-based box with the supplied side length in every dimension.
    Range(T const& sz) {
        for (int dim = 0; dim < n; dim++) {
            begin[dim] = T(0);
            end[dim] = sz;
        }
    }

    /// Test whether this box encloses both endpoints of the other box along every axis.
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

    /// Test membership using half-open bounds: lower endpoints included, upper
    /// endpoints excluded.
    CUDA_EXPORT
    inline bool contains(Array<T, n> const& p) const {
        for (int dim = 0; dim < n; dim++) {
            if (p[dim] < begin[dim] || p[dim] >= end[dim]) {
                return false;
            }
        }
        return true;
    }

    /// Test membership using half-open bounds: lower endpoints included, upper
    /// endpoints excluded.
    CUDA_EXPORT
    inline bool contains(Vector3<T> const& p) const
        requires(n == 3)
    {
        Array<T, n> components;
        for (int dim = 0; dim < n; ++dim) {
            components[dim] = p[dim];
        }
        return contains(components);
    }

    /// Test half-open point membership in the box or its adjacent unit-period images.
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

    /// Test half-open point membership in the box or its adjacent unit-period images.
    inline bool periodicContains(Vector3<T> const& p) const
        requires(n == 3)
    {
        Array<T, n> components;
        for (int dim = 0; dim < n; ++dim) {
            components[dim] = p[dim];
        }
        return periodicContains(components);
    }

    /// Format each axis's lower and upper bounds for diagnostics.
    inline std::string toString() const {
        std::string str;
        for (int dim = 0; dim < n; dim++) {
            str += std::to_string(dim) + ":(";
            str += std::to_string(begin[dim]) + ",";
            str += std::to_string(end[dim]) + ") ";
        }
        return str;
    }

    /// Return the first axis with the largest extent.
    inline int longestDim() const {
        int maxDim = 0;
        T maxSpan = end[0] - begin[0];
        for (int dim = 0; dim < n; dim++) {
            T const span = end[dim] - begin[dim];
            if (span > maxSpan) {
                maxSpan = span;
                maxDim = dim;
            }
        }
        return maxDim;
    }

    /// Return the last axis with the smallest extent.
    inline int shortestDim() const {
        int minDim = 0;
        T minSpan = end[0] - begin[0];
        for (int dim = 0; dim < n; dim++) {
            T const span = end[dim] - begin[dim];
            if (span <= minSpan) {
                minSpan = span;
                minDim = dim;
            }
        }
        return minDim;
    }

    /// Bisect the first longest axis and return the two resulting boxes.
    inline std::pair<Range<T, n>, Range<T, n>> split() const {
        auto left = *this;
        auto right = *this;
        int maxDim = 0;
        T maxSpan = end[0] - begin[0];
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

    /// Cyclically permute three coordinates from (x,y,z) to (y,z,x).
    Range<T, n> shiftUp() const {
        Range<T, n> rc;
        rc.begin = ::shiftUp(begin);
        rc.end = ::shiftUp(end);
        return rc;
    }

    /// Cyclically permute three coordinates from (x,y,z) to (z,x,y).
    Range<T, n> shiftDown() const {
        Range<T, n> rc;
        rc.begin = ::shiftDown(begin);
        rc.end = ::shiftDown(end);
        return rc;
    }

    /// Flatten a three-dimensional coordinate relative to the box origin, with z
    /// varying fastest.
    CUDA_EXPORT
    inline T index(T xi, T yi, T zi) const {
        auto const spanz = end[2] - begin[2];
        auto const spany = end[1] - begin[1];
        return spanz * (spany * (xi - begin[0]) + (yi - begin[1])) + (zi - begin[2]);
    }

    /// Flatten a three-dimensional coordinate relative to the box origin, with z
    /// varying fastest.
    CUDA_EXPORT
    inline T index(Array<T, n> const& i) const {
        return index(i.data());
    }

    /// Flatten a three-dimensional coordinate relative to the box origin, with z
    /// varying fastest.
    CUDA_EXPORT
    inline T index(T const* i) const {
        auto const spanz = end[2] - begin[2];
        auto const spany = end[1] - begin[1];
        return spanz * (spany * (i[0] - begin[0]) + (i[1] - begin[1])) + (i[2] - begin[2]);
    }

    /// Return a copy with the two selected coordinate axes exchanged.
    inline Range<T, n> transpose(int dim1, int dim2) const {
        auto rc = *this;
        std::swap(rc.begin[dim1], rc.begin[dim2]);
        std::swap(rc.end[dim1], rc.end[dim2]);
        return rc;
    }

    /// Multiply axis extents, returning zero when any extent is negative.
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

    /// Archive `begin`, `end` in transport order.
    template <class A>
    void serialize(A&& arc, unsigned) {
        for (int dim = 0; dim < n; dim++) {
            arc& begin[dim];
            arc& end[dim];
        }
    }

    /// Expand both ends of every axis by dx.
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

/// Construct the unit box with bounds zero and one on every axis.
template <class T, int n = kDimensions>
inline Range<T, n> unitBox() {
    Range<T, n> r;
    for (int dim = 0; dim < n; dim++) {
        r.begin[dim] = T(0);
        r.end[dim] = T(1);
    }
    return r;
}

#include <cosmictiger/math/position.hpp>

/// Construct position-storage bounds from zero to the largest representable coordinate.
inline Range<Position> fixed32UnitBox() {
    Range<Position> r;
    for (int dim = 0; dim < kDimensions; dim++) {
        r.begin[dim] = 0.0;
        r.end[dim] = Position::max();
    }
    return r;
}

/// Convert double box endpoints into the configured fixed-point position representation.
inline Range<Position> rngdbl2rngfixed32(Range<double> const& other) {
    Range<Position> rc;
    for (int dim = 0; dim < kDimensions; dim++) {
        rc.begin[dim] = other.begin[dim];
        rc.end[dim] = other.end[dim];
    }
    return rc;
}

using RangeFixed = Fixed<int, 29>;

struct SphBox : public Range<RangeFixed> {
    /// Default-construct a fixed-coordinate range without assigning its endpoints.
    SphBox() {}
    /// Copy both endpoint arrays from the supplied fixed-coordinate box.
    CUDA_EXPORT
    SphBox& operator=(SphBox const& other) {
        begin = other.begin;
        end = other.end;
        return *this;
    }
    /// Copy both endpoint arrays from the supplied fixed-coordinate box.
    CUDA_EXPORT
    SphBox& operator=(SphBox&& other) {
        begin = other.begin;
        end = other.end;
        return *this;
    }
    /// Construct a fixed-coordinate box by copying both endpoint arrays.
    CUDA_EXPORT
    SphBox(SphBox const& other) {
        begin = other.begin;
        end = other.end;
    }
    /// Construct a fixed-coordinate box by copying both endpoint arrays.
    CUDA_EXPORT
    SphBox(SphBox&& other) {
        begin = other.begin;
        end = other.end;
    }
    /// Test point membership across adjacent periodic images with one fixed-point tick
    /// of boundary tolerance.
    CUDA_EXPORT
    bool contains(Vector3<Position> const& pt, FlopCounter<>* flops = nullptr) const {
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                for (int k = -1; k <= 1; k++) {
                    bool contains = true;
                    Vector3<double> iValue;
                    iValue[0] = i;
                    iValue[1] = j;
                    iValue[2] = k;
                    for (int dim = 0; dim < kDimensions; dim++) {
                        if (flops) {
                            *flops += 1;    // Image offset conversion to fixed-point.
                        }
                        if (RangeFixed(pt[dim]) + RangeFixed::min() <
                            begin[dim] + RangeFixed(iValue[dim])) {
                            contains = false;
                            break;
                        }
                        if (flops) {
                            *flops += 1;
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
    /// Archive this record in the field order required by distributed transport.
    template <class A>
    void serialize(A&& arc, unsigned i) {
        Range<RangeFixed>::serialize(arc, i);
    }
};

/// Return the signed minimum-image separation between a range endpoint and a particle
/// coordinate.
CUDA_EXPORT
inline float distance(RangeFixed a, Position b) {
    if constexpr (kPositionBits == 64) {
        // Range endpoints are exact fixed-point values. Subtract in integer
        // space before converting, including across the periodic boundary.
        return distance(Position(a), b);
    }
    float f = a.toDouble() - b.toDouble();
    f -= double(f > 0.5);
    f += double(f < -0.5);
    return f;
}

/// Return the signed minimum-image separation between a range endpoint and a particle
/// coordinate.
CUDA_EXPORT
inline float distance(Position b, RangeFixed a) {
    return -distance(a, b);
}

/// Return the squared minimum periodic distance from a particle to the box, preserving
/// fixed-coordinate boundary precision.
CUDA_EXPORT inline float boxDistanceSquared(SphBox const& box, Vector3<Position> const& x) {
    float d2 = 0.0f;
    for (int dim = 0; dim < kDimensions; dim++) {
        if (box.end[dim].toDouble() - box.begin[dim].toDouble() >= 1.0) {
            continue;
        }
        PositionInteger const lower = Position(box.begin[dim]).raw();
        PositionInteger const length = Position(box.end[dim]).raw() - lower;
        PositionInteger const offset = x[dim].raw() - lower;
        PositionInteger ticks = 0;
        if (offset > length) {
            PositionInteger const fromUpper = offset - length;
            PositionInteger const fromLower = PositionInteger{0} - offset;
            ticks = fromUpper < fromLower ? fromUpper : fromLower;
        }
        float const delta = float(double(ticks) * Position::kToFloatFactor_);
        d2 = fmaf(delta, delta, d2);
    }
    return d2;
}

/// Test whether the minimum periodic box distance is strictly smaller than the sphere
/// radius.
CUDA_EXPORT inline bool boxIntersectsSphere(
    SphBox const& box, Vector3<Position> const& x, float r) {
    return boxDistanceSquared(box, x) < sqr(r);
}

#ifndef __CUDACC__
/// Return the squared minimum periodic distance from a particle to the box, preserving
/// fixed-coordinate boundary precision.
inline SimdFloat boxDistanceSquared(Range<SimdDouble8> const& box, Vector3<SimdPosition> const& x) {
    if constexpr (kPositionBits == 64) {
        // The SIMD box endpoints originate in RangeFixed. Reuse the integer
        // comparisons and subtraction so conversion to double cannot erase
        // a particle's low bits at an endpoint.
        return SimdFloat([&](auto lane) {
            SphBox scalarBox;
            Vector3<Position> point;
            for (int dim = 0; dim < kDimensions; ++dim) {
                scalarBox.begin[dim] = box.begin[dim][lane];
                scalarBox.end[dim] = box.end[dim][lane];
                point[dim].setInteger(static_cast<PositionInteger>(x[dim][lane]));
            }
            return boxDistanceSquared(scalarBox, point);
        });
    }
    SimdFloat d2 = 0.0f;
    for (int dim = 0; dim < kDimensions; dim++) {
        SimdDouble8 xValue =
            simdCast<SimdDouble8>(x[dim]) * math::broadcast<SimdDouble8>(Position::kToFloatFactor_);
        // Positions are unsigned coordinates in [0, 1), not signed displacements.
        xValue = math::select(xValue < SimdDouble8(0.0), xValue + SimdDouble8(1.0), xValue);
        // Q0.32 values and Q32.32 box endpoints are exact in double.
        // Measure from the lower endpoint modulo one, then test containment
        // in that same periodic interval before choosing the nearest endpoint.
        SimdDouble8 offset = xValue - box.begin[dim];
        offset -= simdFloor(offset);
        SimdDouble8 const length = box.end[dim] - box.begin[dim];
        auto const outside = (length < SimdDouble8(1.0)) & (offset > length);
        SimdDouble8 const nearest = math::min(offset - length, SimdDouble8(1.0) - offset);
        SimdFloat const delta = simdCast<SimdFloat>(math::select(outside, nearest, SimdDouble8(0.0)));
        d2 = math::fma(delta, delta, d2);
    }
    return d2;
}

/// Test whether the minimum periodic box distance is strictly smaller than the sphere
/// radius.
inline auto boxIntersectsSphere(
    Range<SimdDouble8> const& box, Vector3<SimdPosition> const& x, SimdFloat r) {
    return boxDistanceSquared(box, x) < sqr(r);
}
#endif
#endif /* RANGE_HPP_ */
