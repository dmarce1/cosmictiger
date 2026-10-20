/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <cosmictiger/math/range.hpp>

namespace detail {

template <int n>
__device__ void exerciseRange(Range<double, n> const& box, Range<double, n> const& other,
    Range<double, n>* intersections, bool* intersects) {
    intersections[0] = box.intersection(other);
    intersections[1] = box.periodicIntersection(other);
    *intersects = box.periodicIntersects(other);
}

}    // namespace detail

__global__ void rangeCudaCompileRegression(Range<double, 2> const* boxes2,
    Range<double, 6> const* boxes6, Range<double, 2>* intersections2,
    Range<double, 6>* intersections6, bool* intersects) {
    detail::exerciseRange(boxes2[0], boxes2[1], intersections2, intersects);
    detail::exerciseRange(boxes6[0], boxes6[1], intersections6, intersects + 1);
}
