/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

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
