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

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>

#include <cosmictiger/math.hpp>
#include <cosmictiger/range.hpp>

void cosmictigerAssert(char const* expression, bool condition, char const* file, int line) {
    if (!condition) {
        std::fprintf(stderr, "Assertion failed: %s at %s:%d\n", expression, file, line);
        std::abort();
    }
}

namespace detail {

void require(bool condition, char const* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void testErfcexp() {
    double const arguments[] = {0.0, 1.0, 8.0, -1.0};
    double const tolerance = 8.0 * std::numeric_limits<double>::epsilon();
    for (double const argument : arguments) {
        double complement;
        double exponential;
        erfcexp(argument, &complement, &exponential);
        double const expectedComplement = std::erfc(argument);
        double const expectedExponential = std::exp(-argument * argument);
        require(std::abs(complement - expectedComplement) <= tolerance * expectedComplement,
            "erfcexp must return the complementary error function");
        require(std::abs(exponential - expectedExponential) <= tolerance * expectedExponential,
            "erfcexp must return exp(-x*x)");
    }
}

void testPeriodicBoxSphere() {
    Fixed32Range box;
    Range<SimdDouble8> simdBox;
    for (int dim = 0; dim < NDIM; dim++) {
        box.begin[dim] = dim == XDIM ? 0.70 : 0.20;
        box.end[dim] = dim == XDIM ? 0.80 : 0.30;
        simdBox.begin[dim] = box.begin[dim].toDouble();
        simdBox.end[dim] = box.end[dim].toDouble();
    }
    double const coordinates[8][NDIM] = {{0.75, 0.25, 0.25}, {0.81, 0.25, 0.25}, {0.85, 0.25, 0.25},
        {0.25, 0.25, 0.25}, {0.75, 0.35, 0.25}, {0.75, 0.25, 0.35}, {0.70, 0.25, 0.25},
        {0.50, 0.25, 0.25}};
    float const radii[8] = {0.02f, 0.02f, 0.02f, 0.02f, 0.02f, 0.06f, 0.01f, 0.01f};
    bool const expected[8] = {true, true, false, false, false, true, true, false};
    Array<SimdInt, NDIM> simdPoints{};
    SimdFloat simdRadii{};
    for (int lane = 0; lane < SIMD_FLOAT_SIZE; lane++) {
        Array<Fixed32, NDIM> point;
        for (int dim = 0; dim < NDIM; dim++) {
            point[dim] = coordinates[lane][dim];
            simdSetLane(simdPoints[dim], lane, point[dim].raw());
        }
        simdSetLane(simdRadii, lane, radii[lane]);
        require(boxIntersectsSphere(box, point, radii[lane]) == expected[lane],
            "Scalar box-sphere test disagrees with the geometry");
    }
    auto const intersections = boxIntersectsSphere(simdBox, simdPoints, simdRadii);
    auto const distances = boxDistanceSquared(simdBox, simdPoints);
    for (int lane = 0; lane < SIMD_FLOAT_SIZE; lane++) {
        require(bool(intersections[lane]) == expected[lane],
            "SIMD box-sphere test must preserve unsigned fixed-point positions");
        Array<Fixed32, NDIM> point;
        for (int dim = 0; dim < NDIM; dim++) {
            point[dim] = coordinates[lane][dim];
        }
        require(distances[lane] == boxDistanceSquared(box, point),
            "Scalar and SIMD point-box distances must agree exactly");
    }
    box.begin[XDIM] = 0.95;
    box.end[XDIM] = 0.99;
    simdBox.begin[XDIM] = box.begin[XDIM].toDouble();
    simdBox.end[XDIM] = box.end[XDIM].toDouble();
    Array<Fixed32, NDIM> point;
    for (int dim = 0; dim < NDIM; dim++) {
        point[dim] = dim == XDIM ? 0.01 : 0.25;
        simdPoints[dim] = simdBroadcast<SimdInt>(point[dim].raw());
    }
    require(boxIntersectsSphere(box, point, 0.03f),
        "Box-sphere test must account for the periodic boundary");
    require(simdAllOf(boxIntersectsSphere(simdBox, simdPoints, SimdFloat(0.03f))),
        "SIMD box-sphere test must account for the periodic boundary");

    for (int dim = 0; dim < NDIM; dim++) {
        box.begin[dim] = 0.25;
        box.end[dim] = 0.5;
        simdBox.begin[dim] = 0.25;
        simdBox.end[dim] = 0.5;
        point[dim] = dim == XDIM ? 0.5 : 0.375;
    }
    point[XDIM].raw()++;
    for (int dim = 0; dim < NDIM; dim++) {
        simdPoints[dim] = simdBroadcast<SimdInt>(point[dim].raw());
    }
    float const oneTickSquared = float(Fixed32::kToFloatFactor_ * Fixed32::kToFloatFactor_);
    require(boxDistanceSquared(box, point) == oneTickSquared,
        "Point-box containment must retain all 32 position bits");
    require(boxDistanceSquared(simdBox, simdPoints)[0] == oneTickSquared,
        "SIMD point-box containment must retain all 32 position bits");

    // Exercise float rounding and unsigned coordinates near periodic boundaries.
    for (int sample = 0; sample < 256; sample++) {
        for (int dim = 0; dim < NDIM; dim++) {
            box.begin[dim] = ((sample * 37 + dim * 43) % 193) / 256.0;
            box.end[dim] = box.begin[dim].toDouble() + 0.125;
            simdBox.begin[dim] = box.begin[dim].toDouble();
            simdBox.end[dim] = box.end[dim].toDouble();
        }
        Array<Array<Fixed32, NDIM>, SIMD_FLOAT_SIZE> points;
        for (int lane = 0; lane < SIMD_FLOAT_SIZE; lane++) {
            for (int dim = 0; dim < NDIM; dim++) {
                points[lane][dim] = ((sample * 97 + lane * 71 + dim * 83) % 1009) / 1009.0;
                simdSetLane(simdPoints[dim], lane, points[lane][dim].raw());
            }
        }
        auto const squaredDistances = boxDistanceSquared(simdBox, simdPoints);
        for (int lane = 0; lane < SIMD_FLOAT_SIZE; lane++) {
            require(squaredDistances[lane] == boxDistanceSquared(box, points[lane]),
                "Periodic scalar/SIMD distance rounding differs");
            double reference = 0.0;
            for (int dim = 0; dim < NDIM; dim++) {
                double nearest = 1.0;
                double const coordinate = points[lane][dim].toDouble();
                for (int shift = -1; shift <= 1; shift++) {
                    double const lower = box.begin[dim].toDouble() + shift;
                    double const upper = box.end[dim].toDouble() + shift;
                    nearest = std::min(
                        nearest, std::max(0.0, std::max(lower - coordinate, coordinate - upper)));
                }
                reference += nearest * nearest;
            }
            require(std::abs(squaredDistances[lane] - reference) < 2.0e-7,
                "Periodic point-box distance differs from an independent interval reference");
        }
    }
}

template <int n>
void testRanges() {
    Range<double, n> box(0.25);
    Range<double, n> other = box;
    other.begin[n - 1] = 0.5;
    other.end[n - 1] = 0.75;
    require(!box.periodicIntersects(other), "Separated boxes must not periodically intersect");

    other = box;
    other.begin[n - 1] = 0.125;
    other.end[n - 1] = 0.375;
    require(box.periodicIntersects(other), "Overlapping boxes must periodically intersect");
    auto const intersection = box.intersection(other);
    for (int dim = 0; dim < n; dim++) {
        require(intersection.begin[dim] == (dim == n - 1 ? 0.125 : 0.0),
            "Intersection lower endpoint is incorrect");
        require(intersection.end[dim] == 0.25, "Intersection upper endpoint is incorrect");
    }

    for (double const shift : {-1.0, 1.0}) {
        other = box;
        other.begin[n - 1] = shift + 0.125;
        other.end[n - 1] = shift + 0.375;
        require(box.periodicIntersects(other), "Periodically shifted boxes must intersect");
        auto const periodic = box.periodicIntersection(other);
        require(periodic.begin[n - 1] == shift + 0.125,
            "Periodic intersection lower endpoint is incorrect");
        require(periodic.end[n - 1] == shift + 0.25,
            "Periodic intersection upper endpoint is incorrect");
    }

    other = box;
    other.begin[n - 1] = 0.25;
    other.end[n - 1] = 0.5;
    require(!box.periodicIntersects(other), "Touching half-open boxes must not intersect");
}

}    // namespace detail

int main() {
    try {
        detail::testErfcexp();
        detail::testPeriodicBoxSphere();
        detail::testRanges<6>();
        detail::testRanges<3>();
        detail::testRanges<2>();
        detail::testRanges<1>();
    } catch (std::exception const& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return EXIT_FAILURE;
    }
    std::puts("Numerical utility regressions passed");
    return EXIT_SUCCESS;
}
