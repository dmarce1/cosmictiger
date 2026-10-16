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
