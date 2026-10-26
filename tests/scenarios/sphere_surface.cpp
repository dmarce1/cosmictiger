/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <hpx/hpx_init.hpp>

#include <cosmictiger/scenarios/sphere.hpp>

namespace detail {

using Shells = Vector<Array<Vector<float>, kDimensions>>;

static void require(bool const condition, std::string const& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

static double dot(Array<Vector<float>, kDimensions> const& shell, int const i, int const j) {
    double sum = 0.0;
    for (int dim = 0; dim < kDimensions; dim++) {
        sum += double(shell[dim][i]) * shell[dim][j];
    }
    return sum;
}

static void checkUnitSphere(Array<Vector<float>, kDimensions> const& shell) {
    int const n = shell[kXDimension].size();
    for (int i = 0; i < n; i++) {
        double const norm = std::sqrt(dot(shell, i, i));
        require(std::isfinite(norm) && std::abs(norm - 1.0) < 1e-5,
            "point " + std::to_string(i) + " of " + std::to_string(n) + " is off the unit sphere");
    }
}

// Every pair dot product of the shell must match one of the expected values,
// counted per point; this fixes the antipodal pair, tetrahedron and octahedron.
static void checkPolyhedron(
    Array<Vector<float>, kDimensions> const& shell, std::vector<double> const& expected) {
    int const n = shell[kXDimension].size();
    for (int i = 0; i < n; i++) {
        std::vector<double> dots;
        for (int j = 0; j < n; j++) {
            if (j != i) {
                dots.push_back(dot(shell, i, j));
            }
        }
        std::sort(dots.begin(), dots.end());
        for (int k = 0; k < static_cast<int>(dots.size()); k++) {
            require(std::abs(dots[k] - expected[k]) < 1e-3,
                std::to_string(n) + "-point equilibrium dot product " + std::to_string(dots[k]) +
                    " differs from " + std::to_string(expected[k]));
        }
    }
}

// Random points crowd to separations of order 1/n; the relaxed shell must keep
// every nearest neighbor near the hexagonal spacing sqrt(8 pi / (sqrt(3) n)).
static void checkEvenSpacing(Array<Vector<float>, kDimensions> const& shell) {
    int const n = shell[kXDimension].size();
    double const spacing = std::sqrt(4.0 * std::numbers::pi / n);
    double minNearest = std::numeric_limits<double>::max();
    double maxNearest = 0.0;
    Array<double, kDimensions> centroid;
    for (int dim = 0; dim < kDimensions; dim++) {
        centroid[dim] = 0.0;
    }
    for (int i = 0; i < n; i++) {
        double nearest = std::numeric_limits<double>::max();
        for (int j = 0; j < n; j++) {
            if (j != i) {
                nearest = std::min(nearest, std::sqrt(std::max(0.0, 2.0 - 2.0 * dot(shell, i, j))));
            }
        }
        minNearest = std::min(minNearest, nearest);
        maxNearest = std::max(maxNearest, nearest);
        for (int dim = 0; dim < kDimensions; dim++) {
            centroid[dim] += shell[dim][i] / double(n);
        }
    }
    std::printf("n = %i nearest neighbor %.3f..%.3f of sqrt(4 pi / n)\n", n, minNearest / spacing,
        maxNearest / spacing);
    require(minNearest > 0.8 * spacing,
        std::to_string(n) + "-point shell has crowded points: " + std::to_string(minNearest));
    require(maxNearest < 1.25 * spacing,
        std::to_string(n) + "-point shell has a gap: " + std::to_string(maxNearest));
    double const offset = std::sqrt(centroid[kXDimension] * centroid[kXDimension] +
        centroid[kYDimension] * centroid[kYDimension] +
        centroid[kZDimension] * centroid[kZDimension]);
    require(
        offset < 1e-2, std::to_string(n) + "-point shell is lopsided: " + std::to_string(offset));
}

static Shells makeShells(std::vector<int> const& counts) {
    Shells shells;
    for (int const count : counts) {
        Array<Vector<float>, kDimensions> shell;
        for (int dim = 0; dim < kDimensions; dim++) {
            shell[dim].resize(count);
        }
        shells.push_back(std::move(shell));
    }
    return shells;
}

static void checkSphereSurface() {
    std::vector<int> const counts = {0, 1, 2, 4, 6, 300, 1500};
    Shells shells = makeShells(counts);
    solveSphereSurfaceProblemCpu(shells);
    for (auto const& shell : shells) {
        checkUnitSphere(shell);
    }
    checkPolyhedron(shells[2], {-1.0});
    checkPolyhedron(shells[3], {-1.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0});
    checkPolyhedron(shells[4], {-1.0, 0.0, 0.0, 0.0, 0.0});
    checkEvenSpacing(shells[5]);
    checkEvenSpacing(shells[6]);
    Shells repeat = makeShells(counts);
    solveSphereSurfaceProblem(repeat);
    for (int i = 0; i < static_cast<int>(shells.size()); i++) {
        for (int dim = 0; dim < kDimensions; dim++) {
            require(repeat[i][dim] == shells[i][dim],
                "the CPU solver is not deterministic for shell " + std::to_string(i));
        }
    }
}

}    // namespace detail

TEST(SphereSurface, RelaxationAndDeterminism) {
    detail::checkSphereSurface();
}

int hpx_main(int, char*[]) {
    int const status = RUN_ALL_TESTS();
    hpx::finalize();
    return status;
}

int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    if (GTEST_FLAG_GET(list_tests)) {
        return RUN_ALL_TESTS();
    }
    std::vector<std::string> configuration = {"hpx.os_threads=2"};
#ifdef HPX_EARLY
    return hpx::init(argc, argv, configuration);
#else
    hpx::init_params parameters;
    parameters.cfg = std::move(configuration);
    return hpx::init(argc, argv, parameters);
#endif
}
