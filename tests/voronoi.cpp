/*
 * Copyright (C) 2026 Dominic C. Marcello
 * Distributed under the Boost Software License, Version 1.0.
 * See LICENSE_1_0.txt.
 */

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>

#if defined(__SSE__)
#include <xmmintrin.h>
#endif

#include <gtest/gtest.h>

#include <cosmictiger/geometry/voronoi.hpp>

namespace detail {

void require(bool condition, std::string const& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void near(double actual, double expected, double tolerance, std::string const& message) {
    require(std::abs(actual - expected) <= tolerance, message);
}

template <typename Function>
void throws(Function function, std::string const& message) {
    bool caught = false;
    try {
        function();
    } catch (std::runtime_error const&) {
        caught = true;
    }
    require(caught, message);
}

std::vector<voronoi::Site> images(std::vector<voronoi::Point> const& points) {
    std::vector<voronoi::Site> sites;
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            for (int z = -1; z <= 1; z++) {
                for (size_t i = 0; i < points.size(); i++) {
                    sites.push_back({i, {points[i][0] + x, points[i][1] + y, points[i][2] + z},
                        int(i % 3), {x, y, z}});
                }
            }
        }
    }
    return sites;
}

std::vector<size_t> centralIndices(std::vector<voronoi::Site> const& sites) {
    std::vector<size_t> indices;
    for (size_t i = 0; i < sites.size(); i++) {
        if (sites[i].image == std::array<int, 3>{}) {
            indices.push_back(i);
        }
    }
    return indices;
}

void checkCells(std::vector<voronoi::Cell> const& cells, double expectedVolume) {
    double totalVolume = 0;
    for (auto const& cell : cells) {
        require(cell.complete, "periodic central cells must be bounded");
        require(cell.volume > 0, "positive volume");
        require(std::isfinite(cell.searchRadius) && cell.searchRadius >= 2 * cell.maxRadius,
            "conservative halo certificate");
        totalVolume += cell.volume;
        voronoi::Point closure = {};
        for (auto const& face : cell.faces) {
            require(face.vertices.size() >= 3, "polygon has at least three vertices");
            double outward = 0;
            for (size_t dim = 0; dim < 3; dim++) {
                closure[dim] += face.areaVector[dim];
                outward +=
                    face.areaVector[dim] * (face.neighbor.position[dim] - cell.site.position[dim]);
            }
            require(outward > 0, "face orientation must be outward");
            for (auto const& point : face.vertices) {
                long double distanceA = 0;
                long double distanceB = 0;
                for (size_t dim = 0; dim < 3; dim++) {
                    long double const a =
                        static_cast<long double>(point[dim]) - cell.site.position[dim];
                    long double const b =
                        static_cast<long double>(point[dim]) - face.neighbor.position[dim];
                    distanceA += a * a;
                    distanceB += b * b;
                }
                near(double(distanceA), double(distanceB), 1e-12,
                    "dual vertex must lie on the bisecting plane");
            }
            if (face.neighbor.image == std::array<int, 3>{}) {
                auto const neighbor = std::find_if(cells.begin(), cells.end(),
                    [&](auto const& other) { return other.site.id == face.neighbor.id; });
                require(neighbor != cells.end(), "central neighbor is available");
                auto const reverse = std::find_if(
                    neighbor->faces.begin(), neighbor->faces.end(), [&](auto const& other) {
                        return other.neighbor.id == cell.site.id &&
                            other.neighbor.image == std::array<int, 3>{};
                    });
                require(reverse != neighbor->faces.end(), "shared face must be reciprocal");
                require(
                    face.centroid == reverse->centroid, "shared face centroid must be canonical");
                auto reversed = reverse->vertices;
                std::reverse(reversed.begin() + 1, reversed.end());
                require(face.vertices == reversed, "shared face coordinates must be identical");
                for (size_t dim = 0; dim < 3; dim++) {
                    require(face.areaVector[dim] == -reverse->areaVector[dim],
                        "shared area vectors must be exact opposites");
                }
            }
        }
        for (double const residual : closure) {
            near(residual, 0, 2e-13, "closed polyhedral surface");
        }
    }
    near(
        totalVolume, expectedVolume, 2e-12 * expectedVolume, "periodic cells must tile the domain");
}

void periodicCube() {
    auto const sites = images({{0.5, 0.5, 0.5}});
    auto const cells = voronoi::buildCells(sites, centralIndices(sites));
    checkCells(cells, 1);
    require(cells.size() == 1 && cells[0].faces.size() == 6, "single periodic cell is a cube");
    near(cells[0].volume, 1, 1e-15, "single cube volume");
    require(cells[0].centroid == voronoi::Point{0.5, 0.5, 0.5}, "single cube centroid");
    for (auto const& face : cells[0].faces) {
        require(face.vertices.size() == 4, "cospherical degeneracies collapse to square faces");
    }
}

void periodicLattice() {
    std::vector<voronoi::Point> points;
    for (int x = 0; x < 2; x++) {
        for (int y = 0; y < 2; y++) {
            for (int z = 0; z < 2; z++) {
                points.push_back({0.25 + 0.5 * x, 0.25 + 0.5 * y, 0.25 + 0.5 * z});
            }
        }
    }
    auto const sites = images(points);
    auto const cells = voronoi::buildCells(sites, centralIndices(sites));
    checkCells(cells, 1);
    for (auto const& cell : cells) {
        near(cell.volume, 0.125, 1e-15, "regular lattice cell volume");
        require(cell.faces.size() == 6, "regular lattice cube faces");
        require(cell.centroid == cell.site.position, "regular lattice centroid");
        std::array<size_t, 1> const selected = {cell.inputIndex};
        auto const separate = voronoi::buildCells(sites, selected);
        require(cell.faces.size() == separate[0].faces.size(), "ownership subset keeps topology");
        for (size_t face = 0; face < cell.faces.size(); face++) {
            require(cell.faces[face].vertices == separate[0].faces[face].vertices,
                "ownership subset keeps canonical cospherical vertices");
        }
    }
}

void periodicRandom() {
    std::mt19937 generator(712);
    std::uniform_real_distribution<double> random(0, 1);
    std::vector<voronoi::Point> points;
    for (size_t i = 0; i < 48; i++) {
        points.push_back({random(generator), random(generator), random(generator)});
    }
    auto sites = images(points);
    auto const cells = voronoi::buildCells(sites, centralIndices(sites));
    checkCells(cells, 1);
    std::shuffle(sites.begin(), sites.end(), generator);
    auto reordered = voronoi::buildCells(sites, centralIndices(sites));
    std::sort(reordered.begin(), reordered.end(),
        [](auto const& a, auto const& b) { return a.site.id < b.site.id; });
    require(cells.size() == reordered.size(), "all selected sites retained after shuffle");
    for (size_t i = 0; i < cells.size(); i++) {
        require(
            cells[i].volume == reordered[i].volume && cells[i].centroid == reordered[i].centroid,
            "input ordering must not change geometry");
        require(
            cells[i].faces.size() == reordered[i].faces.size(), "input order invariant topology");
    }
}

void tinyFaceAndThinCell() {
    std::vector<voronoi::Site> sites = {{0, {0, 0, 0}}, {1, {2, 0, 0}}, {2, {-2, 0, 0}},
        {3, {0, 2, 0}}, {4, {0, -2, 0}}, {5, {0, 0, 2}}, {6, {0, 0, -2}},
        {7, {2 - 1e-14, 2 - 1e-14, 2 - 1e-14}}};
    std::vector<size_t> const owned = {0};
    auto cells = voronoi::buildCells(sites, owned);
    require(cells[0].complete && cells[0].faces.size() == 7,
        "a nearly cospherical generator creates a genuine tiny seventh face");
    auto const tinyFace = std::find_if(cells[0].faces.begin(), cells[0].faces.end(),
        [](auto const& face) { return face.neighbor.id == 7; });
    require(tinyFace != cells[0].faces.end() && tinyFace->areaVector[0] > 0 &&
            tinyFace->areaVector[0] < 1e-26,
        "tiny positive face must survive without tolerance welding");
    near(cells[0].volume, 8, 1e-14, "tiny face cube volume");
    sites.pop_back();
    sites[5].position[2] = 2e-18;
    sites[6].position[2] = -2e-18;
    cells = voronoi::buildCells(sites, owned);
    require(cells[0].complete && cells[0].faces.size() == 6, "thin cell remains closed");
    near(cells[0].volume, 8e-18, 1e-31, "thin cell volume");
    for (double const coordinate : cells[0].centroid) {
        near(coordinate, 0, 1e-30, "thin symmetric cell centroid");
    }
}

void incompleteAndInvalid() {
    std::vector<voronoi::Site> sites = {
        {0, {0, 0, 0}}, {1, {1, 0, 0}}, {2, {0, 1, 0}}, {3, {0, 0, 1}}};
    std::vector<size_t> owned = {0, 1, 2, 3};
    auto cells = voronoi::buildCells(sites, owned);
    for (auto const& cell : cells) {
        require(!cell.complete && std::isinf(cell.searchRadius), "hull requires halo expansion");
    }
    sites[3].position = {1, 1, 0};
    cells = voronoi::buildCells(sites, owned);
    for (auto const& cell : cells) {
        require(!cell.complete, "coplanar inputs must be incomplete without backend failure");
    }
    sites[3].position = sites[0].position;
    throws([&] { voronoi::buildCells(sites, owned); }, "coincident sites must be rejected");
    sites[3].position = {0, 0, 1};
    sites[3].id = sites[0].id;
    throws([&] { voronoi::buildCells(sites, owned); }, "repeated identities must be rejected");
    sites[3].id = 3;
    sites[3].position[0] = std::numeric_limits<double>::quiet_NaN();
    throws([&] { voronoi::buildCells(sites, owned); }, "nonfinite sites must be rejected");
    require(voronoi::buildCells({}, {}).empty(), "empty ownership is valid");
}

class FloatingEnvironmentGuard {
public:
    FloatingEnvironmentGuard() {
        require(std::fegetenv(&savedEnvironment_) == 0, "capture floating-point environment");
#if defined(__SSE__)
        savedControl_ = _mm_getcsr();
#endif
    }

    ~FloatingEnvironmentGuard() {
        std::fesetenv(&savedEnvironment_);
#if defined(__SSE__)
        _mm_setcsr(savedControl_);
#endif
    }

private:
    std::fenv_t savedEnvironment_{};
#if defined(__SSE__)
    unsigned savedControl_{};
#endif
};

void floatingPointEnvironment() {
    // GoogleTest may execute another case after an exception from buildCells.
    FloatingEnvironmentGuard const restoreOnFailure;
    auto const sites = images({{0.5, 0.5, 0.5}});
    auto const indices = centralIndices(sites);
    std::fenv_t savedEnvironment;
    require(std::fegetenv(&savedEnvironment) == 0, "capture floating-point environment");
#if defined(__SSE__)
    auto const savedControl = _mm_getcsr();
#endif
    require(std::fesetround(FE_DOWNWARD) == 0, "set nondefault caller rounding mode");
#if defined(__SSE__)
    _mm_setcsr(_mm_getcsr() | (1U << 15) | (1U << 6));
    auto const callerControl = _mm_getcsr();
#endif
    auto const cells = voronoi::buildCells(sites, indices);
    bool const roundingRestored = std::fegetround() == FE_DOWNWARD;
#if defined(__SSE__)
    bool const controlRestored = _mm_getcsr() == callerControl;
#endif
    std::fesetenv(&savedEnvironment);
#if defined(__SSE__)
    _mm_setcsr(savedControl);
    require(controlRestored, "restore caller FTZ, DAZ, exception and rounding controls");
#endif
    require(roundingRestored, "restore caller rounding environment");
    checkCells(cells, 1);
}

}    // namespace detail

TEST(Voronoi, PeriodicCube) {
    detail::periodicCube();
}

TEST(Voronoi, PeriodicLattice) {
    detail::periodicLattice();
}

TEST(Voronoi, PeriodicRandom) {
    detail::periodicRandom();
}

TEST(Voronoi, TinyFaceAndThinCell) {
    detail::tinyFaceAndThinCell();
}

TEST(Voronoi, IncompleteAndInvalid) {
    detail::incompleteAndInvalid();
}

TEST(Voronoi, FloatingPointEnvironment) {
    detail::floatingPointEnvironment();
}
