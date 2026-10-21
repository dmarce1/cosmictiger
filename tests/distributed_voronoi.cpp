/*
 * Copyright (C) 2026 Dominic C. Marcello
 * Distributed under the Boost Software License, Version 1.0.
 * See LICENSE_1_0.txt.
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cosmictiger/geometry/distributed_voronoi.hpp>
#include <cosmictiger/io/silo_voronoi.hpp>

namespace detail {

void require(bool condition, char const* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::vector<voronoi::Cell> partitioned(std::vector<voronoi::Site> sites, int ranks,
    bool interleaved = false, size_t* maximumImported = nullptr) {
    std::vector<std::vector<voronoi::Site>> owned(ranks);
    for (auto& site : sites) {
        site.owner = interleaved ? int(site.id % (ranks - 1)) :
                                   std::min(ranks - 1, int(site.position[0] * ranks));
        owned[site.owner].push_back(site);
    }
    std::vector<voronoi::SiteIndex> indices;
    std::vector<voronoi::SitePartition> partitions;
    for (int rank = 0; rank < ranks; ++rank) {
        indices.emplace_back(owned[rank]);
        partitions.push_back(indices.back().partition(rank));
    }
    std::vector<voronoi::Cell> cells;
    for (auto const& local : owned) {
        size_t imported = 0;
        auto localCells = voronoi::buildDistributedCells(
            local, partitions, [&](std::vector<voronoi::SiteQueryBatch> const& batches) {
                std::vector<voronoi::Site> result;
                for (auto const& batch : batches) {
                    auto received = indices.at(batch.owner).query(batch.queries);
                    result.insert(result.end(), received.begin(), received.end());
                    imported += received.size();
                }
                return result;
            });
        require(localCells.size() == local.size(), "Owned cell ordering/count");
        if (maximumImported) {
            *maximumImported = std::max(*maximumImported, imported);
        }
        cells.insert(cells.end(), localCells.begin(), localCells.end());
    }
    std::sort(cells.begin(), cells.end(),
        [](auto const& a, auto const& b) { return a.site.id < b.site.id; });
    double volume = 0;
    for (auto const& cell : cells) {
        require(cell.complete && cell.volume > 0, "Closed positive distributed cell");
        volume += cell.volume;
    }
    require(std::abs(volume - 1.0) < 1e-10, "Periodic cells partition unit volume");
    return cells;
}

void compare(std::vector<voronoi::Cell> const& a, std::vector<voronoi::Cell> const& b) {
    require(a.size() == b.size(), "Partition invariant count");
    for (size_t i = 0; i < a.size(); ++i) {
        require(a[i].site.id == b[i].site.id, "Partition invariant generator");
        require(std::abs(a[i].volume - b[i].volume) < 1e-12, "Partition invariant cell volume");
        for (size_t axis = 0; axis < 3; ++axis) {
            require(std::abs(a[i].centroid[axis] - b[i].centroid[axis]) < 1e-11,
                "Partition invariant centroid");
        }
        require(a[i].faces.size() == b[i].faces.size(), "Partition invariant face count");
        std::map<std::pair<uint64_t, std::array<int, 3>>, voronoi::Point> faces;
        for (auto const& face : a[i].faces) {
            faces[{face.neighbor.id, face.neighbor.image}] = face.areaVector;
        }
        for (auto const& face : b[i].faces) {
            auto const match = faces.find({face.neighbor.id, face.neighbor.image});
            require(match != faces.end(), "Partition invariant neighbor identity");
            for (size_t axis = 0; axis < 3; ++axis) {
                require(std::abs(match->second[axis] - face.areaVector[axis]) < 1e-11,
                    "Partition invariant oriented face");
            }
        }
    }
}

void compareConnectivity(std::vector<voronoi::Cell> const& a, std::vector<voronoi::Cell> const& b) {
    require(a.size() == b.size(), "Partition invariant connectivity count");
    for (size_t i = 0; i < a.size(); ++i) {
        std::map<std::pair<uint64_t, std::array<int, 3>>, std::vector<voronoi::Point>> faces;
        for (auto const& face : a[i].faces) {
            auto vertices = face.vertices;
            std::rotate(vertices.begin(), std::min_element(vertices.begin(), vertices.end()),
                vertices.end());
            faces[{face.neighbor.id, face.neighbor.image}] = std::move(vertices);
        }
        for (auto const& face : b[i].faces) {
            auto vertices = face.vertices;
            std::rotate(vertices.begin(), std::min_element(vertices.begin(), vertices.end()),
                vertices.end());
            auto const match = faces.find({face.neighbor.id, face.neighbor.image});
            require(match != faces.end() && match->second == vertices,
                "Partition invariant exact face coordinates and oriented connectivity");
        }
    }
}

void checkOutput(std::vector<voronoi::Cell> const& cells) {
    auto const output = voronoi::prepareVoronoiOutput(cells, {});
    double total = 0;
    for (auto const volume : output.fragmentVolumes) {
        require(std::isfinite(volume) && volume > 0, "Positive finite full-view fragment volume");
        total += volume;
    }
    require(std::abs(total - 1.0) < 1e-10, "Actual Delaunay dual output fills unit box");
    voronoi::OutputBounds const crop{{0.13, 0.27, 0.31}, {0.71, 0.83, 0.69}};
    auto const cropped = voronoi::prepareVoronoiOutput(cells, crop);
    total = 0;
    for (auto const volume : cropped.fragmentVolumes) {
        require(std::isfinite(volume) && volume > 0, "Positive finite cropped fragment volume");
        total += volume;
    }
    require(std::abs(total - 0.58 * 0.56 * 0.38) < 1e-10, "Actual Delaunay dual output fills crop");
}

}    // namespace detail

void testRoundedPerturbedLatticeConnectivity() {
    // Fixed-point particle positions can produce much smaller dual features.
    // These seeds exercise a tiny face whose rounded area changes sign, and
    // periodic fragments with coincident external faces on the view boundary.
    for (unsigned const seed : {4U, 12U}) {
        std::mt19937_64 random(seed);
        std::uniform_real_distribution<double> noise(-1e-9, 1e-9);
        std::vector<voronoi::Site> sites;
        for (int x = 0; x < 4; ++x) {
            for (int y = 0; y < 4; ++y) {
                for (int z = 0; z < 4; ++z) {
                    voronoi::Point position{(x + 0.5) / 4, (y + 0.5) / 4, (z + 0.5) / 4};
                    for (auto& coordinate : position) {
                        coordinate =
                            std::ldexp(std::round(std::ldexp(coordinate + noise(random), 32)), -32);
                    }
                    sites.push_back({sites.size(), position, 0, {}});
                }
            }
        }
        auto const reference = detail::partitioned(sites, 1);
        detail::checkOutput(reference);
        for (int const ranks : {2, 4}) {
            auto const cells = detail::partitioned(sites, ranks);
            detail::compare(reference, cells);
            detail::compareConnectivity(reference, cells);
            detail::checkOutput(cells);
        }
    }
}

int main() {
    try {
        std::vector<voronoi::Site> lattice;
        for (int x = 0; x < 4; ++x) {
            for (int y = 0; y < 4; ++y) {
                for (int z = 0; z < 4; ++z) {
                    lattice.push_back({lattice.size(), {x / 4.0, y / 4.0, z / 4.0}, 0, {}});
                }
            }
        }
        auto const regular = detail::partitioned(lattice, 1);
        detail::compare(regular, detail::partitioned(lattice, 2));
        detail::compare(regular, detail::partitioned(lattice, 4));
        detail::checkOutput(regular);
        std::mt19937_64 random(195);
        std::uniform_real_distribution<double> uniform(0, 1);
        std::vector<voronoi::Site> irregular;
        for (size_t i = 0; i < 90; ++i) {
            irregular.push_back({i, {uniform(random), uniform(random), uniform(random)}, 0, {}});
        }
        auto const reference = detail::partitioned(irregular, 1);
        size_t maximumImported = 0;
        detail::compare(reference, detail::partitioned(irregular, 4, false, &maximumImported));
        detail::compare(reference, detail::partitioned(irregular, 4, true));
        detail::require(maximumImported < irregular.size() * 27,
            "Spatial halo avoids gathering every periodic image");
        detail::checkOutput(reference);
        // Thin/clustered data need adaptive halo expansion, including ranks
        // whose initial neighborhood is lower-dimensional.
        std::vector<voronoi::Site> clustered;
        for (size_t i = 0; i < 24; ++i) {
            clustered.push_back({i,
                {0.01 + 0.03 * uniform(random), 0.02 + 0.02 * uniform(random),
                    0.03 + 0.01 * uniform(random)},
                0, {}});
        }
        clustered.push_back({24, {0.8, 0.7, 0.6}, 0, {}});
        auto const sparse = detail::partitioned(clustered, 1);
        detail::compare(sparse, detail::partitioned(clustered, 4));
        detail::checkOutput(sparse);
        for (size_t count = 1; count <= 3; ++count) {
            std::vector<voronoi::Site> small;
            for (size_t i = 0; i < count; ++i) {
                small.push_back({i, {0.1 + 0.2 * i, 0.2, 0.3}, 0, {}});
            }
            detail::checkOutput(detail::partitioned(small, 4));
        }
        testRoundedPerturbedLatticeConnectivity();
        std::puts("Distributed Voronoi partition, halo, periodic and crop checks passed");
        return 0;
    } catch (std::exception const& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
