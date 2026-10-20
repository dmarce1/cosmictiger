/*
 * Copyright (C) 2026 Dominic C. Marcello
 * Distributed under the Boost Software License, Version 1.0.
 * See LICENSE_1_0.txt.
 */

#ifndef COSMICTIGER_GEOMETRY_VORONOI_HPP_
#define COSMICTIGER_GEOMETRY_VORONOI_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace voronoi {

using Point = std::array<double, 3>;

// Images are explicit: position already includes the periodic translation.
struct Site {
    uint64_t id = 0;
    Point position = {};
    int owner = 0;
    std::array<int, 3> image = {};
    template <typename Archive>
    void serialize(Archive& archive, unsigned) {
        archive & id & owner;
        for (size_t dimension = 0; dimension < 3; ++dimension) {
            archive& position[dimension] & image[dimension];
        }
    }
};

struct Face {
    Site neighbor;
    // Counterclockwise as seen from outside the owning cell.
    std::vector<Point> vertices;
    Point areaVector = {};
    Point centroid = {};
};

struct Cell {
    Site site;
    size_t inputIndex = 0;
    std::vector<Face> faces;
    Point centroid = {};
    double volume = 0;
    double maxRadius = 0;
    // Twice the maximum incident Delaunay circumradius; remote sites must
    // have been searched beyond this distance before a cell is certified.
    double searchRadius = 0;
    // Bounded with respect to the supplied sites, not a remote-halo certificate.
    bool complete = false;
};

// Builds selected cells in ownedIndices order. The caller supplies periodic
// images and remote halo sites. Hull cells are returned with complete=false.
// Coordinates and identities must be unique; invalid/duplicate sites throw.
// Site IDs and image offsets determine canonical ordering independently of
// input order. All returned coordinates use the input coordinate system.
std::vector<Cell> buildCells(std::span<Site const> sites, std::span<size_t const> ownedIndices);

}    // namespace voronoi

#endif
