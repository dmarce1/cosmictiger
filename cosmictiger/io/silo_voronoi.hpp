/*
 Copyright (C) 2026 Dominic C. Marcello
 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt.
 */

#ifndef COSMICTIGER_IO_SILO_VORONOI_HPP_
#define COSMICTIGER_IO_SILO_VORONOI_HPP_

#include <array>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <silo.h>

#include <cosmictiger/geometry/voronoi.hpp>

namespace voronoi {

struct OutputBounds {
    Point lower = {0.0, 0.0, 0.0};
    Point upper = {1.0, 1.0, 1.0};
};

struct PreparedVoronoiOutput {
    std::array<std::vector<double>, 3> coordinates;
    std::vector<int> nodeCounts;
    std::vector<int> nodeList;
    std::vector<int> faceCounts;
    std::vector<int> faceList;
    std::vector<char> externalFaces;
    // Index in the Cell span passed to prepareVoronoiOutput, not Cell::inputIndex.
    std::vector<size_t> parentIndices;
    std::vector<long long> parentIds;
    std::vector<std::array<int, 3>> imageOffsets;
    // Volumes use the same scaled length units as coordinates.
    std::vector<double> fullCellVolumes;
    std::vector<double> fragmentVolumes;
};

// Physical cells are unwrapped and owned by their generator. Translate periodic
// images, then clip to the requested view, never to a locality's ownership box.
// Exact vertex keys preserve arbitrarily small genuine geometric features.
PreparedVoronoiOutput prepareVoronoiOutput(std::span<Cell const> cells, OutputBounds const& bounds,
    bool periodic = true, double lengthScale = 1.0);

// Scale each actual value in wide precision, avoiding an intermediate double
// scale cubed. Reject overflow and loss of a nonzero value to underflow.
PreparedVoronoiOutput rescaleVoronoiOutput(PreparedVoronoiOutput const& output, double lengthScale);

// Operates in the current Silo directory. Empty blocks belong in the master's
// EMPTY list and are intentionally not written as zero-zone UCD meshes.
void writeSiloVoronoi(DBfile* db, char const* meshName, char const* zonesName,
    PreparedVoronoiOutput const& output, DBoptlist const* options = nullptr);

template <typename Value>
void writeSiloVoronoiVariable(DBfile* db, char const* name, char const* meshName,
    std::span<Value const> values, DBoptlist const* options = nullptr) {
    static_assert(std::is_same_v<Value, float> || std::is_same_v<Value, double> ||
        std::is_same_v<Value, int> || std::is_same_v<Value, long long>);
    if (values.empty() || values.size() > size_t(std::numeric_limits<int>::max())) {
        throw std::runtime_error("Invalid Voronoi Silo variable count");
    }
    int const datatype = std::is_same_v<Value, float> ? DB_FLOAT :
        std::is_same_v<Value, double>                 ? DB_DOUBLE :
        std::is_same_v<Value, int>                    ? DB_INT :
                                                        DB_LONG_LONG;
    if (DBPutUcdvar1(db, name, meshName, values.data(), int(values.size()), nullptr, 0, datatype,
            DB_ZONECENT, options) < 0) {
        throw std::runtime_error("Cannot write Voronoi Silo variable");
    }
}

// Names are fixed so a multivar master can enumerate them without opening leaves.
inline constexpr std::array<char const*, 6> kOutputDiagnosticNames = {"generator_id",
    "image_offset_x", "image_offset_y", "image_offset_z", "cell_volume", "fragment_volume"};
void writeSiloVoronoiDiagnostics(DBfile* db, char const* meshName,
    PreparedVoronoiOutput const& output, char const* volumeUnit = nullptr);

}    // namespace voronoi

#endif
