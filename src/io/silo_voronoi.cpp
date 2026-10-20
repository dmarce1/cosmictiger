/*
 Copyright (C) 2026 Dominic C. Marcello
 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt.
 */

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <utility>

#include <cosmictiger/io/silo_voronoi.hpp>

namespace voronoi {
namespace detail {

    using Polygon = std::vector<Point>;
    using Polyhedron = std::vector<Polygon>;

    int siloCount(size_t value) {
        if (value > size_t(std::numeric_limits<int>::max())) {
            throw std::runtime_error("Voronoi output exceeds Silo's per-block integer limit");
        }
        return int(value);
    }

    void appendPoint(Polygon& polygon, Point const& point) {
        if (polygon.empty() || polygon.back() != point) {
            polygon.push_back(point);
        }
    }

    void closePolygon(Polygon& polygon) {
        if (polygon.size() > 1 && polygon.front() == polygon.back()) {
            polygon.pop_back();
        }
    }

    Point intersectEdge(Point first, Point second, int axis, double bound) {
        // The same edge approached in the opposite direction must produce exactly
        // the same coordinate, including when adjacent cells are clipped separately.
        if (second < first) {
            std::swap(first, second);
        }
        if (first[axis] == bound) {
            return first;
        }
        if (second[axis] == bound) {
            return second;
        }
        long double const fraction = (static_cast<long double>(bound) - first[axis]) /
            (static_cast<long double>(second[axis]) - first[axis]);
        Point point;
        for (int dim = 0; dim < 3; dim++) {
            point[dim] = double(static_cast<long double>(first[dim]) +
                fraction * (static_cast<long double>(second[dim]) - first[dim]));
        }
        point[axis] = bound;
        return point;
    }

    Polyhedron clip(Polyhedron const& input, int axis, double bound, int sign) {
        Polyhedron output;
        std::set<Point> capVertices;
        bool removed = false;
        for (auto const& polygon : input) {
            Polygon clipped;
            for (size_t i = 0; i < polygon.size(); i++) {
                Point const& first = polygon[i];
                Point const& second = polygon[(i + 1) % polygon.size()];
                bool const firstInside = sign * (first[axis] - bound) <= 0.0;
                bool const secondInside = sign * (second[axis] - bound) <= 0.0;
                removed = removed || !firstInside;
                if (firstInside) {
                    appendPoint(clipped, first);
                }
                if (firstInside != secondInside) {
                    appendPoint(clipped, intersectEdge(first, second, axis, bound));
                }
            }
            closePolygon(clipped);
            if (clipped.size() >= 3) {
                for (auto const& point : clipped) {
                    if (point[axis] == bound) {
                        capVertices.insert(point);
                    }
                }
                output.push_back(std::move(clipped));
            }
        }
        if (removed && capVertices.size() >= 3) {
            int const u = (axis + 1) % 3;
            int const v = (axis + 2) % 3;
            long double centerU = 0.0L;
            long double centerV = 0.0L;
            for (auto const& point : capVertices) {
                centerU += point[u];
                centerV += point[v];
            }
            centerU /= capVertices.size();
            centerV /= capVertices.size();
            Polygon cap(capVertices.begin(), capVertices.end());
            std::sort(cap.begin(), cap.end(), [&](Point const& first, Point const& second) {
                long double const angleFirst = std::atan2(first[v] - centerV, first[u] - centerU);
                long double const angleSecond =
                    std::atan2(second[v] - centerV, second[u] - centerU);
                return angleFirst == angleSecond ? first < second : angleFirst < angleSecond;
            });
            if (sign < 0) {
                std::reverse(cap.begin(), cap.end());
            }
            output.push_back(std::move(cap));
        }
        return output;
    }

    long double volume(Polyhedron const& polyhedron) {
        if (polyhedron.empty()) {
            return 0.0L;
        }
        Point const reference = polyhedron.front().front();
        long double sum = 0.0L;
        for (auto const& polygon : polyhedron) {
            for (size_t i = 1; i + 1 < polygon.size(); i++) {
                std::array<long double, 3> first, second, third;
                for (int dim = 0; dim < 3; dim++) {
                    first[dim] = static_cast<long double>(polygon[0][dim]) - reference[dim];
                    second[dim] = static_cast<long double>(polygon[i][dim]) - reference[dim];
                    third[dim] = static_cast<long double>(polygon[i + 1][dim]) - reference[dim];
                }
                sum += first[0] * (second[1] * third[2] - second[2] * third[1]) +
                    first[1] * (second[2] * third[0] - second[0] * third[2]) +
                    first[2] * (second[0] * third[1] - second[1] * third[0]);
            }
        }
        return sum / 6.0L;
    }

    std::pair<std::vector<int>, bool> canonicalFace(std::vector<int> nodes) {
        auto const start = std::min_element(nodes.begin(), nodes.end());
        std::rotate(nodes.begin(), start, nodes.end());
        std::vector<int> reversed(nodes);
        std::reverse(reversed.begin() + 1, reversed.end());
        if (reversed < nodes) {
            return {std::move(reversed), true};
        }
        return {std::move(nodes), false};
    }

}    // namespace detail

PreparedVoronoiOutput prepareVoronoiOutput(
    std::span<Cell const> cells, OutputBounds const& bounds, bool periodic, double lengthScale) {
    if (!std::isfinite(lengthScale) || lengthScale <= 0.0) {
        throw std::runtime_error("Invalid Voronoi output length scale");
    }
    for (int dim = 0; dim < 3; dim++) {
        if (!std::isfinite(bounds.lower[dim]) || !std::isfinite(bounds.upper[dim]) ||
            bounds.lower[dim] >= bounds.upper[dim]) {
            throw std::runtime_error("Invalid Voronoi output bounds");
        }
    }
    PreparedVoronoiOutput output;
    std::map<Point, int> vertexIds;
    std::map<std::vector<int>, int> faceIds;
    std::vector<int> faceUses;
    std::vector<int> faceOrientations;
    for (size_t parent = 0; parent < cells.size(); parent++) {
        auto const& cell = cells[parent];
        if (!cell.complete || !std::isfinite(cell.volume) || cell.volume <= 0.0 ||
            cell.faces.empty() || cell.site.id > uint64_t(std::numeric_limits<long long>::max())) {
            throw std::runtime_error("Invalid or incomplete physical Voronoi cell for output");
        }
        Point minimum = {INFINITY, INFINITY, INFINITY};
        Point maximum = {-INFINITY, -INFINITY, -INFINITY};
        for (auto const& face : cell.faces) {
            if (face.vertices.size() < 3) {
                throw std::runtime_error("Degenerate physical Voronoi face");
            }
            for (auto const& point : face.vertices) {
                for (int dim = 0; dim < 3; dim++) {
                    if (!std::isfinite(point[dim])) {
                        throw std::runtime_error("Nonfinite physical Voronoi vertex");
                    }
                    minimum[dim] = std::min(minimum[dim], point[dim]);
                    maximum[dim] = std::max(maximum[dim], point[dim]);
                }
            }
        }
        std::array<int, 3> first = {}, last = {};
        for (int dim = 0; dim < 3; dim++) {
            if (periodic) {
                double const lo = std::ceil(bounds.lower[dim] - maximum[dim]);
                double const hi = std::floor(bounds.upper[dim] - minimum[dim]);
                if (lo < std::numeric_limits<int>::min() || lo > std::numeric_limits<int>::max() ||
                    hi < std::numeric_limits<int>::min() || hi > std::numeric_limits<int>::max()) {
                    throw std::runtime_error("Voronoi image offset exceeds integer range");
                }
                first[dim] = int(lo);
                last[dim] = int(hi);
            }
        }
        for (long long ix = first[0]; ix <= last[0]; ix++) {
            for (long long iy = first[1]; iy <= last[1]; iy++) {
                for (long long iz = first[2]; iz <= last[2]; iz++) {
                    std::array<int, 3> const image = {int(ix), int(iy), int(iz)};
                    bool intersects = true;
                    for (int dim = 0; dim < 3; dim++) {
                        intersects = intersects && maximum[dim] + image[dim] > bounds.lower[dim] &&
                            minimum[dim] + image[dim] < bounds.upper[dim];
                    }
                    if (!intersects) {
                        continue;
                    }
                    detail::Polyhedron polyhedron;
                    for (auto const& face : cell.faces) {
                        detail::Polygon polygon;
                        for (auto point : face.vertices) {
                            for (int dim = 0; dim < 3; dim++) {
                                point[dim] += image[dim];
                            }
                            detail::appendPoint(polygon, point);
                        }
                        detail::closePolygon(polygon);
                        polyhedron.push_back(std::move(polygon));
                    }
                    for (int dim = 0; dim < 3 && !polyhedron.empty(); dim++) {
                        polyhedron = detail::clip(polyhedron, dim, bounds.lower[dim], -1);
                        polyhedron = detail::clip(polyhedron, dim, bounds.upper[dim], 1);
                    }
                    long double const fragmentVolume = detail::volume(polyhedron);
                    if (fragmentVolume == 0.0L) {
                        continue;
                    }
                    if (!(fragmentVolume > 0.0L) || polyhedron.size() < 4) {
                        throw std::runtime_error("Invalid clipped Voronoi polyhedron winding");
                    }
                    double const storedVolume = double(fragmentVolume);
                    if (!std::isfinite(storedVolume) || !(storedVolume > 0.0)) {
                        throw std::runtime_error("Invalid Voronoi fragment volume");
                    }
                    output.faceCounts.push_back(detail::siloCount(polyhedron.size()));
                    output.parentIndices.push_back(parent);
                    output.parentIds.push_back(static_cast<long long>(cell.site.id));
                    output.imageOffsets.push_back(image);
                    output.fullCellVolumes.push_back(cell.volume);
                    output.fragmentVolumes.push_back(storedVolume);
                    std::map<std::pair<int, int>, int> edges;
                    for (auto const& polygon : polyhedron) {
                        std::vector<int> nodes;
                        for (auto const& point : polygon) {
                            auto const [entry, inserted] =
                                vertexIds.emplace(point, detail::siloCount(vertexIds.size()));
                            if (inserted) {
                                for (int dim = 0; dim < 3; dim++) {
                                    output.coordinates[dim].push_back(point[dim]);
                                }
                            }
                            nodes.push_back(entry->second);
                        }
                        for (size_t i = 0; i < nodes.size(); i++) {
                            int const firstNode = nodes[i];
                            int const secondNode = nodes[(i + 1) % nodes.size()];
                            if (firstNode == secondNode) {
                                throw std::runtime_error("Zero-length Voronoi output edge");
                            }
                            edges[{firstNode, secondNode}]++;
                        }
                        auto [key, reversed] = detail::canonicalFace(std::move(nodes));
                        auto const [entry, inserted] =
                            faceIds.emplace(key, detail::siloCount(faceIds.size()));
                        int const face = entry->second;
                        if (inserted) {
                            output.nodeCounts.push_back(detail::siloCount(key.size()));
                            output.nodeList.insert(output.nodeList.end(), key.begin(), key.end());
                            faceUses.push_back(0);
                            faceOrientations.push_back(0);
                        }
                        faceOrientations[face] += reversed ? -1 : 1;
                        if (++faceUses[face] > 2 ||
                            (faceUses[face] == 2 && faceOrientations[face] != 0)) {
                            throw std::runtime_error("Nonmanifold Voronoi output face");
                        }
                        output.faceList.push_back(reversed ? ~face : face);
                    }
                    for (auto const& [edge, count] : edges) {
                        auto const opposite = edges.find({edge.second, edge.first});
                        if (count != 1 || opposite == edges.end() || opposite->second != 1) {
                            throw std::runtime_error(
                                "Voronoi output cell is not a closed oriented shell");
                        }
                    }
                }
            }
        }
    }
    detail::siloCount(output.nodeList.size());
    detail::siloCount(output.faceList.size());
    detail::siloCount(output.faceCounts.size());
    for (int const uses : faceUses) {
        output.externalFaces.push_back(uses == 1);
    }
    if (lengthScale != 1.0) {
        return rescaleVoronoiOutput(output, lengthScale);
    }
    return output;
}

PreparedVoronoiOutput rescaleVoronoiOutput(
    PreparedVoronoiOutput const& output, double lengthScale) {
    if (!std::isfinite(lengthScale) || !(lengthScale > 0.0)) {
        throw std::runtime_error("Invalid Voronoi output length scale");
    }
    PreparedVoronoiOutput scaled = output;
    for (auto& coordinates : scaled.coordinates) {
        for (double& value : coordinates) {
            double const result = double(static_cast<long double>(value) * lengthScale);
            if (!std::isfinite(result) || (value != 0.0 && result == 0.0)) {
                throw std::runtime_error("Unrepresentable scaled Voronoi coordinate");
            }
            value = result;
        }
    }
    for (auto* volumes : {&scaled.fullCellVolumes, &scaled.fragmentVolumes}) {
        for (double& value : *volumes) {
            double const result =
                double(static_cast<long double>(value) * lengthScale * lengthScale * lengthScale);
            if (!std::isfinite(result) || !(result > 0.0)) {
                throw std::runtime_error("Unrepresentable scaled Voronoi volume");
            }
            value = result;
        }
    }
    return scaled;
}

void writeSiloVoronoi(DBfile* db, char const* meshName, char const* zonesName,
    PreparedVoronoiOutput const& output, DBoptlist const* options) {
    if (output.faceCounts.empty()) {
        throw std::runtime_error("Empty Voronoi block must be represented as EMPTY in its master");
    }
    int const zones = detail::siloCount(output.faceCounts.size());
    if (DBPutPHZonelist(db, zonesName, detail::siloCount(output.nodeCounts.size()),
            output.nodeCounts.data(), detail::siloCount(output.nodeList.size()),
            output.nodeList.data(), output.externalFaces.data(), zones, output.faceCounts.data(),
            detail::siloCount(output.faceList.size()), output.faceList.data(), 0, 0, zones - 1,
            nullptr) < 0) {
        throw std::runtime_error("Cannot write Voronoi polyhedral zonelist");
    }
    std::unique_ptr<DBoptlist, decltype(&DBFreeOptlist)> meshOptions(
        DBMakeOptlist((options ? options->numopts : 0) + 2), DBFreeOptlist);
    if (!meshOptions) {
        throw std::runtime_error("Cannot allocate Voronoi mesh options");
    }
    if (options) {
        for (int i = 0; i < options->numopts; i++) {
            if (options->options[i] != DBOPT_PHZONELIST &&
                options->options[i] != DBOPT_TV_CONNECTIVITY &&
                DBAddOption(meshOptions.get(), options->options[i], options->values[i]) < 0) {
                throw std::runtime_error("Cannot copy Voronoi mesh option");
            }
        }
    }
    int varyingConnectivity = 1;
    if (DBAddOption(meshOptions.get(), DBOPT_PHZONELIST, const_cast<char*>(zonesName)) < 0 ||
        DBAddOption(meshOptions.get(), DBOPT_TV_CONNECTIVITY, &varyingConnectivity) < 0) {
        throw std::runtime_error("Cannot set Voronoi mesh options");
    }
    void const* coordinates[3] = {
        output.coordinates[0].data(), output.coordinates[1].data(), output.coordinates[2].data()};
    if (DBPutUcdmesh(db, meshName, 3, nullptr, coordinates,
            detail::siloCount(output.coordinates[0].size()), zones, nullptr, nullptr, DB_DOUBLE,
            meshOptions.get()) < 0) {
        throw std::runtime_error("Cannot write Voronoi UCD mesh");
    }
}

void writeSiloVoronoiDiagnostics(
    DBfile* db, char const* meshName, PreparedVoronoiOutput const& output, char const* volumeUnit) {
    writeSiloVoronoiVariable(
        db, kOutputDiagnosticNames[0], meshName, std::span<long long const>(output.parentIds));
    for (int dim = 0; dim < 3; dim++) {
        std::vector<int> offsets;
        offsets.reserve(output.imageOffsets.size());
        for (auto const& image : output.imageOffsets) {
            offsets.push_back(image[dim]);
        }
        writeSiloVoronoiVariable(
            db, kOutputDiagnosticNames[dim + 1], meshName, std::span<int const>(offsets));
    }
    std::unique_ptr<DBoptlist, decltype(&DBFreeOptlist)> volumeOptions(
        DBMakeOptlist(1), DBFreeOptlist);
    if (!volumeOptions ||
        (volumeUnit &&
            DBAddOption(volumeOptions.get(), DBOPT_UNITS, const_cast<char*>(volumeUnit)) < 0)) {
        throw std::runtime_error("Cannot set Voronoi volume units");
    }
    writeSiloVoronoiVariable(db, kOutputDiagnosticNames[4], meshName,
        std::span<double const>(output.fullCellVolumes), volumeOptions.get());
    writeSiloVoronoiVariable(db, kOutputDiagnosticNames[5], meshName,
        std::span<double const>(output.fragmentVolumes), volumeOptions.get());
}

}    // namespace voronoi
