/*
 Copyright (C) 2026 Dominic C. Marcello
 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <silo.h>

#include <cosmictiger/io/silo_voronoi.hpp>

namespace detail {

void require(bool condition, std::string const& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void requireNear(double actual, double expected, std::string const& message) {
    require(std::abs(actual - expected) <=
            2e-13 * std::max(std::abs(expected), std::numeric_limits<double>::min()),
        message);
}

voronoi::Cell boxCell(voronoi::Point lower, voronoi::Point upper, uint64_t id) {
    voronoi::Cell cell;
    cell.site.id = id;
    cell.complete = true;
    cell.volume = 1.0;
    for (int axis = 0; axis < 3; axis++) {
        cell.site.position[axis] = cell.centroid[axis] = (lower[axis] + upper[axis]) * 0.5;
        cell.volume *= upper[axis] - lower[axis];
        for (int side = 0; side < 2; side++) {
            voronoi::Face face;
            int const u = (axis + 1) % 3;
            int const v = (axis + 2) % 3;
            for (auto const& corner :
                std::array<std::array<int, 2>, 4>{{{0, 0}, {1, 0}, {1, 1}, {0, 1}}}) {
                voronoi::Point point;
                point[axis] = side ? upper[axis] : lower[axis];
                point[u] = corner[0] ? upper[u] : lower[u];
                point[v] = corner[1] ? upper[v] : lower[v];
                face.vertices.push_back(point);
            }
            if (!side) {
                std::reverse(face.vertices.begin(), face.vertices.end());
            }
            cell.faces.push_back(std::move(face));
        }
    }
    return cell;
}

// Independently check each stored zone is a closed, consistently oriented shell.
double checkTopology(voronoi::PreparedVoronoiOutput const& output) {
    std::vector<size_t> starts(output.nodeCounts.size());
    for (size_t face = 1; face < starts.size(); face++) {
        starts[face] = starts[face - 1] + output.nodeCounts[face - 1];
    }
    size_t offset = 0;
    double total = 0.0;
    for (size_t zone = 0; zone < output.faceCounts.size(); zone++) {
        std::map<std::pair<int, int>, int> edges;
        long double sixVolume = 0.0;
        int const firstReference = output.faceList[offset];
        int const firstFace = firstReference < 0 ? ~firstReference : firstReference;
        int const originNode = output.nodeList[starts[firstFace]];
        for (int localFace = 0; localFace < output.faceCounts[zone]; localFace++) {
            int const reference = output.faceList[offset++];
            int const face = reference < 0 ? ~reference : reference;
            std::vector<int> nodes(output.nodeList.begin() + starts[face],
                output.nodeList.begin() + starts[face] + output.nodeCounts[face]);
            if (reference < 0) {
                std::reverse(nodes.begin(), nodes.end());
            }
            for (size_t i = 0; i < nodes.size(); i++) {
                int const first = nodes[i];
                int const second = nodes[(i + 1) % nodes.size()];
                require(first != second, "no zero-length output edge");
                edges[{first, second}]++;
            }
            for (size_t i = 1; i + 1 < nodes.size(); i++) {
                std::array<long double, 3> a, b, c;
                for (int dim = 0; dim < 3; dim++) {
                    long double const origin = output.coordinates[dim][originNode];
                    a[dim] = output.coordinates[dim][nodes[0]] - origin;
                    b[dim] = output.coordinates[dim][nodes[i]] - origin;
                    c[dim] = output.coordinates[dim][nodes[i + 1]] - origin;
                }
                sixVolume += a[0] * (b[1] * c[2] - b[2] * c[1]) +
                    a[1] * (b[2] * c[0] - b[0] * c[2]) + a[2] * (b[0] * c[1] - b[1] * c[0]);
            }
        }
        for (auto const& [edge, count] : edges) {
            auto const other = edges.find({edge.second, edge.first});
            require(count == 1 && other != edges.end() && other->second == 1,
                "closed oriented polyhedron edges");
        }
        double const volume = double(sixVolume / 6.0L);
        require(volume > 0.0, "positive stored zone volume");
        requireNear(volume, output.fragmentVolumes[zone], "stored winding matches fragment volume");
        total += volume;
    }
    return total;
}

void checkReadback(char const* filename, voronoi::PreparedVoronoiOutput const& expected) {
    std::unique_ptr<DBfile, decltype(&DBClose)> file(
        DBOpen(filename, DB_UNKNOWN, DB_READ), DBClose);
    require(bool(file), "open Voronoi Silo file");
    require(DBSetDir(file.get(), "locality_2") >= 0, "read locality subdirectory");
    std::unique_ptr<DBucdmesh, decltype(&DBFreeUcdmesh)> mesh(
        DBGetUcdmesh(file.get(), "gas"), DBFreeUcdmesh);
    require(mesh && mesh->ndims == 3 && mesh->datatype == DB_DOUBLE && mesh->phzones,
        "read native polyhedral UCD mesh");
    require(mesh->nnodes == int(expected.coordinates[0].size()) &&
            mesh->phzones->nzones == int(expected.faceCounts.size()),
        "node and zone count");
    require(mesh->phzones->lo_offset == 0 && mesh->phzones->hi_offset == mesh->phzones->nzones - 1,
        "all written zones are real, not ghost zones");
    require(mesh->dtime == 2.75, "preserve mesh time");
    voronoi::PreparedVoronoiOutput stored;
    stored.fragmentVolumes = expected.fragmentVolumes;
    for (int dim = 0; dim < 3; dim++) {
        auto const* coordinates = static_cast<double const*>(mesh->coords[dim]);
        stored.coordinates[dim].assign(coordinates, coordinates + mesh->nnodes);
        require(mesh->units[dim] && std::string(mesh->units[dim]) == "cm", "coordinate units");
        require(stored.coordinates[dim] == expected.coordinates[dim], "exact coordinate readback");
    }
    auto const* zones = mesh->phzones;
    stored.nodeCounts.assign(zones->nodecnt, zones->nodecnt + zones->nfaces);
    stored.nodeList.assign(zones->nodelist, zones->nodelist + zones->lnodelist);
    stored.faceCounts.assign(zones->facecnt, zones->facecnt + zones->nzones);
    stored.faceList.assign(zones->facelist, zones->facelist + zones->lfacelist);
    require(stored.nodeCounts == expected.nodeCounts && stored.nodeList == expected.nodeList &&
            stored.faceCounts == expected.faceCounts && stored.faceList == expected.faceList,
        "exact topology readback");
    checkTopology(stored);
    for (auto const& [name, datatype] :
        std::array<std::pair<char const*, int>, 3>{{{"rho", DB_DOUBLE},
            {"kernel_support_radius", DB_FLOAT}, {"generator_id", DB_LONG_LONG}}}) {
        std::unique_ptr<DBucdvar, decltype(&DBFreeUcdvar)> variable(
            DBGetUcdvar(file.get(), name), DBFreeUcdvar);
        bool const matchingType = variable &&
            (variable->datatype == datatype ||
                (datatype == DB_LONG_LONG && variable->datatype == DB_LONG && sizeof(long) == 8));
        require(matchingType && variable->centering == DB_ZONECENT &&
                variable->nels == int(expected.faceCounts.size()),
            std::string("zone field datatype/count/centering: ") + name +
                " datatype=" + std::to_string(variable ? variable->datatype : -1));
        for (size_t i = 0; i < expected.parentIndices.size(); i++) {
            if (datatype == DB_DOUBLE) {
                require(static_cast<double const*>(variable->vals[0])[i] ==
                        2.0 + expected.parentIndices[i],
                    "density follows parent-cell map");
            } else if (datatype == DB_FLOAT) {
                require(static_cast<float const*>(variable->vals[0])[i] == 0.25f,
                    "support-radius field");
            } else {
                long long const id = variable->datatype == DB_LONG ?
                    static_cast<long const*>(variable->vals[0])[i] :
                    static_cast<long long const*>(variable->vals[0])[i];
                require(id == expected.parentIds[i], "integer generator identity");
            }
        }
    }
    std::unique_ptr<DBucdvar, decltype(&DBFreeUcdvar)> volume(
        DBGetUcdvar(file.get(), "fragment_volume"), DBFreeUcdvar);
    require(volume && volume->units && std::string(volume->units) == "cm^3", "volume units");
}

void testDriver(int driver, char const* filename, voronoi::PreparedVoronoiOutput const& output) {
    std::unique_ptr<DBfile, decltype(&DBClose)> file(
        DBCreateReal(filename, DB_CLOBBER, DB_LOCAL, "Voronoi polyhedra", driver), DBClose);
    require(bool(file), "create polyhedral Silo file");
    require(DBMkDir(file.get(), "locality_2") >= 0 && DBSetDir(file.get(), "locality_2") >= 0,
        "write in locality subdirectory");
    std::unique_ptr<DBoptlist, decltype(&DBFreeOptlist)> options(DBMakeOptlist(4), DBFreeOptlist);
    double time = 2.75;
    char units[] = "cm";
    DBAddOption(options.get(), DBOPT_DTIME, &time);
    DBAddOption(options.get(), DBOPT_XUNITS, units);
    DBAddOption(options.get(), DBOPT_YUNITS, units);
    DBAddOption(options.get(), DBOPT_ZUNITS, units);
    voronoi::writeSiloVoronoi(file.get(), "gas", "gas_zones", output, options.get());
    require(options->numopts == 4, "writer does not mutate caller's option list");
    std::vector<double> density;
    for (size_t const parent : output.parentIndices) {
        density.push_back(2.0 + parent);
    }
    std::vector<float> radius(density.size(), 0.25f);
    voronoi::writeSiloVoronoiVariable(file.get(), "rho", "gas", std::span<double const>(density));
    voronoi::writeSiloVoronoiVariable(
        file.get(), "kernel_support_radius", "gas", std::span<float const>(radius));
    voronoi::writeSiloVoronoiDiagnostics(file.get(), "gas", output, "cm^3");
    require(DBClose(file.release()) >= 0, "close Voronoi Silo file");
    checkReadback(filename, output);
}

void testClipping() {
    std::array<voronoi::Cell, 1> const periodic = {boxCell({-0.5, -0.5, -0.5}, {0.5, 0.5, 0.5}, 7)};
    auto const fragments = voronoi::prepareVoronoiOutput(periodic, {});
    require(fragments.faceCounts.size() == 8, "periodic corner cell splits into eight fragments");
    requireNear(checkTopology(fragments), 1.0, "periodic fragments fill fundamental box");
    for (size_t i = 0; i < fragments.parentIds.size(); i++) {
        require(fragments.parentIds[i] == 7 && fragments.parentIndices[i] == 0,
            "all fragments retain parent association");
        requireNear(
            fragments.fullCellVolumes[i], 1.0, "full physical volume is not fragment volume");
        requireNear(fragments.fragmentVolumes[i], 0.125, "periodic fragment volume");
    }
    std::array<voronoi::Cell, 2> cells = {
        boxCell({0, 0, 0}, {0.5, 1, 1}, 17), boxCell({0.5, 0, 0}, {1, 1, 1}, 23)};
    auto const regular = voronoi::prepareVoronoiOutput(cells, {}, true, 2.0);
    require(regular.coordinates[0].size() == 12 && regular.nodeCounts.size() == 11,
        "exact shared vertex and face deduplication");
    require(std::count(regular.externalFaces.begin(), regular.externalFaces.end(), 0) == 1,
        "one shared internal face");
    requireNear(checkTopology(regular), 8.0, "scaled volume fills box");
    voronoi::OutputBounds const bounds{{0.1, 0.17, 0.23}, {0.9, 0.83, 0.77}};
    auto const crop = voronoi::prepareVoronoiOutput(cells, bounds);
    require(crop.coordinates[0].size() == 12 && crop.nodeCounts.size() == 11,
        "independently clipped cells retain shared faces");
    requireNear(checkTopology(crop), 0.8 * 0.66 * 0.54, "cropped cells fill view volume");
    cells[0].site.position[0] = 0.1;
    voronoi::OutputBounds const offCenter{{0.25, 0, 0}, {0.45, 1, 1}};
    auto const outside = voronoi::prepareVoronoiOutput(cells, offCenter);
    require(outside.faceCounts.size() == 1 && outside.parentIds[0] == 17,
        "include intersecting cell whose generator lies outside view");
    requireNear(checkTopology(outside), 0.2, "outside-generator crop volume");
    auto const absent = voronoi::prepareVoronoiOutput(cells, {{2, 2, 2}, {3, 3, 3}}, false);
    require(absent.faceCounts.empty() && absent.coordinates[0].empty(),
        "empty view produces empty block");
    std::array<voronoi::Cell, 1> const tiny = {boxCell({0, 0, 0}, {1e-15, 1e-15, 1e-15}, 9)};
    auto const small = voronoi::prepareVoronoiOutput(tiny, {}, false);
    require(small.coordinates[0].size() == 8, "no tolerance merge of tiny genuine cells");
    requireNear(checkTopology(small), 1e-45, "tiny cell volume survives");
    for (int const driver : {DB_PDB, DB_HDF5}) {
        std::string const suffix = driver == DB_PDB ? "_pdb.silo" : "_hdf5.silo";
        testDriver(driver, ("silo_voronoi" + suffix).c_str(), regular);
        testDriver(driver, ("silo_voronoi_periodic" + suffix).c_str(), fragments);
        testDriver(driver, ("silo_voronoi_crop" + suffix).c_str(), crop);
    }
}

void testComputedCells() {
    std::mt19937 generator(712);
    std::uniform_real_distribution<double> random(0.0, 1.0);
    for (int fixture = 0; fixture < 3; fixture++) {
        bool const randomSites = fixture == 2;
        std::vector<voronoi::Point> points;
        if (randomSites) {
            for (int i = 0; i < 48; i++) {
                points.push_back({random(generator), random(generator), random(generator)});
            }
        } else {
            int const side = fixture + 2;
            for (int x = 0; x < side; x++) {
                for (int y = 0; y < side; y++) {
                    for (int z = 0; z < side; z++) {
                        points.push_back({(x + 0.5) / side, (y + 0.5) / side, (z + 0.5) / side});
                    }
                }
            }
        }
        std::vector<voronoi::Site> sites;
        std::vector<size_t> owned;
        for (int x = -1; x <= 1; x++) {
            for (int y = -1; y <= 1; y++) {
                for (int z = -1; z <= 1; z++) {
                    for (size_t i = 0; i < points.size(); i++) {
                        if (x == 0 && y == 0 && z == 0) {
                            owned.push_back(sites.size());
                        }
                        sites.push_back({i, {points[i][0] + x, points[i][1] + y, points[i][2] + z},
                            0, {x, y, z}});
                    }
                }
            }
        }
        auto const cells = voronoi::buildCells(sites, owned);
        auto const complete = voronoi::prepareVoronoiOutput(cells, {});
        requireNear(checkTopology(complete), 1.0, "computed periodic Voronoi cells fill unit box");
        voronoi::OutputBounds const bounds{{0.1, 0.17, 0.23}, {0.9, 0.83, 0.77}};
        auto const cropped = voronoi::prepareVoronoiOutput(cells, bounds);
        requireNear(
            checkTopology(cropped), 0.8 * 0.66 * 0.54, "computed Voronoi cells fill cropped view");
        if (randomSites) {
            for (int const driver : {DB_PDB, DB_HDF5}) {
                testDriver(driver,
                    driver == DB_PDB ? "silo_voronoi_random_pdb.silo" :
                                       "silo_voronoi_random_hdf5.silo",
                    cropped);
            }
        }
    }
}

void testRescaling() {
    voronoi::PreparedVoronoiOutput source;
    source.coordinates = {{{1.0}, {0.0}, {-1.0}}};
    source.fullCellVolumes = {0.125};
    source.fragmentVolumes = {0.0625};
    source.parentIds = {17};
    auto const scaled = voronoi::rescaleVoronoiOutput(source, 2.0);
    require(scaled.coordinates[0][0] == 2.0 && scaled.coordinates[2][0] == -2.0 &&
            scaled.fullCellVolumes[0] == 1.0 && scaled.fragmentVolumes[0] == 0.5 &&
            scaled.parentIds == source.parentIds && source.coordinates[0][0] == 1.0,
        "normal rescaling preserves parent data and source geometry");
    auto const rejected = [&](double scale) {
        bool threw = false;
        try {
            voronoi::rescaleVoronoiOutput(source, scale);
        } catch (std::runtime_error const&) {
            threw = true;
        }
        require(threw, "reject nonrepresentable rescaling");
    };
    rejected(0.0);
    rejected(-1.0);
    rejected(std::numeric_limits<double>::infinity());
    rejected(std::numeric_limits<double>::quiet_NaN());
    source.coordinates[0][0] = 1e300;
    rejected(1e10);
    source.coordinates[0][0] = 1e-300;
    rejected(1e-100);
    source.coordinates = {{{0.0}, {0.0}, {0.0}}};
    source.fullCellVolumes[0] = 1e300;
    rejected(1e10);
    source.fullCellVolumes[0] = 1e-300;
    source.fragmentVolumes[0] = 1e-300;
    rejected(1e-100);
    auto const expanded = voronoi::rescaleVoronoiOutput(source, 1e150);
    requireNear(expanded.fullCellVolumes[0], 1e150,
        "valid actual volume survives overflowing double scale cubed");
    source.fullCellVolumes[0] = 1e300;
    source.fragmentVolumes[0] = 1e300;
    auto const contracted = voronoi::rescaleVoronoiOutput(source, 1e-150);
    requireNear(contracted.fullCellVolumes[0], 1e-150,
        "valid actual volume survives underflowing double scale cubed");
}

}    // namespace detail

int main() {
    try {
        detail::testClipping();
        detail::testComputedCells();
        detail::testRescaling();
    } catch (std::exception const& error) {
        std::fprintf(stderr, "Silo Voronoi test: %s\n", error.what());
        return 1;
    }
    return 0;
}
