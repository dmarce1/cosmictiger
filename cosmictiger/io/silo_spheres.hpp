/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#ifndef COSMICTIGER_SILO_SPHERES_HPP_
#define COSMICTIGER_SILO_SPHERES_HPP_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <silo.h>

namespace detail {

inline void writeSiloSpheres(DBfile* db, char const* meshName, char const* zonesName,
    std::span<double const> coefficients, DBoptlist* options) {
    if (coefficients.empty() || coefficients.size() % 4 != 0 ||
        coefficients.size() > size_t(std::numeric_limits<int>::max())) {
        throw std::runtime_error("Invalid gas sphere count for Silo CSG mesh");
    }
    int const count = int(coefficients.size() / 4);
    std::vector<int> boundaryTypes(count, DBCSG_SPHERE_PR);
    std::vector<int> regionTypes(count, DBCSG_INNER);
    std::vector<int> leftIds(count);
    std::vector<int> rightIds(count, -1);
    std::vector<int> zoneIds(count);
    double extents[6];
    for (int dim = 0; dim < 3; dim++) {
        extents[dim] = std::numeric_limits<double>::max();
        extents[dim + 3] = -std::numeric_limits<double>::max();
    }
    for (int i = 0; i < count; i++) {
        double const radius = coefficients[4 * i + 3];
        if (!std::isfinite(radius) || radius <= 0.0) {
            throw std::runtime_error("Invalid gas smoothing radius in Silo CSG mesh");
        }
        for (int dim = 0; dim < 3; dim++) {
            double const center = coefficients[4 * i + dim];
            if (!std::isfinite(center) || !std::isfinite(center - radius) ||
                !std::isfinite(center + radius)) {
                throw std::runtime_error("Invalid gas sphere extent in Silo CSG mesh");
            }
            extents[dim] = std::min(extents[dim], center - radius);
            extents[dim + 3] = std::max(extents[dim + 3], center + radius);
        }
        leftIds[i] = i;
        zoneIds[i] = i;
    }
    if (DBPutCSGZonelist(db, zonesName, count, regionTypes.data(), leftIds.data(), rightIds.data(),
            nullptr, 0, DB_INT, count, zoneIds.data(), nullptr) < 0) {
        throw std::runtime_error("Cannot write gas CSG zonelist");
    }
    if (DBPutCsgmesh(db, meshName, 3, count, boundaryTypes.data(), nullptr, coefficients.data(),
            int(coefficients.size()), DB_DOUBLE, extents, zonesName, options) < 0) {
        throw std::runtime_error("Cannot write gas CSG mesh");
    }
}

template <typename Value>
void writeSiloSphereVariable(DBfile* db, char const* name, char const* meshName,
    std::span<Value const> values, DBoptlist* options) {
    static_assert(std::is_same_v<Value, float> || std::is_same_v<Value, double>);
    if (values.empty() || values.size() > size_t(std::numeric_limits<int>::max())) {
        throw std::runtime_error("Invalid gas CSG variable count");
    }
    char const* names[1] = {name};
    void const* variables[1] = {values.data()};
    int const datatype = std::is_same_v<Value, float> ? DB_FLOAT : DB_DOUBLE;
    if (DBPutCsgvar(db, name, meshName, 1, names, variables, int(values.size()), datatype,
            DB_ZONECENT, options) < 0) {
        throw std::runtime_error("Cannot write gas CSG variable");
    }
}

}    // namespace detail

#endif
