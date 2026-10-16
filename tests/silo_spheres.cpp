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

#include <array>
#include <cstdio>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>

#include <silo.h>

#include <cosmictiger/silo_spheres.hpp>

namespace detail {

void requireSiloSphere(bool condition, std::string const& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void checkSiloSphereOutput(char const* filename) {
    std::unique_ptr<DBfile, decltype(&DBClose)> file(
        DBOpen(filename, DB_UNKNOWN, DB_READ), DBClose);
    requireSiloSphere(bool(file), "open CSG file");
    std::unique_ptr<DBcsgmesh, decltype(&DBFreeCsgmesh)> mesh(
        DBGetCsgmesh(file.get(), "gas"), DBFreeCsgmesh);
    requireSiloSphere(bool(mesh), "read gas CSG mesh");
    requireSiloSphere(mesh->ndims == 3 && mesh->nbounds == 2 && mesh->datatype == DB_DOUBLE &&
            mesh->lcoeffs == 8 && mesh->coeffs && mesh->typeflags,
        "gas sphere mesh shape");
    std::array<double, 8> const coefficients = {1.0, -2.0, 3.5, 0.25, -4.0, 5.0, -6.0, 1.5};
    auto const* storedCoefficients = static_cast<double const*>(mesh->coeffs);
    for (int i = 0; i < 2; i++) {
        requireSiloSphere(
            mesh->typeflags[i] == DBCSG_SPHERE_PR, "one sphere boundary per particle");
    }
    for (int i = 0; i < 8; i++) {
        requireSiloSphere(storedCoefficients[i] == coefficients[i], "sphere center or radius");
    }
    std::array<double, 3> const minima = {-5.5, -2.25, -7.5};
    std::array<double, 3> const maxima = {1.25, 6.5, 3.75};
    for (int dim = 0; dim < 3; dim++) {
        requireSiloSphere(
            mesh->min_extents[dim] == minima[dim] && mesh->max_extents[dim] == maxima[dim],
            "sphere mesh extents");
        requireSiloSphere(
            mesh->units[dim] && std::string(mesh->units[dim]) == "cm", "sphere coordinate units");
    }
    requireSiloSphere(mesh->dtime == 2.75, "sphere mesh time");
    requireSiloSphere(mesh->zones && mesh->zones->nregs == 2 && mesh->zones->nzones == 2 &&
            mesh->zones->typeflags && mesh->zones->leftids && mesh->zones->rightids &&
            mesh->zones->zonelist,
        "one region and zone per sphere");
    for (int i = 0; i < 2; i++) {
        requireSiloSphere(mesh->zones->typeflags[i] == DBCSG_INNER &&
                mesh->zones->leftids[i] == i && mesh->zones->rightids[i] == -1 &&
                mesh->zones->zonelist[i] == i,
            "sphere interior topology");
    }

    std::unique_ptr<DBcsgvar, decltype(&DBFreeCsgvar)> rho(
        DBGetCsgvar(file.get(), "rho"), DBFreeCsgvar);
    requireSiloSphere(bool(rho) && rho->datatype == DB_DOUBLE && rho->centering == DB_ZONECENT &&
            rho->nels == 2 && rho->nvals == 1 && rho->vals && rho->vals[0],
        "double zone variable");
    auto const* density = static_cast<double const*>(rho->vals[0]);
    requireSiloSphere(density[0] == 8.0 && density[1] == 3.25, "density values");
    requireSiloSphere(rho->units && std::string(rho->units) == "g/cm^3", "density units");

    std::unique_ptr<DBcsgvar, decltype(&DBFreeCsgvar)> radius(
        DBGetCsgvar(file.get(), "kernel_support_radius"), DBFreeCsgvar);
    requireSiloSphere(bool(radius) && radius->datatype == DB_FLOAT &&
            radius->centering == DB_ZONECENT && radius->nels == 2 && radius->nvals == 1 &&
            radius->vals && radius->vals[0],
        "float zone variable");
    auto const* radii = static_cast<float const*>(radius->vals[0]);
    requireSiloSphere(
        radii[0] == 0.25f && radii[1] == 1.5f, "radius field matches sphere coefficients");
}

void testSiloSpheres(char const* filename, int driver) {
    std::unique_ptr<DBfile, decltype(&DBClose)> file(
        DBCreateReal(filename, DB_CLOBBER, DB_LOCAL, "CSG spheres", driver), DBClose);
    requireSiloSphere(bool(file), "create CSG file");
    std::array<double, 8> const coefficients = {1.0, -2.0, 3.5, 0.25, -4.0, 5.0, -6.0, 1.5};
    std::unique_ptr<DBoptlist, decltype(&DBFreeOptlist)> meshOptions(
        DBMakeOptlist(4), DBFreeOptlist);
    requireSiloSphere(bool(meshOptions), "create mesh options");
    char coordinateUnit[] = "cm";
    double time = 2.75;
    requireSiloSphere(DBAddOption(meshOptions.get(), DBOPT_XUNITS, coordinateUnit) >= 0 &&
            DBAddOption(meshOptions.get(), DBOPT_YUNITS, coordinateUnit) >= 0 &&
            DBAddOption(meshOptions.get(), DBOPT_ZUNITS, coordinateUnit) >= 0 &&
            DBAddOption(meshOptions.get(), DBOPT_DTIME, &time) >= 0,
        "set mesh options");
    writeSiloSpheres(
        file.get(), "gas", "gas_zones", std::span<double const>(coefficients), meshOptions.get());
    std::array<double, 2> const density = {8.0, 3.25};
    std::unique_ptr<DBoptlist, decltype(&DBFreeOptlist)> densityOptions(
        DBMakeOptlist(1), DBFreeOptlist);
    requireSiloSphere(bool(densityOptions), "create density options");
    char densityUnit[] = "g/cm^3";
    requireSiloSphere(
        DBAddOption(densityOptions.get(), DBOPT_UNITS, densityUnit) >= 0, "set density unit");
    writeSiloSphereVariable(
        file.get(), "rho", "gas", std::span<double const>(density), densityOptions.get());
    std::array<float, 2> const radii = {0.25f, 1.5f};
    writeSiloSphereVariable(
        file.get(), "kernel_support_radius", "gas", std::span<float const>(radii), nullptr);
    requireSiloSphere(DBClose(file.release()) >= 0, "close CSG file");
    checkSiloSphereOutput(filename);
}

}    // namespace detail

int main() {
    try {
        detail::testSiloSpheres("silo_spheres_pdb.silo", DB_PDB);
        detail::testSiloSpheres("silo_spheres_hdf5.silo", DB_HDF5);
    } catch (std::exception const& error) {
        std::fprintf(stderr, "Silo sphere test: %s\n", error.what());
        return 1;
    }
    return 0;
}
