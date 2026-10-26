/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <filesystem>
#include <stdexcept>
#include <string>
#include <system_error>

#include <silo.h>

#include <cosmictiger/core/options.hpp>
#include <cosmictiger/io/output.hpp>
#include <cosmictiger/math/constants.hpp>
#include <cosmictiger/math/math.hpp>
#include <cosmictiger/math/simulation_units.hpp>
#include <cosmictiger/particles/particles.hpp>

std::string outputPath(std::string const& filename) {
    auto const& options = getOptions();
    std::filesystem::path const requested(filename);
    if (!options.run.test.empty() || options.output.directory.empty()) {
        return filename;
    }
    auto const path = requested.is_absolute() ?
        requested :
        std::filesystem::path(options.output.directory) / requested;
    auto const parent = path.parent_path();
    if (!parent.empty()) {
        std::error_code error;
        std::filesystem::create_directories(parent, error);
        if (error) {
            throw std::runtime_error(
                "Cannot create output directory " + parent.string() + ": " + error.message());
        }
    }
    return path.string();
}

Vector<float> outputGetSlice();

HPX_PLAIN_ACTION(outputGetSlice, OutputGetSliceAction);

void outputParticles(std::string const filename, Vector<OutputParticle> const& parts) {
    DBfile* db = DBCreateReal(filename.c_str(), DB_CLOBBER, DB_LOCAL, "Meshless", DB_PDB);
    Vector<double> x(parts.size());
    Vector<double> y(parts.size());
    Vector<double> z(parts.size());
    for (int i = 0; i < parts.size(); i++) {
        x[i] = parts[i].x[XDIM].toDouble();
        y[i] = parts[i].x[YDIM].toDouble();
        z[i] = parts[i].x[ZDIM].toDouble();
    }
    int const numberOfParticles = parts.size();
    double* coords[NDIM] = {x.data(), y.data(), z.data()};
    DBPutPointmesh(db, "points", NDIM, coords, numberOfParticles, DB_DOUBLE, NULL);
    x = decltype(x)();
    y = decltype(y)();
    z = decltype(z)();
    for (int dim = 0; dim < NDIM; dim++) {
        Vector<float> v(parts.size());
        for (int i = 0; i < parts.size(); i++) {
            v[i] = parts[i].v[dim];
        }
        std::string nm = std::string() + "v_" + char('x' + char(dim));
        DBPutPointvar1(db, nm.c_str(), "points", v.data(), numberOfParticles, DB_FLOAT, NULL);
    }
    Vector<int> r(parts.size());
    for (int i = 0; i < parts.size(); i++) {
        r[i] = parts[i].r;
    }
    DBPutPointvar1(db, "rung", "points", r.data(), numberOfParticles, DB_INT, NULL);
    DBClose(db);
}

void outputTracers(int number) {
    std::string const filename = outputPath("tracers." + std::to_string(number) + ".silo");
    outputParticles(filename, particlesGetTracers());
}

void outputSlice(int number, double time) {
    constexpr int kNdim = 2;
    auto pixels = outputGetSlice();
    int const res = getOptions().analysis.sliceResolution;
    std::string const filename = outputPath("slice." + std::to_string(number) + ".silo");
    DBfile* db = DBCreateReal(filename.c_str(), DB_CLOBBER, DB_LOCAL, "Data", DB_PDB);
    Vector<float> x(res + 1);
    Vector<float> y(res + 1);
    float const hubble = getOptions().cosmology.hubble;
    float const c0 =
        (units::scalesFromOptions(getOptions()).lengthScale() * double(hubble) / units::kMegaparsec)
            .value();
    for (int i = 0; i < res + 1; i++) {
        y[i] = x[i] = (float) i / res * c0;
    }
    char const* coordnames[] = {"x", "y"};
    float const* coords[] = {x.data(), y.data()};
    int const dims1[] = {res + 1, res + 1};
    int const dims2[] = {res, res};
    auto optlist = DBMakeOptlist(5);
    float ftime = time;
    char label[6];
    strcpy(label, "Mpc/h");
    DBAddOption(optlist, DBOPT_CYCLE, &number);
    DBAddOption(optlist, DBOPT_TIME, &ftime);
    DBAddOption(optlist, DBOPT_DTIME, &time);
    DBAddOption(optlist, DBOPT_XLABEL, label);
    DBAddOption(optlist, DBOPT_YLABEL, label);
    DBPutQuadmesh(db, "mesh", coordnames, coords, dims1, kNdim, DB_FLOAT, DB_COLLINEAR, optlist);
    DBPutQuadvar1(db, "intensity", "mesh", pixels.data(), dims2, kNdim, NULL, 0, DB_FLOAT,
        DB_ZONECENT, optlist);
    DBFreeOptlist(optlist);
    DBClose(db);
}

Vector<float> outputGetSlice() {
    Vector<hpx::future<Vector<float>>> futs;
    int const res = getOptions().analysis.sliceResolution;
    Vector<float> pixels(sqr(res), 0.0);
    for (auto const& c : hpxChildren()) {
        futs.push_back(hpx::async<OutputGetSliceAction>(c));
    }
    for (int i = 0; i < particlesSize(); i++) {
        double const z = particlesPos(ZDIM, i).toDouble();
        if (z < getOptions().analysis.sliceSize) {
            double const x = particlesPos(XDIM, i).toDouble();
            double const y = particlesPos(YDIM, i).toDouble();
            int const xi = x * res;
            int const yi = y * res;
            int const j = xi * res + yi;
            pixels[j] += particlesMass(i);
        }
    }
    for (auto& f : futs) {
        auto v = f.get();
        for (int i = 0; i < sqr(res); i++) {
            pixels[i] += v[i];
        }
    }
    return pixels;
}
