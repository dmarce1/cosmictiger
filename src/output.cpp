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

#include <silo.h>

#include <cosmictiger/constants.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/output.hpp>
#include <cosmictiger/particles.hpp>

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
    int const numberParticles = parts.size();
    double* coords[NDIM] = {x.data(), y.data(), z.data()};
    DBPutPointmesh(db, "points", NDIM, coords, numberParticles, DB_DOUBLE, NULL);
    x = decltype(x)();
    y = decltype(y)();
    z = decltype(z)();
    for (int dim = 0; dim < NDIM; dim++) {
        Vector<float> v(parts.size());
        for (int i = 0; i < parts.size(); i++) {
            v[i] = parts[i].v[dim];
        }
        std::string nm = std::string() + "v_" + char('x' + char(dim));
        DBPutPointvar1(db, nm.c_str(), "points", v.data(), numberParticles, DB_FLOAT, NULL);
    }
    Vector<int> r(parts.size());
    for (int i = 0; i < parts.size(); i++) {
        r[i] = parts[i].r;
    }
    DBPutPointvar1(db, "rung", "points", r.data(), numberParticles, DB_INT, NULL);
    DBClose(db);
}

void outputTracers(int number) {
    profilerEnter("output_tracers");
    std::string filename = "tracers." + std::to_string(number) + ".silo";
    outputParticles(filename, particlesGetTracers());
    profilerExit();
}

void outputSlice(int number, double time) {
    profilerEnter("output_slice");
    constexpr int kNdim = 2;
    auto pixels = outputGetSlice();
    int const res = getOptions().sliceRes;
    std::string const filename =
        std::string("slice.") + std::to_string(number) + std::string(".silo");
    DBfile* db = DBCreateReal(filename.c_str(), DB_CLOBBER, DB_LOCAL, "Data", DB_PDB);
    Vector<float> x(res + 1);
    Vector<float> y(res + 1);
    float const hubble = getOptions().hubble;
    float const c0 = getOptions().codeToCm * hubble / constants::kMpcToCm;
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
    profilerExit();
}

Vector<float> outputGetSlice() {
    Vector<hpx::future<Vector<float>>> futs;
    int const res = getOptions().sliceRes;
    Vector<float> pixels(sqr(res), 0.0);
    for (auto const& c : hpxChildren()) {
        futs.push_back(hpx::async<OutputGetSliceAction>(c));
    }
    for (int i = 0; i < particlesSize(); i++) {
        double const z = particlesPos(ZDIM, i).toDouble();
        if (z < getOptions().sliceSize) {
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
