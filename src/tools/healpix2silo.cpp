/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2021  Dominic C. Marcello

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

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include <boost/program_options.hpp>
#include <chealpix.h>
#include <silo.h>

#include <cosmictiger/core/safe_io.hpp>

void sphToCart(double psi, double lambda, double* x, double* y) {
    double theta = psi;
    double theta0;
    int iters = 0;
    do {
        theta0 = theta;
        theta -= (2.0 * theta + std::sin(2.0 * theta) - M_PI * std::sin(psi)) /
            (2.0 + 2.0 * std::cos(2.0 * theta));
        iters++;
    } while (std::abs(theta0 - theta) > 1.0e-6);
    *x = 2.0 * lambda * std::cos(theta) / (M_PI);
    *y = std::sin(theta);
}

bool cartToSph(double x, double y, double* psi, double* lambda) {
    auto const theta = std::asin(y);
    auto const arg = (2.0 * theta + std::sin(2.0 * theta)) / M_PI;

    *psi = std::asin(arg);
    *lambda = M_PI * x / (2.0 * std::cos(theta));
    if (*lambda < -M_PI || *lambda > M_PI) {
        return false;
    } else {
        return true;
    }
}

int main(int argc, char** argv) {
    std::string infile, outfile;

    namespace po = boost::program_options;
    po::options_description commandOpts("options");

    commandOpts.add_options()                                                                  //
        ("help", "produce help message")                                                       //
        ("input.file", po::value<std::string>(&infile)->default_value(""), "input file")       //
        ("output.file", po::value<std::string>(&outfile)->default_value(""), "output file")    //
        ;

    boost::program_options::variables_map vm;
    po::store(po::parse_command_line(argc, argv, commandOpts), vm);
    po::notify(vm);
    if (vm.count("help")) {
        std::cout << commandOpts << "\n";
        return 0;
    }

    if (infile == "") {
        printf("Input file not specified. Use --input.file=\n");
        return -1;
    }

    if (outfile == "") {
        printf("Output file not specified. Use --output.file=\n");
        return -1;
    }

    bool foundFile = true;
    int rank = 0;
    int nside;
    int npix;
    int cycle;
    double time;
    std::vector<float> healpixData;
    PRINT("Reading file\n");
    int maxPix = 0;
    int minPix = 1000000000;
    std::string const fname = infile;
    FILE* fp = fopen(fname.c_str(), "rb");
    if (rank == 0 && fp == nullptr) {
        printf("Unable to open %s\n", infile.c_str());
        abort();
    }
    if (fp == nullptr) {
        foundFile = false;
    } else {
        foundFile = true;
        FREAD(&nside, sizeof(int), 1, fp);
        if (rank == 0) {
            PRINT("Nside = %i\n", nside);
            npix = 12 * nside * nside;
            healpixData.resize(npix, 0.0);
        }
        int size;
        FREAD(&size, sizeof(int), 1, fp);
        for (int pix = 0; pix < size; pix++) {
            float value;
            FREAD(&value, sizeof(float), 1, fp);
            healpixData[pix] += value;
        }
        if (fread(&cycle, sizeof(int), 1, fp) == 0) {
            cycle = 0;
            time = 0.0;
        } else {
            FREAD(&time, sizeof(double), 1, fp);
        }
        fclose(fp);
    }
    PRINT("Done reading file\n");

    int const res = (nside * std::sqrt(1.5) + 0.5);

    int order = 3;
    std::vector<float> mwData;
    size_t iter = 0;
    size_t maxIter = res * res * 8;
    PRINT("Converting data\n");
    for (int iy = -res; iy < res; iy++) {
        for (int ix = -2 * res; ix < 2 * res; ix++) {
            double value = 0.0;
            for (int nx = 0; nx < order; nx++) {
                for (int ny = 0; ny < order; ny++) {
                    double const x = (ix + (nx + 0.5) / order) / double(res);
                    double const y = (iy + (ny + 0.5) / order) / double(res);
                    double psi, lambda;
                    if (cartToSph(x, y, &psi, &lambda)) {
                        long ipring;
                        ang2pix_ring(nside, psi + M_PI / 2.0, lambda, &ipring);
                        value += healpixData[ipring];
                    }
                }
            }
            mwData.push_back(value);
            if (iter % 1000 == 0) {
                PRINT("%i%%\r", iter * 100 / maxIter);
            }
            iter++;
        }
    }

    auto db = DBCreate(outfile.c_str(), DB_CLOBBER, DB_LOCAL, NULL, DB_PDB);
    auto optlist = DBMakeOptlist(3);
    float ftime = time;
    DBAddOption(optlist, DBOPT_CYCLE, &cycle);
    DBAddOption(optlist, DBOPT_TIME, &ftime);
    DBAddOption(optlist, DBOPT_DTIME, &time);
    std::vector<float> x, y;
    for (int ix = -2 * res; ix <= 2 * res; ix++) {
        double const x0 = double(ix) / double(res);
        x.push_back(x0);
    }
    for (int iy = -res; iy <= res; iy++) {
        double const y0 = double(iy) / double(res);
        y.push_back(y0);
    }

    constexpr int kNdim = 2;
    int const dims1[kNdim] = {4 * res + 1, 2 * res + 1};
    int const dims2[kNdim] = {4 * res, 2 * res};
    float const* coords[kNdim] = {x.data(), y.data()};
    char const* coordNames[kNdim] = {"x", "y"};
    DBPutQuadmesh(db, "mesh", coordNames, coords, dims1, kNdim, DB_FLOAT, DB_COLLINEAR, optlist);
    DBPutQuadvar1(db, "intensity", "mesh", mwData.data(), dims2, kNdim, nullptr, 0, DB_FLOAT,
        DB_ZONECENT, optlist);
    DBFreeOptlist(optlist);
    DBClose(db);
    return 0;
}
