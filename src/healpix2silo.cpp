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

#include <cosmictiger/safe_io.hpp>

#include <boost/program_options.hpp>
#include <chealpix.h>

#include <silo.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

void sph_to_cart(double psi, double lambda, double* x, double* y) {
	double theta = psi;
	double theta0;
	int iters = 0;
	do {
		theta0 = theta;
		theta -= (2.0 * theta + std::sin(2.0 * theta) - M_PI * std::sin(psi)) / (2.0 + 2.0 * std::cos(2.0 * theta));
		iters++;
	} while (std::abs(theta0 - theta) > 1.0e-6);
	*x = 2.0 * lambda * std::cos(theta) / (M_PI);
	*y = std::sin(theta);
}

bool cart_to_sph(double x, double y, double* psi, double* lambda) {
	const auto theta = std::asin(y);
	const auto arg = (2.0 * theta + std::sin(2.0 * theta)) / M_PI;

	*psi = std::asin(arg);
	*lambda = M_PI * x / (2.0 * std::cos(theta));
	if (*lambda < -M_PI || *lambda > M_PI) {
		return false;
	} else {
		return true;
	}
}

int main(int argc, char **argv) {
	std::string infile, outfile;

	namespace po = boost::program_options;
	po::options_description command_opts("options");

	command_opts.add_options()                                                                       //
	("help", "produce help message")                                                                 //
	("in", po::value < std::string > (&infile)->default_value(""), "input file") //
	("out", po::value < std::string > (&outfile)->default_value(""), "output file") //
			;

	boost::program_options::variables_map vm;
	po::store(po::parse_command_line(argc, argv, command_opts), vm);
	po::notify(vm);

	if (infile == "") {
		printf("Input file not specified. Use --in=\n");
		return -1;
	}

	if (outfile == "") {
		printf("Output file not specified. Use --out=\n");
		return -1;
	}

	bool found_file = true;
	int rank = 0;
	int Nside;
	int npix;
	int cycle;
	double time;
	std::vector<float> healpix_data;
	PRINT("Reading file\n");
	int max_pix = 0;
	int min_pix = 1000000000;
	const std::string fname = infile;
	FILE* fp = fopen(fname.c_str(), "rb");
	if (rank == 0 && fp == nullptr) {
		printf("Unable to open %s\n", infile.c_str());
		abort();
	}
	if (fp == nullptr) {
		found_file = false;
	} else {
		found_file = true;
		FREAD(&Nside, sizeof(int), 1, fp);
		if (rank == 0) {
			PRINT("Nside = %i\n", Nside);
			npix = 12 * Nside * Nside;
			healpix_data.resize(npix, 0.0);
		}
		int size;
		FREAD(&size, sizeof(int), 1, fp);
		for (int pix = 0; pix < size; pix++) {
			float value;
			FREAD(&value, sizeof(float), 1, fp);
			healpix_data[pix] += value;
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

	const int res = (Nside * std::sqrt(1.5) + 0.5);

	int ORDER = 3;
	std::vector<float> mw_data;
	size_t iter = 0;
	size_t max_iter = res * res * 8;
	PRINT("Converting data\n");
	for (int iy = -res; iy < res; iy++) {
		for (int ix = -2 * res; ix < 2 * res; ix++) {
			double value = 0.0;
			for (int nx = 0; nx < ORDER; nx++) {
				for (int ny = 0; ny < ORDER; ny++) {
					const double x = (ix + (nx + 0.5) / ORDER) / double(res);
					const double y = (iy + (ny + 0.5) / ORDER) / double(res);
					double psi, lambda;
					if (cart_to_sph(x, y, &psi, &lambda)) {
						long ipring;
						ang2pix_ring(Nside, psi + M_PI / 2.0, lambda, &ipring);
						value += healpix_data[ipring];
					}
				}
			}
			mw_data.push_back(value);
			if (iter % 1000 == 0) {
				PRINT("%i%%\r", iter * 100 / max_iter);
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
		const double x0 = double(ix) / double(res);
		x.push_back(x0);
	}
	for (int iy = -res; iy <= res; iy++) {
		const double y0 = double(iy) / double(res);
		y.push_back(y0);
	}

	constexpr int ndim = 2;
	const int dims1[ndim] = { 4 * res + 1, 2 * res + 1 };
	const int dims2[ndim] = { 4 * res, 2 * res };
	const float* coords[ndim] = { x.data(), y.data() };
	const char* coord_names[ndim] = { "x", "y" };
	DBPutQuadmesh(db, "mesh", coord_names, coords, dims1, ndim, DB_FLOAT, DB_COLLINEAR, optlist);
	DBPutQuadvar1(db, "intensity", "mesh", mw_data.data(), dims2, ndim, nullptr, 0, DB_FLOAT, DB_ZONECENT, optlist);
	DBFreeOptlist(optlist);
	DBClose(db);
	return 0;
}
