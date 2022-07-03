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

#include <cosmictiger/output.hpp>
#include <cosmictiger/particles.hpp>
#include <cosmictiger/constants.hpp>
#include <cosmictiger/math.hpp>

#include <silo.h>

vector<float> output_get_slice();

HPX_PLAIN_ACTION (output_get_slice);

void output_particles(const std::string filename, const vector<output_particle>& parts) {
	DBfile *db = DBCreateReal(filename.c_str(), DB_CLOBBER, DB_LOCAL, "Meshless", DB_PDB);
	vector<double> x(parts.size());
	vector<double> y(parts.size());
	vector<double> z(parts.size());
	for (int i = 0; i < parts.size(); i++) {
		x[i] = parts[i].x[XDIM].to_double();
		y[i] = parts[i].x[YDIM].to_double();
		z[i] = parts[i].x[ZDIM].to_double();
	}
	const int nparts = parts.size();
	double *coords[NDIM] = { x.data(), y.data(), z.data() };
		DBPutPointmesh(db, "points", NDIM, coords, nparts, DB_DOUBLE, NULL);
		x = decltype(x)();
	y = decltype(y)();
	z = decltype(z)();
	for (int dim = 0; dim < NDIM; dim++) {
		vector<float> v(parts.size());
		for (int i = 0; i < parts.size(); i++) {
			v[i] = parts[i].v[dim];
		}
		std::string nm = std::string() + "v_" + char('x' + char(dim));
		DBPutPointvar1(db, nm.c_str(), "points", v.data(), nparts, DB_FLOAT, NULL);
	}
	vector<int> r(parts.size());
	for (int i = 0; i < parts.size(); i++) {
		r[i] = parts[i].r;
	}
	DBPutPointvar1(db, "rung", "points", r.data(), nparts, DB_INT, NULL);
	DBClose(db);
}

void output_tracers(int number) {
	profiler_enter("output_tracers");
	std::string filename = "tracers." + std::to_string(number) + ".silo";
	output_particles(filename, particles_get_tracers());
	profiler_exit();
}

void output_slice(int number, double time) {
	profiler_enter("output_slice");
	constexpr int ndim = 2;
	auto pixels = output_get_slice();
	const int res = get_options().slice_res;
	const std::string filename = std::string("slice.") + std::to_string(number) + std::string(".silo");
	DBfile *db = DBCreateReal(filename.c_str(), DB_CLOBBER, DB_LOCAL, "Data", DB_PDB);
	vector<float> x(res + 1);
	vector<float> y(res + 1);
	const float hubble = get_options().hubble;
	const float c0 = get_options().code_to_cm * hubble / constants::mpc_to_cm;
	for (int i = 0; i < res + 1; i++) {
		y[i] = x[i] = (float) i / res * c0;
	}
	const char* coordnames[] = { "x", "y" };
	const float* coords[] = { x.data(), y.data() };
	const int dims1[] = { res + 1, res + 1 };
	const int dims2[] = { res, res };
	auto optlist = DBMakeOptlist(5);
	float ftime = time;
	char label[6];
	strcpy(label, "Mpc/h");
	DBAddOption(optlist, DBOPT_CYCLE, &number);
	DBAddOption(optlist, DBOPT_TIME, &ftime);
	DBAddOption(optlist, DBOPT_DTIME, &time);
	DBAddOption(optlist, DBOPT_XLABEL, label);
	DBAddOption(optlist, DBOPT_YLABEL, label);
	DBPutQuadmesh(db, "mesh", coordnames, coords, dims1, ndim, DB_FLOAT, DB_COLLINEAR, optlist);
	DBPutQuadvar1(db, "intensity", "mesh", pixels.data(), dims2, ndim, NULL, 0, DB_FLOAT, DB_ZONECENT, optlist);
	DBFreeOptlist(optlist);
	DBClose(db);
	profiler_exit();
}

vector<float> output_get_slice() {
	vector<hpx::future<vector<float>>>futs;
	const int res = get_options().slice_res;
	vector<float> pixels(sqr(res), 0.0);
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async < output_get_slice_action > (c));
	}
	for (int i = 0; i < particles_size(); i++) {
		const double z = particles_pos(ZDIM, i).to_double();
		if (z < get_options().slice_size) {
			const double x = particles_pos(XDIM, i).to_double();
			const double y = particles_pos(YDIM, i).to_double();
			const int xi = x * res;
			const int yi = y * res;
			const int j = xi * res + yi;
			pixels[j] += 1.0;
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

