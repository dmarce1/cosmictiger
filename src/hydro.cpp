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

#include <cosmictiger/sph.hpp>
#include <cosmictiger/sph_particles.hpp>
#include <cosmictiger/time.hpp>
#include <cosmictiger/drift.hpp>

static void output_line(int num) {

	vector<double> x(1000), y(1000, 0.5), z(1000, 0.5);
	for (int i = 0; i < 1000; i++) {
		x[i] = i / 1000.0;
	}
	std::string filename = "lineout." + std::to_string(num) + ".txt";
	FILE* fp = fopen(filename.c_str(), "wt");
	auto values = sph_values_at(x, y, z);
	for (int i = 0; i < 1000; i++) {
		fprintf(fp, "%e %e %e %e %e %e\n", x[i], values[i].rho, values[i].vx, values[i].vy, values[i].vz, values[i].p);
	}
	fclose(fp);
}

sph_run_return sph_step(int minrung, double scale, double tau, double t0, int phase, bool verbose);

void hydro_driver(double tmax) {
	time_type itime = 0;
	int minrung = 0;
	double t = 0.0;
	double t0 = tmax / 64.0;
	int step = 0;
	int main_step = 0;
	float e0;
	do {
		int minrung = min_rung(itime);
		auto rc1 = sph_step(minrung, 1.0, t, t0, 0, false);
		sph_run_return rc2 = sph_step(minrung, 1.0, t, t0, 1, false);
		int maxrung = rc2.max_rung;
		double dt = t0 / (1 << maxrung);
		auto dr = drift(1.0, dt, t, t + dt, tmax);
		itime = inc(itime, maxrung);
		if (t == 0.0) {
			e0 = rc1.ekin + rc1.etherm;
		}
		float etot = rc1.ekin + rc1.etherm;
		rc1.momx /= sqrt(2.0f * dr.kin + 1e-20);
		rc1.momy /= sqrt(2.0f * dr.kin + 1e-20);
		rc1.momz /= sqrt(2.0f * dr.kin + 1e-20);
		t += dt;
		PRINT("%i %e %e %i %i %e %e %e %e %e %e\n", step, t, dt, minrung, maxrung, rc1.ekin, rc1.etherm, (etot - e0) / (rc1.ekin+1e-20), rc1.momx, rc1.momy, rc1.momz);
		step++;
		if (minrung == 0) {
			output_line(main_step);
			main_step++;
		}
	} while (t < tmax);
}

void hydro_sod_test() {
	part_int nparts_total = pow(get_options().parts_dim, 3);
	double rho1 = 0.125;
	double rho0 = 1.0;
	double vx1 = 0.0;
	double vy1 = 0.0;
	double vz1 = 0.0;
	double vx0 = 0.0;
	double vy0 = 0.0;
	double vz0 = 0.0;
	double p1 = 0.1;
	double p0 = 1.0;
	part_int left_dim = pow(0.25 * nparts_total * rho1 / (rho1 + rho0), 1.0 / 3.0) + 0.49999;
	part_int right_dim = pow(0.25 * nparts_total * rho0 / (rho1 + rho0), 1.0 / 3.0) + 0.49999;
	part_int nparts_left = left_dim * sqr(2 * left_dim);
	part_int nparts_right = right_dim * sqr(2 * right_dim);
	nparts_total = nparts_left + nparts_right;
	double dx = 0.5 / left_dim;
	part_int i = 0;
	auto opts = get_options();
	opts.sph_mass = rho1 * 0.5 / nparts_left;
	const double m = opts.sph_mass;
	set_options(opts);
	rho1 = nparts_left * m / 0.5f;
	rho0 = nparts_right * m / 0.5f;
	for (int ix = 0; ix < left_dim; ix++) {
		for (int iy = 0; iy < 2 * left_dim; iy++) {
			for (int iz = 0; iz < 2 * left_dim; iz++) {
				double x = (ix + 0.5) * dx;
				double y = (iy + 0.5) * dx;
				double z = (iz + 0.5) * dx;
				double ent = p1 / pow(rho1, SPH_GAMMA);
				double h = pow(m * SPH_NEIGHBOR_COUNT / (4.0 * M_PI / 3.0 * rho1), 1.0 / 3.0);
				sph_particles_resize(sph_particles_size() + 1);
				sph_particles_smooth_len(i) = h;
				sph_particles_pos(XDIM, i) = x;
				sph_particles_pos(YDIM, i) = y;
				sph_particles_pos(ZDIM, i) = z;
				sph_particles_vel(XDIM, i) = vx1;
				sph_particles_vel(YDIM, i) = vy1;
				sph_particles_vel(ZDIM, i) = vz1;
				sph_particles_rung(i) = 0;
				sph_particles_ent(i) = ent;
				i++;
				//			PRINT("%i\n", i);
			}
		}
	}
	dx = 0.5 / right_dim;
	for (int ix0 = right_dim; ix0 < 2 * right_dim; ix0++) {
		int ix = ix0;
		for (int iy = 0; iy < 2 * right_dim; iy++) {
			for (int iz = 0; iz < 2 * right_dim; iz++) {
				double x = (ix + 0.5) * dx;
				double y = (iy + 0.5) * dx;
				double z = (iz + 0.5) * dx;
				double ent = p0 / pow(rho0, SPH_GAMMA);
				double h = pow(m * SPH_NEIGHBOR_COUNT / (4.0 * M_PI / 3.0 * rho0), 1.0 / 3.0);
				sph_particles_resize(sph_particles_size() + 1);
				sph_particles_smooth_len(i) = h;
				sph_particles_pos(XDIM, i) = x;
				sph_particles_pos(YDIM, i) = y;
				sph_particles_pos(ZDIM, i) = z;
				sph_particles_vel(XDIM, i) = vx0;
				sph_particles_vel(YDIM, i) = vy0;
				sph_particles_vel(ZDIM, i) = vz0;
				sph_particles_rung(i) = 0;
				sph_particles_ent(i) = ent;
				//		PRINT("%i\n", i);
				i++;
			}
		}
	}
	hydro_driver(0.25);
}
