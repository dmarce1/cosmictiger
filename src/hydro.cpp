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
#include <cosmictiger/view.hpp>
#include <cosmictiger/exact_sod.hpp>

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

sph_run_return sph_step(int minrung, double scale, double tau, double t0, int phase, double adot, int max_rung, int iter, double dt, bool verbose = true);

void hydro_driver(double tmax, float vdrag = 0.f, bool adiabatic = false) {
	time_type itime = 0;
	int minrung = 0;
	double t = 0.0;
	double t0 = tmax / 2048.0;
	int step = 0;
	int main_step = 0;
	float e0, ent0;
	do {
		if (minrung == 0) {
			view_output_views(main_step, 1.0);
//			output_line(main_step);
			main_step++;
		}
		int minrung = min_rung(itime);
		auto rc1 = sph_step(minrung, 1.0, t, t0, 0, 0.0, 0, 0, 0.0, false);
		sph_run_return rc2 = sph_step(minrung, 1.0, t, t0, 1, 0.0, 0, 0, 0.0, false);
		int maxrung = rc2.max_rung;
		double dt = t0 / (1 << maxrung);
		auto dr = drift(1.0, dt, t, t + dt, tmax);
		itime = inc(itime, maxrung);
		if (t == 0.0) {
			e0 = rc1.ekin + rc1.etherm;
			ent0 = rc1.ent;
		}
		float etot = rc1.ekin + rc1.etherm;
		rc1.momx /= sqrt(2.0f * dr.kin + 1e-20);
		rc1.momy /= sqrt(2.0f * dr.kin + 1e-20);
		rc1.momz /= sqrt(2.0f * dr.kin + 1e-20);
		t += dt;
		if (minrung != 0) {
			PRINT("%i %e %e %i %i\n", step, t, dt, minrung, maxrung);
		} else {
			PRINT("%i %e %e %i %i %e %e %e %e %e %e %e %e\n", step, t, dt, minrung, maxrung, rc1.ent, rc1.ekin, rc1.etherm, (etot - e0) / (rc1.ekin + 1e-20),
					rc1.momx, rc1.momy, rc1.momz, rc1.vol);
		}
		step++;
	} while (t < tmax);
}

void hydro_sod_test() {
	part_int nparts_total = pow(get_options().parts_dim, 3);
	double rho0 = 0.125;
	double rho1 = 1.0;
	double vx1 = 0.0;
	double vy1 = 0.0;
	double vz1 = 0.0;
	double vx0 = 0.0;
	double vy0 = -0.0;
	double vz0 = -0.0e-1;
	double p0 = 1.0e-1;
	double p1 = 1.0;
	sod_init_t sod;
	part_int left_dim = pow(0.25 * nparts_total * rho1 / (rho1 + rho0) / 2, 1.0 / 3.0) + 0.49999;
	part_int right_dim = pow(0.25 * nparts_total * rho0 / (rho1 + rho0) / 2, 1.0 / 3.0) + 0.49999;
	part_int nparts_left = 2 * left_dim * sqr(2 * left_dim);
	part_int nparts_right = 2 * right_dim * sqr(2 * right_dim);
	nparts_total = nparts_left + nparts_right;
	double dx = 0.5 / left_dim;
	part_int i = 0;
	auto opts = get_options();
	opts.sph_mass = rho1 * 0.5 / nparts_left;
	const double m = opts.sph_mass;
	set_options(opts);
	rho1 = nparts_left * m / 0.5;
	rho0 = nparts_right * m / 0.5;
	sod.rhol = rho1;
	sod.rhor = rho0;
	sod.pl = p1;
	sod.pr = p0;
	sod.gamma = get_options().gamma;
	constexpr float eta = 0.0;
	const double Ne = get_options().neighbor_number;
	PRINT("Sod dimensions are %i and %i rho0 %e rho1 %e\n", 2 * right_dim, 2 * left_dim, rho0 / (m * Ne) * (4.0 * M_PI / 3.0),
			rho1 / (m * Ne) * (4.0 * M_PI / 3.0));
	for (int ix = 0; ix < left_dim; ix++) {
		for (int iy = 0; iy < 2 * left_dim; iy++) {
			for (int iz = 0; iz < 2 * left_dim; iz++) {
				double x = (ix) * dx;
				double y = (iy) * dx;
				double z = (iz) * dx;
				double ent = p1 / pow(rho1, get_options().gamma);
				double h = pow(m * get_options().neighbor_number / (4.0 * M_PI / 3.0 * rho1), 1.0 / 3.0);
				sph_particles_resize(sph_particles_size() + 1);
				sph_particles_smooth_len(i) = h;
				sph_particles_pos(XDIM, i) = x + rand1() * dx * eta;
				sph_particles_pos(YDIM, i) = y + rand1() * dx * eta;
				sph_particles_pos(ZDIM, i) = z + rand1() * dx * eta;
				sph_particles_vel(XDIM, i) = vx0;
				sph_particles_vel(YDIM, i) = vy0;
				sph_particles_vel(ZDIM, i) = vz0;
				sph_particles_rung(i) = 0;
				sph_particles_ent(i) = ent;
				i++;
				//			PRINT("%i\n", i);
				x = (ix + 0.5) * dx;
				y = (iy + 0.5) * dx;
				z = (iz + 0.5) * dx;
				ent = p1 / pow(rho1, get_options().gamma);
				sph_particles_resize(sph_particles_size() + 1);
				sph_particles_smooth_len(i) = h;
				sph_particles_pos(XDIM, i) = x + rand1() * dx * eta;
				sph_particles_pos(YDIM, i) = y + rand1() * dx * eta;
				sph_particles_pos(ZDIM, i) = z + rand1() * dx * eta;
				sph_particles_vel(XDIM, i) = vx0;
				sph_particles_vel(YDIM, i) = vy0;
				sph_particles_vel(ZDIM, i) = vz0;
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
				double x = (ix) * dx;
				double y = (iy) * dx;
				double z = (iz) * dx;
				double ent = p0 / pow(rho0, get_options().gamma);
				double h = pow(m * get_options().neighbor_number / (4.0 * M_PI / 3.0 * rho0), 1.0 / 3.0);
				sph_particles_resize(sph_particles_size() + 1);
				sph_particles_smooth_len(i) = h;
				sph_particles_pos(XDIM, i) = x + rand1() * dx * eta;
				sph_particles_pos(YDIM, i) = y + rand1() * dx * eta;
				sph_particles_pos(ZDIM, i) = z + rand1() * dx * eta;
				sph_particles_vel(XDIM, i) = 0;
				sph_particles_vel(YDIM, i) = 0;
				sph_particles_vel(ZDIM, i) = 0;
				;
				sph_particles_rung(i) = 0;
				sph_particles_ent(i) = ent;
				//		PRINT("%i\n", i);
				i++;
				x = (ix + 0.5) * dx;
				y = (iy + 0.5) * dx;
				z = (iz + 0.5) * dx;
				sph_particles_resize(sph_particles_size() + 1);
				sph_particles_smooth_len(i) = h;
				sph_particles_pos(XDIM, i) = x + rand1() * dx * eta;
				sph_particles_pos(YDIM, i) = y + rand1() * dx * eta;
				sph_particles_pos(ZDIM, i) = z + rand1() * dx * eta;
				sph_particles_vel(XDIM, i) = 0;
				sph_particles_vel(YDIM, i) = 0;
				sph_particles_vel(ZDIM, i) = 0;
				;
				sph_particles_rung(i) = 0;
				sph_particles_ent(i) = ent;
				i++;
			}
		}
	}
	constexpr float t = 0.10;
	constexpr int N = 1000;
	hydro_driver(t);

	FILE* fp = fopen("sod.txt", "wt");
	double l1 = 0.0, l2 = 0.0, lmax = 0.0;
	double norm1 = 0.0, norm2 = 0.0;
	for (int l = 0; l < N; l++) {
		const int i = rand() % sph_particles_size();
		const int j = sph_particles_dm_index(i);
		float x = particles_pos(XDIM, j).to_float();
		const float h = sph_particles_smooth_len(i);
		const float rho = sph_den(1 / (h * h * h));
		sod_state_t state;
		float x0 = x;
		x0 -= 0.5;
		if (x0 > 0.25) {
			x0 = 0.5 - x0;
		} else if (x0 < -0.25) {
			x0 = -0.5 - x0;
		}
		exact_sod(&state, &sod, x0, t, 0);
		double dif = fabs(rho - state.rho);
		norm1 += state.rho;
		norm2 += sqr(state.rho);
		l1 += dif;
		l2 += dif * dif;
		lmax = std::max(lmax, dif);
		fprintf(fp, "%e %e %e\n", x, rho, state.rho);
	}
	l1 /= norm1;
	l2 /= norm2;
	l2 = sqrt(l2);
	PRINT("L1 = %e L2 = %e Lmax = %e\n", l1, l2, lmax);
	fclose(fp);
}
void hydro_helmholtz_test() {
	part_int nparts_total = pow(get_options().parts_dim, 3);
	double rho1 = 0.5;
	double rho0 = 1.0;
	double vx1 = 0.0;
	double vy1 = 0.5;
	double vz1 = 0.0;
	double vx0 = 0.0;
	double vy0 = -0.5;
	double vz0 = 0.0;
	double p1 = 1.0;
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
				double ent = p1 / pow(rho1, get_options().gamma);
				double h = pow(m * get_options().neighbor_number / (4.0 * M_PI / 3.0 * rho1), 1.0 / 3.0);
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
				double ent = p0 / pow(rho0, get_options().gamma);
				double h = pow(m * get_options().neighbor_number / (4.0 * M_PI / 3.0 * rho0), 1.0 / 3.0);
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
	hydro_driver(1.00);
}

void hydro_blast_test() {
	part_int nparts_total = pow(get_options().parts_dim, 3);
	double rho = 1.0;
	double p1 = 1000.0;
	double p0 = 1.0e-3;
	auto opts = get_options();
	part_int ndim = get_options().parts_dim;
	part_int nparts = std::pow(ndim, NDIM);
	ndim = pow(nparts / 2, 1. / 3.);
	nparts = 2 * ndim * ndim * ndim;
	opts.sph_mass = rho / nparts;
	const double m = opts.sph_mass;
	set_options(opts);
	part_int i = 0;
	const double dx = 1.0 / ndim;
	for (int ix = 0; ix < ndim; ix++) {
		for (int iy = 0; iy < ndim; iy++) {
			for (int iz = 0; iz < ndim; iz++) {
				double x = (ix + 0.5) * dx;
				double y = (iy + 0.5) * dx;
				double z = (iz + 0.5) * dx;
				const bool center = (ix == ndim / 2 && iy == ndim / 2 && iz == ndim / 2);
				double ent = center ? p1 : p0 / pow(rho, get_options().gamma);
				if (center) {
					PRINT("CENTER at %e %e %e\n", x, y, z);
				}
				double h = pow(m * get_options().neighbor_number / (4.0 * M_PI / 3.0 * rho), 1.0 / 3.0);
				sph_particles_resize(sph_particles_size() + 1);
				//	PRINT( "%e\n", h);
				sph_particles_smooth_len(i) = h;
				sph_particles_pos(XDIM, i) = x;
				sph_particles_pos(YDIM, i) = y;
				sph_particles_pos(ZDIM, i) = z;
				sph_particles_vel(XDIM, i) = 0.f;
				sph_particles_vel(YDIM, i) = 0.f;
				sph_particles_vel(ZDIM, i) = 0.f;
				sph_particles_rung(i) = 0;
				sph_particles_ent(i) = ent;
				i++;
				x = (ix) * dx;
				y = (iy) * dx;
				z = (iz) * dx;
				ent = p0 / pow(rho, get_options().gamma);
				if (center) {
					PRINT("CENTER at %e %e %e\n", x, y, z);
				}
				sph_particles_resize(sph_particles_size() + 1);
				//	PRINT( "%e\n", h);
				sph_particles_smooth_len(i) = h;
				sph_particles_pos(XDIM, i) = x;
				sph_particles_pos(YDIM, i) = y;
				sph_particles_pos(ZDIM, i) = z;
				sph_particles_vel(XDIM, i) = 0.f;
				sph_particles_vel(YDIM, i) = 0.f;
				sph_particles_vel(ZDIM, i) = 0.f;
				sph_particles_rung(i) = 0;
				sph_particles_ent(i) = ent;
				i++;
				//			PRINT("%i\n", i);
			}
		}
	}
	hydro_driver(2.5);
}

void hydro_wave_test() {
	part_int nparts_total = pow(get_options().parts_dim, 3);
	double rho = 1.0;
	double p0 = 0.845154255;
	auto opts = get_options();
	const part_int ndim = get_options().parts_dim;
	const part_int nparts = std::pow(ndim, NDIM);
	opts.sph_mass = rho / nparts;
	const double m = opts.sph_mass;
	set_options(opts);
	part_int i = 0;
	const double dx = 1.0 / ndim;
	for (int ix = 0; ix < ndim; ix++) {
		for (int iy = 0; iy < ndim; iy++) {
			for (int iz = 0; iz < ndim; iz++) {
				double x = (ix + 0.5) * dx;
				double y = (iy + 0.5) * dx;
				double z = (iz + 0.5) * dx;
				double ent = p0 / pow(rho, get_options().gamma);
				double h = pow(m * get_options().neighbor_number / (4.0 * M_PI / 3.0 * rho), 1.0 / 3.0);
				//	PRINT( "%e\n", h);
				if (ix == ndim / 2) {
					x += 0.99 * dx;
				} else if (ix == ndim / 2 - 1) {
					x -= 0.99 * dx;
				}
				sph_particles_resize(sph_particles_size() + 1);
				sph_particles_smooth_len(i) = h;
				sph_particles_pos(XDIM, i) = x;
				sph_particles_pos(YDIM, i) = y;
				sph_particles_pos(ZDIM, i) = z;
				sph_particles_vel(XDIM, i) = 0.f;
				sph_particles_vel(YDIM, i) = 0.f;
				sph_particles_vel(ZDIM, i) = 0.f;
				sph_particles_rung(i) = 0;
				sph_particles_ent(i) = ent;
				i++;
				//			PRINT("%i\n", i);
			}
		}
	}
	hydro_driver(0.5);
}
