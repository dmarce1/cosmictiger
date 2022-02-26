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
#include <cosmictiger/driver.hpp>

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

void hydro_driver(double tmax, int nsteps = 64) {
	time_type itime = 0;
	int minrung = 0;
	double t = 0.0;
	double t0 = tmax / nsteps;
	int step = 0;
	int main_step = 0;
	float e0, ent0;
	const double m = get_options().sph_mass;
	do {
		double ekin = 0.0;
		double eint = 0.0;
		double xmom = 0.0, ymom = 0.0, zmom = 0.0;
		for (part_int i = 0; i < sph_particles_size(); i++) {
			const int k = sph_particles_dm_index(i);
			const double vx = particles_vel(XDIM, k);
			const double vy = particles_vel(YDIM, k);
			const double vz = particles_vel(ZDIM, k);
			xmom += m * vx;
			ymom += m * vy;
			zmom += m * vz;
			const double e = sph_particles_eint(i);
			ekin += 0.5 * m * sqr(vx, vy, vz);
			eint += m * e;
		}
		FILE* fp = fopen("energy.dat", "at");
		const double etot = ekin + eint;
		double etot0;
		if (t == 0.0) {
			etot0 = etot;
		}
		fprintf(fp, "%e %e %e %e %e %e %e %e\n", t, xmom, ymom, zmom, ekin, eint, etot, (etot - etot0) / etot);
		fclose(fp);
		int minrung = min_rung(itime);
//		if (minrung == 0) {
		view_output_views(main_step, 1.0);
//			output_line(main_step);
		main_step++;
		//	}
		double dummy;
		auto rc1 = sph_step(minrung, 1.0, t, t0, 0, 0.0, 0, 0, 0.0, &dummy, false);
		sph_run_return rc2 = sph_step(minrung, 1.0, t, t0, 1, 0.0, 0, 0, 0.0, &dummy, false);
		int maxrung = rc2.max_rung;
		double dt = t0 / (1 << maxrung);
		auto dr = drift(1.0, dt, t, t + dt, tmax);
		itime = inc(itime, maxrung);
		if (t == 0.0) {
			e0 = rc1.ekin + rc1.etherm;
			ent0 = rc1.ent;
		}
		t += dt;
		PRINT("%i %e %e %i %i\n", step, t, dt, minrung, maxrung);
		step++;
	} while (t < tmax);
	view_output_views(main_step, 1.0);
}

void hydro_rt_test() {
	part_int nparts_total = pow(get_options().parts_dim, 3);
	part_int hi_parts = 2 * nparts_total / 3;
	part_int lo_parts = nparts_total / 3;
	part_int dim = pow(lo_parts * 0.25, (1. / 3.));
	double rho1 = 2.0;
	double rho0 = 1.0;
	lo_parts = 4 * dim * dim * dim;
	hi_parts = 2 * lo_parts;
	const double gy = get_options().gy;
	auto opts = get_options();
	opts.sph_mass = rho0 / lo_parts * 0.5;
	const double m = opts.sph_mass;
	set_options(opts);
	rho1 = hi_parts * m / 0.5;
	rho0 = lo_parts * m / 0.5;

	if (gy >= 0.0) {
		//	PRINT("Error gy must be negative for RT test\n");
		//	abort();
	}
	constexpr double eta = 0.01;
	double dx = .5 / dim;
	int i = 0;
	double v0 = 0.1;
	for (int ix = 0; ix < 2 * dim; ix++) {
		for (int iy = dim / 2; iy <= 3 * dim / 2; iy++) {
			for (int iz = 0; iz < 2 * dim; iz++) {
				double x = (ix + 0.5) * dx;
				double y = (iy + 0.5) * dx;
				double z = (iz + 0.5) * dx;
				double vy = v0 / 4.0 * (1.0 + cos(8.0 * M_PI * (x - 0.5)) * (1.0 + cos(6.0 * M_PI * (x - 0.5))));
				double p = 1.5 + fabs(rho0 * gy * (y - 0.5));
				double eint = p / (get_options().gamma - 1) / rho0;
				double h = pow(m * get_options().neighbor_number / (4.0 * M_PI / 3.0 * rho0), 1.0 / 3.0);
				sph_particles_resize(sph_particles_size() + 1);
				sph_particles_smooth_len(i) = h;
				sph_particles_pos(XDIM, i) = x;
				sph_particles_pos(YDIM, i) = y;
				sph_particles_pos(ZDIM, i) = z;
				sph_particles_vel(XDIM, i) = 0;
				sph_particles_vel(YDIM, i) = vy;
				;
				sph_particles_vel(ZDIM, i) = 0;
				;
				sph_particles_rung(i) = 0;
				sph_particles_eint(i) = eint;
				i++;
			}
		}
	}
	for (int ix = 0; ix < 2 * dim; ix++) {
		for (int iy = 0; iy < 2 * dim; iy++) {
			for (int iz = 0; iz < 2 * dim; iz++) {
				double x = (ix + 0.5) * dx;
				double y = (iy + 0.5) * dx;
				double z = (iz + 0.5) * dx;
				double p;
				double vy = v0 / 4.0 * (1.0 + cos(8.0 * M_PI * (x - 0.5)) * (1.0 + cos(6.0 * M_PI * (x - 0.5))));
				double h = pow(m * get_options().neighbor_number / (4.0 * M_PI / 3.0 * rho1), 1.0 / 3.0);
				if (y > 0.75 || y < 0.25) {
					if (y >= 0.75) {
						p = 1.5 + fabs(rho1 * gy * (y - 0.75)) + fabs(rho0 * gy * .25);
					} else {
						p = 1.5 + fabs(rho1 * gy * (y - 0.25)) + fabs(rho0 * gy * .25);
					}
					double eint = p / (get_options().gamma - 1) / rho1;
					sph_particles_resize(sph_particles_size() + 1);
					sph_particles_smooth_len(i) = h;
					sph_particles_pos(XDIM, i) = x;
					sph_particles_pos(YDIM, i) = y;
					sph_particles_pos(ZDIM, i) = z;
					sph_particles_vel(XDIM, i) = 0;
					;
					sph_particles_vel(YDIM, i) = vy;
					;
					sph_particles_vel(ZDIM, i) = 0;
					;
					sph_particles_rung(i) = 0;
					sph_particles_eint(i) = eint;
					i++;
				}
				x = (ix + 0.0) * dx;
				y = (iy + 0.0) * dx;
				z = (iz + 0.0) * dx;
				vy = v0 / 4.0 * (1.0 + cos(8.0 * M_PI * (x - 0.5)) * (1.0 + cos(6.0 * M_PI * (x - 0.5))));
				if (y > 0.75 || y < 0.25) {
					if (y >= 0.75) {
						p = 1.5 + fabs(rho1 * gy * (y - 0.75)) + fabs(rho0 * gy * .25);
					} else {
						p = 1.5 + fabs(rho1 * gy * (y - 0.25)) + fabs(rho0 * gy * .25);
					}
					double eint = p / (get_options().gamma - 1) / rho1;
					sph_particles_resize(sph_particles_size() + 1);
					sph_particles_smooth_len(i) = h;
					sph_particles_pos(XDIM, i) = x;
					sph_particles_pos(YDIM, i) = y;
					sph_particles_pos(ZDIM, i) = z;
					sph_particles_vel(XDIM, i) = 0;
					;
					sph_particles_vel(YDIM, i) = vy;
					;
					sph_particles_vel(ZDIM, i) = 0;
					;
					sph_particles_rung(i) = 0;
					sph_particles_eint(i) = eint;
					i++;
				}
			}
		}
	}
	/*	for (int ix0 = 0; ix0 < 2 * dim; ix0++) {
	 int ix = ix0;
	 for (int iy = dim; iy < 2 * dim; iy++) {
	 for (int iz = 0; iz < 2 * dim; iz++) {
	 double x = (ix + 0.25) * dx;
	 double y = (iy + 0.25) * dx;
	 double z = (iz + 0.25) * dx;
	 double p = -(1.0 - y) * rho1 * gy;
	 double eint = p / rho1 / (get_options().gamma - 1);
	 double h = pow(m * get_options().neighbor_number / (4.0 * M_PI / 3.0 * rho1), 1.0 / 3.0);
	 sph_particles_resize(sph_particles_size() + 1);
	 sph_particles_smooth_len (i) = h;
	 sph_particles_pos(XDIM, i) = x;
	 sph_particles_pos(YDIM, i) = y;
	 sph_particles_pos(ZDIM, i) = z;
	 sph_particles_vel(XDIM, i) = 0.0;
	 sph_particles_vel(YDIM, i) = 0.0;
	 sph_particles_vel(ZDIM, i) = 0.0;
	 sph_particles_rung(i) = 0;
	 sph_particles_eint(i) = eint;
	 i++;
	 x = (ix + 0.75) * dx;
	 y = (iy + 0.75) * dx;
	 z = (iz + 0.75) * dx;
	 p = -rho1 * (1.0 - y) * gy;
	 eint = p / rho1 / (get_options().gamma - 1);
	 sph_particles_resize(sph_particles_size() + 1);
	 sph_particles_smooth_len(i) = h;
	 sph_particles_pos(XDIM, i) = x;
	 sph_particles_pos(YDIM, i) = y;
	 sph_particles_pos(ZDIM, i) = z;
	 sph_particles_vel(XDIM, i) = 0.0;
	 sph_particles_vel(YDIM, i) = 0.0;
	 sph_particles_vel(ZDIM, i) = 0.0;
	 sph_particles_rung(i) = 0;
	 sph_particles_eint(i) = eint;
	 i++;
	 }
	 }
	 }*/
	constexpr float t = 10.0;
	constexpr int N = 1000;
	hydro_driver(t, 256);

}

void hydro_disc_test() {
	part_int nparts_total = pow(get_options().parts_dim, 3);
	const double rinner = 0.225;
	const double router = 0.275;
	const double rho = nparts_total;
	const double p0 = 1.0e-6;
	const double m = 1.0;
	double h = pow(m * get_options().neighbor_number / (4.0 * M_PI / 3.0 * rho), 1.0 / 3.0);
	int i = 0;
	const int nz = 10;
	int dim = sqrt(nparts_total) / nz;
	double dx = 1.0 / dim;
	for (int iz = 0; iz < dim; iz++) {
		for (int k = 0; k < dim; k++) {
			for (int l = 0; l < dim; l++) {
				double r, x, y, z;
				x = dx * k;
				y = dx * l;
				z = iz * dx;
				r = sqrt(sqr(x - 0.5, y - 0.5, z - 0.5));
				if (r > rinner && r < router) {
					double eint = p0 / rho / (get_options().gamma - 1);
					sph_particles_resize(sph_particles_size() + 1);
					sph_particles_smooth_len(i) = h;
					sph_particles_pos(XDIM, i) = x;
					sph_particles_pos(YDIM, i) = y;
					sph_particles_pos(ZDIM, i) = z;
					const double v = 0.00*sqrt(1 / r);
					const double vx = -(y - 0.5) / r * v;
					const double vy = (x - 0.5) / r * v;
					const double vz = 0.0;
					double sgn;
					if (r > (rinner + router) / 2.0) {
						sgn = 1.0;
					} else {
						sgn = 1.0;
					}
					sph_particles_vel(XDIM, i) = sgn * vx;
					sph_particles_vel(YDIM, i) = sgn * vy;
					sph_particles_vel(ZDIM, i) = vz;
					sph_particles_rung(i) = 0;
					sph_particles_eint(i) = eint;
					i++;
				}
			}
		}
	}
	const double t = 1.0;
	hydro_driver(t, 128);
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
	dx = 0.25 / left_dim;
	for (int ix = 0; ix < 2 * left_dim; ix++) {
		for (int iy = 0; iy < 4 * left_dim; iy++) {
			for (int iz = 0; iz < 4 * left_dim; iz++) {
				double x = (ix) * dx;
				double y = (iy) * dx;
				double z = (iz) * dx;
				double eint = p1 / rho1 / (get_options().gamma - 1);
				double h = pow(m * get_options().neighbor_number / (4.0 * M_PI / 3.0 * rho1), 1.0 / 3.0);
				sph_particles_resize(sph_particles_size() + 1);
				sph_particles_smooth_len(i) = h;
				sph_particles_pos(XDIM, i) = x + rand1() * dx * eta;
				sph_particles_pos(YDIM, i) = y + rand1() * dx * eta;
				sph_particles_pos(ZDIM, i) = z + rand1() * dx * eta;
				sph_particles_vel(XDIM, i) = vx1;
				sph_particles_vel(YDIM, i) = vy1;
				sph_particles_vel(ZDIM, i) = vz1;
				sph_particles_rung(i) = 0;
				sph_particles_eint(i) = eint;
				i++;
				//			PRINT("%i\n", i);
				x = (ix + 0.5) * dx;
				y = (iy + 0.5) * dx;
				z = (iz + 0.5) * dx;
				sph_particles_resize(sph_particles_size() + 1);
				sph_particles_smooth_len(i) = h;
				sph_particles_pos(XDIM, i) = x + rand1() * dx * eta;
				sph_particles_pos(YDIM, i) = y + rand1() * dx * eta;
				sph_particles_pos(ZDIM, i) = z + rand1() * dx * eta;
				sph_particles_vel(XDIM, i) = vx1;
				sph_particles_vel(YDIM, i) = vy1;
				sph_particles_vel(ZDIM, i) = vz1;
				sph_particles_rung(i) = 0;
				sph_particles_eint(i) = eint;
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
				double eint = p0 / rho0 / (get_options().gamma - 1);
				double h = pow(m * get_options().neighbor_number / (4.0 * M_PI / 3.0 * rho0), 1.0 / 3.0);
				sph_particles_resize(sph_particles_size() + 1);
				sph_particles_smooth_len(i) = h;
				sph_particles_pos(XDIM, i) = x + rand1() * dx * eta;
				sph_particles_pos(YDIM, i) = y + rand1() * dx * eta;
				sph_particles_pos(ZDIM, i) = z + rand1() * dx * eta;
				sph_particles_vel(XDIM, i) = vx0;
				sph_particles_vel(YDIM, i) = vy0;
				sph_particles_vel(ZDIM, i) = vz0;
				;
				sph_particles_rung(i) = 0;
				sph_particles_eint(i) = eint;
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
				sph_particles_vel(XDIM, i) = vx0;
				sph_particles_vel(YDIM, i) = vy0;
				sph_particles_vel(ZDIM, i) = vz0;
				;
				sph_particles_rung(i) = 0;
				sph_particles_eint(i) = eint;
				i++;
			}
		}
	}
	constexpr float t = 10.0;
	constexpr int N = 1000;
	hydro_driver(t, 1);
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
	rho1 = nparts_left * m / 0.5f;
	rho0 = nparts_right * m / 0.5f;
	constexpr double eta = 1e-6;
	for (int ix = 0; ix < left_dim; ix++) {
		for (int iy = 0; iy < 2 * left_dim; iy++) {
			for (int iz = 0; iz < 2 * left_dim; iz++) {
				double x = (ix + 0.5) * dx;
				double y = (iy + 0.5) * dx;
				double z = (iz + 0.5) * dx;
				double eint = p1 / rho1 / (get_options().gamma - 1);
				double h = pow(m * get_options().neighbor_number / (4.0 * M_PI / 3.0 * rho1), 1.0 / 3.0);
				sph_particles_resize(sph_particles_size() + 1);
				sph_particles_smooth_len(i) = h;
				sph_particles_pos(XDIM, i) = x;
				sph_particles_pos(YDIM, i) = y;
				sph_particles_pos(ZDIM, i) = z;
				sph_particles_vel(XDIM, i) = vx1 + eta * (2.0 * rand1() - 1.0);
				sph_particles_vel(YDIM, i) = vy1 + eta * (2.0 * rand1() - 1.0);
				sph_particles_vel(ZDIM, i) = vz1 + eta * (2.0 * rand1() - 1.0);
				sph_particles_rung(i) = 0;
				sph_particles_eint(i) = eint;
				i++;
				x = (ix) * dx;
				y = (iy) * dx;
				z = (iz) * dx;
				sph_particles_resize(sph_particles_size() + 1);
				sph_particles_smooth_len(i) = h;
				sph_particles_pos(XDIM, i) = x;
				sph_particles_pos(YDIM, i) = y;
				sph_particles_pos(ZDIM, i) = z;
				sph_particles_vel(XDIM, i) = vx1 + eta * (2.0 * rand1() - 1.0);
				sph_particles_vel(YDIM, i) = vy1 + eta * (2.0 * rand1() - 1.0);
				sph_particles_vel(ZDIM, i) = vz1 + eta * (2.0 * rand1() - 1.0);
				sph_particles_rung(i) = 0;
				sph_particles_eint(i) = eint;
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
				double eint = p0 / rho0 / (get_options().gamma - 1);
				double h = pow(m * get_options().neighbor_number / (4.0 * M_PI / 3.0 * rho0), 1.0 / 3.0);
				sph_particles_resize(sph_particles_size() + 1);
				sph_particles_smooth_len(i) = h;
				sph_particles_pos(XDIM, i) = x;
				sph_particles_pos(YDIM, i) = y;
				sph_particles_pos(ZDIM, i) = z;
				sph_particles_vel(XDIM, i) = vx0 + eta * (2.0 * rand1() - 1.0);
				sph_particles_vel(YDIM, i) = vy0 + eta * (2.0 * rand1() - 1.0);
				sph_particles_vel(ZDIM, i) = vz0 + eta * (2.0 * rand1() - 1.0);
				sph_particles_rung(i) = 0;
				sph_particles_eint(i) = eint;
				//		PRINT("%i\n", i);
				i++;
				x = (ix) * dx;
				y = (iy) * dx;
				z = (iz) * dx;
				sph_particles_resize(sph_particles_size() + 1);
				sph_particles_smooth_len(i) = h;
				sph_particles_pos(XDIM, i) = x;
				sph_particles_pos(YDIM, i) = y;
				sph_particles_pos(ZDIM, i) = z;
				sph_particles_vel(XDIM, i) = vx0 + eta * (2.0 * rand1() - 1.0);
				sph_particles_vel(YDIM, i) = vy0 + eta * (2.0 * rand1() - 1.0);
				sph_particles_vel(ZDIM, i) = vz0 + eta * (2.0 * rand1() - 1.0);
				sph_particles_rung(i) = 0;
				sph_particles_eint(i) = eint;
				//		PRINT("%i\n", i);
				i++;
			}
		}
	}
	hydro_driver(10.00);
}

void hydro_blast_test() {
	part_int nparts_total = pow(get_options().parts_dim, 3);
	double rho0 = 1.0;
	double p1 = 10000.0;
	double p0 = 1.0;
	double sigma = 0.01;
	auto opts = get_options();
	part_int ndim = get_options().parts_dim;
	part_int nparts = std::pow(ndim, NDIM);
	ndim = pow(nparts / 2, 1. / 3.);
	nparts = 2 * ndim * ndim * ndim;
	opts.sph_mass = rho0 / nparts;
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
				double r = sqrt(sqr(x - 0.5, y - 0.5, z - 0.5));
				double p = p0 + p1 * exp(-sqr(r) / sqr(sigma));
				double eint = p / rho0 / (get_options().gamma - 1);
				double vx = 0.0;
				double vy = 0.0;
				double vz = 0.0;
				double h = pow(m * get_options().neighbor_number / (4.0 * M_PI / 3.0 * rho0), 1.0 / 3.0);
				sph_particles_resize(sph_particles_size() + 1);
				sph_particles_smooth_len(i) = h;
				sph_particles_pos(XDIM, i) = x;
				sph_particles_pos(YDIM, i) = y;
				sph_particles_pos(ZDIM, i) = z;
				sph_particles_vel(XDIM, i) = vx;
				sph_particles_vel(YDIM, i) = vy;
				sph_particles_vel(ZDIM, i) = vz;
				sph_particles_rung(i) = 0;
				sph_particles_eint(i) = eint;
				i++;
				x = (ix + 0.0) * dx;
				y = (iy + 0.0) * dx;
				z = (iz + 0.0) * dx;
				r = sqrt(sqr(x - 0.5, y - 0.5, z - 0.5));
				p = p0 + p1 * exp(-sqr(r) / sqr(sigma));
				eint = p / rho0 / (get_options().gamma - 1);
				vx = 0.0;
				vy = 0.0;
				vz = 0.0;
				sph_particles_resize(sph_particles_size() + 1);
				sph_particles_smooth_len(i) = h;
				sph_particles_pos(XDIM, i) = x;
				sph_particles_pos(YDIM, i) = y;
				sph_particles_pos(ZDIM, i) = z;
				sph_particles_vel(XDIM, i) = vx;
				sph_particles_vel(YDIM, i) = vy;
				sph_particles_vel(ZDIM, i) = vz;
				sph_particles_rung(i) = 0;
				sph_particles_eint(i) = eint;
				i++;
			}
		}
	}
	hydro_driver(.25, 1024);
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
				double eint = p0 / rho / (get_options().gamma - 1);
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
				sph_particles_eint(i) = eint;
				i++;
				//			PRINT("%i\n", i);
			}
		}
	}
	hydro_driver(0.5);
}
