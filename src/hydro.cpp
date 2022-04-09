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
#include <cosmictiger/domain.hpp>
#include <cosmictiger/sphere.hpp>
#include <cosmictiger/sedov.hpp>

static inline double pow_n(double y, double n) {
	return std::pow(y, n);
}

static inline double fy(double y, double z, double r) {
	return z;
}

static inline double fz(double y, double z, double r, double n) {
	if (r != 0.0) {
		return -(pow_n(y, n) + 2.0 * z / r);
	}
	return -3.0;
}

static inline double fm(double theta, double dummy, double r, double n) {
	constexpr static double four_pi = double(4) * M_PI;
	return four_pi * pow_n(theta, n) * r * r;
}

double lane_emden(double r0, double dr, double n, double& menc) {
	double dy1, dz1, y, z, r, dy2, dz2, dy3, dz3, dy4, dz4, y0, z0;
	double dm1, m, dm2, dm3, dm4, m0;
	int done = 0;
	y = 1.0;
	z = 0.0;
	m = 0.0;
	int N = static_cast<int>(r0 / dr + 0.5);
	if (N < 1) {
		N = 1;
	}
	r = 0.0;
	do {
		if (r + dr > r0) {
			dr = r0 - r;
			done = 1;
		}
		y0 = y;
		z0 = z;
		m0 = m;
		dy1 = fy(y, z, r) * dr;
		dz1 = fz(y, z, r, n) * dr;
		dm1 = fm(y, z, r, n) * dr;
		y += 0.5 * dy1;
		z += 0.5 * dz1;
		m += 0.5 * dm1;
		if (y <= 0.0) {
			y = 0.0;
			break;
		}
		double rdr2 = r + 0.5 * dr;
		dy2 = fy(y, z, rdr2) * dr;
		dz2 = fz(y, z, rdr2, n) * dr;
		dm2 = fm(y, z, rdr2, n) * dr;
		y = y0 + 0.5 * dy2;
		z = z0 + 0.5 * dz2;
		m = m0 + 0.5 * dm2;
		if (y <= 0.0) {
			y = 0.0;
			break;
		}
		dy3 = fy(y, z, rdr2) * dr;
		dz3 = fz(y, z, rdr2, n) * dr;
		dm3 = fm(y, z, rdr2, n) * dr;
		y = y0 + dy3;
		z = z0 + dz3;
		m = m0 + dm3;
		if (y <= 0.0) {
			y = 0.0;
			break;
		}
		double rdr = r + dr;
		dy4 = fy(y, z, rdr) * dr;
		dz4 = fz(y, z, rdr, n) * dr;
		dm4 = fm(y, z, rdr, n) * dr;
		y = y0 + (dy1 + dy4 + 2.0 * (dy3 + dy2)) / 6.0;
		z = z0 + (dz1 + dz4 + 2.0 * (dz3 + dz2)) / 6.0;
		m = m0 + (dm1 + dm4 + 2.0 * (dm3 + dm2)) / 6.0;
		if (y <= 0.0) {
			y = 0.0;
			break;
		}
		r += dr;
	} while (done == 0);
	menc = m;
	if (y < 0.0) {
		return 0.0;
	}
	return pow(y, n);
}

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
	domains_rebound();
	do {
		int minrung = min_rung(itime);
		double dummy;
		if (minrung == 0) {
			view_output_views(main_step, 1.0);
//			output_line(main_step);
			main_step++;
		}
		kick_return kr;
		if (get_options().gravity) {
			auto tmp = kick_step(minrung, 1.0, 0.0, t0, 0.5, t0 == 0.0, minrung == 0);
			kr = tmp.first;
		}
		if (minrung == 0) {
			double ekin = 0.0;
			double eint = 0.0;
			double rho_max = 0.0;
			double epot = kr.pot / 2.0;
			double xmom = 0.0, ymom = 0.0, zmom = 0.0;
			for (part_int i = 0; i < particles_size(); i++) {
				const double vx = particles_vel(XDIM, i);
				const double vy = particles_vel(YDIM, i);
				const double vz = particles_vel(ZDIM, i);
				xmom += m * vx;
				ymom += m * vy;
				zmom += m * vz;
				ekin += 0.5 * m * sqr(vx, vy, vz);
				if (particles_type(i) == SPH_TYPE) {
					rho_max = std::max(rho_max, (double) sph_particles_rho(particles_cat_index(i)));
					const double e = sph_particles_eint(particles_cat_index(i));
					eint += m * e;
				}
			}
			FILE* fp = fopen("energy.dat", "at");
			const double etot = ekin + eint + epot;
			double etot0;
			if (t == 0.0) {
				etot0 = etot;
			}
			fprintf(fp, "%e %e %e %e %e %e %e %e %e\n", t, xmom, ymom, zmom, ekin, eint, etot, epot, (etot - etot0) / etot);
			fclose(fp);
			fp = fopen("rho.dat", "at");
			fprintf(fp, "%e %e\n", t, rho_max);
			fclose(fp);
		}
		sph_run_return rc2 = sph_step(minrung, 1.0, t, t0, 1, 0.0, 0, 0, 0.0, &dummy, true);
		if (!get_options().gravity) {
			kr.max_rung = rc2.max_rung;
		}
		int maxrung = std::max((int) rc2.max_rung, (int) kr.max_rung);
		double dt = t0 / (1 << maxrung);
		auto dr = drift(1.0, dt, t, t + dt, tmax);
		itime = inc(itime, maxrung);
		if (t == 0.0) {
			e0 = rc2.ekin + rc2.etherm;
			ent0 = rc2.ent;
		}
		t += dt;
		PRINT("%i %e %e %i %i\n", step, t, dt, minrung, maxrung);
		step++;
	} while (t < tmax);
	view_output_views(main_step, 1.0);
}

void hydro_plummer() {
	part_int N = pow(get_options().parts_dim, 3);
	const double a = 0.01;
	auto opts = get_options();
	opts.sph_mass = 1. / N;
	opts.dm_mass = 1. / N;
	const double m = opts.sph_mass;
	set_options(opts);
	const double G = opts.GM;
	const double M0 = N * m;
	const double rho0 = 3.0 * M0 / (4.0 * M_PI) / (a * a * a);
	PRINT("Central density = %e\n", rho0);
	double pot = 0.0;
	double ekin = 0.0;
	double maxr = 5.0 * a;
	while (particles_size() < 0) {
		double x = maxr * (2.0 * rand1() - 1.0);
		double y = maxr * (2.0 * rand1() - 1.0);
		double z = maxr * (2.0 * rand1() - 1.0);
		double r = sqrt(sqr(x, y, z));
		double dist = powf(1.0 + sqr(r / a), -2.5);
		if (rand1() < dist) {
			double h = pow(m * get_options().neighbor_number / (4.0 * M_PI / 3.0 * rho0 * dist), 1.0 / 3.0);
			part_int k = particles_size();
			particles_resize(k + 1);
			double phi = -G * M0 / sqrt(r * r + a * a);
			double v = sqrt(-phi / 2.0);
			particles_pos(XDIM, k) = x - 0.5;
			particles_pos(YDIM, k) = y - 0.5;
			particles_pos(ZDIM, k) = z - 0.5;
			x = rand1();
			y = rand1();
			v *= sqrt(-2.0 * log(x)) * cos(2.0 * M_PI * y);
			x = rand1() - 0.5;
			y = rand1() - 0.5;
			z = rand1() - 0.5;
			double norminv = 1.0 / sqrt(sqr(x, y, z));
			x *= norminv;
			y *= norminv;
			z *= norminv;
			particles_vel(XDIM, k) = x * v;
			particles_vel(YDIM, k) = y * v;
			particles_vel(ZDIM, k) = z * v;
			particles_rung(k) = 0;
		}
	}
	sph_particles_resize(N);
	int k = 0;
	while (k < N) {
		double x = maxr * (2.0 * rand1() - 1.0);
		double y = maxr * (2.0 * rand1() - 1.0);
		double z = maxr * (2.0 * rand1() - 1.0);
		double r = sqrt(sqr(x, y, z));
		double dist = powf(1.0 + sqr(r / a), -2.5);
		if (rand1() < dist) {
			double h = pow(m * get_options().neighbor_number / (4.0 * M_PI / 3.0 * rho0 * dist), 1.0 / 3.0);
//			part_int k = sph_particles_size();
//			sph_particles_resize(k + 1);
			double phi = -G * M0 / sqrt(r * r + a * a);
			double v = sqrt(-phi / 2.0);
			sph_particles_pos(XDIM, k) = x - 0.5;
			sph_particles_pos(YDIM, k) = y - 0.5;
			sph_particles_pos(ZDIM, k) = z - 0.5;
			x = rand1();
			y = rand1();
			v *= sqrt(-2.0 * log(x)) * cos(2.0 * M_PI * y);
			x = rand1() - 0.5;
			y = rand1() - 0.5;
			z = rand1() - 0.5;
			double norminv = 1.0 / sqrt(sqr(x, y, z));
			x *= norminv;
			y *= norminv;
			z *= norminv;
			sph_particles_vel(XDIM, k) = x * v;
			sph_particles_vel(YDIM, k) = y * v;
			sph_particles_vel(ZDIM, k) = z * v;
			const float eint = 1e-3 * (v * v * 0.5);
			sph_particles_rec2(k).A = eint * (get_options().gamma - 1.0) / pow(rho0, get_options().gamma - 1.0);
			sph_particles_rung(k) = 0;
			sph_particles_smooth_len(k) = h;
			ekin += 0.5 * m * sqr(v);
			pot += 0.5 * m * phi;
			k++;
		}
	}
	PRINT("Virial error is %e\n", (2.0 * ekin + pot) / (2.0 * ekin - pot));
	const double tdyn = sqrt(4.0 * M_PI * a * a * a / (3.0 * G * M0));
	hydro_driver(100.0 * tdyn, 1024);
}

void hydro_star_test() {
	part_int nparts_total = pow(get_options().parts_dim, 3);
	const double r0 = 20.0;
	const int N = nparts_total;
	auto opts = get_options();
	PRINT("Making star\n");
	opts.sph_mass = 203. / N / (r0 * r0 * r0) / 5.99071;
	const double m = opts.sph_mass;
	set_options(opts);
	double rho0 = 1.0;
	const double npoly = 1.5;
	const auto rho = [rho0,npoly]( double r ) {
		if( r == 0.0 ) {
			return rho0;
		} else {
			double menc;
			return rho0 * lane_emden(r, r / 100.0, npoly, menc);
		}
	};
	double K = 4.0 * M_PI * opts.GM / (npoly + 1.0) * powf(rho0, 1.0 - 1.0 / npoly) / sqr(r0);
	double d;
	double r = 0.0;
	int Ntot = 0;
	int Nr = 0;
	double rmax = 3.65375 / r0;
	vector<array<vector<float>, NDIM>> X;
	vector<float> radius;
	do {
		d = rho(r * r0);
		double dr1 = pow(m / d, 1.0 / 3.0);
		double dr2;
		r += 0.5 * dr1;
		if (r < rmax) {
			double diff;
			d = rho(r * r0);
			dr2 = pow(m / d, 1.0 / 3.0);
			r += 0.5 * (dr2 - dr1);
			if (r < rmax) {
				dr1 = dr2;
				const int N = 4.0 * M_PI * r * r * dr2 * (d / m) + 0.5;
				array<vector<float>, NDIM> x;
				for (int dim = 0; dim < NDIM; dim++) {
					x[dim].resize(N);
				}
				X.push_back(std::move(x));
				printf("%i %e %i %e\n", Nr, r, N, dr1);
				Ntot += N;
				radius.push_back(r);
				r += 0.5 * dr1;
				Nr++;
			}
		}
	} while (r < rmax);
	printf("Ntot = %i\n", Ntot);
	solve_sphere_surface_problem(X);
	for (int i = 0; i < X.size(); i++) {
		const double d = rho(radius[i] * r0);
		double h = pow(m * get_options().neighbor_number / (4.0 * M_PI / 3.0 * d), 1.0 / 3.0);
		const int N = X[i][XDIM].size();
		for (int l = 0; l < N; l++) {
			const part_int k = sph_particles_size();
			sph_particles_resize(k + 1);
			sph_particles_pos(XDIM, k) = X[i][XDIM][l] * radius[i] + 0.5;
			sph_particles_pos(YDIM, k) = X[i][YDIM][l] * radius[i] + 0.5;
			sph_particles_pos(ZDIM, k) = X[i][ZDIM][l] * radius[i] + 0.5;
			sph_particles_smooth_len(k) = h;
			const double P = K * pow(d, 1.0 + 1.0 / npoly);
			const double E = P * npoly;
			const double eint = E / d;
			sph_particles_rec2(k).A = eint * (get_options().gamma - 1.) / pow(d, get_options().gamma - 1.);
			sph_particles_vel(XDIM, k) = 0;
			sph_particles_vel(YDIM, k) = 0;
			sph_particles_vel(ZDIM, k) = 0;
			sph_particles_rung(k) = 0;
		}
	}
	const double tdyn = sqrt(1.0 / (opts.GM * rho0));
	PRINT("************************************\n");
	PRINT("tdyn = %e\n", 1.0 / sqrt(opts.GM * rho0));
	PRINT("************************************\n");
	opts = get_options();
	opts.damping = 1.0;
	opts.alpha0 = 0.0;
	opts.alpha1 = 0.0;
	set_options(opts);
	hydro_driver(10.0 * tdyn, 25);
	opts.damping = 0.0;
	opts.alpha0 = 0.05;
	opts.alpha1 = 1.5;
	set_options(opts);
	hydro_driver(100.0 * tdyn, 250);
	const int Nsample = 1000;
	double l2 = 0.0;
	double norm = 0.0;
	for (int i = 0; i < Nsample; i++) {
		const part_int k = rand() % sph_particles_size();
		const float d0 = sph_particles_rho(k);
		const float x = sph_particles_pos(XDIM, k).to_float();
		const float y = sph_particles_pos(YDIM, k).to_float();
		const float z = sph_particles_pos(ZDIM, k).to_float();
		const float r = sqrt(sqr(x - 0.5, y - 0.5, z - 0.5));
		float d1 = 0.0;
		if (r < rmax) {
			d1 = rho(r * r0);
		}
		l2 += sqr(d1 - d0);
		norm += d1 * d0;
	}
	l2 /= norm;
	l2 = sqrt(l2);
	PRINT("L2 Error = %e\n", l2);
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
				sph_particles_rec2(i).A = eint * (get_options().gamma - 1.) / pow(rho0, get_options().gamma - 1.);
				;
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
					sph_particles_rec2(i).A = eint * (get_options().gamma - 1.) / pow(rho0, get_options().gamma - 1.);
					;
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
					sph_particles_rec2(i).A = eint * (get_options().gamma - 1.) / pow(rho1, get_options().gamma - 1.);
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
	const double rinner = 0.2;
	const double router = 0.3;
	const double rho = nparts_total;
	const double p0 = 1.0e-30;
	const double m = 1.0;
	double h = pow(m * get_options().neighbor_number / (4.0 * M_PI / 3.0 * rho), 1.0 / 3.0);
	int i = 0;
	const int nz = 3;
	int dim = sqrt(nparts_total) / nz;
	double dx = 1.0 / dim;
	for (int iz = dim / 2 - nz; iz < dim / 2 + nz; iz++) {
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
					const double v = 1.00 * sqrt(1 / r);
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
					sph_particles_rec2(i).A = eint * (get_options().gamma - 1.) / pow(rho, get_options().gamma - 1.);
					;
					i++;
				}
			}
		}
	}
	const double t = 16.0;
	hydro_driver(t, 1024);
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
	double p0 = .1;
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
				double ent = p1 / powf(rho1, get_options().gamma);
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
				sph_particles_rec2(i).A = ent;
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
				sph_particles_rung(i) = 0;
				sph_particles_rec2(i).A = ent;
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
				double ent = p0 / powf(rho0, get_options().gamma);
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
				sph_particles_rec2(i).A = ent;
				;
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
				sph_particles_rec2(i).A = ent;
				;
				i++;
			}
		}
	}
	constexpr float t = .09;
	constexpr int N = 1000;
	hydro_driver(t, 10);
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
				sph_particles_rec2(i).A = eint * (get_options().gamma - 1.) / pow(rho1, get_options().gamma - 1.);
				;

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
				sph_particles_rec2(i).A = eint * (get_options().gamma - 1.) / pow(rho1, get_options().gamma - 1.);
				;

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
				sph_particles_rec2(i).A = eint * (get_options().gamma - 1.) / pow(rho0, get_options().gamma - 1.);
				;

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
				sph_particles_rec2(i).A = eint * (get_options().gamma - 1.) / pow(rho0, get_options().gamma - 1.);
				;
				//		PRINT("%i\n", i);
				i++;
			}
		}
	}
	hydro_driver(10.00, 256);
}

void hydro_blast_test() {
	part_int nparts_total = pow(get_options().parts_dim, 3);
	double rho0 = 1.0;
	double eblast = 1.0;
	double p0 = 1.0e-20f;
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
	double sigma = .015;
	for (int ix = 0; ix < ndim; ix++) {
		for (int iy = 0; iy < ndim; iy++) {
			for (int iz = 0; iz < ndim; iz++) {
				double x = (ix + 0.5) * dx;
				double y = (iy + 0.5) * dx;
				double z = (iz + 0.5) * dx;
				double r = sqrt(sqr(x - 0.5, y - 0.5, z - 0.5));
				double eint = std::max(1.0 / sigma / sigma / sigma / sqrt(8.0 * M_PI * M_PI * M_PI) * exp(-0.5 * sqr(r / sigma)), 1e-20);
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
				sph_particles_rec2(i).A = eint * (get_options().gamma - 1.) / pow(rho0, get_options().gamma - 1.);
				;
				i++;
				x = (ix + 0.0) * dx;
				y = (iy + 0.0) * dx;
				z = (iz + 0.0) * dx;
				r = sqrt(sqr(x - 0.5, y - 0.5, z - 0.5));
				eint = std::max(1.0 / sigma / sigma / sigma / sqrt(8.0 * M_PI * M_PI * M_PI) * exp(-0.5 * sqr(r / sigma)), 1e-20);
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
				sph_particles_rec2(i).A = eint * (get_options().gamma - 1.) / pow(rho0, get_options().gamma - 1.);
				;
				i++;
			}
		}
	}
	const double t = 5e-3;
	hydro_driver(t, 16);
	FILE* fp = fopen("blast.txt", "wt");
	double l1 = 0.0, l2 = 0.0, lmax = 0.0;
	double norm1 = 0.0, norm2 = 0.0;
	constexpr int N = 100000;
	for (int l = 0; l < N; l++) {
		const int i = rand() % sph_particles_size();
		const int j = sph_particles_dm_index(i);
		float x = particles_pos(XDIM, j).to_float() - 0.5;
		float y = particles_pos(YDIM, j).to_float() - 0.5;
		float z = particles_pos(ZDIM, j).to_float() - 0.5;
		const float h = sph_particles_smooth_len(i);
		const float rho = sph_den(1 / (h * h * h));
		double time = t;
		double r = sqrt(sqr(x, y, z));
		double rmax = sqrt(1.5);
		double d, v, p;
		int ndim = 3;
		sedov::solution(time, r, rmax, d, v, p, ndim);
		double dif = fabs(rho - d);
		norm1 += d;
		norm2 += sqr(d);
		l1 += dif;
		l2 += dif * dif;
		lmax = std::max(lmax, dif);
		fprintf(fp, "%e %e %e\n", r, rho, d);
	}
	l1 /= norm1;
	l2 /= norm2;
	l2 = sqrt(l2);
	PRINT("L1 = %e L2 = %e Lmax = %e\n", l1, l2, lmax);
	fclose(fp);

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
				sph_particles_rec2(i).A = eint * (get_options().gamma - 1.) / pow(rho, get_options().gamma - 1.);
				i++;
				//			PRINT("%i\n", i);
			}
		}
	}
	hydro_driver(0.5);
}
