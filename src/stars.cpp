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

#include <cosmictiger/stars.hpp>
#include <cosmictiger/sph_particles.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/constants.hpp>

#include <gsl/gsl_rng.h>

vector<star_particle> stars;

float stars_sample_mass(gsl_rng*);
float stars_luminosity(float mass);
float stars_lifetime(float mass);
float stars_remnant_mass(float mass);
float stars_helium_produced(float mass);

HPX_PLAIN_ACTION (stars_find);
HPX_PLAIN_ACTION (stars_remove);

void stars_save(FILE* fp) {
	size_t size = stars.size();
	fwrite(&size, sizeof(size_t), 1, fp);
	fwrite(stars.data(), sizeof(star_particle), stars.size(), fp);
}

star_particle& stars_get(part_int index) {
	return stars[index];
}

part_int stars_size() {
	return stars.size();
}

void stars_load(FILE* fp) {
	size_t size;
	FREAD(&size, sizeof(size_t), 1, fp);
	stars.resize(size);
	FREAD(stars.data(), sizeof(star_particle), size, fp);
}

double stars_find(float a, float dt, int minrung, int step, float t0) {

	profiler_enter("FUNCTION");

	PRINT("------------------------------------>>> Searching for STARS <<<--------------------------------------------------------\n");
	vector<hpx::future<double>> futs;
	vector<hpx::future<double>> futs2;
	for (auto& c : hpx_children()) {
		futs.push_back(hpx::async<stars_find_action>(c, a, dt, minrung, step, t0));
	}
	mutex_type mutex;
	const int nthreads = hpx_hardware_concurrency();
	vector<gsl_rng *> rnd_gens(nthreads);
	for (int i = 0; i < nthreads; i++) {
		rnd_gens[i] = gsl_rng_alloc(gsl_rng_taus);
		gsl_rng_set(rnd_gens[i], step * nthreads + i);
	}
	double eloss = 0.0;

	static const double sph_mass = get_options().sph_mass;
	static const double code_to_g = get_options().code_to_g;
	static const double code_to_cm = get_options().code_to_cm;
	static const double code_to_s = get_options().code_to_s;
	static const double G = get_options().GM;
	vector<part_int> indices;
	for (int proc = 0; proc < nthreads; proc++) {
		futs2.push_back(hpx::async([proc, t0, nthreads, a, minrung, &mutex,&indices,dt,&rnd_gens]() {
			double eloss = 0.0;
			const double dm_soft = get_options().hsoft;
			const double code_to_s = get_options().code_to_s;
			const double code_to_g = get_options().code_to_g;
			const double code_to_cm = get_options().code_to_cm;
			const double rho0 = 3.0 / (8.0*M_PI*constants::G) * pow(get_options().hubble*constants::H0,2);
			const double rho0_b = rho0 * get_options().omega_b;
			const double rho0_c = rho0 * get_options().omega_c;
			const double code_to_density = code_to_g / pow(code_to_cm,3);
			const double code_to_energy = pow(code_to_cm/code_to_s,2);
			const part_int b = (size_t) proc * sph_particles_size() / nthreads;
			const part_int e = (size_t) (proc+1) * sph_particles_size() / nthreads;
			for( part_int i = b; i < e; i++) {
				char rung = sph_particles_rung(i);
				const part_int kk = sph_particles_dm_index(i);
				const double rho_tot = (particles_rhob(kk) + particles_rhoc(kk))* code_to_density;
#ifdef HOPKINS
				const double rho_b = particles_rhob(kk) * code_to_density;
				const double eint = sph_particles_eint_rho(i) * code_to_energy;
#else
				const double rho_b = sph_particles_rho(i) * code_to_density;
				const double eint = sph_particles_eint(i) * code_to_energy;
#endif
				const double rho_c = rho_tot - rho_b;
				const double tcool = sph_particles_tcool(i);
				bool make_star = false;
				if( rho_b > 0.0f ) {

					const double tdyn = sqrt((3.0*M_PI*a*a*a)/(32.0*constants::G*rho_b));
					const double pre = (get_options().gamma-1.0) * rho_b * eint;
					const double cs = sqrt(pre/rho_b*get_options().gamma);
					const double delta_b = (rho_b-rho0_b);
					const double delta0 = (rho_tot - rho0);
					const double dt = t0 * 0.5 * code_to_s;
					constexpr double eta = 20.0;
					if( rho_b > eta * rho0_b) {
						if(delta_b / delta0 < 0.0 ) {
							PRINT( "%e\n", delta_b / delta0);
							ALWAYS_ASSERT(false);
						}
						if( delta_b / delta0 > 0.0 ) {
							const double a = pow(constants::G,-1.5);
							const double b = pow(rho_b,-0.5);
							const double c = pow(cs,3);
							const double d = pow(delta_b/delta0,1.5);
							const double mj = a * b * c * d;
							const double m0 = get_options().sph_mass * code_to_g;
							if( tcool < tdyn ) {
								if( m0 > mj ) {
									if( sph_particles_divv(i) < 0.0 ) {
										const double eps = dt / tdyn;
										const double p = 1.0 - exp(-eps);
										make_star = ( gsl_rng_uniform(rnd_gens[proc]) < p );
									}
								}
							}
						}
					}
				}
				if( make_star ) {
					star_particle star;
					const part_int kk = sph_particles_dm_index(i);
					star.dm_index = kk;
					star.zform = 1.0 - 1.0 / a;
					const float dt = 0.5 * rung_dt[particles_rung(kk)] * t0;
					sph_particles_rho_rho(i) = -1.0;
					//	particles_vel(XDIM,kk) += sph_particles_gforce(XDIM,i) * dt;
					//	particles_vel(YDIM,kk) += sph_particles_gforce(YDIM,i) * dt;
					//	particles_vel(ZDIM,kk) += sph_particles_gforce(ZDIM,i) * dt;
				particles_type(kk) = STAR_TYPE;
				eloss += sph_particles_eint(i) * sph_mass / (a*a);
				std::lock_guard<mutex_type> lock(mutex);
				indices.push_back(i);
				particles_cat_index(kk) = stars.size();
				stars.push_back(star);
			}
		}
		return eloss;
	}));
	}
	for (auto& f : futs2) {
		eloss += f.get();
	}

	for (auto& i : indices) {
		while (sph_particles_rho_rho(sph_particles_size() - 1) < 0.f && sph_particles_size()) {
			sph_particles_resize(sph_particles_size() - 1, false);
		}
		if (i < sph_particles_size()) {
			const int k = sph_particles_size() - 1;
			if (i != k) {
				const int dmk = sph_particles_dm_index(k);
				sph_particles_swap2(i, k);
				particles_cat_index(dmk) = i;
			}
			sph_particles_resize(k, false);
		}
	}
	for (auto& f : futs) {
		eloss += f.get();
	}
	for (int i = 0; i < nthreads; i++) {
		gsl_rng_free(rnd_gens[i]);
	}
	profiler_exit();
	return eloss;
}

float stars_remnant_mass(float Mi, float Z);

HPX_PLAIN_ACTION (stars_statistics);

stars_stats stars_statistics(float a) {
	profiler_enter(__FUNCTION__);
	vector<hpx::future<stars_stats>> futs;
	for (auto& c : hpx_children()) {
		futs.push_back(hpx::async<stars_statistics_action>(c, a));
	}
	stars_stats stats;
	stats.stars = stars.size();
	if (hpx_rank() == 0) {
		FILE* fp = fopen("stars.txt", "at");
		fprintf(fp, "%e %li \n", 1.f / a - 1.f, stats.stars);
		fclose(fp);
	}
	profiler_exit();
	return stats;
}

#define WIND_RATIO 0.5

void stars_remove(float a, float dt, int minrung, int step) {
//	profiler_enter(__FUNCTION__);

	return;
	/*
	 vector<hpx::future<void>> futs;
	 vector<hpx::future<void>> futs2;
	 for (auto& c : hpx_children()) {
	 futs.push_back(hpx::async<stars_find_action>(c, a, dt, minrung, step));
	 }
	 const int nthreads = hpx_hardware_concurrency();
	 vector<gsl_rng *> rnd_gens(nthreads);
	 for (int i = 0; i < nthreads; i++) {
	 rnd_gens[i] = gsl_rng_alloc(gsl_rng_taus);
	 gsl_rng_set(rnd_gens[i], step * nthreads + i);
	 }
	 static const double sph_mass = get_options().sph_mass;
	 static const double code_to_g = get_options().code_to_g;
	 static const double code_to_cm = get_options().code_to_cm;
	 static const double code_to_s = get_options().code_to_s;
	 vector<part_int> to_gas_indices;
	 mutex_type to_gas_mutex;
	 std::atomic<int> remnants(0);
	 std::atomic<int> clouds(0);
	 std::atomic<int> nstars(0);
	 for (int proc = 0; proc < nthreads; proc++) {
	 futs2.push_back(hpx::async([&rnd_gens,proc, nthreads, a, dt, &to_gas_mutex, &to_gas_indices, minrung, &remnants, &clouds,&nstars]() {
	 const part_int b = (size_t) proc * stars.size() / nthreads;
	 const part_int e = (size_t) (proc+1) * stars.size() / nthreads;
	 for( part_int i = b; i < e; i++) {
	 if( stars[i].type == STAR_TYPE ) {
	 float real_dt = a * dt * code_to_s * constants::seconds_to_years;
	 stars[i].time_remaining -= real_dt;
	 float remnant_mass_ratio = stars_remnant_mass(stars[i].stellar_mass, stars[i].Z);
	 float wind_rate = WIND_RATIO * (1.0 - remnant_mass_ratio);
	 float dt_star = stars[i].total_life;
	 dt_star /= constants::seconds_to_years;
	 dt_star /= code_to_s;
	 dt_star /= a;
	 float pwind = 1.0 - exp(-wind_rate*dt/dt_star);
	 if( gsl_rng_uniform_pos(rnd_gens[proc]) < pwind) {
	 std::lock_guard<mutex_type> lock(to_gas_mutex);
	 //		PRINT( "Stellar wind!\n");
	 to_gas_indices.push_back(i);
	 stars[i].remove = true;
	 } else {
	 if( stars[i].time_remaining < 0.0 && particles_rung(stars[i].dm_index) >= minrung ) {
	 if( gsl_rng_uniform_pos(rnd_gens[proc]) > remnant_mass_ratio ) {
	 std::lock_guard<mutex_type> lock(to_gas_mutex);
	 to_gas_indices.push_back(i);
	 stars[i].remove = true;
	 } else {
	 stars[i].type = REMNANT_TYPE;
	 stars[i].zform = 1.0f / a - 1.f;
	 stars[i].stellar_mass *= remnant_mass_ratio;
	 }
	 }
	 }
	 }
	 if( stars[i].type == REMNANT_TYPE ) {
	 remnants++;
	 } else if( stars[i].type == CLOUD_TYPE) {
	 clouds++;
	 } else if( stars[i].type == STAR_TYPE) {
	 nstars++;
	 }
	 }
	 }));
	 }
	 hpx::wait_all(futs2.begin(), futs2.end());
	 static const auto h0 = 1.0 / get_options().parts_dim;
	 PRINT("Restoring gas\n");
	 for (auto& i : to_gas_indices) {
	 star_particle star = stars[i];
	 const int j = star.dm_index;
	 const int k = sph_particles_size();
	 sph_particles_resize(k + 1, false);
	 sph_particles_dm_index(k) = j;
	 particles_type(j) = SPH_TYPE;
	 particles_cat_index(j) = k;
	 const double T = 10000.0;
	 const double N = sph_mass * code_to_g * ((1. - star.Y) * 2.f + star.Y * .25f * 3.f + 0.5f * star.Z) * constants::avo;
	 const double Cv = 1.5 * constants::kb;
	 double E = Cv * N * T;
	 constexpr float fSn = 6e-4;
	 //		const double fSN = 0.0e-5;
	 E /= sqr(code_to_cm) * code_to_g / sqr(code_to_s);
	 float wind_energy = 0.0;
	 if (star.stellar_mass > 7.5) {
	 //		PRINT("Supernova!\n");
	 //		E += 0.5 * fSn * sph_mass / star.stellar_mass * a * a;
	 //	wind_energy += 0.5 * fSn * sph_mass / star.stellar_mass * a * a;
	 double dZ = 0.02;
	 double dHe = 0.20 * (1.0 - star.Y - star.Z);
	 star.Y += dHe;
	 star.Z += dZ;
	 } else if (star.time_remaining > 0.0) {
	 //		wind_energy = 0.5 * sqr(1e-3) * a * a;
	 }
	 if (wind_energy > 0.0) {
	 const int n = sph_particles_dm_index(k);
	 auto& vx = particles_vel(XDIM, n);
	 auto& vy = particles_vel(YDIM, n);
	 auto& vz = particles_vel(ZDIM, n);
	 float nx = gsl_rng_uniform_pos(rnd_gens[0]);
	 float ny = gsl_rng_uniform_pos(rnd_gens[0]);
	 float nz = gsl_rng_uniform_pos(rnd_gens[0]);
	 float ninv = 1.0 / sqrt(sqr(nx, ny, nz));
	 nx *= ninv;
	 ny *= ninv;
	 nz *= ninv;
	 float vdotn = vx * nx + vy * ny + vz * nz;
	 float dv = sqrt(sqr(vdotn) + 2.0 * wind_energy / sph_mass) - vdotn;
	 float dvx = nx * dv;
	 float dvy = ny * dv;
	 float dvz = nz * dv;
	 particles_vel(XDIM, n) += dvx;
	 particles_vel(YDIM, n) += dvy;
	 particles_vel(ZDIM, n) += dvz;
	 }
	 sph_particles_H2(k) = 0.f;
	 sph_particles_Hp(k) = 1.f - star.Y - star.Z;
	 sph_particles_He0(k) = 0.f;
	 sph_particles_Hep(k) = 0.f;
	 sph_particles_Hn(k) = 0.f;
	 sph_particles_Hepp(k) = star.Y;
	 sph_particles_Z(k) = star.Z;
	 sph_particles_dent_con(k) = 0.f;
	 sph_particles_dent_pred(k) = 0.f;
	 sph_particles_smooth_len(k) = h0;
	 sph_particles_tdyn(k) = 1e38;
	 E *= a * a;
	 sph_particles_ent(k) = -E;
	 for (int dim = 0; dim < NDIM; dim++) {
	 sph_particles_dvel_pred(dim, k) = 0.f;
	 sph_particles_dvel_con(dim, k) = 0.f;
	 }
	 }
	 for (auto& i : to_gas_indices) {
	 while (i < stars.size() && stars[i].remove) {
	 const int j = stars.size() - 1;
	 stars[i] = stars.back();
	 if (!stars[i].remove) {
	 particles_cat_index(stars[i].dm_index) = i;
	 }
	 stars.pop_back();
	 }
	 }
	 hpx::wait_all(futs.begin(), futs.end());
	 PRINT("%i stars, %i remnants, %i clouds\n", (int ) nstars, (int ) remnants, (int ) clouds);
	 for (int i = 0; i < nthreads; i++) {
	 gsl_rng_free(rnd_gens[i]);
	 }
	 //	profiler_exit();*/
}

float stars_sample_mass(gsl_rng* rndgen) {
	float y = gsl_rng_uniform_pos(rndgen);
	constexpr float m1 = 0.08;
	constexpr float m2 = 0.5;
	constexpr float alpha1 = -.7f;
	constexpr float alpha2 = .3f;
	constexpr float alpha3 = 1.3f;
	float mass;
	if (y < 0.0360115f) {
		const float y0 = 0.f;
		const float y1 = powf(m1, 1.f - alpha1);
		y = y0 + (y1 - y0) * gsl_rng_uniform_pos(rndgen);
		mass = powf(y, 1.0f / (1.f - alpha1));
	} else if (y < 0.0360115f + 0.227977f) {
		const float y0 = powf(m1, 1.f - alpha2);
		const float y1 = powf(m2, 1.f - alpha2);
		y = y0 + (y1 - y0) * gsl_rng_uniform_pos(rndgen);
		mass = powf(y, 1.0f / (1.f - alpha2));
	} else {
		const float y0 = 0.f;
		const float y1 = powf(m2, 1.f - alpha3);
		y = y0 + (y1 - y0) * gsl_rng_uniform_pos(rndgen);
		mass = powf(y, 1.0f / (1.f - alpha3));
	}
	if (mass > 265.f) {
		return stars_sample_mass(rndgen);
	}
	return mass;
}

float stars_luminosity(float mass) {
	constexpr float alpha1 = .238175;
	constexpr float alpha2 = 1.f;
	constexpr float alpha3 = sqrtf(2);
	constexpr float alpha4 = 31726.5f;
	constexpr float beta1 = 2.3f;
	constexpr float beta2 = 4.f;
	constexpr float beta3 = 3.5;
	constexpr float beta4 = 1.f;
	if (mass < .43) {
		return alpha1 * powf(mass, beta1);
	} else if (mass < 2) {
		return alpha2 * powf(mass, beta2);
	} else if (mass < 55) {
		return alpha3 * powf(mass, beta3);
	} else {
		return alpha4 * powf(mass, beta4);
	}
}

float stars_lifetime(float mass) {
	float l = mass / stars_luminosity(mass);
	l *= 12e9;
	//PRINT( "Stellar lifetime = %e B yr \n", l/1e9);
	return l;
}

float stars_helium_produced(float mass) {
	const double L0 = 3.839e33;
	const double M0 = 1.989e33;
	const double L = stars_luminosity(mass) * L0;
//	PRINT( "%e\n", L);
	const double t = stars_lifetime(mass) / constants::seconds_to_years;
//	PRINT( "%e\n", t);
	const double E = L * t;
//	PRINT( "%e\n", E);
	const double MHe = E / sqr(constants::c) / M0 / 0.007;
//	PRINT( "%e\n", MHe);
	return MHe / mass;

}

/*
 float stars_remnant_mass(float m) {
 float mf;
 //Cummings et al 2018
 if (m < 0.85) {
 mf = std::min(m, 0.080 * m + 0.489);
 } else if (m < 3.60) {
 mf = 0.187 * m + .184;
 } else if (m < 7.5) {
 mf = 0.107 * x + 0.471 * m;
 }
 //Fryer et al 2012
 else if( m < )
 }
 */

float Mrem_frac_big(float M, float Z) {
// Spera et al 2015
	const auto Mco_coeffs = [](float& B1, float&K1, float& K2, float&delta1, float& delta2,
			float Z) {
		if (Z > 4e-3) {
			B1 = 59.63 - 2.969e3 * Z + 4.988e4 * sqr(Z);
			K1 = 45.04 - 2.176e3 * Z + 3.806e4 * sqr(Z);
			K2 = 1.389e2 - 4.664e3 * Z + 5.106e4 * sqr(Z);
			delta1 = 2.790e-2 - 1.780e-2 * Z + 77.05 * sqr(Z);
			delta2 = 6.730e-3 + 2.690 * Z - 52.39 * sqr(Z);
		} else if (Z > 1e-3) {
			B1 = 40.98 + 3.415e4 * Z - 8.064e6 * sqr(Z);
			K1 = 35.17 + 1.548e4 * Z - 3.759e6 * sqr(Z);
			K2 = 20.36 + 1.162e5 * Z - 2.276e7 * sqr(Z);
			delta1 = 2.500e-2 - 4.346 * Z + 1.340e3 * sqr(Z);
			delta2 = 1.750e-2 + 11.39 * Z - 2.902 * sqr(Z);
		} else {
			B1 = 67.07;
			K1 = 46.89;
			K2 = 1.138e2;
			delta1 = 2.199e-2;
			delta2 = 2.602e-2;

		}
	};

	const auto gmxy = [](float M, float x, float y) {
		return 0.5 / (1.0 + pow(10, (x - M) * y));
	};

	const auto MCO = [Mco_coeffs,gmxy](float M, float Z) {
		float B1, K1, K2, delta1, delta2;
		Mco_coeffs(B1, K1, K2, delta1, delta2, Z);
		return -2.0 + (B1 + 2.0) * (gmxy(M, K1, delta1) + gmxy(M, K2, delta2));
	};

	float mco = MCO(M, Z);
	float Mrem;
	if (Z < 5e-4) {
		float mZ = -6.476e2 * Z + 1.911;
		float qZ = 2.300e3 * Z + 11.67;
		float fmcoz = mZ * mco + qZ;
		float pmco = -2.333 + 0.1559 * mco + 0.2700 * sqr(mco);
		if (mco < 5) {
			Mrem = std::max(pmco, 1.27f);
		} else if (mco < 10.0) {
			Mrem = pmco;
		} else {
			Mrem = std::min(pmco, fmcoz);
		}
	} else {
		float A1, A2, L, eta, m, q;
		if (Z > 1e-3) {
			A1 = 1.340 - 29.46 / (1.0 + pow(Z / 1.110e-3, 2.361));
			A2 = 80.22 - 74.73 * pow(Z, 0.965) / (2.72e-3 + pow(Z, 0.965));
			L = 5.683 + 3.533 / (1.0 + pow(Z / 7.430e-3, 1.993));
			eta = 1.066 - 1.121 / (1.0 + pow(Z / 2.558e-2, 0.609));
		} else {
			A1 = 1.150e5 * Z - 1.258e2;
			A2 = 91.56 - 1.957e4 * Z - 1.558e7 * sqr(Z);
			L = 1.134e4 * Z - 2.143;
			eta = 3.090e-2 - 22.30 * Z + 7.363e4 * sqr(Z);
		}
		if (Z > 2e-3) {
			m = 1.127;
			q = 1.061;
		} else if (Z > 1e-3) {
			m = -43.82 * Z + 1.304;
			q = -1.296e4 * Z + 26.98;
		} else {
			m = -6.476e2 * Z + 1.911;
			q = 2.300e3 * Z + 11.67;
		}
		float hmcoz = A1 + (A2 - A1) / (1.0 + pow(10, (L - mco) * eta));
		float fmcoz = m * mco + q;
		if (mco < 5) {
			Mrem = std::max(hmcoz, 1.27f);
		} else if (mco < 10) {
			Mrem = hmcoz;
		} else {
			Mrem = std::max(hmcoz, fmcoz);
		}
	}
	return std::min(std::max(Mrem / M, 0.0f), 1.0f);
}
float stars_remnant_mass(float Mi, float Z) {
	float Mf;
	if (Mi < 2.85) { // Cummings 2018
		Mf = std::min(0.080f * Mi + 0.489f, Mi);
	} else if (Mi < 3.60) { // Cummings 2018
		Mf = 0.187 * Mi + 0.184;
	} else if (Mi < 8.0) { // Cummings 2018
		Mf = 0.107 * Mi + 0.471;
	} else {
		Mf = Mi * Mrem_frac_big(Mi, Z);
	}
	return Mf / Mi;

}
void stars_test_mass() {
	constexpr int N = 200000;
	vector<float> m;
	float dm = 0.01;
	vector<float> cnt(10000, 0.0);
	for (int i = 0; i < N; i++) {
		static auto* rndgen = gsl_rng_alloc(gsl_rng_taus);
		m.push_back(stars_sample_mass(rndgen));
		//	PRINT( "%e\n", m[i]);
		const int j = m[i] / dm;
		if (j < 10000 && j >= 0) {
			cnt[j] += 1.0;
		}
	}
	FILE* fp = fopen("mass.txt", "wt");
	for (int i = 0; i < 10000; i++) {
		const float m = (i + 0.5) * dm;
		fprintf(fp, "%e %e %e %e %e %e \n", m, cnt[i], stars_lifetime(m), stars_remnant_mass(m, 0.0), stars_remnant_mass(m, 0.002), stars_remnant_mass(m, 0.02));
		//		printf( "%e %e\n", m, cnt[i]);
	}
	fclose(fp);

}
