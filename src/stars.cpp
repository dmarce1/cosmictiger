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

void stars_find(float a, float dt, int minrung, int step) {
	PRINT("Searching for STARS\n");
	vector<hpx::future<void>> futs;
	vector<hpx::future<void>> futs2;
	for (auto& c : hpx_children()) {
		futs.push_back(hpx::async<stars_find_action>(c, a, dt, minrung, step));
	}
	mutex_type mutex;
	std::atomic<int> found(0);
	std::atomic<int> remnants(0);
	vector<part_int> indices;
	const int nthreads = hpx_hardware_concurrency();
	static bool first = true;
	static vector<gsl_rng *> rnd_gens(nthreads);
	if (first) {
		first = false;
		for (int i = 0; i < nthreads; i++) {
			rnd_gens[i] = gsl_rng_alloc(gsl_rng_taus);
			gsl_rng_set(rnd_gens[i], step * nthreads + i);
		}
	}
	static const double sph_mass = get_options().sph_mass;
	static const double code_to_g = get_options().code_to_g;
	static const double code_to_cm = get_options().code_to_cm;
	static const double code_to_s = get_options().code_to_s;
	for (int proc = 0; proc < nthreads; proc++) {
		futs2.push_back(hpx::async([proc, nthreads, a, &found, &mutex,&indices,dt]() {
			const float code_to_s = get_options().code_to_s;
			const part_int b = (size_t) proc * sph_particles_size() / nthreads;
			const part_int e = (size_t) (proc+1) * sph_particles_size() / nthreads;
			for( part_int i = b; i < e; i++) {
				float tdyn = sph_particles_tdyn(i);
				bool make_star = false;
				if( tdyn < 1e38 ) {
					if( tdyn == 0.f ) {
						PRINT( "ERROR %s %i\n", __FILE__, __LINE__);
					}
					float p = 1.f - expf(-std::min(dt/tdyn,88.0f));
					make_star = gsl_rng_uniform_pos(rnd_gens[proc]) < p;
				}
				if( make_star ) {
					sph_particles_tdyn(i) = 0.0;
					star_particle star;
					star.zform = 1.f / a - 1.f;
					star.dm_index = sph_particles_dm_index(i);
					star.Z = sph_particles_formZ(i);
					star.stellar_mass = stars_sample_mass(rnd_gens[proc]) * (star.Z == 0.f ? 25.0f : 1.0f);
					star.time_remaining = stars_lifetime(star.stellar_mass);
					star.remnant = false;
					star.remove = false;
					star.Y = sph_particles_formY(i);
					const int dmi = star.dm_index;
					found++;
					particles_type(dmi) = STAR_TYPE;
					std::lock_guard<mutex_type> lock(mutex);
					particles_cat_index(dmi) = stars.size();
					stars.push_back(star);
					indices.push_back(i);
				}
			}
		}));
	}
	hpx::wait_all(futs2.begin(), futs2.end());
	PRINT("Creating stars\n");
	for (auto& i : indices) {
		while (sph_particles_tdyn(sph_particles_size() - 1) == 0.f && sph_particles_size()) {
			sph_particles_resize(sph_particles_size() - 1, false);
		}
		if (i < sph_particles_size()) {
			const int k = sph_particles_size() - 1;
			if (i != k) {
				const int dmk = sph_particles_dm_index(k);
				sph_particles_swap(i, k);
				particles_cat_index(dmk) = i;
				if (particles_type(dmk) == STAR_TYPE) {
					PRINT("Error %s %i\n", __FILE__, __LINE__);
					abort();
				}
			}
			sph_particles_resize(k, false);
		}
	}
	hpx::wait_all(futs.begin(), futs.end());
	PRINT("%i stars created  for a total of %i stars and remnants\n", (int ) found, stars.size());
}

float stars_remnant_mass(float Mi, float Z);

HPX_PLAIN_ACTION (stars_statistics);

stars_stats stars_statistics(float a) {
	PRINT("Searching for STARS\n");
	vector<hpx::future<stars_stats>> futs;
	for (auto& c : hpx_children()) {
		futs.push_back(hpx::async<stars_statistics_action>(c, a));
	}
	stars_stats stats;
	const int nthreads = hpx_hardware_concurrency();
	for (int proc = 0; proc < nthreads; proc++) {
		futs.push_back(hpx::async([proc, nthreads]() {
			const part_int b = (size_t) proc * stars.size() / nthreads;
			const part_int e = (size_t) (proc+1) * stars.size() / nthreads;
			stars_stats stats;
			for( part_int i = b; i < e; i++) {
				if( stars[i].remnant) {
					stats.remnants++;
				} else {
					stats.stars++;
					if( stars[i].Z == 0.0) {
						stats.popIII++;
					} else if( stars[i].Z < 0.02) {
						stats.popII++;
					} else {
						stats.popI++;
					}
				}
			}
			return stats;
		}));
	}
	for (auto& f : futs) {
		stats += f.get();
	}
	if (hpx_rank() == 0) {
		FILE* fp = fopen("stars.txt", "at");
		fprintf(fp, "%e %li %li %li %li %li\n", 1.f / a - 1.f, stats.stars, stats.popI, stats.popII, stats.popIII, stats.remnants);
		fclose(fp);
	}
	return stats;
}

void stars_remove(float a, float dt, int minrung, int step) {
	PRINT("Searching for STARS\n");
	vector<hpx::future<void>> futs;
	vector<hpx::future<void>> futs2;
	for (auto& c : hpx_children()) {
		futs.push_back(hpx::async<stars_find_action>(c, a, dt, minrung, step));
	}
	const int nthreads = hpx_hardware_concurrency();
	static bool first = true;
	static vector<gsl_rng *> rnd_gens(nthreads);
	if (first) {
		first = false;
		for (int i = 0; i < nthreads; i++) {
			rnd_gens[i] = gsl_rng_alloc(gsl_rng_taus);
			gsl_rng_set(rnd_gens[i], step * nthreads + i);
		}
	}
	static const double sph_mass = get_options().sph_mass;
	static const double code_to_g = get_options().code_to_g;
	static const double code_to_cm = get_options().code_to_cm;
	static const double code_to_s = get_options().code_to_s;
	vector<part_int> to_gas_indices;
	mutex_type to_gas_mutex;
	std::atomic<int> remnants(0);
	for (int proc = 0; proc < nthreads; proc++) {
		futs2.push_back(hpx::async([proc, nthreads, a, dt, &to_gas_mutex, &to_gas_indices, minrung, &remnants]() {
			const part_int b = (size_t) proc * stars.size() / nthreads;
			const part_int e = (size_t) (proc+1) * stars.size() / nthreads;
			for( part_int i = b; i < e; i++) {
				if( stars[i].remnant == false ) {
					float real_dt = a * dt * code_to_s * constants::seconds_to_years;
					//PRINT( "removing %e years from %e\n", real_dt, stars[i].time_remaining);
				stars[i].time_remaining -= real_dt;
				if( stars[i].time_remaining < 0.0 /*&& particles_rung(stars[i].dm_index) >= minrung*/ ) {
					float remnant_mass_ratio = stars_remnant_mass(stars[i].stellar_mass, stars[i].Z);
					if( gsl_rng_uniform_pos(rnd_gens[proc]) > remnant_mass_ratio ) {
						std::lock_guard<mutex_type> lock(to_gas_mutex);
						to_gas_indices.push_back(i);
						stars[i].remove = true;
					} else {
						stars[i].remnant = true;
						stars[i].zform = 1.0f / a - 1.f;
						stars[i].stellar_mass *= remnant_mass_ratio;
					}
				}
			}
			if( stars[i].remnant == true ) {
				remnants++;
			}
		}
	}));
	}
	hpx::wait_all(futs2.begin(), futs2.end());
	static const auto h0 = 1.0e-1 / get_options().parts_dim;
	PRINT("Restoring gas\n");
	for (auto& i : to_gas_indices) {
		star_particle star = stars[i];
		const int j = star.dm_index;
		const int k = sph_particles_size();
		sph_particles_resize(k + 1, false);
		sph_particles_dm_index(k) = j;
		particles_type(j) = SPH_TYPE;
		particles_cat_index(j) = k;
		if (star.stellar_mass > 10.0f) {
			const float Hefrac = stars_helium_produced(star.stellar_mass);
			if (Hefrac > 1.0 - star.Y - star.Z) {
				PRINT("CANNOT CONVERT MORE THAN A STARS HYDROGEN MASS TO HELIUM! %e %e %e\n", Hefrac, star.Y, star.Z);
				abort();
			}
			const double Zyield = 0.02;
			if (Hefrac < Zyield) {
				PRINT("Cannot have less total mass converted than z yield\n");
				abort();
			}
//			star.Y += Hefrac - Zyield;
//			star.Z += Zyield;
			PRINT("***********************************SUPERNOVA************************************\n!\n");
			const int k = star.dm_index;
			float x = 2.f * gsl_rng_uniform_pos(rnd_gens[0]) - 1.f;
			float y = 2.f * gsl_rng_uniform_pos(rnd_gens[0]) - 1.f;
			float z = 2.f * gsl_rng_uniform_pos(rnd_gens[0]) - 1.f;
			const float ninv = 1.f / sqrt(sqr(x, y, z));
			x *= ninv;
			y *= ninv;
			z *= ninv;
			constexpr float vsup = 0e-3;
			float& vx = particles_vel(XDIM, k);
			float& vy = particles_vel(YDIM, k);
			float& vz = particles_vel(ZDIM, k);
			vx += vsup * x * a;
			vy += vsup * y * a;
			vz += vsup * z * a;
		}
		const double T = 5000.0;
		const double N = sph_mass * code_to_g * ((1. - star.Y) * 2.f + star.Y * .25f * 3.f + 0.5f * star.Z) * constants::avo;
		const double Cv = 1.5 * constants::kb;
		double E = Cv * N * T;
//		const double fSN = 0.0e-5;
		if (star.stellar_mass > 7.5) {
			float prem = stars_remnant_mass(stars[i].stellar_mass, stars[i].Z);
			float psn = 1.0f - prem;
			sph_particles_SN(k) = 1.0f / psn;
		}
		E /= sqr(code_to_cm) * code_to_g / sqr(code_to_s);
		E *= a * a;
		sph_particles_ent(k) = -E;
		sph_particles_dent(k) = 0.f;
		sph_particles_H2(k) = 0.f;
		sph_particles_Hp(k) = 1.f - star.Y - star.Z;
		sph_particles_He0(k) = 0.f;
		sph_particles_Hep(k) = 0.f;
		sph_particles_Hn(k) = 0.f;
		sph_particles_Hepp(k) = star.Y;
		sph_particles_Z(k) = star.Z;
		sph_particles_smooth_len(k) = h0;
		sph_particles_tdyn(k) = 1e38;
		for (int dim = 0; dim < NDIM; dim++) {
			sph_particles_dvel(dim, k) = 0.f;
		}
	}
	for (auto& i : to_gas_indices) {
		while (i < stars.size() && stars[i].remove) {
			stars[i] = stars.back();
			if (!stars[i].remove) {
				particles_cat_index(stars[i].dm_index) = i;
			}
			stars.pop_back();
		}
	}
	hpx::wait_all(futs.begin(), futs.end());
	PRINT("%i stars destroyed for a total of %i stars and %i remnants\n", to_gas_indices.size(), stars.size() - (int ) remnants, (int ) remnants);
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

float stars_remnant_mass(float Mi, float Z) {
	float Mf;
	constexpr float Z0 = 0.012;
	Z /= Z0;
	if (Mi < 2.85) { // Cummings 2018
		Mf = std::min(0.080f * Mi + 0.489f, Mi);
	} else if (Mi < 3.60) { // Cummings 2018
		Mf = 0.187 * Mi + 0.184;
	} else if (Mi < 10.0) { // Cummings 2018
		Mf = 0.107 * Mi + 0.471;
	} else if (Mi < 20.0) {
		const float m1 = (0.107 * 10.0f + 0.471);
		Mf = m1 + (Mi - 10.0) / 10.0 * (0.5 * Mi - m1);
	} else {
		Mf = 0.5f * Mi;
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
		fprintf(fp, "%e %e %e %e %e \n", m, cnt[i], stars_remnant_mass(m, 0.0), stars_remnant_mass(m, 0.002), stars_remnant_mass(m, 0.02));
		//		printf( "%e %e\n", m, cnt[i]);
	}
	fclose(fp);

}

