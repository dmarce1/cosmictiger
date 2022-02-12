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

vector<star_particle> stars;

float stars_sample_mass();
float stars_luminosity(float mass);
float stars_lifetime(float mass);

HPX_PLAIN_ACTION (stars_find);

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

void stars_find(float a, float dt, int minrung) {
	PRINT("Searching for STARS\n");
	vector<hpx::future<void>> futs;
	vector<hpx::future<void>> futs2;
	for (auto& c : hpx_children()) {
		futs.push_back(hpx::async<stars_find_action>(c, a, dt, minrung));
	}
	mutex_type mutex;
	std::atomic<int> found(0);
	vector<part_int> indices;
	const int nthreads = hpx_hardware_concurrency();
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
					make_star = rand1() < p;
				}
				if( make_star ) {
					sph_particles_tdyn(i) = 0.0;
					star_particle star;
					star.zform = 1.f / a - 1.f;
					star.dm_index = sph_particles_dm_index(i);
					star.stellar_mass = stars_sample_mass();
					star.time_remaining = stars_lifetime(star.stellar_mass) / constants::seconds_to_years / code_to_s;
					star.remnant = false;
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
	vector<part_int> to_gas_indices;
	mutex_type to_gas_mutex;
	for (int proc = 0; proc < nthreads; proc++) {
		futs2.push_back(hpx::async([proc, nthreads, a, dt, &to_gas_mutex, &to_gas_indices, minrung]() {
			const part_int b = (size_t) proc * stars.size() / nthreads;
			const part_int e = (size_t) (proc+1) * stars.size() / nthreads;
			for( part_int i = b; i < e; i++) {
				if( stars[i].remnant == false ) {
					stars[i].time_remaining -= a * dt;
					if( stars[i].time_remaining < 0.0 && particles_rung(stars[i].dm_index) >= minrung ) {
						if( rand1() < 0.4f ) {
							std::lock_guard<mutex_type> lock(to_gas_mutex);
							to_gas_indices.push_back(i);
						} else {
							stars[i].remnant = true;
						}
					}
				}
			}
		}));
	}
	hpx::wait_all(futs2.begin(), futs2.end());
	PRINT( "Creating stars\n");
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
	const static auto Y = get_options().Y;
	static const auto h0 = 1.0e-3 / get_options().parts_dim;
	static const double sph_mass = get_options().sph_mass;
	static const double code_to_g = get_options().code_to_g;
	static const double code_to_cm = get_options().code_to_cm;
	static const double code_to_s = get_options().code_to_s;
	PRINT( "Restoring gas\n");
	for (auto& i : to_gas_indices) {
		star_particle star = stars[i];
		const int j = star.dm_index;
		const int k = sph_particles_size();
		sph_particles_resize(k + 1, false);
		sph_particles_dm_index(k) = j;
		particles_type(j) = SPH_TYPE;
		const double T = 5000.0;
		const double N = sph_mass * code_to_g * ((1. - Y) * 2.f + Y * .25f * 3.f) * constants::avo;
		const double Cv = 1.5 * constants::kb;
		double E = Cv * N * T;
		E /= sqr(code_to_cm) * code_to_g / sqr(code_to_s);
		E *= a * a;
		sph_particles_ent(k) = -E;
		sph_particles_dent(k) = 0.f;
		sph_particles_H2(k) = 0.f;
		sph_particles_Hp(k) = 1.f - Y;
		sph_particles_Hep(k) = 0.f;
		sph_particles_Hn(k) = 0.f;
		sph_particles_Hepp(k) = Y;
		sph_particles_smooth_len(k) = h0;
		sph_particles_tdyn(k) = 1e38;
		particles_cat_index(j) = k;
		for (int dim = 0; dim < NDIM; dim++) {
			sph_particles_dvel(dim, k) = 0.f;
		}
	}
	for (auto& i : to_gas_indices) {
		if (i < stars.size()) {
			stars[i] = stars.back();
			particles_cat_index(stars[i].dm_index) = i;
			stars.pop_back();
		}
	}
	hpx::wait_all(futs.begin(), futs.end());
	PRINT("%i stars created and %i stars destroyed for a total of %i\n", (int ) found, to_gas_indices.size(), stars.size());
}

float stars_sample_mass() {
	float y = rand1();
	constexpr float m1 = 0.08;
	constexpr float m2 = 0.5;
	constexpr float alpha1 = -.7f;
	constexpr float alpha2 = .3f;
	constexpr float alpha3 = 1.3f;
	float mass;
	if (y < 0.0360115f) {
		const float y0 = 0.f;
		const float y1 = powf(m1, 1.f - alpha1);
		y = y0 + (y1 - y0) * rand1();
		mass = powf(y, 1.0f / (1.f - alpha1));
	} else if (y < 0.0360115f + 0.227977f) {
		const float y0 = powf(m1, 1.f - alpha2);
		const float y1 = powf(m2, 1.f - alpha2);
		y = y0 + (y1 - y0) * rand1();
		mass = powf(y, 1.0f / (1.f - alpha2));
	} else {
		const float y0 = 0.f;
		const float y1 = powf(m2, 1.f - alpha3);
		y = y0 + (y1 - y0) * rand1();
		mass = powf(y, 1.0f / (1.f - alpha3));
	}
	if (mass < 0.f) {
		PRINT("%e\n", mass);
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
	const float l = mass / stars_luminosity(mass);
	return l * 12e9;
}

void stars_test_mass() {
	constexpr int N = 200000;
	vector<float> m;
	float dm = 0.01;
	vector<float> cnt(10000, 0.0);
	for (int i = 0; i < N; i++) {
		m.push_back(stars_sample_mass());
		//	PRINT( "%e\n", m[i]);
		const int j = m[i] / dm;
		if (j < 10000 && j >= 0) {
			cnt[j] += 1.0;
		}
	}
	FILE* fp = fopen("mass", "wt");
	for (int i = 0; i < 10000; i++) {
		const float m = (i + 0.5) * dm;
		fprintf(fp, "%e %e %e\n", m, cnt[i], stars_lifetime(m));
//		printf( "%e %e\n", m, cnt[i]);
	}
	fclose(fp);

}

