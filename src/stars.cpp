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

vector<star_particle> stars;

star_particle& stars_get(part_int index) {
	if (index > stars.size()) {
		PRINT("Attempt to read %i star with only %i\n", index, stars.size());
		abort();
	}
	return stars[index];
}

HPX_PLAIN_ACTION (stars_find);

void stars_save(FILE* fp) {
	size_t size = stars.size();
	fwrite(&size, sizeof(size_t), 1, fp);
	fwrite(stars.data(), sizeof(star_particle), stars.size(), fp);
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

void stars_find(float a, float dt) {
	PRINT("Searching for STARS\n");
	vector<hpx::future<void>> futs;
	vector<hpx::future<void>> futs2;
	for (auto& c : hpx_children()) {
		futs.push_back(hpx::async<stars_find_action>(c, a, dt));
	}
	mutex_type mutex;
	std::atomic<int> found(0);
	vector<part_int> indices;
	const int nthreads = hpx_hardware_concurrency();
	for (int proc = 0; proc < nthreads; proc++) {
		futs2.push_back(hpx::async([proc, nthreads, a, &found, &mutex,&indices,dt]() {
			const part_int b = (size_t) proc * sph_particles_size() / nthreads;
			const part_int e = (size_t) (proc+1) * sph_particles_size() / nthreads;
			for( part_int i = b; i < e; i++) {
				float tdyn = sph_particles_tdyn(i);
				bool make_star = false;
				if( tdyn < 1e38 ) {
					float p = 1.f - expf(-std::min(dt/tdyn,88.0f));
					make_star = rand1() < p;
				}
				if( make_star ) {
					sph_particles_tdyn(i) = 0.0;
					star_particle star;
					star.energy = sph_particles_energy(i);
					star.zform = 1.f / a - 1.f;
					star.dm_index = sph_particles_dm_index(i);
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
	for (auto& i : indices) {
		while (sph_particles_tdyn(sph_particles_size() - 1) == 0.f && sph_particles_size()) {
			sph_particles_resize(sph_particles_size() - 1);
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
	/*	for( part_int i = 0; i < particles_size(); i++) {
	 if( particles_type(i) == STAR_TYPE) {
	 if( particles_cat_index(i) >= stars.size()) {
	 PRINT( "Error %s %i\n", __FILE__, __LINE__);
	 abort();
	 }
	 }
	 }*/
	hpx::wait_all(futs.begin(), futs.end());
	PRINT("%i stars found for a total of %i\n", (int ) found, stars.size());
}
