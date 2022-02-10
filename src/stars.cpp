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
	return stars[index];
}

HPX_PLAIN_ACTION (stars_find);

void stars_save(FILE* fp) {
	size_t size = stars.size();
	fwrite(&size, sizeof(size_t), 1, fp);
	fwrite(stars.data(), sizeof(star_particle), stars.size(), fp);
}

void stars_load(FILE* fp) {
	size_t size;
	FREAD(&size, sizeof(size_t), 1, fp);
	stars.resize(size);
	FREAD(stars.data(), sizeof(star_particle), size, fp);
}

void stars_find(float a) {
	PRINT("Searching for STARS\n");
	vector<hpx::future<void>> futs;
	for (auto& c : hpx_children()) {
		futs.push_back(hpx::async<stars_find_action>(c, a));
	}
	mutex_type mutex;
	std::atomic<part_int> next_index(sph_particles_size());
	std::atomic<int> found(0);
	const int nthreads = hpx_hardware_concurrency();
	for (int proc = 0; proc < nthreads; proc++) {
		futs.push_back(hpx::async([proc, nthreads, &next_index, a, &found, &mutex]() {
			const part_int b = (size_t) proc * sph_particles_size() / nthreads;
			const part_int e = (size_t) (proc+1) * sph_particles_size() / nthreads;
			for( part_int i = b; i < e; i++) {
				if( sph_particles_time_to_star(i) < 0.0 ) {
					part_int k = --next_index;
					star_particle star;
					star.energy = sph_particles_energy(i);
					star.zform = 1.f / a - 1.f;
					star.dm_index = sph_particles_dm_index(i);
					const int dmk = sph_particles_dm_index(k);
					const int dmi = star.dm_index;
					sph_particles_swap(k,i);
					particles_cat_index(dmk) = i;
					found++;
					std::lock_guard<mutex_type> lock(mutex);
					particles_cat_index(dmi) = stars.size();
					stars.push_back(star);
				}
			}
		}));
	}
	hpx::wait_all(futs.begin(), futs.end());
	PRINT("%i stars found for a total of %i\n", (int ) found, stars.size());
	sph_particles_resize(next_index);
}
