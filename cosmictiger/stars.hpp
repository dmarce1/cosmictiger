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

#include <cosmictiger/particles.hpp>



struct star_particle {
	float zform;
	float time_remaining;
	float total_life;
	float stellar_mass;
	float Y;
	float Z;
	int type;
	bool remove;
	int dm_index;
};


struct stars_stats {
	size_t stars;
	size_t popI;
	size_t popII;
	size_t popIII;
	stars_stats() {
		popI = popII = popIII = stars = 0;
	}
	stars_stats& operator+=(const stars_stats& other) {
		stars += other.stars;
		popI += other.popI;
		popII += other.popII;
		popIII += other.popIII;
		return *this;
	}
	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & stars;
		arc & popI;
		arc & popII;
		arc & popIII;
	}
};



star_particle& stars_get(part_int index);
part_int stars_size();
double stars_find(float a, float dt, int minrung, int step, float tau);
void stars_remove(float a, float dt, int minrung, int step);
stars_stats stars_statistics(float);
void stars_save(FILE* fp);
void stars_load(FILE* fp);
void stars_test_mass();
