/*
 * map.hpp
 *
 *  Created on: Aug 1, 2021
 *      Author: dmarce1
 */

#ifndef MAP_HPP_
#define MAP_HPP_

#include <unordered_map>

class healpix_map {
	struct pixel {
		float i;
		pixel() {
			i = 0.0;
		}
	};
	std::unordered_map<int, std::unordered_map<int, pixel>> map;
public:
	int map_add_particle(float x, float y, float z, float x1, float y1, float z1, float vx, float vy, float vz, float t, float dt);
	void add_map(healpix_map&& other);
	void map_flush(float t);
	void map_load(FILE* fp);
	void map_save(FILE* fp);
};

void map_init(float tmax);
void map_add_map(healpix_map&& other);
void map_flush(float t);
void map_load(FILE* fp);
void map_save(FILE* fp);

#endif /* MAP_HPP_ */
