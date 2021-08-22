/*
 * map.hpp
 *
 *  Created on: Aug 1, 2021
 *      Author: dmarce1
 */

#ifndef MAP_HPP_
#define MAP_HPP_

#include <unordered_map>
#include <fstream>

namespace hpx {
namespace serialization {

template<class A>
void serialize(A & arc, std::atomic<float> & value, unsigned int) {
	float number = value;
	arc & number;
	value = number;
}

}
}

class healpix_map {
	using pixel = std::atomic<float>;
public:
	using pixel_map = vector<std::atomic<float>>;
private:
	std::unordered_map<int, pixel_map> map;
	mutex_type mutex;
public:
	int map_add_particle(float x, float y, float z, float x1, float y1, float z1, float vx, float vy, float vz, float t, float dt);
	void add_map(healpix_map&& other);
	vector<float> map_flush(float t);
	void map_load(FILE* fp);
	void map_save(std::ofstream&);
};

void map_init(float tmax);
void map_add_map(healpix_map&& other);
vector<float> map_flush(float t);
void map_load(FILE* fp);
void map_save(std::ofstream&);

#endif /* MAP_HPP_ */
