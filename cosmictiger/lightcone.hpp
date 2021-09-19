#pragma once


using lc_real = double;


class lc_group: public std::atomic<long long> {
public:
	lc_group() {
	}
	lc_group(const lc_group& other) {
		std::atomic<long long>::operator=((long long) other);
	}
	lc_group(lc_group&& other) {
		std::atomic<long long>::operator=((long long) other);
	}
	lc_group& operator=(const lc_group& other) {
		std::atomic<long long>::operator=((long long) other);
		return *this;
	}
	lc_group& operator=(lc_group&& other) {
		std::atomic<long long>::operator=((long long) other);
		return *this;
	}
	lc_group& operator=(long long other) {
		std::atomic<long long>::operator=(other);
		return *this;
	}
	template<class A>
	void serialize(A&& arc, unsigned) {
		long long number = (long long) *((std::atomic<long long>*) (this));
		arc & number;
		*((std::atomic<long long>*) (this)) = number;
	}
};

struct lc_particle {
	array<lc_real, NDIM> pos;
	array<float, NDIM> vel;
	lc_group group;
	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & pos;
		arc & vel;
		arc & group;
	}
};

#include <cosmictiger/lc_group_archive.hpp>

void lc_init(double, double);
int lc_add_particle(lc_real x0, lc_real y0, lc_real z0, lc_real x1, lc_real y1, lc_real z1, float vx, float vy, float vz, float t, float dt, vector<lc_particle>& this_part_buffer);
void lc_add_parts(vector<lc_particle>&&);
void lc_buffer2homes();
size_t lc_time_to_flush(double, double);
void lc_particle_boundaries();
void lc_form_trees(double tmax, double link_len);
size_t lc_find_groups();
void lc_groups2homes();
void lc_parts2groups(double a, double link_len);
void lc_save(FILE* fp);
void lc_load(FILE* fp);
vector<float> lc_flush_final();



