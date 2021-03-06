/*
 * particle.hpp
 *
 *  Created on: Jan 23, 2021
 *      Author: dmarce1
 */

#ifndef COSMICTIGER_PARTICLE_HPP_
#define COSMICTIGER_PARTICLE_HPP_

#include <cosmictiger/defs.hpp>
#include <cosmictiger/global.hpp>
#include <cosmictiger/memory.hpp>
#include <cosmictiger/fixed.hpp>
#include <cosmictiger/range.hpp>
#include <cosmictiger/array.hpp>
#include <cosmictiger/vector.hpp>
#ifdef _CUDA_ARCH_
#include <cosmictiger/hpx.hpp>
#endif

struct range;

#define NO_GROUP (0x7FFFFFFFFFFF)

using group_t = unsigned long long int;

#include <cosmictiger/domain.hpp>
#include <cosmictiger/array.hpp>
#include <atomic>
#include <vector>
#include <unordered_map>

using rung_t = int8_t;

using part_int = unsigned;

struct particle {
	array<fixed32, NDIM> x;
	array<float, NDIM> v;
	rung_t rung;
	unsigned long long group;
#ifdef TEST_FORCE
	array<float, NDIM> g;
	float phi;
#endif
	template<class A>
	void serialize(A&& arc, unsigned) {
		for (int dim = 0; dim < NDIM; dim++) {
			arc & x[dim];
			arc & v[dim];
#ifdef TEST_FORCE
			arc & g[dim];
#endif
		}
		arc & rung;
		static bool groups = global().opts.groups;
		if (groups) {
			arc & group;
		}
#ifdef TEST_FORCE
		arc & phi;
#endif
	}
};

#ifndef __CUDACC__
struct particle_send_entry {
	vector<particle> parts;
	mutex_type mutex;
};

using particle_send_type = std::unordered_map<int,particle_send_entry>;
#endif

CUDA_EXPORT bool operator<(const particle& a, const particle& b);

struct particle_set {
	std::pair<fixed32, fixed32> find_range(part_int b, part_int e, int xdim) const;
	fixed32 find_middle(part_int b, part_int e, int xdim) const;
	particle get_particle(part_int i);
	void set_particle(const particle&, part_int i);
	void free_particles(part_int*, int cnt);
	part_int count_parts_below(part_int, part_int, int, fixed32) const;CUDA_EXPORT
	particle_set();
	static void sort_parts(particle* begin, particle* end);
	static void sort_indices(part_int* begin, part_int* end);
	particle_set(part_int);
	void resize(part_int);
	void resize_pos(part_int);CUDA_EXPORT
	~particle_set();CUDA_EXPORT
	fixed32 pos_ldg(int, part_int index) const;CUDA_EXPORT
	fixed32 pos(int, part_int index) const;CUDA_EXPORT
	float vel(int dim, part_int index) const;CUDA_EXPORT
	rung_t rung(part_int index) const;
	part_int sort_range(part_int begin, part_int end, double xmid, int xdim);CUDA_EXPORT
	fixed32& pos(int dim, part_int index);CUDA_EXPORT
	float& vel(int dim, part_int index);CUDA_EXPORT
	void set_rung(rung_t t, part_int index);
	void generate_random(int seed);
	void load_from_file(FILE* fp);
	void save_to_file(FILE* fp);
	void generate_grid();CUDA_EXPORT
	group_t group(part_int) const;CUDA_EXPORT
	group_t& group(part_int);CUDA_EXPORT
	group_t get_last_group(part_int) const;CUDA_EXPORT
	void set_last_group(part_int, group_t);CUDA_EXPORT
	part_int size() const {
		return size_;
	}
	part_int pos_size() const {
		return pos_size_;
	}
	void finish_groups();
	void init_groups();CUDA_EXPORT
	float force(int dim, part_int index) const;CUDA_EXPORT
	float& force(int dim, part_int index);CUDA_EXPORT
	float pot(part_int index) const;CUDA_EXPORT
	float& pot(part_int index);
#ifndef __CUDACC__
	bool gather_sends(particle_send_type&, vector<part_int>&, domain_bounds);
protected:
#endif
	array<fixed32*, NDIM> xptr_;
	array<float, NDIM>* uptr_;
	rung_t* rptr_;
	group_t* idptr_;
	uint32_t* lidptr1_;
	uint16_t* lidptr2_;
	array<float*, NDIM> gptr_;
	float* eptr_;
	part_int size_;
	part_int pos_size_;
	part_int cap_;
	part_int pos_cap_;
	bool virtual_;

public:
	CUDA_EXPORT
	particle_set get_virtual_particle_set() const {
		particle_set v;
		v.cap_ = cap_;
		v.pos_cap_ = cap_;
		v.rptr_ = rptr_;
		v.idptr_ = idptr_;
		v.lidptr1_ = lidptr1_;
		v.lidptr2_ = lidptr2_;
		v.uptr_ = uptr_;
		for (int dim = 0; dim < NDIM; dim++) {
			v.xptr_[dim] = xptr_[dim];
		}
//		v.rptr_ = rptr_;
#ifdef TEST_FORCE
		v.gptr_ = gptr_;
		v.eptr_ = eptr_;
#endif
		v.size_ = size_;
		v.pos_size_ = size_;
		v.virtual_ = true;
		return v;
	}
}
;

CUDA_EXPORT
inline fixed32 particle_set::pos_ldg(int dim, part_int index) const {

	assert(index < pos_size_);
	union fixed_union {
		fixed32 f;
		int i;
	};
	fixed_union x;
	x.i = LDG((int* )(xptr_[dim] + index));
	return x.f;
}

CUDA_EXPORT
inline fixed32 particle_set::pos(int dim, part_int index) const {

	assert(index < pos_size_);
	return xptr_[dim][index];
}

inline float particle_set::vel(int dim, part_int index) const {

	assert(index < size_);
	return uptr_[index][dim];
}

CUDA_EXPORT
inline rung_t particle_set::rung(part_int index) const {
	if (index >= size_) {
		PRINT("%i\n", size_);
	}
	assert(index < size_);
	/*if (rptr_[index] != uptr_[index].p.r) {
	 PRINT("%i %i\n", rptr_[index], uptr_[index].p.r);
	 }
	 return uptr_[index].p.r;*/
	return rptr_[index];
}

CUDA_EXPORT
inline fixed32& particle_set::pos(int dim, part_int index) {

	assert(index < pos_size_);
	return xptr_[dim][index];
}

CUDA_EXPORT
inline float& particle_set::vel(int dim, part_int index) {

	assert(index < size_);
	return uptr_[index][dim];
}

CUDA_EXPORT
inline void particle_set::set_rung(rung_t t, part_int index) {

	assert(index < size_);
	rptr_[index] = t;
	//uptr_[index].p.r = t;
}

CUDA_EXPORT inline float particle_set::force(int dim, part_int index) const {
	assert(index < size_);
	return gptr_[dim][index];

}
CUDA_EXPORT inline float& particle_set::force(int dim, part_int index) {
	assert(index < size_);
	return gptr_[dim][index];

}
CUDA_EXPORT inline float particle_set::pot(part_int index) const {
	assert(index < size_);
	return eptr_[index];

}
CUDA_EXPORT inline float& particle_set::pot(part_int index) {
	assert(index < size_);
	return eptr_[index];

}

CUDA_EXPORT inline group_t particle_set::group(part_int index) const {
	return idptr_[index];
}

CUDA_EXPORT inline group_t& particle_set::group(part_int index) {
	return idptr_[index];
}

CUDA_EXPORT inline group_t particle_set::get_last_group(part_int index) const {
	return lidptr1_[index] | (group_t(lidptr2_[index]) << 32ULL);
}

CUDA_EXPORT inline void particle_set::set_last_group(part_int index, group_t g) {
	lidptr1_[index] = g & 0xFFFFFFFFULL;
	lidptr2_[index] = g >> 32ULL;
}

using part_iters = pair<part_int,part_int>;
#endif /* COSMICTIGER_PARTICLE_HPP_ */
