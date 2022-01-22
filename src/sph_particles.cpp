#define SPH_PARTICLES_CPP
#include <cosmictiger/sph_particles.hpp>

static part_int capacity = 0;
static part_int size = 0;

struct sph_particle {
	float h;
	part_int dm;
	bool operator<(const sph_particle& other) const {
		return dm < other.dm;
	}
};

struct sph_particle_ref {
	part_int index;
	sph_particle_ref(int i) {
		index = i;
	}
	operator sph_particle() {
		sph_particle part;
		part.h = sph_particles_smooth_len(index);
		part.dm = sph_particles_dm_index(index);
		return part;
	}
	sph_particle_ref& operator=(sph_particle_ref other) {
		sph_particles_smooth_len(index) = sph_particles_smooth_len(other.index);
		sph_particles_dm_index(index) = sph_particles_dm_index(other.index);
		return *this;
	}
	sph_particle_ref& operator=(const sph_particle& other) {
		sph_particles_smooth_len(index) = other.h;
		sph_particles_dm_index(index) = other.dm;
		return *this;
	}
	bool operator<(const sph_particle_ref& other) const {
		return sph_particles_dm_index(index) < sph_particles_dm_index(other.index);
	}

	bool operator<(const sph_particle& other) const {
		return sph_particles_dm_index(index) < other.dm;
	}


};


void swap(sph_particle_ref a, sph_particle_ref b) {
	sph_particle c = a;
	a = b;
	b = c;
}

class sph_iterator;

namespace std {
template<>
struct iterator_traits<sph_iterator> {
	using iterator_category = std::random_access_iterator_tag;
	using difference_type = int;
	using value_type = sph_particle;
	using reference = sph_particle_ref;  // or also value_type&
};
}

struct sph_iterator {
	part_int index;
	using iterator_category = std::random_access_iterator_tag;
	using difference_type = int;
	using value_type = sph_particle;
	using reference = sph_particle_ref;  // or also value_type&
	reference operator*() const {
		return sph_particle_ref(index);
	}

	// Prefix increment
	sph_iterator& operator++() {
		index++;
		return *this;
	}

	// Prefix increment
	sph_iterator& operator++(int) {
		index++;
		return *this;
	}
	sph_iterator& operator--() {
		index--;
		return *this;
	}

	// Prefix increment
	sph_iterator& operator--(int) {
		index--;
		return *this;
	}
	sph_iterator operator-(int i) const {
		sph_iterator rc;
		rc.index = index - i;
		return rc;
	}
	sph_iterator operator+(int i) const {
		sph_iterator rc;
		rc.index = index + i;
		return rc;
	}
	bool operator!=(sph_iterator other) const {
		return index != other.index;
	}
	bool operator==(sph_iterator other) const {
		return index == other.index;
	}
	bool operator<(sph_iterator other) const {
		return index < other.index;
	}
};

int operator-(sph_iterator i, sph_iterator j) {
	return i.index - j.index;
}

sph_iterator sph_particles_begin() {
	sph_iterator i;
	i.index = 0;
	return i;
}

sph_iterator sph_particles_end() {
	sph_iterator i;
	i.index = sph_particles_size();
	return i;
}

void sph_particles_sort() {
	std::sort(sph_particles_begin(), sph_particles_end());
}

part_int sph_particles_size() {
	return size;
}

template<class T>
void particles_array_resize(T*& ptr, part_int new_capacity, bool reg) {
	T* new_ptr;
	if (capacity > 0) {
	}
#ifdef USE_CUDA
	if( reg ) {
		cudaMallocManaged(&new_ptr,sizeof(T) * new_capacity);
	} else {
		new_ptr = (T*) malloc(sizeof(T) * new_capacity);
	}
#else
	new_ptr = (T*) malloc(sizeof(T) * new_capacity);
#endif
	if (capacity > 0) {
		hpx_copy(PAR_EXECUTION_POLICY, ptr, ptr + size, new_ptr).get();
#ifdef USE_CUDA
		if( reg ) {
			cudaFree(ptr);
		} else {
			free(ptr);
		}
#else
		free(ptr);
#endif
	}
	ptr = new_ptr;

}

void sph_particles_resize(part_int sz) {
	if (sz > capacity) {
		part_int new_capacity = std::max(capacity, (part_int) 100);
		while (new_capacity < sz) {
			new_capacity = size_t(101) * new_capacity / size_t(100);
		}
		PRINT("Resizing sph_particles to %li from %li\n", new_capacity, capacity);
		particles_array_resize(sph_particles_dm, new_capacity, false);
		particles_array_resize(sph_particles_h, new_capacity, false);
		capacity = new_capacity;
	}
	size = sz;
}

void sph_particles_free() {
	free(sph_particles_dm);
	free(sph_particles_h);
}
