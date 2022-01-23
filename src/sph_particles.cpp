#define SPH_PARTICLES_CPP
#include <cosmictiger/sph_particles.hpp>
#include <cosmictiger/hpx.hpp>

static part_int capacity = 0;
static part_int size = 0;

part_int sph_particles_half_sort(pair<part_int> rng, part_int im) {
	part_int begin = rng.first;
	part_int end = rng.second;
	part_int lo = begin;
	part_int hi = end;
	while (lo < hi) {
		if (sph_particles_dm_index(lo) >= im) {
			while (lo != hi) {
				hi--;
				if (sph_particles_dm_index(lo) < im) {
					std::swap(sph_particles_dm_index(lo), sph_particles_dm_index(hi));
					std::swap(sph_particles_smooth_len(lo), sph_particles_smooth_len(hi));
					std::swap(sph_particles_ent(lo), sph_particles_ent(hi));
					break;
				}
			}
		}
		lo++;
	}
	return hi;

}

void sph_particles_sort(pair<part_int> rng, pair<part_int> irng, int level) {
	const part_int mid = (irng.first + irng.second) / 2;
	part_int midi = sph_particles_half_sort(rng, mid);
	pair<part_int> lo = rng;
	pair<part_int> hi = rng;
	pair<part_int> ilo = irng;
	pair<part_int> ihi = irng;
	lo.second = midi;
	hi.first = midi;
	ilo.second = mid;
	ihi.first = mid;
	if (lo.second - lo.first <= 1) {
		if (hi.second - hi.first > 1) {
			sph_particles_sort(hi, ihi, level + 1);
		}
	} else if (hi.second - hi.first <= 1) {
		if (lo.second - lo.first > 1) {
			sph_particles_sort(lo, ilo, level + 1);
		}
	} else {
		if ((1 << level) < 4 * hpx_hardware_concurrency()) {
			auto fut = hpx::async(sph_particles_sort, lo, ilo, level + 1);
			sph_particles_sort(hi, ihi, level + 1);
			fut.get();
		} else {
			sph_particles_sort(lo, ilo, level + 1);
			sph_particles_sort(hi, ihi, level + 1);
		}
	}
}

void sph_particles_sort() {
	pair<part_int> part_range;
	pair<part_int> index_range;
	part_range.first = 0;
	part_range.second = sph_particles_size();
	index_range.first = 0;
	index_range.second = particles_size();
	sph_particles_sort(part_range, index_range, 0);
	const int nthreads = hpx_hardware_concurrency();
	std::vector<hpx::future<void>> futs;
	for (int proc = 0; proc < nthreads; proc++) {
		futs.push_back(hpx::async([nthreads, proc]() {
			const int b = (size_t) proc * (size_t)sph_particles_size() / (size_t) nthreads;
			const int e = (size_t) (proc+1) * (size_t)sph_particles_size() / (size_t) nthreads;
			for (int i = b; i < e; i++) {
				particles_sph_index(sph_particles_dm_index(i)) = i;
			}
		}));
	}
	hpx::wait_all(futs.begin(), futs.end());
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
		particles_array_resize(sph_particles_r, new_capacity, false);
		particles_array_resize(sph_particles_drdh, new_capacity, false);
		particles_array_resize(sph_particles_e, new_capacity, false);
		particles_array_resize(sph_particles_de, new_capacity, false);
		for (int dim = 0; dim < NDIM; dim++) {
			particles_array_resize(sph_particles_dv[NDIM], new_capacity, false);
		}
		capacity = new_capacity;
	}
	part_int new_parts = sz - size;
	part_int offset = particles_size();
	particles_resize(particles_size() + new_parts);
	for (int i = 0; i < new_parts; i++) {
		particles_sph_index(offset + i) = size + i;
		sph_particles_dm_index(size + i) = offset + i;
	}
	size = sz;
}

void sph_particles_free() {
	free(sph_particles_dm);
	free(sph_particles_h);
	free(sph_particles_r);
	free(sph_particles_drdh);
	free(sph_particles_e);
	free(sph_particles_de);
	for (int dim = 0; dim < NDIM; dim++) {
		free(sph_particles_dv[NDIM]);
	}
}
