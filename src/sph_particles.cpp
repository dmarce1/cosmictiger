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

#define SPH_PARTICLES_CPP
#include <cosmictiger/sph_particles.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/stars.hpp>
#include <cosmictiger/sph.hpp>
#include <cosmictiger/constants.hpp>

static part_int capacity = 0;
static part_int size = 0;

struct line_id_type {
	int proc;
	part_int index;
	inline bool operator==(line_id_type other) const {
		return proc == other.proc && index == other.index;
	}
};

struct line_id_hash {
	inline size_t operator()(line_id_type id) const {
		const part_int line_size = get_options().part_cache_line_size;
		const part_int i = id.index / line_size;
		return i * (hpx_size() - 1) + ((id.proc < hpx_rank()) ? id.proc : id.proc - 1);
	}
};

struct line_id_hash_lo {
	inline size_t operator()(line_id_type id) const {
		line_id_hash hash;
		return hash(id) % PART_CACHE_SIZE;
	}
};

struct line_id_hash_hi {
	inline size_t operator()(line_id_type id) const {
		line_id_hash hash;
		return hash(id) / PART_CACHE_SIZE;
	}
};

static const array<fixed32, NDIM>* sph_particles_cache_read_line(line_id_type line_id);
static const float* sph_particles_entr_cache_read_line(line_id_type line_id);
static const float* sph_particles_fcold_cache_read_line(line_id_type line_id);
static const char* sph_particles_rung_cache_read_line(line_id_type line_id);
static const array<float, NDIM>* sph_particles_vels_cache_read_line(line_id_type line_id);
static const float* sph_particles_kappas_cache_read_line(line_id_type line_id);
static const aux_quantities* sph_particles_aux_cache_read_line(line_id_type line_id);

static vector<array<fixed32, NDIM>> sph_particles_fetch_cache_line(part_int index);
static vector<char> sph_particles_fetch_rung_cache_line(part_int index);
static vector<array<float, NDIM>> sph_particles_fetch_vels_cache_line(part_int index);
static vector<float> sph_particles_fetch_kappas_cache_line(part_int index);
static vector<float> sph_particles_fetch_entr_cache_line(part_int index);
static vector<float> sph_particles_fetch_fcold_cache_line(part_int index);
static vector<aux_quantities> sph_particles_fetch_aux_cache_line(part_int index);

HPX_PLAIN_ACTION (sph_particles_fetch_cache_line);
HPX_PLAIN_ACTION (sph_particles_fetch_entr_cache_line);
HPX_PLAIN_ACTION (sph_particles_fetch_fcold_cache_line);
HPX_PLAIN_ACTION (sph_particles_fetch_rung_cache_line);
HPX_PLAIN_ACTION (sph_particles_fetch_vels_cache_line);
HPX_PLAIN_ACTION (sph_particles_fetch_kappas_cache_line);
HPX_PLAIN_ACTION (sph_particles_cache_free_entr);
HPX_PLAIN_ACTION (sph_particles_cache_free1);
HPX_PLAIN_ACTION (sph_particles_cache_free2);
HPX_PLAIN_ACTION (sph_particles_fetch_aux_cache_line);

static array<std::unordered_map<line_id_type, hpx::shared_future<vector<array<fixed32, NDIM>>> , line_id_hash_hi>,PART_CACHE_SIZE> part_cache;
static array<std::unordered_map<line_id_type, hpx::shared_future<vector<float>>, line_id_hash_hi>, PART_CACHE_SIZE> entr_part_cache;
static array<std::unordered_map<line_id_type, hpx::shared_future<vector<float>>, line_id_hash_hi>, PART_CACHE_SIZE> fcold_part_cache;
static array<std::unordered_map<line_id_type, hpx::shared_future<vector<char>>, line_id_hash_hi>, PART_CACHE_SIZE> rung_part_cache;
static array<std::unordered_map<line_id_type, hpx::shared_future<vector<array<float, NDIM>>> , line_id_hash_hi>, PART_CACHE_SIZE> vels_part_cache;
static array<std::unordered_map<line_id_type, hpx::shared_future<vector<float>>, line_id_hash_hi>, PART_CACHE_SIZE> kappas_part_cache;
static array<std::unordered_map<line_id_type, hpx::shared_future<vector<aux_quantities>>, line_id_hash_hi>, PART_CACHE_SIZE> aux_cache;
static array<spinlock_type, PART_CACHE_SIZE> mutexes;
static array<spinlock_type, PART_CACHE_SIZE> entr_mutexes;
static array<spinlock_type, PART_CACHE_SIZE> fcold_mutexes;
static array<spinlock_type, PART_CACHE_SIZE> rung_mutexes;
static array<spinlock_type, PART_CACHE_SIZE> vels_mutexes;
static array<spinlock_type, PART_CACHE_SIZE> kappas_mutexes;
static array<spinlock_type, PART_CACHE_SIZE> aux_mutexes;
static int group_cache_epoch = 0;

HPX_PLAIN_ACTION (sph_particles_max_smooth_len);

float sph_particles_max_smooth_len() {
	vector<hpx::future<float>> futs;
	for (auto& c : hpx_children()) {
		futs.push_back(hpx::async<sph_particles_max_smooth_len_action>(c));
	}
	const int nthreads = hpx_hardware_concurrency();
	for (int proc = 0; proc < nthreads; proc++) {
		futs.push_back(hpx::async([nthreads, proc]() {
			float maxh = 0.0;
			const part_int b = (size_t) proc * sph_particles_size() / nthreads;
			const part_int e = (size_t) (proc+1) * sph_particles_size() / nthreads;
			for( int i = b; i < e; i++) {
				maxh = std::max(maxh, sph_particles_smooth_len(i));
			}
			return maxh;
		}));
	}
	float maxh = 0.f;
	for (auto& f : futs) {
		maxh = std::max(maxh, f.get());
	}
	PRINT("done\n");
	return maxh;
}

HPX_PLAIN_ACTION (sph_particles_apply_updates);
std::pair<double, double> sph_particles_apply_updates(int minrung, int phase, float t0, float tau, float w) {
	if (!(phase == 2 || tau > 0.0)) {
		return std::pair<double, double>();
	}
	profiler_enter(__FUNCTION__);
	double err = 0.0;
	double norm = 0.0;
	vector<hpx::future<std::pair<double, double>>>futs;
	for (auto& c : hpx_children()) {
		futs.push_back(hpx::async<sph_particles_apply_updates_action>(c, minrung, phase, t0, tau, w));
	}
	const int nthreads = hpx_hardware_concurrency();
	for (int proc = 0; proc < nthreads; proc++) {
		futs.push_back(hpx::async([t0,nthreads, proc, phase, minrung, tau, w]() {
			const auto chem = get_options().chem;
			const auto stars = get_options().stars;
			double error = 0.0;
			double norm = 0.0;
			const part_int b = (size_t) proc * sph_particles_size() / nthreads;
			const part_int e = (size_t) (proc+1) * sph_particles_size() / nthreads;
			for( int i = b; i < e; i++) {
				const part_int k = sph_particles_dm_index(i);
				const auto rung1 = sph_particles_oldrung(i);
				const auto rung2 = sph_particles_rung(i);
				const float dt1 = tau > 0.0 ? 0.5f * t0 / (1<<rung1) : 0.f;
				const float dt2 = 0.5f * t0 / (1<<rung2);
				const float dt = dt1 + dt2;
				if( phase == 0 ) {
					if( rung2 >= minrung) {
						for( int dim =0; dim < NDIM; dim++) {
							sph_particles_dvel0(dim,i) = sph_particles_dvel(dim,i);
						}
						for( int dim =0; dim < NDIM; dim++) {
							particles_vel(dim,k) += sph_particles_dvel(dim,i)* dt2;
						}
					}
				} else if( phase == 1 ) {
					if( rung1 >= minrung) {
						for( int dim =0; dim < NDIM; dim++) {
							particles_vel(dim,k) += (sph_particles_dvel(dim,i) - sph_particles_dvel0(dim,i))* dt1;
						}
					}
				} else if( phase == 2 ) {
					if( rung2 >= minrung) {
						for( int dim =0; dim < NDIM; dim++) {
							particles_vel(dim,k) += sph_particles_dvel(dim,i)* dt2;
						}
						sph_particles_entr(i) += sph_particles_dentr(i)* 2.0 * dt2;
						if( stars ) {
							sph_particles_cold_mass(i) += sph_particles_dcold_mass(i)* 2.0 * dt2;
						}
						if( chem ) {
							for( int fi = 0; fi < NCHEMFRACS; fi++) {
								auto& frac = sph_particles_frac(fi,i);
								auto dfrac = sph_particles_dchem(i)[fi];
								/*if( frac + dfrac * 2.0 * dt2 < 0.0) {
								 if(frac + 2.0 * dt2 * dfrac > -1e-20f) {
								 frac = 1e-30f - dfrac * 2.0 * dt2 *1.0000001;
								 } else {
								 PRINT( "%e %e\n", frac, 2.0 * dt2 * dfrac);
								 ALWAYS_ASSERT(false);
								 }
								 }*/
								frac += dfrac * dt2 * 2.0;
							}
						}

					}

				}
			}
			return std::make_pair(error, norm);
		}));
	}
	for (auto& f : futs) {
		auto tmp = f.get();
		err = std::max(err, tmp.first);
		norm += tmp.second;
	}
	profiler_exit();
	return std::make_pair(err, 1.0);

}

float sph_particles_temperature(part_int i, float a) {
	const double code_to_energy_density = get_options().code_to_g / (get_options().code_to_cm * sqr(get_options().code_to_s));		// 7
	const double code_to_energy = sqr(get_options().code_to_cm) / (sqr(get_options().code_to_s));		// 7
	const double code_to_density = pow(get_options().code_to_cm, -3) * get_options().code_to_g;										// 10
	const double code_to_entropy = code_to_energy / pow(code_to_density, get_options().gamma - 1.0);
	const double h = sph_particles_smooth_len(i);
	const double Hp = sph_particles_Hp(i);
	const double Hn = sph_particles_Hn(i);
	const double H2 = sph_particles_H2(i);
	const double Y = sph_particles_Y(i);
	const double Hep = sph_particles_Hep(i);
	const double Hepp = sph_particles_Hepp(i);
	const double H = sph_particles_H(i);
	const double He = Y - Hep - Hepp;
	double rho = sph_den(1 / (h * h * h));
	double n = (H + 2.f * Hp + .5f * H2 + .25f * He + .5f * Hep + .75f * Hepp) * 1.0 / (1.0 - sph_particles_Z(i));
	rho *= code_to_density * pow(a, -3);
	n *= constants::avo * rho;									// 8
	double gamma = sph_particles_gamma(i);
	double cv = 1.0 / (gamma - 1.0);															// 4
	cv *= double(constants::kb);																							// 1
	double entr = sph_particles_entr(i);
	entr *= code_to_entropy;
	const double eint = entr * pow(rho, get_options().gamma - 1.0) / (gamma - 1.0);
	double T = rho * eint / (n * cv);
	if (H < 0.0) {
		if (H < -5.0e-3) {
			PRINT("NEGATIVE H\n");
			PRINT("%e %e %e %e %e %e %e\n", H, Hp, Hn, H2, He, Hep, Hepp);
			abort();
		}
	}
	if (T > TMAX) {
		PRINT("T == %e %e %e %e %e %e\n", T, sph_particles_entr(i), eint, eint, rho, h);
		abort();
	}
	if (T < 0.0) {
		PRINT("T == %e %e %e %e %e %e\n", T, sph_particles_entr(i), eint, eint, rho, h);
	}
	return T;
}

float sph_particles_mmw(part_int i) {
	const double Hp = sph_particles_Hp(i);
	const double Hn = sph_particles_Hn(i);
	const double H2 = sph_particles_H2(i);
	const double Y = sph_particles_Y(i);
	const double Hep = sph_particles_Hep(i);
	const double Hepp = sph_particles_Hepp(i);
	const double H = sph_particles_H(i);
	const double He = Y - Hep - Hepp;
	double n = H + 2.f * Hp + .5f * H2 + .25f * He + .5f * Hep + .75f * Hepp;
//	PRINT( "%e\n", 1.0 / n);
	return 1.0 / n;
}

/*float sph_particles_lambda_e(part_int i, float a, float T) {
 const double code_to_energy_density = get_options().code_to_g / (get_options().code_to_cm * sqr(get_options().code_to_s));		// 7
 const double code_to_density = pow(get_options().code_to_cm, -3) * get_options().code_to_g;										// 10
 const double h = sph_particles_smooth_len(i);
 const double Hp = sph_particles_Hp(i);
 const double Hn = sph_particles_Hn(i);
 const double Hep = sph_particles_Hep(i);
 const double Hepp = sph_particles_Hepp(i);
 double rho = sph_den(1 / (h * h * h));
 double ne = Hp - Hn + 0.25f * Hep + 0.5f * Hepp;
 rho *= code_to_density * pow(a, -3);
 ne *= constants::avo * rho;									// 8
 constexpr float colog = logf(37.8f);
 static const double lambda_e0 = pow(3.0, 1.5) / (4.0 * sqrt(M_PI) * pow(constants::e, 4.) * colog);
 double lambda_e = lambda_e0 * sqr(constants::kb * T) / (ne + 1e-10);
 lambda_e /= get_options().code_to_cm;
 lambda_e /= a;
 if (T > 1e6 && lambda_e > 0.0) {
 //		PRINT("lambda_e %e %e %e\n", lambda_e, lambda_e0, ne);
 }
 return lambda_e;
 }*/

HPX_PLAIN_ACTION (sph_particles_energy_to_entropy);

void sph_particles_energy_to_entropy(float a) {
}

void sph_particles_swap(part_int i, part_int j) {
	static const bool chem = get_options().chem;
	static const bool gravity = get_options().gravity;
	static const bool diff = get_options().diffusion;
	static const bool cond = get_options().conduction;
	static const bool stars = get_options().stars;
	std::swap(sph_particles_r1[i], sph_particles_r1[j]);
	std::swap(sph_particles_r2[i], sph_particles_r2[j]);
	std::swap(sph_particles_r3[i], sph_particles_r3[j]);
	std::swap(sph_particles_r5[i], sph_particles_r5[j]);
	std::swap(sph_particles_r6[i], sph_particles_r6[j]);
	std::swap(sph_particles_kap[i], sph_particles_kap[j]);
	std::swap(sph_particles_dm[i], sph_particles_dm[j]);
}

part_int sph_particles_sort(pair<part_int> rng, fixed32 xmid, int xdim) {
	part_int begin = rng.first;
	part_int end = rng.second;
	part_int lo = begin;
	part_int hi = end;
	part_int j = end;
	while (lo < hi) {
		if (sph_particles_pos(xdim, lo) >= xmid) {
			while (lo != hi) {
				hi--;
				if (sph_particles_pos(xdim, hi) < xmid) {
					sph_particles_swap(lo, hi);
					break;
				}
			}
		}
		lo++;
	}
	return hi;

}

part_int sph_particles_size() {
	return size;
}

template<class T>
void sph_particles_array_resize(T*& ptr, part_int new_capacity, bool reg) {
	T* new_ptr;
#ifdef USE_CUDA
	if( reg ) {
		cudaMallocManaged(&new_ptr,sizeof(T) * new_capacity);
	} else {
		new_ptr = (T*) malloc(sizeof(T) * new_capacity);
	}
#else
	new_ptr = (T*) malloc(sizeof(T) * new_capacity);
	memset(new_ptr, T(0), new_capacity);
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

void sph_particles_resize(part_int sz, bool parts2) {
	const bool chem = get_options().chem;
	const bool stars = get_options().stars;
	if (sz > capacity) {
		part_int new_capacity = std::max(capacity, (part_int) 100);
		while (new_capacity < sz) {
			new_capacity = size_t(101) * new_capacity / size_t(100);
		}
		//	PRINT("Resizing sph_particles to %li from %li\n", new_capacity, capacity);
		sph_particles_array_resize(sph_particles_r1, new_capacity, true);
		sph_particles_array_resize(sph_particles_r2, new_capacity, true);
		sph_particles_array_resize(sph_particles_r3, new_capacity, true);
		sph_particles_array_resize(sph_particles_r5, new_capacity, true);
		sph_particles_array_resize(sph_particles_r6, new_capacity, true);
		sph_particles_array_resize(sph_particles_fp, new_capacity, true);
		sph_particles_array_resize(sph_particles_kap, new_capacity, true);
		sph_particles_array_resize(sph_particles_or, new_capacity, true);
		sph_particles_array_resize(sph_particles_sa, new_capacity, true);
		sph_particles_array_resize(sph_particles_dm, new_capacity, true);
		sph_particles_array_resize(sph_particles_da, new_capacity, true);
		sph_particles_array_resize(sph_particles_e0, new_capacity, true);
		sph_particles_array_resize(sph_particles_de2, new_capacity, true);
		for (int dim = 0; dim < NDIM; dim++) {
			sph_particles_array_resize(sph_particles_dv2[dim], new_capacity, true);
			sph_particles_array_resize(sph_particles_g[dim], new_capacity, true);
		}
		capacity = new_capacity;
	}
	part_int new_parts = sz - size;
	part_int offset = particles_size();
	if (parts2) {
		particles_resize(particles_size() + new_parts);
	}
	int oldsz = size;
	size = sz;
	for (int i = 0; i < new_parts; i++) {
		if (parts2) {
			particles_cat_index(offset + i) = oldsz + i;
			particles_type(offset + i) = SPH_TYPE;
			sph_particles_dm_index(oldsz + i) = offset + i;
		}
		sph_particles_alpha(oldsz + i) = get_options().alpha0;
		sph_particles_rec3(oldsz + i).divv = 0.f;
		sph_particles_semiactive(oldsz + i) = 0.0f;
		if (stars) {
			sph_particles_r2[(oldsz + i)].fcold = 0.f;
		}
		sph_particles_dentr(oldsz + i) = 0.f;
		for (int dim = 0; dim < NDIM; dim++) {
			sph_particles_dvel(dim, oldsz + i) = 0.0f;
			sph_particles_dvel0(dim, oldsz + i) = 0.0f;
			sph_particles_gforce(dim, oldsz + i) = 0.0f;
		}
	}
}

void sph_particles_free() {
	const bool stars = get_options().stars;
	free(sph_particles_dm);
	bool cuda = false;
#ifdef USE_CUDA
#ifdef SPH_TOTAL_ENERGY
	cuda = true;
#endif
#endif
	CUDA_CHECK(cudaFree(sph_particles_r2));
	CUDA_CHECK(cudaFree(sph_particles_r5));
	CUDA_CHECK(cudaFree(sph_particles_r6));
	CUDA_CHECK(cudaFree(sph_particles_r3));
	CUDA_CHECK(cudaFree(sph_particles_r1));
	if (stars) {
		//		CUDA_CHECK(cudaFree(sph_particles_fZ));
//			CUDA_CHECK(cudaFree(sph_particles_fY));
//			CUDA_CHECK(cudaFree(sph_particles_sn));
//			CUDA_CHECK(cudaFree(sph_particles_dz));
	}
}

void sph_particles_resolve_with_particles() {
	const int nthread = hpx_hardware_concurrency();
	std::vector<hpx::future<void>> futs;
	for (int proc = 0; proc < nthread; proc++) {
		futs.push_back(hpx::async([proc, nthread] {
			const part_int b = (size_t) proc * sph_particles_size() / nthread;
			const part_int e = (size_t) (proc + 1) * sph_particles_size() / nthread;
			for( part_int i = b; i < e; i++) {
				const int j = sph_particles_dm_index(i);
				particles_cat_index(j) = i;
			}
		}));
	}
	hpx::wait_all(futs.begin(), futs.end());
}

void sph_particles_global_read_pos(particle_global_range range, fixed32* x, fixed32* y, fixed32* z, part_int offset) {
	const part_int line_size = get_options().part_cache_line_size;
	if (range.range.first != range.range.second) {
		if (range.proc == hpx_rank()) {
			const part_int dif = offset - range.range.first;
			const part_int sz = range.range.second - range.range.first;
			for (part_int i = range.range.first; i < range.range.second; i++) {
				const int j = offset + i - range.range.first;
				x[j] = sph_particles_pos(XDIM, i);
				y[j] = sph_particles_pos(YDIM, i);
				z[j] = sph_particles_pos(ZDIM, i);
			}
		} else {
			line_id_type line_id;
			line_id.proc = range.proc;
			const part_int start_line = (range.range.first / line_size) * line_size;
			const part_int stop_line = ((range.range.second - 1) / line_size) * line_size;
			part_int dest_index = offset;
			for (part_int line = start_line; line <= stop_line; line += line_size) {
				line_id.index = line;
				const auto* ptr = sph_particles_cache_read_line(line_id);
				const auto begin = std::max(line_id.index, range.range.first);
				const auto end = std::min(line_id.index + line_size, range.range.second);
				for (part_int i = begin; i < end; i++) {
					const part_int src_index = i - line_id.index;
					x[dest_index] = ptr[src_index][XDIM];
					y[dest_index] = ptr[src_index][YDIM];
					z[dest_index] = ptr[src_index][ZDIM];
					dest_index++;
				}
			}
		}
	}
}

static const array<fixed32, NDIM>* sph_particles_cache_read_line(line_id_type line_id) {
	const part_int line_size = get_options().part_cache_line_size;
	const size_t bin = line_id_hash_lo()(line_id);
	std::unique_lock<spinlock_type> lock(mutexes[bin]);
	auto iter = part_cache[bin].find(line_id);
	const array<fixed32, NDIM>* ptr;
	if (iter == part_cache[bin].end()) {
		auto prms = std::make_shared<hpx::lcos::local::promise<vector<array<fixed32, NDIM>>> >();
		part_cache[bin][line_id] = prms->get_future();
		lock.unlock();
		hpx::apply([prms,line_id]() {
			auto line_fut = hpx::async<sph_particles_fetch_cache_line_action>(HPX_PRIORITY_HI, hpx_localities()[line_id.proc],line_id.index);
			prms->set_value(line_fut.get());
		});
		lock.lock();
		iter = part_cache[bin].find(line_id);
	}
	auto fut = iter->second;
	lock.unlock();
	return fut.get().data();
}

static vector<array<fixed32, NDIM>> sph_particles_fetch_cache_line(part_int index) {
	const part_int line_size = get_options().part_cache_line_size;
	vector<array<fixed32, NDIM>> line(line_size);
	const part_int begin = (index / line_size) * line_size;
	const part_int end = std::min(sph_particles_size(), begin + line_size);
	for (part_int i = begin; i < end; i++) {
		for (int dim = 0; dim < NDIM; dim++) {
			line[i - begin][dim] = sph_particles_pos(dim, i);
		}
	}
	return line;
}

void sph_particles_global_read_fcold(particle_global_range range, float* cfrac, part_int offset) {
	const part_int line_size = get_options().part_cache_line_size;
	const int sz = offset + range.range.second - range.range.first;
	if (range.range.first != range.range.second) {
		if (range.proc == hpx_rank()) {
			const part_int dif = offset - range.range.first;
			const part_int sz = range.range.second - range.range.first;
			for (part_int i = range.range.first; i < range.range.second; i++) {
				const int j = offset + i - range.range.first;
				cfrac[j] = sph_particles_cold_mass(i);
			}
		} else {
			line_id_type line_id;
			line_id.proc = range.proc;
			const part_int start_line = (range.range.first / line_size) * line_size;
			const part_int stop_line = ((range.range.second - 1) / line_size) * line_size;
			part_int dest_index = offset;
			for (part_int line = start_line; line <= stop_line; line += line_size) {
				line_id.index = line;
				const auto* ptr = sph_particles_fcold_cache_read_line(line_id);
				const auto begin = std::max(line_id.index, range.range.first);
				const auto end = std::min(line_id.index + line_size, range.range.second);
				for (part_int i = begin; i < end; i++) {
					const part_int src_index = i - line_id.index;
					const float& part = ptr[src_index];
					cfrac[dest_index] = part;
					dest_index++;
				}
			}
		}
	}
}

static const float* sph_particles_fcold_cache_read_line(line_id_type line_id) {
	const part_int line_size = get_options().part_cache_line_size;
	const size_t bin = line_id_hash_lo()(line_id);
	std::unique_lock<spinlock_type> lock(fcold_mutexes[bin]);
	auto iter = fcold_part_cache[bin].find(line_id);
	if (iter == fcold_part_cache[bin].end()) {
		auto prms = std::make_shared<hpx::lcos::local::promise<vector<float>> >();
		fcold_part_cache[bin][line_id] = prms->get_future();
		lock.unlock();
		hpx::apply([prms,line_id]() {
			auto line_fut = hpx::async<sph_particles_fetch_fcold_cache_line_action>(HPX_PRIORITY_HI, hpx_localities()[line_id.proc],line_id.index);
			prms->set_value(line_fut.get());
		});
		lock.lock();
		iter = fcold_part_cache[bin].find(line_id);
	}
	auto fut = iter->second;
	lock.unlock();
	return fut.get().data();
}

static vector<float> sph_particles_fetch_fcold_cache_line(part_int index) {
	const part_int line_size = get_options().part_cache_line_size;
	vector<float> line(line_size);
	const part_int begin = (index / line_size) * line_size;
	const part_int end = std::min(sph_particles_size(), begin + line_size);
	for (part_int i = begin; i < end; i++) {
		line[i - begin] = sph_particles_cold_mass(i);
	}
	return line;
}

void sph_particles_global_read_entr(particle_global_range range, float* ent, part_int offset) {
	const part_int line_size = get_options().part_cache_line_size;
	const int sz = offset + range.range.second - range.range.first;
	if (range.range.first != range.range.second) {
		if (range.proc == hpx_rank()) {
			const part_int dif = offset - range.range.first;
			const part_int sz = range.range.second - range.range.first;
			for (part_int i = range.range.first; i < range.range.second; i++) {
				const int j = offset + i - range.range.first;
				ent[j] = sph_particles_entr(i);
			}
		} else {
			line_id_type line_id;
			line_id.proc = range.proc;
			const part_int start_line = (range.range.first / line_size) * line_size;
			const part_int stop_line = ((range.range.second - 1) / line_size) * line_size;
			part_int dest_index = offset;
			for (part_int line = start_line; line <= stop_line; line += line_size) {
				line_id.index = line;
				const auto* ptr = sph_particles_entr_cache_read_line(line_id);
				const auto begin = std::max(line_id.index, range.range.first);
				const auto end = std::min(line_id.index + line_size, range.range.second);
				for (part_int i = begin; i < end; i++) {
					const part_int src_index = i - line_id.index;
					ent[dest_index] = ptr[src_index];
					dest_index++;
				}
			}
		}
	}
}

static const float* sph_particles_entr_cache_read_line(line_id_type line_id) {
	const part_int line_size = get_options().part_cache_line_size;
	const size_t bin = line_id_hash_lo()(line_id);
	std::unique_lock<spinlock_type> lock(entr_mutexes[bin]);
	auto iter = entr_part_cache[bin].find(line_id);
	if (iter == entr_part_cache[bin].end()) {
		auto prms = std::make_shared<hpx::lcos::local::promise<vector<float>> >();
		entr_part_cache[bin][line_id] = prms->get_future();
		lock.unlock();
		hpx::apply([prms,line_id]() {
			auto line_fut = hpx::async<sph_particles_fetch_entr_cache_line_action>(HPX_PRIORITY_HI, hpx_localities()[line_id.proc],line_id.index);
			prms->set_value(line_fut.get());
		});
		lock.lock();
		iter = entr_part_cache[bin].find(line_id);
	}
	auto fut = iter->second;
	lock.unlock();
	return fut.get().data();
}

static vector<float> sph_particles_fetch_entr_cache_line(part_int index) {
	const part_int line_size = get_options().part_cache_line_size;
	vector<float> line(line_size);
	const part_int begin = (index / line_size) * line_size;
	const part_int end = std::min(sph_particles_size(), begin + line_size);
	for (part_int i = begin; i < end; i++) {
		line[i - begin] = sph_particles_entr(i);
	}
	return line;
}

void sph_particles_global_read_rungs(particle_global_range range, char* rungs, part_int offset) {
	const part_int line_size = get_options().part_cache_line_size;
	if (range.range.first != range.range.second) {
		if (range.proc == hpx_rank()) {
			const part_int dif = offset - range.range.first;
			const part_int sz = range.range.second - range.range.first;
			for (part_int i = range.range.first; i < range.range.second; i++) {
				const int j = offset + i - range.range.first;
				rungs[j] = sph_particles_rung(i);
			}
		} else {
			line_id_type line_id;
			line_id.proc = range.proc;
			const part_int start_line = (range.range.first / line_size) * line_size;
			const part_int stop_line = ((range.range.second - 1) / line_size) * line_size;
			part_int dest_index = offset;
			for (part_int line = start_line; line <= stop_line; line += line_size) {
				line_id.index = line;
				const auto* ptr = sph_particles_rung_cache_read_line(line_id);
				const auto begin = std::max(line_id.index, range.range.first);
				const auto end = std::min(line_id.index + line_size, range.range.second);
				for (part_int i = begin; i < end; i++) {
					const part_int src_index = i - line_id.index;
					rungs[dest_index] = ptr[src_index];
					dest_index++;
				}
			}
		}
	}
}

static const char* sph_particles_rung_cache_read_line(line_id_type line_id) {
	const part_int line_size = get_options().part_cache_line_size;
	const size_t bin = line_id_hash_lo()(line_id);
	std::unique_lock<spinlock_type> lock(rung_mutexes[bin]);
	auto iter = rung_part_cache[bin].find(line_id);
	if (iter == rung_part_cache[bin].end()) {
		auto prms = std::make_shared<hpx::lcos::local::promise<vector<char>> >();
		rung_part_cache[bin][line_id] = prms->get_future();
		lock.unlock();
		hpx::apply([prms,line_id]() {
			auto line_fut = hpx::async<sph_particles_fetch_rung_cache_line_action>(HPX_PRIORITY_HI, hpx_localities()[line_id.proc],line_id.index);
			prms->set_value(line_fut.get());
		});
		lock.lock();
		iter = rung_part_cache[bin].find(line_id);
	}
	auto fut = iter->second;
	lock.unlock();
	return fut.get().data();
}

static vector<char> sph_particles_fetch_rung_cache_line(part_int index) {
	const part_int line_size = get_options().part_cache_line_size;
	vector<char> line(line_size);
	const part_int begin = (index / line_size) * line_size;
	const part_int end = std::min(sph_particles_size(), begin + line_size);
	for (part_int i = begin; i < end; i++) {
		line[i - begin] = sph_particles_rung(i);
	}
	return line;
}

void sph_particles_global_read_vels(particle_global_range range, float* vx, float* vy, float* vz, part_int offset) {
	const part_int line_size = get_options().part_cache_line_size;
	if (range.range.first != range.range.second) {
		if (range.proc == hpx_rank()) {
			const part_int dif = offset - range.range.first;
			const part_int sz = range.range.second - range.range.first;
			for (part_int i = range.range.first; i < range.range.second; i++) {
				const int j = offset + i - range.range.first;
				const int k = sph_particles_dm_index(i);
				vx[j] = particles_vel(XDIM, k);
				vy[j] = particles_vel(YDIM, k);
				vz[j] = particles_vel(ZDIM, k);
			}
		} else {
			line_id_type line_id;
			line_id.proc = range.proc;
			const part_int start_line = (range.range.first / line_size) * line_size;
			const part_int stop_line = ((range.range.second - 1) / line_size) * line_size;
			part_int dest_index = offset;
			for (part_int line = start_line; line <= stop_line; line += line_size) {
				line_id.index = line;
				const auto* ptr = sph_particles_vels_cache_read_line(line_id);
				const auto begin = std::max(line_id.index, range.range.first);
				const auto end = std::min(line_id.index + line_size, range.range.second);
				for (part_int i = begin; i < end; i++) {
					const part_int src_index = i - line_id.index;
					vx[dest_index] = ptr[src_index][XDIM];
					vy[dest_index] = ptr[src_index][YDIM];
					vz[dest_index] = ptr[src_index][ZDIM];
					dest_index++;
				}
			}
		}
	}
}

static const array<float, NDIM>* sph_particles_vels_cache_read_line(line_id_type line_id) {
	const part_int line_size = get_options().part_cache_line_size;
	const size_t bin = line_id_hash_lo()(line_id);
	std::unique_lock<spinlock_type> lock(vels_mutexes[bin]);
	auto iter = vels_part_cache[bin].find(line_id);
	if (iter == vels_part_cache[bin].end()) {
		auto prms = std::make_shared<hpx::lcos::local::promise<vector<array<float, NDIM>>> >();
		vels_part_cache[bin][line_id] = prms->get_future();
		lock.unlock();
		hpx::apply([prms,line_id]() {
			auto line_fut = hpx::async<sph_particles_fetch_vels_cache_line_action>(HPX_PRIORITY_HI, hpx_localities()[line_id.proc],line_id.index);
			prms->set_value(line_fut.get());
		});
		lock.lock();
		iter = vels_part_cache[bin].find(line_id);
	}
	auto fut = iter->second;
	lock.unlock();
	return fut.get().data();
}

static vector<array<float, NDIM>> sph_particles_fetch_vels_cache_line(part_int index) {
	const part_int line_size = get_options().part_cache_line_size;
	vector<array<float, NDIM>> line(line_size);
	const part_int begin = (index / line_size) * line_size;
	const part_int end = std::min(sph_particles_size(), begin + line_size);
	for (part_int i = begin; i < end; i++) {
		const int k = sph_particles_dm_index(i);
		line[i - begin][XDIM] = particles_vel(k, XDIM);
		line[i - begin][YDIM] = particles_vel(k, YDIM);
		line[i - begin][ZDIM] = particles_vel(k, ZDIM);
	}
	return line;
}

void sph_particles_global_read_kappas(particle_global_range range, float* kappas, part_int offset) {
	const part_int line_size = get_options().part_cache_line_size;
	if (range.range.first != range.range.second) {
		if (range.proc == hpx_rank()) {
			const part_int dif = offset - range.range.first;
			const part_int sz = range.range.second - range.range.first;
			for (part_int i = range.range.first; i < range.range.second; i++) {
				const int j = offset + i - range.range.first;
				kappas[j] = sph_particles_kappa(i);
			}
		} else {
			line_id_type line_id;
			line_id.proc = range.proc;
			const part_int start_line = (range.range.first / line_size) * line_size;
			const part_int stop_line = ((range.range.second - 1) / line_size) * line_size;
			part_int dest_index = offset;
			for (part_int line = start_line; line <= stop_line; line += line_size) {
				line_id.index = line;
				const auto* ptr = sph_particles_kappas_cache_read_line(line_id);
				const auto begin = std::max(line_id.index, range.range.first);
				const auto end = std::min(line_id.index + line_size, range.range.second);
				for (part_int i = begin; i < end; i++) {
					const part_int src_index = i - line_id.index;
					kappas[dest_index] = ptr[src_index];
					dest_index++;
				}
			}
		}
	}
}

static const float* sph_particles_kappas_cache_read_line(line_id_type line_id) {
	const part_int line_size = get_options().part_cache_line_size;
	const size_t bin = line_id_hash_lo()(line_id);
	std::unique_lock<spinlock_type> lock(kappas_mutexes[bin]);
	auto iter = kappas_part_cache[bin].find(line_id);
	if (iter == kappas_part_cache[bin].end()) {
		auto prms = std::make_shared<hpx::lcos::local::promise<vector<float>> >();
		kappas_part_cache[bin][line_id] = prms->get_future();
		lock.unlock();
		hpx::apply([prms,line_id]() {
			auto line_fut = hpx::async<sph_particles_fetch_kappas_cache_line_action>(HPX_PRIORITY_HI, hpx_localities()[line_id.proc],line_id.index);
			prms->set_value(line_fut.get());
		});
		lock.lock();
		iter = kappas_part_cache[bin].find(line_id);
	}
	auto fut = iter->second;
	lock.unlock();
	return fut.get().data();
}

static vector<float> sph_particles_fetch_kappas_cache_line(part_int index) {
	const part_int line_size = get_options().part_cache_line_size;
	vector<float> line(line_size);
	const part_int begin = (index / line_size) * line_size;
	const part_int end = std::min(sph_particles_size(), begin + line_size);
	for (part_int i = begin; i < end; i++) {
		line[i - begin] = sph_particles_kappa(i);
	}
	return line;
}

void sph_particles_global_read_aux(particle_global_range range, float* h, float* alpha, float* pre, float* fpre1, float* fpre2, float* shearv,
		array<float, NCHEMFRACS>* fracs, float* fpot, part_int offset) {
	const part_int line_size = get_options().part_cache_line_size;
	if (range.range.first != range.range.second) {
		if (range.proc == hpx_rank()) {
			const part_int dif = offset - range.range.first;
			const part_int sz = range.range.second - range.range.first;
			for (part_int i = range.range.first; i < range.range.second; i++) {
				const int j = offset + i - range.range.first;
				if (h) {
					h[j] = sph_particles_smooth_len(i);
				}
				if (alpha) {
					alpha[j] = sph_particles_alpha(i);
				}
				if (fpre1) {
					fpre1[j] = sph_particles_fpre1(i);
				}
				if (fpre2) {
					fpre2[j] = sph_particles_fpre2(i);
				}
				if (pre) {
					pre[j] = sph_particles_pre(i);
				}
				if (shearv) {
					shearv[j] = sph_particles_shear(i);
				}
				if (fracs) {
					fracs[j] = sph_particles_chem(i);
				}
				if (fpot) {
					fpot[j] = sph_particles_fpot(i);
				}

			}
		}
	} else {
		line_id_type line_id;
		line_id.proc = range.proc;
		const part_int start_line = (range.range.first / line_size) * line_size;
		const part_int stop_line = ((range.range.second - 1) / line_size) * line_size;
		part_int dest_index = offset;
		for (part_int line = start_line; line <= stop_line; line += line_size) {
			line_id.index = line;
			const auto* ptr = sph_particles_aux_cache_read_line(line_id);
			const auto begin = std::max(line_id.index, range.range.first);
			const auto end = std::min(line_id.index + line_size, range.range.second);
			for (part_int i = begin; i < end; i++) {
				const part_int src_index = i - line_id.index;
				const int j = dest_index;
				if (h) {
					h[j] = ptr[src_index].h;
				}
				if (alpha) {
					alpha[j] = ptr[src_index].alpha;
				}
				if (fpre1) {
					fpre1[j] = ptr[src_index].fpre1;
				}
				if (fpre2) {
					fpre2[j] = ptr[src_index].fpre2;
				}
				if (pre) {
					pre[j] = ptr[src_index].pre;
				}
				if (shearv) {
					shearv[j] = ptr[src_index].shearv;
				}
				if (fracs) {
					fracs[j] = ptr[src_index].fracs;
				}
				if (fpot) {
					fpot[j] = ptr[src_index].fpot;
				}
				dest_index++;
			}
		}
	}
}

static const aux_quantities* sph_particles_aux_cache_read_line(line_id_type line_id) {
	const part_int line_size = get_options().part_cache_line_size;
	const size_t bin = line_id_hash_lo()(line_id);
	std::unique_lock<spinlock_type> lock(aux_mutexes[bin]);
	auto iter = aux_cache[bin].find(line_id);
	if (iter == aux_cache[bin].end()) {
		auto prms = std::make_shared<hpx::lcos::local::promise<vector<aux_quantities>> >();
		aux_cache[bin][line_id] = prms->get_future();
		lock.unlock();
		hpx::apply([prms,line_id]() {
			auto line_fut = hpx::async<sph_particles_fetch_aux_cache_line_action>(HPX_PRIORITY_HI, hpx_localities()[line_id.proc],line_id.index);
			prms->set_value(line_fut.get());
		});
		lock.lock();
		iter = aux_cache[bin].find(line_id);
	}
	auto fut = iter->second;
	lock.unlock();
	return fut.get().data();
}

static vector<aux_quantities> sph_particles_fetch_aux_cache_line(part_int index) {
	const part_int line_size = get_options().part_cache_line_size;
	vector<aux_quantities> line(line_size);
	const part_int begin = (index / line_size) * line_size;
	const part_int end = std::min(sph_particles_size(), begin + line_size);
	for (part_int i = begin; i < end; i++) {
		line[i - begin] = sph_particles_aux_quantities(i);

	}
	return line;
}
/*
 void sph_particles_global_read_sns(particle_global_range range, float* sn, part_int offset) {
 const part_int line_size = get_options().part_cache_line_size;
 if (range.range.first != range.range.second) {
 if (range.proc == hpx_rank()) {
 const part_int dif = offset - range.range.first;
 const part_int sz = range.range.second - range.range.first;
 for (part_int i = range.range.first; i < range.range.second; i++) {
 const int j = offset + i - range.range.first;
 sn[j] = sph_particles_SN(i);
 }
 } else {
 line_id_type line_id;
 line_id.proc = range.proc;
 const part_int start_line = (range.range.first / line_size) * line_size;
 const part_int stop_line = ((range.range.second - 1) / line_size) * line_size;
 part_int dest_index = offset;
 for (part_int line = start_line; line <= stop_line; line += line_size) {
 line_id.index = line;
 const auto* ptr = sph_particles_sn_cache_read_line(line_id);
 const auto begin = std::max(line_id.index, range.range.first);
 const auto end = std::min(line_id.index + line_size, range.range.second);
 for (part_int i = begin; i < end; i++) {
 const part_int src_index = i - line_id.index;
 sn[dest_index] = ptr[src_index];
 dest_index++;
 }
 }
 }
 }
 }
 static const float* sph_particles_sn_cache_read_line(line_id_type line_id) {
 const part_int line_size = get_options().part_cache_line_size;
 const size_t bin = line_id_hash_lo()(line_id);
 std::unique_lock<spinlock_type> lock(fvel_mutexes[bin]);
 auto iter = sn_cache[bin].find(line_id);
 if (iter == sn_cache[bin].end()) {
 auto prms = std::make_shared<hpx::lcos::local::promise<vector<float>> >();
 sn_cache[bin][line_id] = prms->get_future();
 lock.unlock();
 hpx::apply([prms,line_id]() {
 auto line_fut = hpx::async<sph_particles_fetch_sn_cache_line_action>(HPX_PRIORITY_HI, hpx_localities()[line_id.proc],line_id.index);
 prms->set_value(line_fut.get());
 });
 lock.lock();
 iter = sn_cache[bin].find(line_id);
 }
 auto fut = iter->second;
 lock.unlock();
 return fut.get().data();
 }
 static vector<float> sph_particles_fetch_sn_cache_line(part_int index) {
 const part_int line_size = get_options().part_cache_line_size;
 vector<float> line(line_size);
 const part_int begin = (index / line_size) * line_size;
 const part_int end = std::min(sph_particles_size(), begin + line_size);
 for (part_int i = begin; i < end; i++) {
 line[i - begin] = sph_particles_SN(i);
 }
 return line;
 }
 */

void sph_particles_cache_free1() {
	profiler_enter(__FUNCTION__);
	static const auto stars = get_options().stars;
	vector<hpx::future<void>> futs;
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async<sph_particles_cache_free1_action>(HPX_PRIORITY_HI, c));
	}
	const int nthreads = hpx_hardware_concurrency();
	entr_part_cache = decltype(entr_part_cache)();
	vels_part_cache = decltype(vels_part_cache)();
	aux_cache = decltype(aux_cache)();
	hpx::wait_all(futs.begin(), futs.end());
	profiler_exit();
}


void sph_particles_cache_free_entr() {
	profiler_enter(__FUNCTION__);
	static const auto stars = get_options().stars;
	vector<hpx::future<void>> futs;
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async<sph_particles_cache_free1_action>(HPX_PRIORITY_HI, c));
	}
	const int nthreads = hpx_hardware_concurrency();
	entr_part_cache = decltype(entr_part_cache)();
	hpx::wait_all(futs.begin(), futs.end());
	profiler_exit();
}

void sph_particles_cache_free2() {
	profiler_enter(__FUNCTION__);
	static const auto stars = get_options().stars;
	vector<hpx::future<void>> futs;
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async<sph_particles_cache_free2_action>(HPX_PRIORITY_HI, c));
	}
	const int nthreads = hpx_hardware_concurrency();
	part_cache = decltype(part_cache)();
	entr_part_cache = decltype(entr_part_cache)();
	rung_part_cache = decltype(rung_part_cache)();
	fcold_part_cache = decltype(fcold_part_cache)();
	vels_part_cache = decltype(vels_part_cache)();
	kappas_part_cache = decltype(kappas_part_cache)();
	aux_cache = decltype(aux_cache)();
	hpx::wait_all(futs.begin(), futs.end());
	profiler_exit();
}

void sph_particles_load(FILE* fp) {
	const bool chem = get_options().chem;
	const bool diff = get_options().diffusion;
	const bool cond = get_options().conduction;
	const bool stars = get_options().stars;
	FREAD(sph_particles_r1, sizeof(sph_record1), sph_particles_size(), fp);
	FREAD(sph_particles_r2, sizeof(sph_record2), sph_particles_size(), fp);
	FREAD(sph_particles_r3, sizeof(sph_record3), sph_particles_size(), fp);
	FREAD(sph_particles_r5, sizeof(sph_record5), sph_particles_size(), fp);
	FREAD(sph_particles_r6, sizeof(sph_record6), sph_particles_size(), fp);
	FREAD(sph_particles_kap, sizeof(float), sph_particles_size(), fp);
	FREAD(&sph_particles_dm_index(0), sizeof(part_int), sph_particles_size(), fp);
	if (stars) {
		stars_load(fp);
	}
}

void sph_particles_save(FILE* fp) {
	const bool chem = get_options().chem;
	const bool diff = get_options().diffusion;
	const bool cond = get_options().conduction;
	const bool stars = get_options().stars;
	fwrite(sph_particles_r1, sizeof(sph_record1), sph_particles_size(), fp);
	fwrite(sph_particles_r2, sizeof(sph_record2), sph_particles_size(), fp);
	fwrite(sph_particles_r3, sizeof(sph_record3), sph_particles_size(), fp);
	fwrite(sph_particles_r5, sizeof(sph_record5), sph_particles_size(), fp);
	fwrite(sph_particles_r6, sizeof(sph_record6), sph_particles_size(), fp);
	fwrite(sph_particles_kap, sizeof(float), sph_particles_size(), fp);
	fwrite(&sph_particles_dm_index(0), sizeof(part_int), sph_particles_size(), fp);
	if (stars) {
		stars_save(fp);
	}
}

HPX_PLAIN_ACTION (sph_particles_reset_converged);

void sph_particles_reset_converged() {
	const int nthread = hpx_hardware_concurrency();
	std::vector<hpx::future<void>> futs;
	for (auto& c : hpx_children()) {
		futs.push_back(hpx::async<sph_particles_reset_converged_action>(c));
	}
	for (int proc = 0; proc < nthread; proc++) {
		futs.push_back(hpx::async([proc, nthread] {
			const part_int b = (size_t) proc * sph_particles_size() / nthread;
			const part_int e = (size_t) (proc + 1) * sph_particles_size() / nthread;
			for( part_int i = b; i < e; i++) {
				sph_particles_converged(i) = false;
				sph_particles_semiactive(i) = false;
			}
		}));
	}
	hpx::wait_all(futs.begin(), futs.end());
}

