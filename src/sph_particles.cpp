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
static const array<float, NDIM>* sph_particles_gforce_cache_read_line(line_id_type line_id);
static const array<float, NDIM + 1>* sph_particles_force_cache_read_line(line_id_type line_id);
static const sph_particle* sph_particles_sph_cache_read_line(line_id_type line_id, float);
static const sph_particle0* sph_particles_sph0_cache_read_line(line_id_type line_id);
static const pair<char, float>* sph_particles_rung_cache_read_line(line_id_type line_id);
static const aux_quantities* sph_particles_aux_cache_read_line(line_id_type line_id);

static vector<array<fixed32, NDIM>> sph_particles_fetch_cache_line(part_int index);
static vector<array<float, NDIM>> sph_particles_fetch_gforce_cache_line(part_int index);
static vector<array<float, NDIM + 1>> sph_particles_fetch_force_cache_line(part_int index);
static vector<pair<char, float>> sph_particles_fetch_rung_cache_line(part_int index);
static vector<sph_particle> sph_particles_fetch_sph_cache_line(part_int index, float);
static vector<sph_particle0> sph_particles_fetch_sph0_cache_line(part_int index);
static vector<aux_quantities> sph_particles_fetch_aux_cache_line(part_int index);

HPX_PLAIN_ACTION (sph_particles_fetch_cache_line);
HPX_PLAIN_ACTION (sph_particles_fetch_gforce_cache_line);
HPX_PLAIN_ACTION (sph_particles_fetch_force_cache_line);
HPX_PLAIN_ACTION (sph_particles_fetch_sph_cache_line);
HPX_PLAIN_ACTION (sph_particles_fetch_sph0_cache_line);
HPX_PLAIN_ACTION (sph_particles_fetch_rung_cache_line);
HPX_PLAIN_ACTION (sph_particles_cache_free);
HPX_PLAIN_ACTION (sph_particles_fetch_aux_cache_line);

static array<std::unordered_map<line_id_type, hpx::shared_future<vector<array<fixed32, NDIM>>> , line_id_hash_hi>,PART_CACHE_SIZE> part_cache;
static array<std::unordered_map<line_id_type, hpx::shared_future<vector<array<float, NDIM>>> , line_id_hash_hi>,PART_CACHE_SIZE> gforce_cache;
static array<std::unordered_map<line_id_type, hpx::shared_future<vector<array<float, NDIM + 1>>> , line_id_hash_hi>,PART_CACHE_SIZE> force_cache;
static array<std::unordered_map<line_id_type, hpx::shared_future<vector<sph_particle>>, line_id_hash_hi>, PART_CACHE_SIZE> sph_part_cache;
static array<std::unordered_map<line_id_type, hpx::shared_future<vector<sph_particle0>>, line_id_hash_hi>, PART_CACHE_SIZE> sph0_part_cache;
static array<std::unordered_map<line_id_type, hpx::shared_future<vector<pair<char, float>>> , line_id_hash_hi>, PART_CACHE_SIZE> rung_part_cache;
static array<std::unordered_map<line_id_type, hpx::shared_future<vector<aux_quantities>>, line_id_hash_hi>, PART_CACHE_SIZE> aux_cache;
static array<spinlock_type, PART_CACHE_SIZE> mutexes;
static array<spinlock_type, PART_CACHE_SIZE> sph_mutexes;
static array<spinlock_type, PART_CACHE_SIZE> sph0_mutexes;
static array<spinlock_type, PART_CACHE_SIZE> rung_mutexes;
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
	return maxh;
}

HPX_PLAIN_ACTION (sph_particles_apply_updates);
std::pair<double, double> sph_particles_apply_updates(int minrung, int phase, float t0, float tau, float w) {

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
			double error = 0.0;
			double norm = 0.0;
			const part_int b = (size_t) proc * sph_particles_size() / nthreads;
			const part_int e = (size_t) (proc+1) * sph_particles_size() / nthreads;
			for( int i = b; i < e; i++) {
				const int k = sph_particles_dm_index(i);
				const auto rung2 = particles_rung(k);
				const auto rung1 = sph_particles_oldrung(i);
				const float dt2 = 0.5f * t0 / (1<<rung2);
				const float dt1 = 0.5f * t0 / (1<<rung1);
				if( rung2 >= minrung) {
					switch(phase) {
						case 0: {
							sph_particles_eint(i) +=sph_particles_deint_pred(i) *dt2;
							for( int dim =0; dim < NDIM; dim++) {
								particles_vel(dim,k) += sph_particles_dvel_pred(dim,i)* dt2;
							}
						}
						break;
						case 1: {
							sph_particles_eint(i) -= sph_particles_deint_pred(i) *dt2;
							for( int dim =0; dim < NDIM; dim++) {
								particles_vel(dim,k) -= sph_particles_dvel_pred(dim,i)* dt2;
							}
							sph_particles_eint(i) += sph_particles_deint_con(i) *dt2;
							for( int dim =0; dim < NDIM; dim++) {
								particles_vel(dim,k) += sph_particles_dvel_con(dim,i)* dt2;
							}
						}
						break;
						case 2: {
							sph_particles_deint_pred(i) = sph_particles_deint_con(i);
							sph_particles_eint(i) +=sph_particles_deint_con(i) *dt2;
							for( int dim =0; dim < NDIM; dim++) {
								sph_particles_dvel_pred(dim,i) = sph_particles_dvel_con(dim,i);
								particles_vel(dim,k) += sph_particles_dvel_con(dim,i)* dt2;
							}
						}
						break;
					}
				}
//				sph_particles_alpha(i) = std::max(sph_particles_alpha(i), SPH_ALPHA0);
//				sph_particles_alpha(i) = std::min(sph_particles_alpha(i), SPH_ALPHA1);
			}
			return std::make_pair(error,norm);
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

float sph_particles_coloumb_log(part_int i, float a) {
	const double code_to_energy_density = get_options().code_to_g / (get_options().code_to_cm * sqr(get_options().code_to_s));		// 7
	const double code_to_density = pow(get_options().code_to_cm, -3) * get_options().code_to_g;										// 10
	const double h = sph_particles_smooth_len(i);
	const double Hp = sph_particles_Hp(i);
	const double Hn = sph_particles_Hn(i);
	const double Hep = sph_particles_Hep(i);
	const double Hepp = sph_particles_Hepp(i);
	double rho = sph_den(1 / (h * h * h));
	double ne = Hp - Hn + Hep + 2.0f * Hepp;
	rho *= code_to_density * pow(a, -3);
	ne *= constants::avo * rho;
	ne = std::max(ne, 1e-20);
	double T = std::max(sph_particles_temperature(i, a), 1000.f);
	T = std::max(T, 1.0);
	const double part1 = 23.5;
	const double part2 = -log(sqrt(ne) * pow(T, -1.2));
	const double part3 = -sqrt((1e-5 + sqr(log(T) - 2)) / 16.0);
	return part1 + part2 + part3;
}

float sph_particles_temperature(part_int i, float a) {
	const double code_to_energy_density = get_options().code_to_g / (get_options().code_to_cm * sqr(get_options().code_to_s));		// 7
	const double code_to_energy = sqr(get_options().code_to_cm) / (sqr(get_options().code_to_s));		// 7
	const double code_to_density = pow(get_options().code_to_cm, -3) * get_options().code_to_g;										// 10
	const double h = sph_particles_smooth_len(i);
	const double Hp = sph_particles_Hp(i);
	const double Hn = sph_particles_Hn(i);
	const double H2 = sph_particles_H2(i);
	const double Y = sph_particles_Y(i);
	const double Hep = sph_particles_Hep(i);
	const double Hepp = sph_particles_Hepp(i);
	const double H = 1.0 - Y - Hp - Hn - H2;
	const double He = Y - Hep - Hepp;
	double rho = sph_den(1 / (h * h * h));
	double n = H + 2.f * Hp + .5f * H2 + .25f * He + .5f * Hep + .75f * Hepp;
	rho *= code_to_density * pow(a, -3);
	n *= constants::avo * rho;									// 8
	double gamma = sph_particles_gamma(i);
	double cv = 1.0 / (gamma - 1.0);															// 4
	cv *= double(constants::kb);																							// 1
	double eint = sph_particles_eint(i);
	eint /= sqr(a);
	eint *= code_to_energy;
	double T = rho * eint / (n * cv);
	if (H < 0.0) {
		PRINT("NEGATIVE H\n");
		PRINT("%e %e %e %e %e %e %e\n", H, Hp, Hn, H2, He, Hep, Hepp);
		//	abort();
	}
	if (T > TMAX) {
		PRINT("T == %e %e %e %e %e %e\n", T, sph_particles_eint(i), eint, eint, rho, h);
		abort();
	}
	if (T < 0.0) {
		PRINT("T == %e %e %e %e %e %e\n", T, sph_particles_eint(i), eint, eint, rho, h);
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
	const double H = 1.0 - Y - Hp - Hn - 2.0 * H2;
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
	static const bool diff = get_options().diffusion;
	static const bool cond = get_options().conduction;
	static const bool stars = get_options().stars;
	static const bool xsph = get_options().xsph != 0.0;
	std::swap(sph_particles_a[i], sph_particles_a[j]);
	std::swap(sph_particles_e[i], sph_particles_e[j]);
	std::swap(sph_particles_de1[i], sph_particles_de1[j]);
	std::swap(sph_particles_h[i], sph_particles_h[j]);
	std::swap(sph_particles_dm[i], sph_particles_dm[j]);
	std::swap(sph_particles_fp[i], sph_particles_fp[j]);
	std::swap(sph_particles_f0[i], sph_particles_f0[j]);
	std::swap(sph_particles_dvv[i], sph_particles_dvv[j]);
	std::swap(sph_particles_dvv0[i], sph_particles_dvv0[j]);
	std::swap(sph_particles_s2[i], sph_particles_s2[j]);
	std::swap(sph_particles_cv[i], sph_particles_cv[j]);
	if (cond) {
		std::swap(sph_particles_gt[i], sph_particles_gt[j]);
	}
	if (diff) {
		std::swap(sph_particles_ea[i], sph_particles_ea[j]);
	}
	for (int dim = 0; dim < NDIM; dim++) {
		std::swap(sph_particles_dv1[dim][i], sph_particles_dv1[dim][j]);
		if (xsph) {
			std::swap(sph_particles_dvx[dim][i], sph_particles_dvx[dim][j]);
		}
	}
	if (chem) {
		std::swap(sph_particles_c0[i], sph_particles_c0[j]);
	}
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
	const bool xsph = get_options().xsph != 0.0;
	if (sz > capacity) {
		part_int new_capacity = std::max(capacity, (part_int) 100);
		while (new_capacity < sz) {
			new_capacity = size_t(101) * new_capacity / size_t(100);
		}
		//	PRINT("Resizing sph_particles to %li from %li\n", new_capacity, capacity);
		sph_particles_array_resize(sph_particles_ea, new_capacity, true);
		sph_particles_array_resize(sph_particles_e0, new_capacity, true);
		sph_particles_array_resize(sph_particles_s2, new_capacity, true);
		sph_particles_array_resize(sph_particles_cv, new_capacity, true);
		sph_particles_array_resize(sph_particles_or, new_capacity, true);
		sph_particles_array_resize(sph_particles_dm, new_capacity, false);
		sph_particles_array_resize(sph_particles_e, new_capacity, true);
		sph_particles_array_resize(sph_particles_h, new_capacity, true);
		sph_particles_array_resize(sph_particles_fp, new_capacity, true);
		sph_particles_array_resize(sph_particles_dvv0, new_capacity, true);
		sph_particles_array_resize(sph_particles_de1, new_capacity, true);
		sph_particles_array_resize(sph_particles_de2, new_capacity, true);
		sph_particles_array_resize(sph_particles_sa, new_capacity, true);
		sph_particles_array_resize(sph_particles_f0, new_capacity, true);
		sph_particles_array_resize(sph_particles_dvv, new_capacity, true);
		sph_particles_array_resize(sph_particles_gt, new_capacity, true);
		sph_particles_array_resize(sph_particles_a, new_capacity, true);
		for (int dim = 0; dim < NDIM; dim++) {
			sph_particles_array_resize(sph_particles_dv1[dim], new_capacity, false);
			sph_particles_array_resize(sph_particles_dv2[dim], new_capacity, true);
			sph_particles_array_resize(sph_particles_g[dim], new_capacity, true);
			if (xsph) {
				sph_particles_array_resize(sph_particles_dvx[dim], new_capacity, true);
			}
		}
		if (chem) {
			sph_particles_array_resize(sph_particles_c0, new_capacity, true);
			sph_particles_array_resize(sph_particles_dchem2, new_capacity, true);
			sph_particles_array_resize(sph_particles_dchem1, new_capacity, true);
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
		sph_particles_deint_con(oldsz + i) = 0.0f;
		sph_particles_deint_pred(oldsz + i) = 0.0f;
		sph_particles_alpha(oldsz + i) = get_options().alpha0;
		sph_particles_fpre(oldsz + i) = 1.f;
		sph_particles_divv(oldsz + i) = 0.f;
		for (int dim = 0; dim < NDIM; dim++) {
			sph_particles_gforce(dim, oldsz + i) = 0.0f;
			sph_particles_dvel_con(dim, oldsz + i) = 0.0f;
			sph_particles_dvel_pred(dim, oldsz + i) = 0.0f;
			if (xsph) {
				sph_particles_xvel(dim, oldsz + i) = 0.0f;
			}
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
	if (cuda) {
#ifdef USE_CUDA
		CUDA_CHECK(cudaFree(sph_particles_e));
		CUDA_CHECK(cudaFree(sph_particles_de1));
		CUDA_CHECK(cudaFree(sph_particles_de2));
		CUDA_CHECK(cudaFree(sph_particles_dvv));
		CUDA_CHECK(cudaFree(sph_particles_sa));
		CUDA_CHECK(cudaFree(sph_particles_f0));
		CUDA_CHECK(cudaFree(sph_particles_a));
		CUDA_CHECK(cudaFree(sph_particles_fp));
		if( stars ) {
			//		CUDA_CHECK(cudaFree(sph_particles_fZ));
//			CUDA_CHECK(cudaFree(sph_particles_fY));
//			CUDA_CHECK(cudaFree(sph_particles_sn));
//			CUDA_CHECK(cudaFree(sph_particles_dz));
		}
		CUDA_CHECK(cudaFree(sph_particles_h));
		for (int dim = 0; dim < NDIM; dim++) {
			CUDA_CHECK(cudaFree(sph_particles_dv1[NDIM]));
			CUDA_CHECK(cudaFree(sph_particles_dv2[NDIM]));
			CUDA_CHECK(cudaFree(sph_particles_g[NDIM]));
		}
#endif
	} else {
		free(sph_particles_h);
		free(sph_particles_e);
		free(sph_particles_fp);
		free(sph_particles_de1);
		free(sph_particles_de2);
		free(sph_particles_dvv);
		free(sph_particles_sa);
		free(sph_particles_f0);
		free(sph_particles_a);
		for (int dim = 0; dim < NDIM; dim++) {
			free(sph_particles_dv1[NDIM]);
			free(sph_particles_dv2[NDIM]);
			free(sph_particles_g[NDIM]);
		}
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

void sph_particles_global_read_gforce(particle_global_range range, float* gx, float* gy, float* gz, part_int offset) {
	const part_int line_size = get_options().part_cache_line_size;
	if (range.range.first != range.range.second) {
		if (range.proc == hpx_rank()) {
			const part_int dif = offset - range.range.first;
			const part_int sz = range.range.second - range.range.first;
			for (part_int i = range.range.first; i < range.range.second; i++) {
				const int j = offset + i - range.range.first;
				gx[j] = sph_particles_gforce(XDIM, i);
				gy[j] = sph_particles_gforce(YDIM, i);
				gz[j] = sph_particles_gforce(ZDIM, i);
			}
		} else {
			line_id_type line_id;
			line_id.proc = range.proc;
			const part_int start_line = (range.range.first / line_size) * line_size;
			const part_int stop_line = ((range.range.second - 1) / line_size) * line_size;
			part_int dest_index = offset;
			for (part_int line = start_line; line <= stop_line; line += line_size) {
				line_id.index = line;
				const auto* ptr = sph_particles_gforce_cache_read_line(line_id);
				const auto begin = std::max(line_id.index, range.range.first);
				const auto end = std::min(line_id.index + line_size, range.range.second);
				for (part_int i = begin; i < end; i++) {
					const part_int src_index = i - line_id.index;
					gx[dest_index] = ptr[src_index][XDIM];
					gy[dest_index] = ptr[src_index][YDIM];
					gz[dest_index] = ptr[src_index][ZDIM];
					dest_index++;
				}
			}
		}
	}
}

static const array<float, NDIM>* sph_particles_gforce_cache_read_line(line_id_type line_id) {
	const part_int line_size = get_options().part_cache_line_size;
	const size_t bin = line_id_hash_lo()(line_id);
	std::unique_lock<spinlock_type> lock(mutexes[bin]);
	auto iter = gforce_cache[bin].find(line_id);
	const array<float, NDIM>* ptr;
	if (iter == gforce_cache[bin].end()) {
		auto prms = std::make_shared<hpx::lcos::local::promise<vector<array<float, NDIM>>> >();
		gforce_cache[bin][line_id] = prms->get_future();
		lock.unlock();
		hpx::apply([prms,line_id]() {
			auto line_fut = hpx::async<sph_particles_fetch_gforce_cache_line_action>(HPX_PRIORITY_HI, hpx_localities()[line_id.proc],line_id.index);
			prms->set_value(line_fut.get());
		});
		lock.lock();
		iter = gforce_cache[bin].find(line_id);
	}
	auto fut = iter->second;
	lock.unlock();
	return fut.get().data();
}

static vector<array<float, NDIM>> sph_particles_fetch_gforce_cache_line(part_int index) {
	const part_int line_size = get_options().part_cache_line_size;
	vector<array<float, NDIM>> line(line_size);
	const part_int begin = (index / line_size) * line_size;
	const part_int end = std::min(sph_particles_size(), begin + line_size);
	for (part_int i = begin; i < end; i++) {
		for (int dim = 0; dim < NDIM; dim++) {
			line[i - begin][dim] = sph_particles_gforce(dim, i);
		}
	}
	return line;
}

void sph_particles_global_read_force(particle_global_range range, float* ax, float* ay, float* az, float* divv, part_int offset) {
	const part_int line_size = get_options().part_cache_line_size;
	if (range.range.first != range.range.second) {
		if (range.proc == hpx_rank()) {
			const part_int dif = offset - range.range.first;
			const part_int sz = range.range.second - range.range.first;
			for (part_int i = range.range.first; i < range.range.second; i++) {
				const int j = offset + i - range.range.first;
				if (ax) {
					ax[j] = sph_particles_dvel_con(XDIM, i);
					ay[j] = sph_particles_dvel_con(YDIM, i);
					az[j] = sph_particles_dvel_con(ZDIM, i);
				}
				divv[j] = sph_particles_divv(i);
			}
		} else {
			line_id_type line_id;
			line_id.proc = range.proc;
			const part_int start_line = (range.range.first / line_size) * line_size;
			const part_int stop_line = ((range.range.second - 1) / line_size) * line_size;
			part_int dest_index = offset;
			for (part_int line = start_line; line <= stop_line; line += line_size) {
				line_id.index = line;
				const auto* ptr = sph_particles_force_cache_read_line(line_id);
				const auto begin = std::max(line_id.index, range.range.first);
				const auto end = std::min(line_id.index + line_size, range.range.second);
				for (part_int i = begin; i < end; i++) {
					const part_int src_index = i - line_id.index;
					if (ax) {
						ax[dest_index] = ptr[src_index][XDIM];
						ay[dest_index] = ptr[src_index][YDIM];
						az[dest_index] = ptr[src_index][ZDIM];
					}
					divv[dest_index] = ptr[src_index][NDIM];
					dest_index++;
				}
			}
		}
	}
}

static const array<float, NDIM + 1>* sph_particles_force_cache_read_line(line_id_type line_id) {
	const part_int line_size = get_options().part_cache_line_size;
	const size_t bin = line_id_hash_lo()(line_id);
	std::unique_lock<spinlock_type> lock(mutexes[bin]);
	auto iter = force_cache[bin].find(line_id);
	const array<float, NDIM + 1>* ptr;
	if (iter == force_cache[bin].end()) {
		auto prms = std::make_shared<hpx::lcos::local::promise<vector<array<float, NDIM + 1>>> >();
		force_cache[bin][line_id] = prms->get_future();
		lock.unlock();
		hpx::apply([prms,line_id]() {
			auto line_fut = hpx::async<sph_particles_fetch_force_cache_line_action>(HPX_PRIORITY_HI, hpx_localities()[line_id.proc],line_id.index);
			prms->set_value(line_fut.get());
		});
		lock.lock();
		iter = force_cache[bin].find(line_id);
	}
	auto fut = iter->second;
	lock.unlock();
	return fut.get().data();
}

static vector<array<float, NDIM + 1>> sph_particles_fetch_force_cache_line(part_int index) {
	const part_int line_size = get_options().part_cache_line_size;
	vector<array<float, NDIM + 1>> line(line_size);
	const part_int begin = (index / line_size) * line_size;
	const part_int end = std::min(sph_particles_size(), begin + line_size);
	for (part_int i = begin; i < end; i++) {
		for (int dim = 0; dim < NDIM; dim++) {
			line[i - begin][dim] = sph_particles_dvel_con(dim, i);
		}
		line[i - begin][NDIM] = sph_particles_divv(i);
	}
	return line;
}

void sph_particles_global_read_sph(particle_global_range range, float a, float* ent, float* vx, float* vy, float* vz, float* gamma, float* alpha, float*mmw,
		array<float, NCHEMFRACS>* chems, part_int offset) {
	const part_int line_size = get_options().part_cache_line_size;
	const int sz = offset + range.range.second - range.range.first;
	if (range.range.first != range.range.second) {
		if (range.proc == hpx_rank()) {
			const part_int dif = offset - range.range.first;
			const part_int sz = range.range.second - range.range.first;
			for (part_int i = range.range.first; i < range.range.second; i++) {
				const int j = offset + i - range.range.first;
				if (ent) {
					ent[j] = sph_particles_eint(i);
				}
				if (vx) {
					vx[j] = sph_particles_vel(XDIM, i);
					vy[j] = sph_particles_vel(YDIM, i);
					vz[j] = sph_particles_vel(ZDIM, i);
				}
				if (gamma) {
					gamma[j] = sph_particles_gamma(i);
				}
				if (alpha) {
					alpha[j] = sph_particles_alpha(i);
				}
				if (chems) {
					for (int f = 0; f < NCHEMFRACS; f++) {
						chems[j][f] = sph_particles_c0[i][f];
					}
				}
				if (mmw) {
					mmw[j] = sph_particles_mmw(i);
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
				const auto* ptr = sph_particles_sph_cache_read_line(line_id, a);
				const auto begin = std::max(line_id.index, range.range.first);
				const auto end = std::min(line_id.index + line_size, range.range.second);
				for (part_int i = begin; i < end; i++) {
					const part_int src_index = i - line_id.index;
					const sph_particle& part = ptr[src_index];
					if (ent) {
						ent[dest_index] = part.eint;
					}
					if (vx) {
						vx[dest_index] = part.v[XDIM];
						vy[dest_index] = part.v[YDIM];
						vz[dest_index] = part.v[ZDIM];
					}
					if (gamma) {
						gamma[dest_index] = part.gamma;
					}
					if (alpha) {
						alpha[dest_index] = part.alpha;
					}
					if (chems) {
						chems[dest_index] = part.chem;
					}
					if (mmw) {
						mmw[dest_index] = part.mmw;
					}
					dest_index++;
				}
			}
		}
	}
}

static const sph_particle* sph_particles_sph_cache_read_line(line_id_type line_id, float a) {
	const part_int line_size = get_options().part_cache_line_size;
	const size_t bin = line_id_hash_lo()(line_id);
	std::unique_lock<spinlock_type> lock(sph_mutexes[bin]);
	auto iter = sph_part_cache[bin].find(line_id);
	if (iter == sph_part_cache[bin].end()) {
		auto prms = std::make_shared<hpx::lcos::local::promise<vector<sph_particle>> >();
		sph_part_cache[bin][line_id] = prms->get_future();
		lock.unlock();
		hpx::apply([prms,line_id,a]() {
			auto line_fut = hpx::async<sph_particles_fetch_sph_cache_line_action>(HPX_PRIORITY_HI, hpx_localities()[line_id.proc],line_id.index, a);
			prms->set_value(line_fut.get());
		});
		lock.lock();
		iter = sph_part_cache[bin].find(line_id);
	}
	auto fut = iter->second;
	lock.unlock();
	return fut.get().data();
}

static vector<sph_particle> sph_particles_fetch_sph_cache_line(part_int index, float a) {
	const part_int line_size = get_options().part_cache_line_size;
	vector<sph_particle> line(line_size);
	const part_int begin = (index / line_size) * line_size;
	const part_int end = std::min(sph_particles_size(), begin + line_size);
	for (part_int i = begin; i < end; i++) {
		line[i - begin] = sph_particles_get_particle(i, a);
	}
	return line;
}

void sph_particles_global_read_sph0(particle_global_range range, float* eint0, array<float, NCHEMFRACS>* chem0, part_int offset) {
	const part_int line_size = get_options().part_cache_line_size;
	const int sz = offset + range.range.second - range.range.first;
	if (range.range.first != range.range.second) {
		if (range.proc == hpx_rank()) {
			const part_int dif = offset - range.range.first;
			const part_int sz = range.range.second - range.range.first;
			for (part_int i = range.range.first; i < range.range.second; i++) {
				const int j = offset + i - range.range.first;
				if (eint0) {
					eint0[j] = sph_particles_eint0(i);
				}
				if (chem0) {
					for (int f = 0; f < NCHEMFRACS; f++) {
						chem0[j][f] = sph_particles_chem0(i)[f];
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
				const auto* ptr = sph_particles_sph0_cache_read_line(line_id);
				const auto begin = std::max(line_id.index, range.range.first);
				const auto end = std::min(line_id.index + line_size, range.range.second);
				for (part_int i = begin; i < end; i++) {
					const part_int src_index = i - line_id.index;
					const sph_particle0& part = ptr[src_index];
					if (eint0) {
						eint0[dest_index] = part.eint0;
					}
					if (chem0) {
						chem0[dest_index] = part.chem0;
					}
					dest_index++;
				}
			}
		}
	}
}

static const sph_particle0* sph_particles_sph0_cache_read_line(line_id_type line_id) {
	const part_int line_size = get_options().part_cache_line_size;
	const size_t bin = line_id_hash_lo()(line_id);
	std::unique_lock<spinlock_type> lock(sph_mutexes[bin]);
	auto iter = sph0_part_cache[bin].find(line_id);
	if (iter == sph0_part_cache[bin].end()) {
		auto prms = std::make_shared<hpx::lcos::local::promise<vector<sph_particle0>> >();
		sph0_part_cache[bin][line_id] = prms->get_future();
		lock.unlock();
		hpx::apply([prms,line_id]() {
			auto line_fut = hpx::async<sph_particles_fetch_sph0_cache_line_action>(HPX_PRIORITY_HI, hpx_localities()[line_id.proc],line_id.index);
			prms->set_value(line_fut.get());
		});
		lock.lock();
		iter = sph0_part_cache[bin].find(line_id);
	}
	auto fut = iter->second;
	lock.unlock();
	return fut.get().data();
}

static vector<sph_particle0> sph_particles_fetch_sph0_cache_line(part_int index) {
	const part_int line_size = get_options().part_cache_line_size;
	vector<sph_particle0> line(line_size);
	const part_int begin = (index / line_size) * line_size;
	const part_int end = std::min(sph_particles_size(), begin + line_size);
	for (part_int i = begin; i < end; i++) {
		line[i - begin] = sph_particles_get_particle0(i);
	}
	return line;
}

void sph_particles_global_read_rungs_and_smoothlens(particle_global_range range, char* rungs, float* hs, part_int offset) {
	const part_int line_size = get_options().part_cache_line_size;
	if (range.range.first != range.range.second) {
		if (range.proc == hpx_rank()) {
			const part_int dif = offset - range.range.first;
			const part_int sz = range.range.second - range.range.first;
			for (part_int i = range.range.first; i < range.range.second; i++) {
				const int j = offset + i - range.range.first;
				if (rungs) {
					rungs[j] = sph_particles_rung(i);
				}
				if (hs) {
					hs[j] = sph_particles_smooth_len(i);
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
				const auto* ptr = sph_particles_rung_cache_read_line(line_id);
				const auto begin = std::max(line_id.index, range.range.first);
				const auto end = std::min(line_id.index + line_size, range.range.second);
				for (part_int i = begin; i < end; i++) {
					const part_int src_index = i - line_id.index;
					if (rungs) {
						rungs[dest_index] = ptr[src_index].first;
					}
					if (hs) {
						hs[dest_index] = ptr[src_index].second;
					}
					dest_index++;
				}
			}
		}
	}
}

static const pair<char, float>* sph_particles_rung_cache_read_line(line_id_type line_id) {
	const part_int line_size = get_options().part_cache_line_size;
	const size_t bin = line_id_hash_lo()(line_id);
	std::unique_lock<spinlock_type> lock(rung_mutexes[bin]);
	auto iter = rung_part_cache[bin].find(line_id);
	if (iter == rung_part_cache[bin].end()) {
		auto prms = std::make_shared<hpx::lcos::local::promise<vector<pair<char, float>>> >();
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

static vector<pair<char, float>> sph_particles_fetch_rung_cache_line(part_int index) {
	const part_int line_size = get_options().part_cache_line_size;
	vector<pair<char, float>> line(line_size);
	const part_int begin = (index / line_size) * line_size;
	const part_int end = std::min(sph_particles_size(), begin + line_size);
	for (part_int i = begin; i < end; i++) {
		line[i - begin].first = sph_particles_rung(i);
		line[i - begin].second = sph_particles_smooth_len(i);
	}
	return line;
}

void sph_particles_global_read_aux(particle_global_range range, float* fpre, float* divv, float* balsara, float* shearv, float* gradT, part_int offset) {
	const part_int line_size = get_options().part_cache_line_size;
	if (range.range.first != range.range.second) {
		if (range.proc == hpx_rank()) {
			const part_int dif = offset - range.range.first;
			const part_int sz = range.range.second - range.range.first;
			for (part_int i = range.range.first; i < range.range.second; i++) {
				const int j = offset + i - range.range.first;
				if (fpre) {
					fpre[j] = sph_particles_fpre(i);
				}
				if (divv) {
					divv[j] = sph_particles_divv(i);
				}
				if (shearv) {
					shearv[j] = sph_particles_shear(i);
				}
				if (balsara) {
					balsara[j] = sph_particles_balsara(i);
				}
				if (gradT) {
					gradT[j] = sph_particles_gradT(i);
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
				if (fpre) {
					fpre[j] = ptr[src_index].fpre;
				}
				if (divv) {
					divv[j] = ptr[src_index].divv;
				}
				if (shearv) {
					shearv[j] = ptr[src_index].shearv;
				}
				if (shearv) {
					balsara[j] = ptr[src_index].balsara;
				}
				if (gradT) {
					gradT[j] = ptr[src_index].gradT;
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
void sph_particles_cache_free() {
	profiler_enter(__FUNCTION__);
	static const auto stars = get_options().stars;
	vector<hpx::future<void>> futs;
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async<sph_particles_cache_free_action>(HPX_PRIORITY_HI, c));
	}
	const int nthreads = hpx_hardware_concurrency();
	part_cache = decltype(part_cache)();
	sph_part_cache = decltype(sph_part_cache)();
	sph0_part_cache = decltype(sph0_part_cache)();
	rung_part_cache = decltype(rung_part_cache)();
	aux_cache = decltype(aux_cache)();
	hpx::wait_all(futs.begin(), futs.end());
	profiler_exit();
}

void sph_particles_load(FILE* fp) {
	const bool chem = get_options().chem;
	const bool diff = get_options().diffusion;
	const bool cond = get_options().conduction;
	const bool stars = get_options().stars;
	const bool xsph = get_options().xsph != 0.0;
	FREAD(&sph_particles_dm_index(0), sizeof(part_int), sph_particles_size(), fp);
	FREAD(&sph_particles_divv(0), sizeof(float), sph_particles_size(), fp);
	FREAD(&sph_particles_smooth_len(0), sizeof(float), sph_particles_size(), fp);
	FREAD(&sph_particles_eint(0), sizeof(float), sph_particles_size(), fp);
	FREAD(&sph_particles_divv(0), sizeof(float), sph_particles_size(), fp);
	FREAD(&sph_particles_divv0(0), sizeof(float), sph_particles_size(), fp);
	FREAD(&sph_particles_deint_pred(0), sizeof(float), sph_particles_size(), fp);
	FREAD(&sph_particles_alpha(0), sizeof(float), sph_particles_size(), fp);
	FREAD(&sph_particles_cv[0], sizeof(float), sph_particles_size(), fp);
	FREAD(&sph_particles_s2[0], sizeof(float), sph_particles_size(), fp);
	FREAD(&sph_particles_fp[0], sizeof(float), sph_particles_size(), fp);
	FREAD(&sph_particles_f0[0], sizeof(float), sph_particles_size(), fp);
	if (cond) {
		FREAD(&sph_particles_gt[0], sizeof(float), sph_particles_size(), fp);
	}
	if (diff) {
		FREAD(&sph_particles_ea[0], sizeof(float), sph_particles_size(), fp);
	}
	for (int dim = 0; dim < NDIM; dim++) {
		if (xsph) {
			FREAD(&sph_particles_xvel(dim, 0), sizeof(float), sph_particles_size(), fp);
		}
		FREAD(&sph_particles_dvel_pred(dim, 0), sizeof(float), sph_particles_size(), fp);
	}
	if (chem) {
		FREAD(sph_particles_c0, sizeof(sph_particles_c0[0]), sph_particles_size(), fp);
	}
	if (stars) {
		stars_load(fp);
	}
}

void sph_particles_save(FILE* fp) {
	const bool chem = get_options().chem;
	const bool diff = get_options().diffusion;
	const bool cond = get_options().conduction;
	const bool stars = get_options().stars;
	const bool xsph = get_options().xsph != 0.0;
	fwrite(&sph_particles_dm_index(0), sizeof(part_int), sph_particles_size(), fp);
	fwrite(&sph_particles_divv(0), sizeof(float), sph_particles_size(), fp);
	fwrite(&sph_particles_smooth_len(0), sizeof(float), sph_particles_size(), fp);
	fwrite(&sph_particles_eint(0), sizeof(float), sph_particles_size(), fp);
	fwrite(&sph_particles_divv(0), sizeof(float), sph_particles_size(), fp);
	fwrite(&sph_particles_divv0(0), sizeof(float), sph_particles_size(), fp);
	fwrite(&sph_particles_deint_pred(0), sizeof(float), sph_particles_size(), fp);
	fwrite(&sph_particles_alpha(0), sizeof(float), sph_particles_size(), fp);
	fwrite(&sph_particles_s2[0], sizeof(float), sph_particles_size(), fp);
	fwrite(&sph_particles_cv[0], sizeof(float), sph_particles_size(), fp);
	fwrite(&sph_particles_fp[0], sizeof(float), sph_particles_size(), fp);
	fwrite(&sph_particles_f0[0], sizeof(float), sph_particles_size(), fp);
	if (cond) {
		fwrite(&sph_particles_gt[0], sizeof(float), sph_particles_size(), fp);
	}
	if (diff) {
		fwrite(&sph_particles_ea[0], sizeof(float), sph_particles_size(), fp);
	}
	for (int dim = 0; dim < NDIM; dim++) {
		if (xsph) {
			fwrite(&sph_particles_xvel(dim, 0), sizeof(float), sph_particles_size(), fp);
		}
		fwrite(&sph_particles_dvel_pred(dim, 0), sizeof(float), sph_particles_size(), fp);
	}
	if (chem) {
		fwrite(sph_particles_c0, sizeof(sph_particles_c0[0]), sph_particles_size(), fp);
	}
	if (stars) {
		stars_save(fp);
	}
}

