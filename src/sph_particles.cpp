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

struct difco_data {
	float difco;
	float kappa;
	char rung;
	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & difco;
		arc & kappa;
		arc & rung;
	}
};

struct fs_return {
	float fvel;
	float fpre;
	float fpot;
	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & fvel;
		arc & fpre;
		arc & fpot;
	}
};

static const array<fixed32, NDIM>* sph_particles_cache_read_line(line_id_type line_id);
static const array<float, NDIM>* sph_particles_gforce_cache_read_line(line_id_type line_id);
static const sph_particle* sph_particles_sph_cache_read_line(line_id_type line_id, float);
static const pair<char, float>* sph_particles_rung_cache_read_line(line_id_type line_id);
static const fs_return* sph_particles_fvel_cache_read_line(line_id_type line_id);
//static const float* sph_particles_sn_cache_read_line(line_id_type line_id);
static const difco_data* sph_particles_difco_cache_read_line(line_id_type line_id);
static const dif_vector* sph_particles_difvecs_cache_read_line(line_id_type line_id);

static vector<array<fixed32, NDIM>> sph_particles_fetch_cache_line(part_int index);
static vector<array<float, NDIM>> sph_particles_fetch_gforce_cache_line(part_int index);
static vector<pair<char, float>> sph_particles_fetch_rung_cache_line(part_int index);
static vector<sph_particle> sph_particles_fetch_sph_cache_line(part_int index, float);
static vector<fs_return> sph_particles_fetch_fvel_cache_line(part_int index);
//static vector<float> sph_particles_fetch_sn_cache_line(part_int index);
static vector<difco_data> sph_particles_fetch_difco_cache_line(part_int index);
static vector<dif_vector> sph_particles_fetch_difvecs_cache_line(part_int index);

HPX_PLAIN_ACTION (sph_particles_fetch_cache_line);
HPX_PLAIN_ACTION (sph_particles_fetch_gforce_cache_line);
HPX_PLAIN_ACTION (sph_particles_fetch_sph_cache_line);
HPX_PLAIN_ACTION (sph_particles_fetch_rung_cache_line);
HPX_PLAIN_ACTION (sph_particles_cache_free);
HPX_PLAIN_ACTION (sph_particles_fetch_fvel_cache_line);
//HPX_PLAIN_ACTION (sph_particles_fetch_sn_cache_line);
HPX_PLAIN_ACTION (sph_particles_fetch_difco_cache_line);
HPX_PLAIN_ACTION (sph_particles_fetch_difvecs_cache_line);

static array<std::unordered_map<line_id_type, hpx::shared_future<vector<array<fixed32, NDIM>>> , line_id_hash_hi>,PART_CACHE_SIZE> part_cache;
static array<std::unordered_map<line_id_type, hpx::shared_future<vector<array<float, NDIM>>> , line_id_hash_hi>,PART_CACHE_SIZE> gforce_cache;
static array<std::unordered_map<line_id_type, hpx::shared_future<vector<sph_particle>>, line_id_hash_hi>, PART_CACHE_SIZE> sph_part_cache;
static array<std::unordered_map<line_id_type, hpx::shared_future<vector<pair<char, float>>> , line_id_hash_hi>, PART_CACHE_SIZE> rung_part_cache;
static array<std::unordered_map<line_id_type, hpx::shared_future<vector<fs_return>>, line_id_hash_hi>, PART_CACHE_SIZE> fvel_cache;
//static array<std::unordered_map<line_id_type, hpx::shared_future<vector<float>>, line_id_hash_hi>, PART_CACHE_SIZE> sn_cache;
static array<std::unordered_map<line_id_type, hpx::shared_future<vector<difco_data>>, line_id_hash_hi>, PART_CACHE_SIZE> difco_cache;
static array<std::unordered_map<line_id_type, hpx::shared_future<vector<dif_vector>>, line_id_hash_hi>, PART_CACHE_SIZE> difvec_cache;
static array<spinlock_type, PART_CACHE_SIZE> mutexes;
static array<spinlock_type, PART_CACHE_SIZE> gforce_mutexes;
static array<spinlock_type, PART_CACHE_SIZE> sph_mutexes;
static array<spinlock_type, PART_CACHE_SIZE> rung_mutexes;
static array<spinlock_type, PART_CACHE_SIZE> fvel_mutexes;
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

void sph_particles_apply_updates(int minrung, int phase, float t0) {

	profiler_enter(__FUNCTION__);
	vector<hpx::future<void>> futs;
	for (auto& c : hpx_children()) {
		futs.push_back(hpx::async<sph_particles_apply_updates_action>(c, minrung, phase, t0));
	}
	const int nthreads = hpx_hardware_concurrency();
	for (int proc = 0; proc < nthreads; proc++) {
		futs.push_back(hpx::async([t0,nthreads, proc, phase, minrung]() {
			const part_int b = (size_t) proc * sph_particles_size() / nthreads;
			const part_int e = (size_t) (proc+1) * sph_particles_size() / nthreads;
			for( int i = b; i < e; i++) {
				const int k = sph_particles_dm_index(i);
				const auto rung = particles_rung(k);
				const float dt = 0.5 * t0 / (1<<rung);
				if( rung >= minrung) {
					switch(phase) {
						case 0:
						sph_particles_eint(i) +=sph_particles_deint_pred(i) *dt;
						for( int dim =0; dim < NDIM; dim++) {
							particles_vel(dim,k) += sph_particles_dvel_pred(dim,i)* dt;
						}
						break;
						case 1:
						sph_particles_eint(i) -=sph_particles_deint_pred(i) *dt;
						for( int dim =0; dim < NDIM; dim++) {
							particles_vel(dim,k) -= sph_particles_dvel_pred(dim,i)* dt;
						}
						sph_particles_eint(i) +=sph_particles_deint_con(i) *dt;
						for( int dim =0; dim < NDIM; dim++) {
							particles_vel(dim,k) += sph_particles_dvel_con(dim,i)* dt;
						}
						break;
						case 2:
						sph_particles_deint_pred(i) = sph_particles_deint_con(i);
						for( int dim =0; dim < NDIM; dim++) {
							sph_particles_dvel_pred(dim,i) = sph_particles_dvel_con(dim,i);
						}
						sph_particles_eint(i) +=sph_particles_deint_con(i) *dt;
						for( int dim =0; dim < NDIM; dim++) {
							particles_vel(dim,k) += sph_particles_dvel_con(dim,i)* dt;
						}
						break;
					}}
			}
		}));
	}
	for (auto& f : futs) {
		f.get();
	}
	profiler_exit();

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
	T = std::max(T,1.0);
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

float sph_particles_lambda_e(part_int i, float a, float T) {
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
}

HPX_PLAIN_ACTION (sph_particles_energy_to_entropy);

void sph_particles_energy_to_entropy(float a) {
}

void sph_particles_swap(part_int i, part_int j) {
	const bool chem = get_options().chem;
	const bool stars = get_options().stars;
	std::swap(sph_particles_a[i], sph_particles_a[j]);
	std::swap(sph_particles_e[i], sph_particles_e[j]);
	std::swap(sph_particles_cond[i], sph_particles_cond[j]);
	std::swap(sph_particles_de1[i], sph_particles_de1[j]);
	std::swap(sph_particles_dvv[i], sph_particles_dvv[j]);
	std::swap(sph_particles_fp[i], sph_particles_fp[j]);
	std::swap(sph_particles_fv[i], sph_particles_fv[j]);
	std::swap(sph_particles_f0[i], sph_particles_f0[j]);
	std::swap(sph_particles_dc[i], sph_particles_dc[j]);
	std::swap(sph_particles_h[i], sph_particles_h[j]);
	std::swap(sph_particles_dm[i], sph_particles_dm[j]);
	if (stars) {
		std::swap(sph_particles_ts[i], sph_particles_ts[j]);
	}
	for (int dim = 0; dim < NDIM; dim++) {
		std::swap(sph_particles_g[dim][i], sph_particles_g[dim][j]);
		std::swap(sph_particles_dv1[dim][i], sph_particles_dv1[dim][j]);
	}
	if (chem) {
		for (int l = 0; l < NCHEMFRACS; l++) {
			std::swap(sph_particles_chem[l][i], sph_particles_chem[l][j]);
		}
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
	if (sz > capacity) {
		part_int new_capacity = std::max(capacity, (part_int) 100);
		while (new_capacity < sz) {
			new_capacity = size_t(101) * new_capacity / size_t(100);
		}
		//	PRINT("Resizing sph_particles to %li from %li\n", new_capacity, capacity);
		sph_particles_array_resize(sph_particles_dm, new_capacity, false);
		sph_particles_array_resize(sph_particles_e, new_capacity, true);
		sph_particles_array_resize(sph_particles_h, new_capacity, true);
		sph_particles_array_resize(sph_particles_fp, new_capacity, true);
		sph_particles_array_resize(sph_particles_or, new_capacity, true);
		sph_particles_array_resize(sph_particles_cond, new_capacity, true);
		sph_particles_array_resize(sph_particles_de1, new_capacity, true);
		sph_particles_array_resize(sph_particles_de2, new_capacity, true);
		sph_particles_array_resize(sph_particles_sa, new_capacity, true);
		sph_particles_array_resize(sph_particles_fv, new_capacity, true);
		sph_particles_array_resize(sph_particles_f0, new_capacity, true);
		sph_particles_array_resize(sph_particles_dc, new_capacity, true);
		sph_particles_array_resize(sph_particles_dvv, new_capacity, true);
		sph_particles_array_resize(sph_particles_a, new_capacity, true);
		sph_particles_array_resize(sph_particles_dvec, new_capacity, true);
		sph_particles_array_resize(sph_particles_vec0, new_capacity, true);
		sph_particles_array_resize(sph_particles_vec, new_capacity, false);
#ifdef CHECK_MUTUAL_SORT
		sph_particles_array_resize(sph_particles_tst, new_capacity, false);
#endif
		for (int dim = 0; dim < NDIM; dim++) {
			sph_particles_array_resize(sph_particles_dv1[dim], new_capacity, true);
			sph_particles_array_resize(sph_particles_dv2[dim], new_capacity, true);
			sph_particles_array_resize(sph_particles_g[dim], new_capacity, true);
		}
		if (chem) {
			for (int f = 0; f < NCHEMFRACS; f++) {
				sph_particles_array_resize(sph_particles_chem[f], new_capacity, false);
			}
		}
		if (stars) {
			sph_particles_array_resize(sph_particles_ts, new_capacity, true);
			sph_particles_array_resize(sph_particles_tc, new_capacity, true);
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
		sph_particles_alpha(oldsz + i) = SPH_ALPHA0;
		if (stars) {
			sph_particles_tdyn(i) = 1e38f;
		}
		for (int dim = 0; dim < NDIM; dim++) {
			sph_particles_gforce(dim, oldsz + i) = 0.0f;
			sph_particles_dvel_con(dim, oldsz + i) = 0.0f;
			sph_particles_dvel_pred(dim, oldsz + i) = 0.0f;
		}
	}
}

void sph_particles_free() {
	const bool stars = get_options().stars;
	free(sph_particles_dm);
	free(sph_particles_vec);
#ifdef CHECK_MUTUAL_SORT
	free(sph_particles_tst);
#endif
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
		CUDA_CHECK(cudaFree(sph_particles_or));
		CUDA_CHECK(cudaFree(sph_particles_dc));
		CUDA_CHECK(cudaFree(sph_particles_a));
		CUDA_CHECK(cudaFree(sph_particles_fp));
		if( stars ) {
			//		CUDA_CHECK(cudaFree(sph_particles_fZ));
//			CUDA_CHECK(cudaFree(sph_particles_fY));
//			CUDA_CHECK(cudaFree(sph_particles_sn));
//			CUDA_CHECK(cudaFree(sph_particles_dz));
			CUDA_CHECK(cudaFree(sph_particles_tc));
		}
		CUDA_CHECK(cudaFree(sph_particles_vec0));
		CUDA_CHECK(cudaFree(sph_particles_dvec));
		CUDA_CHECK(cudaFree(sph_particles_cond));
		CUDA_CHECK(cudaFree(sph_particles_fv));
		CUDA_CHECK(cudaFree(sph_particles_h));
		for (int dim = 0; dim < NDIM; dim++) {
			CUDA_CHECK(cudaFree(sph_particles_dv1[NDIM]));
			CUDA_CHECK(cudaFree(sph_particles_dv2[NDIM]));
			CUDA_CHECK(cudaFree(sph_particles_g[NDIM]));
		}
#endif
	} else {
		if (get_options().chem) {
			for (int f = 0; f < NCHEMFRACS; f++) {
				free(sph_particles_chem[f]);
			}
		}
		free(sph_particles_h);
		free(sph_particles_e);
		free(sph_particles_dc);
		free(sph_particles_fp);
		free(sph_particles_or);
		free(sph_particles_de1);
		free(sph_particles_de2);
		free(sph_particles_cond);
		free(sph_particles_dvv);
		free(sph_particles_sa);
		free(sph_particles_f0);
		free(sph_particles_fv);
		free(sph_particles_vec0);
		free(sph_particles_dvec);
		free(sph_particles_a);
		if (stars) {
			free(sph_particles_ts);
//			free(sph_particles_dz);
			free(sph_particles_tc);
		}
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

void sph_particles_global_read_sph(particle_global_range range, float a, float* ent, float* vx, float* vy, float* vz, float* gamma, float* T, float* lambda_e,
		float* mmw, float* colog, float* alpha, part_int offset) {
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
				if (T) {
					T[j] = sph_particles_temperature(i, a);
					lambda_e[j] = sph_particles_lambda_e(i, a, T[j]);
				}
				if (mmw) {
					mmw[j] = sph_particles_mmw(i);
				}
				if (colog) {
					colog[j] = sph_particles_coloumb_log(i, a);
				}
				if (alpha) {
					alpha[j] = sph_particles_alpha(i);
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
					if (T) {
						T[dest_index] = part.T;
						lambda_e[dest_index] = part.lambda_e;
					}
					if (mmw) {
						mmw[dest_index] = part.mmw;
					}
					if (colog) {
						colog[dest_index] = part.colog;
					}
					if (alpha) {
						alpha[dest_index] = part.alpha;
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

void sph_particles_global_read_fvels(particle_global_range range, float* fvels, float* fpres, float* fpots, part_int offset) {
	const part_int line_size = get_options().part_cache_line_size;
	if (range.range.first != range.range.second) {
		if (range.proc == hpx_rank()) {
			const part_int dif = offset - range.range.first;
			const part_int sz = range.range.second - range.range.first;
			for (part_int i = range.range.first; i < range.range.second; i++) {
				const int j = offset + i - range.range.first;
				fvels[j] = sph_particles_fvel(i);
				fpres[j] = sph_particles_fpre(i);
				if (fpots)
					fpots[j] = sph_particles_fpot(i);
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
			const auto* ptr = sph_particles_fvel_cache_read_line(line_id);
			const auto begin = std::max(line_id.index, range.range.first);
			const auto end = std::min(line_id.index + line_size, range.range.second);
			for (part_int i = begin; i < end; i++) {
				const part_int src_index = i - line_id.index;
				fvels[dest_index] = ptr[src_index].fvel;
				fpres[dest_index] = ptr[src_index].fpre;
				fpots[dest_index] = ptr[src_index].fpot;
				dest_index++;
			}
		}
	}
}

static const fs_return* sph_particles_fvel_cache_read_line(line_id_type line_id) {
	const part_int line_size = get_options().part_cache_line_size;
	const size_t bin = line_id_hash_lo()(line_id);
	std::unique_lock<spinlock_type> lock(fvel_mutexes[bin]);
	auto iter = fvel_cache[bin].find(line_id);
	if (iter == fvel_cache[bin].end()) {
		auto prms = std::make_shared<hpx::lcos::local::promise<vector<fs_return>> >();
		fvel_cache[bin][line_id] = prms->get_future();
		lock.unlock();
		hpx::apply([prms,line_id]() {
			auto line_fut = hpx::async<sph_particles_fetch_fvel_cache_line_action>(HPX_PRIORITY_HI, hpx_localities()[line_id.proc],line_id.index);
			prms->set_value(line_fut.get());
		});
		lock.lock();
		iter = fvel_cache[bin].find(line_id);
	}
	auto fut = iter->second;
	lock.unlock();
	return fut.get().data();
}

static vector<fs_return> sph_particles_fetch_fvel_cache_line(part_int index) {
	const part_int line_size = get_options().part_cache_line_size;
	vector<fs_return> line(line_size);
	const part_int begin = (index / line_size) * line_size;
	const part_int end = std::min(sph_particles_size(), begin + line_size);
	for (part_int i = begin; i < end; i++) {
		line[i - begin].fvel = sph_particles_fvel(i);
		line[i - begin].fpre = sph_particles_fpre(i);
		line[i - begin].fpot = sph_particles_fpot(i);

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
	rung_part_cache = decltype(rung_part_cache)();
	fvel_cache = decltype(fvel_cache)();
	difco_cache = decltype(difco_cache)();
	difvec_cache = decltype(difvec_cache)();
	hpx::wait_all(futs.begin(), futs.end());
	profiler_exit();
}

void sph_particles_load(FILE* fp) {
	const bool chem = get_options().chem;
	const bool stars = get_options().stars;
	FREAD(&sph_particles_dm_index(0), sizeof(part_int), sph_particles_size(), fp);
	FREAD(&sph_particles_divv(0), sizeof(float), sph_particles_size(), fp);
	FREAD(&sph_particles_smooth_len(0), sizeof(float), sph_particles_size(), fp);
	FREAD(&sph_particles_eint(0), sizeof(float), sph_particles_size(), fp);
	FREAD(&sph_particles_fvel(0), sizeof(float), sph_particles_size(), fp);
	FREAD(&sph_particles_fpre(0), sizeof(float), sph_particles_size(), fp);
	FREAD(&sph_particles_fpot(0), sizeof(float), sph_particles_size(), fp);
	FREAD(&sph_particles_deint_pred(0), sizeof(float), sph_particles_size(), fp);
	FREAD(&sph_particles_difco(0), sizeof(float), sph_particles_size(), fp);
	FREAD(&sph_particles_kappa(0), sizeof(float), sph_particles_size(), fp);
	FREAD(&sph_particles_alpha(0), sizeof(float), sph_particles_size(), fp);
	for (int dim = 0; dim < NDIM; dim++) {
		FREAD(&sph_particles_dvel_pred(dim, 0), sizeof(float), sph_particles_size(), fp);
	}
	if (chem) {
		for (int f = 0; f < NCHEMFRACS; f++) {
			FREAD(sph_particles_chem[f], sizeof(float), sph_particles_size(), fp);
		}
	}
	if (stars) {
		FREAD(sph_particles_ts, sizeof(float), sph_particles_size(), fp);
		stars_load(fp);
	}
}

void sph_particles_save(FILE* fp) {
	const bool chem = get_options().chem;
	const bool stars = get_options().stars;
	fwrite(&sph_particles_dm_index(0), sizeof(part_int), sph_particles_size(), fp);
	fwrite(&sph_particles_divv(0), sizeof(float), sph_particles_size(), fp);
	fwrite(&sph_particles_smooth_len(0), sizeof(float), sph_particles_size(), fp);
	fwrite(&sph_particles_eint(0), sizeof(float), sph_particles_size(), fp);
	fwrite(&sph_particles_fvel(0), sizeof(float), sph_particles_size(), fp);
	fwrite(&sph_particles_fpre(0), sizeof(float), sph_particles_size(), fp);
	fwrite(&sph_particles_fpot(0), sizeof(float), sph_particles_size(), fp);
	fwrite(&sph_particles_deint_pred(0), sizeof(float), sph_particles_size(), fp);
	fwrite(&sph_particles_difco(0), sizeof(float), sph_particles_size(), fp);
	fwrite(&sph_particles_kappa(0), sizeof(float), sph_particles_size(), fp);
	fwrite(&sph_particles_alpha(0), sizeof(float), sph_particles_size(), fp);
	for (int dim = 0; dim < NDIM; dim++) {
		fwrite(&sph_particles_dvel_pred(dim, 0), sizeof(float), sph_particles_size(), fp);
	}
	if (chem) {
		for (int f = 0; f < NCHEMFRACS; f++) {
			fwrite(sph_particles_chem[f], sizeof(float), sph_particles_size(), fp);
		}
	}
	if (stars) {
		fwrite(sph_particles_ts, sizeof(float), sph_particles_size(), fp);
		stars_save(fp);
	}
}

void sph_particles_global_read_difcos(particle_global_range range, float* difcos, float* kappas, char* oldrungs, part_int offset) {
	const part_int line_size = get_options().part_cache_line_size;
	if (range.range.first != range.range.second) {
		if (range.proc == hpx_rank()) {
			const part_int dif = offset - range.range.first;
			const part_int sz = range.range.second - range.range.first;
			for (part_int i = range.range.first; i < range.range.second; i++) {
				const int j = offset + i - range.range.first;
				difcos[j] = sph_particles_difco(i);
				oldrungs[j] = sph_particles_old_rung(i);
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
				const auto* ptr = sph_particles_difco_cache_read_line(line_id);
				const auto begin = std::max(line_id.index, range.range.first);
				const auto end = std::min(line_id.index + line_size, range.range.second);
				for (part_int i = begin; i < end; i++) {
					const part_int src_index = i - line_id.index;
					difcos[dest_index] = ptr[src_index].difco;
					oldrungs[dest_index] = ptr[src_index].rung;
					kappas[dest_index] = ptr[src_index].kappa;
					dest_index++;
				}
			}
		}
	}
}

static const difco_data* sph_particles_difco_cache_read_line(line_id_type line_id) {
	const part_int line_size = get_options().part_cache_line_size;
	const size_t bin = line_id_hash_lo()(line_id);
	std::unique_lock<spinlock_type> lock(fvel_mutexes[bin]);
	auto iter = difco_cache[bin].find(line_id);
	if (iter == difco_cache[bin].end()) {
		auto prms = std::make_shared<hpx::lcos::local::promise<vector<difco_data>> >();
		difco_cache[bin][line_id] = prms->get_future();
		lock.unlock();
		hpx::apply([prms,line_id]() {
			auto line_fut = hpx::async<sph_particles_fetch_difco_cache_line_action>(HPX_PRIORITY_HI, hpx_localities()[line_id.proc],line_id.index);
			prms->set_value(line_fut.get());
		});
		lock.lock();
		iter = difco_cache[bin].find(line_id);
	}
	auto fut = iter->second;
	lock.unlock();
	return fut.get().data();
}

static vector<difco_data> sph_particles_fetch_difco_cache_line(part_int index) {
	const part_int line_size = get_options().part_cache_line_size;
	vector<difco_data> line(line_size);
	const part_int begin = (index / line_size) * line_size;
	const part_int end = std::min(sph_particles_size(), begin + line_size);
	for (part_int i = begin; i < end; i++) {
		line[i - begin].difco = sph_particles_difco(i);
		line[i - begin].kappa = sph_particles_kappa(i);
		line[i - begin].rung = sph_particles_old_rung(i);
	}
	return line;
}

void sph_particles_global_read_difvecs(particle_global_range range, dif_vector* difvecs, part_int offset) {
	const part_int line_size = get_options().part_cache_line_size;
	if (range.range.first != range.range.second) {
		if (range.proc == hpx_rank()) {
			const part_int dif = offset - range.range.first;
			const part_int sz = range.range.second - range.range.first;
			for (part_int i = range.range.first; i < range.range.second; i++) {
				const int j = offset + i - range.range.first;
				difvecs[j] = sph_particles_dif_vec(i);
			}
		} else {
			line_id_type line_id;
			line_id.proc = range.proc;
			const part_int start_line = (range.range.first / line_size) * line_size;
			const part_int stop_line = ((range.range.second - 1) / line_size) * line_size;
			part_int dest_index = offset;
			for (part_int line = start_line; line <= stop_line; line += line_size) {
				line_id.index = line;
				const auto* ptr = sph_particles_difvecs_cache_read_line(line_id);
				const auto begin = std::max(line_id.index, range.range.first);
				const auto end = std::min(line_id.index + line_size, range.range.second);
				for (part_int i = begin; i < end; i++) {
					const part_int src_index = i - line_id.index;
					difvecs[dest_index] = ptr[src_index];
					dest_index++;
				}
			}
		}
	}
}

static const dif_vector* sph_particles_difvecs_cache_read_line(line_id_type line_id) {
	const part_int line_size = get_options().part_cache_line_size;
	const size_t bin = line_id_hash_lo()(line_id);
	std::unique_lock<spinlock_type> lock(fvel_mutexes[bin]);
	auto iter = difvec_cache[bin].find(line_id);
	if (iter == difvec_cache[bin].end()) {
		auto prms = std::make_shared<hpx::lcos::local::promise<vector<dif_vector>> >();
		difvec_cache[bin][line_id] = prms->get_future();
		lock.unlock();
		hpx::apply([prms,line_id]() {
			auto line_fut = hpx::async<sph_particles_fetch_difvecs_cache_line_action>(HPX_PRIORITY_HI, hpx_localities()[line_id.proc],line_id.index);
			prms->set_value(line_fut.get());
		});
		lock.lock();
		iter = difvec_cache[bin].find(line_id);
	}
	auto fut = iter->second;
	lock.unlock();
	return fut.get().data();
}

static vector<dif_vector> sph_particles_fetch_difvecs_cache_line(part_int index) {
	const part_int line_size = get_options().part_cache_line_size;
	vector<dif_vector> line(line_size);
	const part_int begin = (index / line_size) * line_size;
	const part_int end = std::min(sph_particles_size(), begin + line_size);
	for (part_int i = begin; i < end; i++) {
		line[i - begin] = sph_particles_dif_vec(i);
	}
	return line;
}
