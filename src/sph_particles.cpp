/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

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

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <unordered_map>

#define SPH_PARTICLES_CPP
#include <cosmictiger/constants.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/sph.hpp>
#include <cosmictiger/sph_particles.hpp>
#include <cosmictiger/stars.hpp>

static part_int capacity = 0;
static part_int size = 0;

struct line_id_type {
	int proc;
	part_int index;
	inline bool operator==(line_id_type other) const { return proc == other.proc && index == other.index; }
};

struct line_id_hash {
	inline size_t operator()(line_id_type id) const {
		part_int const line_size = get_options().part_cache_line_size;
		part_int const i = id.index / line_size;
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

static array<fixed32, NDIM> const* sph_particles_cache_read_line(line_id_type line_id);
static pair<float> const* sph_particles_energy_and_smoothlen_cache_read_line(line_id_type line_id);
static pair<float, char> const* sph_particles_fcold_cache_read_line(line_id_type line_id);
static char const* sph_particles_rung_cache_read_line(line_id_type line_id);
static array<float, NDIM> const* sph_particles_vels_cache_read_line(line_id_type line_id);
static float const* sph_particles_kappas_cache_read_line(line_id_type line_id);
static float const* sph_particles_rho_cache_read_line(line_id_type line_id);
static aux_quantities const* sph_particles_aux_cache_read_line(line_id_type line_id);

static vector<array<fixed32, NDIM>> sph_particles_fetch_cache_line(part_int index);
static vector<char> sph_particles_fetch_rung_cache_line(part_int index);
static vector<array<float, NDIM>> sph_particles_fetch_vels_cache_line(part_int index);
static vector<float> sph_particles_fetch_rho_cache_line(part_int index);
static vector<float> sph_particles_fetch_kappas_cache_line(part_int index);
static vector<pair<float>> sph_particles_fetch_energy_and_smoothlen_cache_line(part_int index);
static vector<pair<float, char>> sph_particles_fetch_fcold_cache_line(part_int index);
static vector<aux_quantities> sph_particles_fetch_aux_cache_line(part_int index);

HPX_PLAIN_ACTION(sph_particles_fetch_cache_line);
HPX_PLAIN_ACTION(sph_particles_fetch_energy_and_smoothlen_cache_line);
HPX_PLAIN_ACTION(sph_particles_fetch_fcold_cache_line);
HPX_PLAIN_ACTION(sph_particles_fetch_rung_cache_line);
HPX_PLAIN_ACTION(sph_particles_fetch_vels_cache_line);
HPX_PLAIN_ACTION(sph_particles_fetch_rho_cache_line);
HPX_PLAIN_ACTION(sph_particles_fetch_kappas_cache_line);
#ifdef ENTROPY
HPX_PLAIN_ACTION(sph_particles_cache_free_entr);
#else
HPX_PLAIN_ACTION(sph_particles_cache_free_eint);
#endif
HPX_PLAIN_ACTION(sph_particles_cache_free1);
HPX_PLAIN_ACTION(sph_particles_cache_free2);
HPX_PLAIN_ACTION(sph_particles_fetch_aux_cache_line);

static array<std::unordered_map<line_id_type, std::shared_ptr<hpx::shared_future<vector<array<fixed32, NDIM>>>>, line_id_hash_hi>, PART_CACHE_SIZE> part_cache;
static array<std::unordered_map<line_id_type, std::shared_ptr<hpx::shared_future<vector<pair<float>>>>, line_id_hash_hi>, PART_CACHE_SIZE> entr_part_cache;
static array<std::unordered_map<line_id_type, std::shared_ptr<hpx::shared_future<vector<pair<float, char>>>>, line_id_hash_hi>, PART_CACHE_SIZE>
	fcold_part_cache;
static array<std::unordered_map<line_id_type, std::shared_ptr<hpx::shared_future<vector<char>>>, line_id_hash_hi>, PART_CACHE_SIZE> rung_part_cache;
static array<std::unordered_map<line_id_type, std::shared_ptr<hpx::shared_future<vector<array<float, NDIM>>>>, line_id_hash_hi>, PART_CACHE_SIZE>
	vels_part_cache;
static array<std::unordered_map<line_id_type, std::shared_ptr<hpx::shared_future<vector<float>>>, line_id_hash_hi>, PART_CACHE_SIZE> kappas_part_cache;
static array<std::unordered_map<line_id_type, std::shared_ptr<hpx::shared_future<vector<float>>>, line_id_hash_hi>, PART_CACHE_SIZE> rho_part_cache;
static array<std::unordered_map<line_id_type, std::shared_ptr<hpx::shared_future<vector<aux_quantities>>>, line_id_hash_hi>, PART_CACHE_SIZE> aux_cache;
static array<spinlock_type, PART_CACHE_SIZE> mutexes;
static array<spinlock_type, PART_CACHE_SIZE> entr_mutexes;
static array<spinlock_type, PART_CACHE_SIZE> fcold_mutexes;
static array<spinlock_type, PART_CACHE_SIZE> rung_mutexes;
static array<spinlock_type, PART_CACHE_SIZE> vels_mutexes;
static array<spinlock_type, PART_CACHE_SIZE> kappas_mutexes;
static array<spinlock_type, PART_CACHE_SIZE> rho_mutexes;
static array<spinlock_type, PART_CACHE_SIZE> aux_mutexes;
static int group_cache_epoch = 0;

template <typename Action, typename T>
static T const*
readSphCacheLine(line_id_type lineId,
				 array<std::unordered_map<line_id_type, std::shared_ptr<hpx::shared_future<vector<T>>>, line_id_hash_hi>, PART_CACHE_SIZE>& cache,
				 array<spinlock_type, PART_CACHE_SIZE>& locks) {
	size_t const bin = line_id_hash_lo()(lineId);
	std::shared_ptr<hpx::shared_future<vector<T>>> future;
	{
		std::lock_guard<spinlock_type> lock(locks[bin]);
		auto const entry = cache[bin].find(lineId);
		if (entry != cache[bin].end()) {
			future = entry->second;
		}
	}
	if (!future) {
		auto promise = std::make_shared<hpx::promise<vector<T>>>();
		auto candidate = std::make_shared<hpx::shared_future<vector<T>>>(promise->get_future().share());
		bool inserted;
		{
			std::lock_guard<spinlock_type> lock(locks[bin]);
			auto const entry = cache[bin].emplace(lineId, candidate);
			inserted = entry.second;
			future = entry.first->second;
		}
		if (inserted) {
			try {
				hpx::apply([promise, lineId]() {
					try {
						auto remote = hpx::async<Action>(hpx_localities()[lineId.proc], lineId.index);
						promise->set_value(remote.get());
					} catch (...) {
						promise->set_exception(std::current_exception());
					}
				});
			} catch (...) {
				promise->set_exception(std::current_exception());
			}
		}
	}
	return future->get().data();
}

HPX_PLAIN_ACTION(sph_particles_max_smooth_len);

float sph_particles_max_smooth_len() {
	vector<hpx::future<float>> futs;
	for (auto& c : hpx_children()) {
		futs.push_back(hpx::async<sph_particles_max_smooth_len_action>(c));
	}
	int const nthreads = hpx_hardware_concurrency();
	for (int proc = 0; proc < nthreads; proc++) {
		futs.push_back(hpx::async([nthreads, proc]() {
			float maxh = 0.0;
			part_int const b = (size_t)proc * sph_particles_size() / nthreads;
			part_int const e = (size_t)(proc + 1) * sph_particles_size() / nthreads;
			for (int i = b; i < e; i++) {
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

HPX_PLAIN_ACTION(sph_particles_apply_updates);
std::pair<double, double> sph_particles_apply_updates(int minrung, int phase, float t0, float tau, float w) {
	if (!(phase == 2 || tau > 0.0)) {
		return std::pair<double, double>();
	}
	profiler_enter(__FUNCTION__);
	double err = 0.0;
	double norm = 0.0;
	vector<hpx::future<std::pair<double, double>>> futs;
	for (auto& c : hpx_children()) {
		futs.push_back(hpx::async<sph_particles_apply_updates_action>(c, minrung, phase, t0, tau, w));
	}
	int const nthreads = hpx_hardware_concurrency();
	for (int proc = 0; proc < nthreads; proc++) {
		futs.push_back(hpx::async([t0, nthreads, proc, phase, minrung, tau, w]() {
			auto const chem = get_options().chem;
			auto const stars = get_options().stars;
			double error = 0.0;
			double norm = 0.0;
			part_int const b = (size_t)proc * sph_particles_size() / nthreads;
			part_int const e = (size_t)(proc + 1) * sph_particles_size() / nthreads;
			for (int i = b; i < e; i++) {
				part_int const k = sph_particles_dm_index(i);
				auto const rung2 = sph_particles_rung(i);
				float const dt2 = 0.5f * t0 / (1 << rung2);
				if (phase == 0) {
					if (rung2 >= minrung && !sph_particles_isstar(i)) {
						for (int dim = 0; dim < NDIM; dim++) {
							sph_particles_dvel0(dim, i) = sph_particles_dvel(dim, i);
						}
						for (int dim = 0; dim < NDIM; dim++) {
							particles_vel(dim, k) += sph_particles_dvel(dim, i) * dt2;
						}
#ifdef ENTROPY
						ALWAYS_ASSERT(sph_particles_entr(i) > 0.0);
						sph_particles_entr0(i) = sph_particles_dentr2(i);
						sph_particles_entr(i) += sph_particles_dentr2(i) * dt2;
						if (sph_particles_dentr2(i) < -1.0e-7 / sph_particles_entr(i)) {
							ALWAYS_ASSERT(sph_particles_dentr2(i) >= 0.f);
						}
						ALWAYS_ASSERT(sph_particles_entr(i) > 0.0);
#else
						ALWAYS_ASSERT(sph_particles_eint(i) > 0.0);
						sph_particles_eint0(i) = sph_particles_deint2(i);
						sph_particles_eint(i) += sph_particles_deint2(i) * dt2;
						ALWAYS_ASSERT(sph_particles_eint(i) > 0.0);
#endif
					}
				} else if (phase == 1) {
					if (rung2 >= minrung && !sph_particles_isstar(i)) {
						for (int dim = 0; dim < NDIM; dim++) {
							particles_vel(dim, k) += (sph_particles_dvel(dim, i) - sph_particles_dvel0(dim, i)) * dt2;
						}
#ifdef ENTROPY
						sph_particles_entr(i) += (-sph_particles_entr0(i)) * dt2;
						if (sph_particles_entr(i) <= 0.0) {
							PRINT("%e %e\n", sph_particles_entr(i), (-sph_particles_entr0(i)) * dt2);
						}
						ALWAYS_ASSERT(sph_particles_entr(i) > 0.0);
						sph_particles_entr(i) += (sph_particles_dentr2(i)) * dt2;
						if (sph_particles_entr(i) <= 0.0) {
							PRINT("%e %e\n", sph_particles_entr(i), (sph_particles_dentr2(i)) * dt2);
						}
						ALWAYS_ASSERT(sph_particles_entr(i) > 0.0);
#else
						sph_particles_eint(i) += (-sph_particles_eint0(i)) * dt2;
						if (sph_particles_eint(i) <= 0.0) {
							PRINT("%e %e\n", sph_particles_eint(i), (-sph_particles_eint0(i)) * dt2);
						}
						ALWAYS_ASSERT(sph_particles_eint(i) > 0.0);
						sph_particles_eint(i) += (sph_particles_deint2(i)) * dt2;
						if (sph_particles_eint(i) <= 0.0) {
							PRINT("%e %e\n", sph_particles_eint(i), (sph_particles_deint2(i)) * dt2);
						}
						ALWAYS_ASSERT(sph_particles_eint(i) > 0.0);
#endif
					}
				} else if (phase == 2) {
					if (rung2 >= minrung && !sph_particles_isstar(i)) {
						for (int dim = 0; dim < NDIM; dim++) {
							particles_vel(dim, k) += sph_particles_dvel(dim, i) * dt2;
						}
#ifdef ENTROPY
						sph_particles_entr(i) += sph_particles_dentr1(i) * 2.0 * dt2;
						ALWAYS_ASSERT(sph_particles_entr(i) > 0.0);
						sph_particles_entr(i) += sph_particles_dentr2(i) * dt2;
						ALWAYS_ASSERT(sph_particles_entr(i) > 0.0);
#else
						sph_particles_eint(i) += sph_particles_deint1(i) * 2.0 * dt2;
						ALWAYS_ASSERT(sph_particles_eint(i) > 0.0);
						sph_particles_eint(i) += sph_particles_deint2(i) * dt2;
						ALWAYS_ASSERT(sph_particles_eint(i) > 0.0);
#endif
						if (chem) {
							for (int fi = 0; fi < NCHEMFRACS; fi++) {
								auto& frac = sph_particles_frac(fi, i);
								auto dfrac = sph_particles_dchem(i)[fi];
								if (frac + dfrac * dt2 * 2.0 < 0.0f) {
									PRINT("-----> %e %e\n", frac, frac + dfrac * dt2 * 2.0);
								}
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

HPX_PLAIN_ACTION(sph_particles_softlens2smoothlens);

void sph_particles_softlens2smoothlens(int minrung) {
	profiler_enter(__FUNCTION__);
	double err = 0.0;
	double norm = 0.0;
	vector<hpx::future<void>> futs;
	for (auto& c : hpx_children()) {
		futs.push_back(hpx::async<sph_particles_softlens2smoothlens_action>(c, minrung));
	}
	int const nthreads = hpx_hardware_concurrency();
	for (int proc = 0; proc < nthreads; proc++) {
		futs.push_back(hpx::async([nthreads, proc, minrung]() {
			part_int const b = (size_t)proc * sph_particles_size() / nthreads;
			part_int const e = (size_t)(proc + 1) * sph_particles_size() / nthreads;
			for (int i = b; i < e; i++) {
				part_int const k = sph_particles_dm_index(i);
				//		if( particles_rung(k) >= minrung || particles_semiactive(k)) {
				sph_particles_smooth_len(i) = particles_softlen(k);
				//	}
			}
		}));
	}
	for (auto& f : futs) {
		f.get();
	}
	profiler_exit();
}

float sph_particles_temperature(part_int i, float a) {
	if (sph_particles_isstar(i)) {
		return 0.0;
	} else {
		double const code_to_energy_density = get_options().code_to_g / (get_options().code_to_cm * sqr(get_options().code_to_s)); // 7
		double const code_to_energy = sqr(get_options().code_to_cm) / (sqr(get_options().code_to_s));							   // 7
		double const code_to_density = pow(get_options().code_to_cm, -3) * get_options().code_to_g;								   // 10
		double const code_to_entropy = code_to_energy / pow(code_to_density, get_options().gamma - 1.0);
		double const h = sph_particles_smooth_len(i);
		double const Hp = sph_particles_Hp(i);
		double const Hn = sph_particles_Hn(i);
		double const H2 = sph_particles_H2(i);
		double const Y = sph_particles_Y(i);
		double const Hep = sph_particles_Hep(i);
		double const Hepp = sph_particles_Hepp(i);
		double const H = sph_particles_H(i);
		double const He = Y - Hep - Hepp;
#ifdef HOPKINS
		double rho = sph_particles_rho_rho(i);
#else
		double rho = sph_particles_rho(i);
#endif
		double const hfrac = 1.0 - sph_particles_cold_mass(i);
		double n = (H + 2.f * Hp + .5f * H2 + .25f * He + .5f * Hep + .75f * Hepp) * 1.0 / (1.0 - sph_particles_Z(i));
		rho *= code_to_density * pow(a, -3);
		n *= constants::avo * rho; // 8
		double gamma = sph_particles_gamma(i);
		double cv = 1.0 / (gamma - 1.0); // 4
		cv *= double(constants::kb);	 // 1
#ifdef ENTROPY
		double entr = sph_particles_entr(i);
		entr *= code_to_entropy;
		double const eint = entr * pow(rho * hfrac, get_options().gamma - 1.0) / (gamma - 1.0);
#else
		double const eint = sph_particles_eint(i) * code_to_energy;
#endif
		double T = rho * eint / (n * cv);
		if (H < 0.0) {
			if (H < -5.0e-3) {
				PRINT("NEGATIVE H\n");
				PRINT("%e %e %e %e %e %e %e\n", H, Hp, Hn, H2, He, Hep, Hepp);
				abort();
			}
		}
		if (T > TMAX) {
			PRINT("T == %e %e %e %e %e\n", T, eint, eint, rho, h);
			abort();
		}
		if (T < 0.0) {
			PRINT("T == %e %e %e %e %e\n", T, eint, eint, rho, h);
		}
		return T;
	}
}

float sph_particles_mmw(part_int i) {
	double const Hp = sph_particles_Hp(i);
	double const Hn = sph_particles_Hn(i);
	double const H2 = sph_particles_H2(i);
	double const Y = sph_particles_Y(i);
	double const Hep = sph_particles_Hep(i);
	double const Hepp = sph_particles_Hepp(i);
	double const H = sph_particles_H(i);
	double const He = Y - Hep - Hepp;
	double n = H + 2.f * Hp + .5f * H2 + .25f * He + .5f * Hep + .75f * Hepp;
	//	PRINT( "%e\n", 1.0 / n);
	return 1.0 / n;
}

namespace {

template <typename Visitor> void visitSphCheckpointArrays(Visitor&& visit) {
	size_t const count = sph_particles_size();
	// Keep stage fields as well as the physical fields: fine rungs may be staggered at checkpoints.
	visit(sph_particles_r1, count);
	visit(sph_particles_r2, count);
	visit(sph_particles_r5, count);
	visit(sph_particles_r6, count);
	visit(sph_particles_dm, count);
	visit(sph_particles_dv, count);
	visit(sph_particles_kap, count);
	visit(sph_particles_e0, count);
	visit(sph_particles_de3, count);
	visit(sph_particles_de2, count);
	visit(sph_particles_fp2, count);
	visit(sph_particles_pre, count);
	visit(sph_particles_fp1, count);
	visit(sph_particles_s2, count);
	visit(sph_particles_fc, count);
	visit(sph_particles_rh, count);
	visit(sph_particles_tc, count);
	visit(sph_particles_or, count);
	visit(sph_particles_sa, count);
	visit(sph_particles_st, count);
	for (int dim = 0; dim < NDIM; dim++) {
		visit(sph_particles_dv2[dim], count);
		visit(sph_particles_g[dim], count);
	}
}

uint32_t sphCheckpointMode() {
#ifdef ENTROPY
	return 1;
#else
	return 0;
#endif
}

} // namespace

void sph_particles_swap(part_int i, part_int j) {
	visitSphCheckpointArrays([i, j](auto* values, size_t) { std::swap(values[i], values[j]); });
}

void sph_particles_swap2(part_int i, part_int j) { sph_particles_swap(i, j); }

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

part_int sph_particles_size() { return size; }

template <typename T> void sph_particles_array_resize(T*& ptr, part_int new_capacity, bool reg) {
	T* new_ptr;
#ifdef USE_CUDA
	if (reg) {
		cudaMallocManaged(&new_ptr, sizeof(T) * new_capacity);
	} else {
		new_ptr = (T*)malloc(sizeof(T) * new_capacity);
	}
#else
	new_ptr = (T*)malloc(sizeof(T) * new_capacity);
	memset(new_ptr, T(0), new_capacity);
#endif
	if (capacity > 0) {
		hpx_copy(PAR_EXECUTION_POLICY, ptr, ptr + size, new_ptr).get();
#ifdef USE_CUDA
		if (reg) {
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
	bool const chem = get_options().chem;
	bool const stars = get_options().stars;
	if (sz > capacity) {
		part_int new_capacity = std::max(capacity, (part_int)100);
		while (new_capacity < sz) {
			new_capacity = size_t(101) * new_capacity / size_t(100);
		}
		//	PRINT("Resizing sph_particles to %li from %li\n", new_capacity, capacity);
		sph_particles_array_resize(sph_particles_rh, new_capacity, true);
		sph_particles_array_resize(sph_particles_st, new_capacity, true);
		sph_particles_array_resize(sph_particles_fc, new_capacity, true);
		sph_particles_array_resize(sph_particles_dv, new_capacity, true);
		sph_particles_array_resize(sph_particles_r1, new_capacity, true);
		sph_particles_array_resize(sph_particles_r2, new_capacity, true);
		sph_particles_array_resize(sph_particles_r5, new_capacity, true);
		sph_particles_array_resize(sph_particles_r6, new_capacity, true);
		sph_particles_array_resize(sph_particles_kap, new_capacity, true);
		sph_particles_array_resize(sph_particles_or, new_capacity, true);
		sph_particles_array_resize(sph_particles_sa, new_capacity, true);
		sph_particles_array_resize(sph_particles_dm, new_capacity, true);
		sph_particles_array_resize(sph_particles_e0, new_capacity, true);
		sph_particles_array_resize(sph_particles_de2, new_capacity, true);
		sph_particles_array_resize(sph_particles_de3, new_capacity, true);
		sph_particles_array_resize(sph_particles_fp1, new_capacity, true);
		sph_particles_array_resize(sph_particles_fp2, new_capacity, true);
		sph_particles_array_resize(sph_particles_pre, new_capacity, true);
		sph_particles_array_resize(sph_particles_s2, new_capacity, true);
		sph_particles_array_resize(sph_particles_tc, new_capacity, false);
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
		part_int const slot = oldsz + i;
		sph_particles_r1[slot] = {};
		sph_particles_r2[slot] = {};
		sph_particles_r5[slot] = {};
		sph_particles_r6[slot] = {};
		sph_particles_dm[slot] = NO_INDEX;
		sph_particles_dv[slot] = 0.0f;
		sph_particles_kap[slot] = 0.0f;
		sph_particles_e0[slot] = 0.0f;
		sph_particles_de3[slot] = 0.0f;
		sph_particles_de2[slot] = 0.0f;
		sph_particles_fp2[slot] = 0.0f;
		sph_particles_pre[slot] = 0.0f;
		sph_particles_fp1[slot] = 0.0f;
		sph_particles_s2[slot] = 0.0f;
		sph_particles_fc[slot] = 0.0f;
		sph_particles_rh[slot] = 0.0f;
		sph_particles_tc[slot] = 0.0f;
		sph_particles_or[slot] = 0;
		sph_particles_sa[slot] = 0;
		sph_particles_st[slot] = 0;
		for (int dim = 0; dim < NDIM; dim++) {
			sph_particles_dv2[dim][slot] = 0.0f;
			sph_particles_g[dim][slot] = 0.0f;
		}
		if (parts2) {
			particles_cat_index(offset + i) = oldsz + i;
			particles_type(offset + i) = SPH_TYPE;
			sph_particles_dm_index(oldsz + i) = offset + i;
		}
		sph_particles_alpha(oldsz + i) = get_options().alpha0;
		sph_particles_divv(oldsz + i) = 0.f;
		sph_particles_semiactive(oldsz + i) = 0.0f;
		if (stars) {
			sph_particles_cold_mass((oldsz + i)) = 0.f;
		}
		sph_particles_isstar((oldsz + i)) = false;
#ifdef ENTROPY
		sph_particles_dentr1(oldsz + i) = 0.f;
		sph_particles_dentr2(oldsz + i) = 0.f;
#else
		sph_particles_deint1(oldsz + i) = 0.f;
		sph_particles_deint2(oldsz + i) = 0.f;
#endif
		for (int dim = 0; dim < NDIM; dim++) {
			sph_particles_dvel(dim, oldsz + i) = 0.0f;
			sph_particles_dvel0(dim, oldsz + i) = 0.0f;
			sph_particles_gforce(dim, oldsz + i) = 0.0f;
		}
	}
}

void sph_particles_free() {
	bool const stars = get_options().stars;
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
	CUDA_CHECK(cudaFree(sph_particles_r1));
	if (stars) {
		//		CUDA_CHECK(cudaFree(sph_particles_fZ));
		//			CUDA_CHECK(cudaFree(sph_particles_fY));
		//			CUDA_CHECK(cudaFree(sph_particles_sn));
		//			CUDA_CHECK(cudaFree(sph_particles_dz));
	}
}

void sph_particles_resolve_with_particles() {
	int const nthread = hpx_hardware_concurrency();
	std::vector<hpx::future<void>> futs;
	for (int proc = 0; proc < nthread; proc++) {
		futs.push_back(hpx::async([proc, nthread] {
			part_int const b = (size_t)proc * sph_particles_size() / nthread;
			part_int const e = (size_t)(proc + 1) * sph_particles_size() / nthread;
			for (part_int i = b; i < e; i++) {
				int const j = sph_particles_dm_index(i);
				particles_cat_index(j) = i;
			}
		}));
	}
	hpx::wait_all(futs.begin(), futs.end());
}

void sph_particles_global_read_pos(particle_global_range range, fixed32* x, fixed32* y, fixed32* z, part_int offset) {
	part_int const line_size = get_options().part_cache_line_size;
	if (range.range.first != range.range.second) {
		if (range.proc == hpx_rank()) {
			part_int const dif = offset - range.range.first;
			part_int const sz = range.range.second - range.range.first;
			for (part_int i = range.range.first; i < range.range.second; i++) {
				int const j = offset + i - range.range.first;
				x[j] = sph_particles_pos(XDIM, i);
				y[j] = sph_particles_pos(YDIM, i);
				z[j] = sph_particles_pos(ZDIM, i);
			}
		} else {
			line_id_type line_id;
			line_id.proc = range.proc;
			part_int const start_line = (range.range.first / line_size) * line_size;
			part_int const stop_line = ((range.range.second - 1) / line_size) * line_size;
			part_int dest_index = offset;
			for (part_int line = start_line; line <= stop_line; line += line_size) {
				line_id.index = line;
				auto const* ptr = sph_particles_cache_read_line(line_id);
				auto const begin = std::max(line_id.index, range.range.first);
				auto const end = std::min(line_id.index + line_size, range.range.second);
				for (part_int i = begin; i < end; i++) {
					part_int const src_index = i - line_id.index;
					x[dest_index] = ptr[src_index][XDIM];
					y[dest_index] = ptr[src_index][YDIM];
					z[dest_index] = ptr[src_index][ZDIM];
					dest_index++;
				}
			}
		}
	}
}

static array<fixed32, NDIM> const* sph_particles_cache_read_line(line_id_type line_id) {
	return readSphCacheLine<sph_particles_fetch_cache_line_action>(line_id, part_cache, mutexes);
}

static vector<array<fixed32, NDIM>> sph_particles_fetch_cache_line(part_int index) {
	part_int const line_size = get_options().part_cache_line_size;
	vector<array<fixed32, NDIM>> line(line_size);
	part_int const begin = (index / line_size) * line_size;
	part_int const end = std::min(sph_particles_size(), begin + line_size);
	for (part_int i = begin; i < end; i++) {
		for (int dim = 0; dim < NDIM; dim++) {
			line[i - begin][dim] = sph_particles_pos(dim, i);
		}
	}
	return line;
}

void sph_particles_global_read_fcold(particle_global_range range, float* cfrac, char* isstar, part_int offset) {
	part_int const line_size = get_options().part_cache_line_size;
	int const sz = offset + range.range.second - range.range.first;
	if (range.range.first != range.range.second) {
		if (range.proc == hpx_rank()) {
			part_int const dif = offset - range.range.first;
			part_int const sz = range.range.second - range.range.first;
			for (part_int i = range.range.first; i < range.range.second; i++) {
				int const j = offset + i - range.range.first;
				if (cfrac) {
					cfrac[j] = sph_particles_cold_mass(i);
				}
				if (isstar) {
					isstar[j] = sph_particles_isstar(i);
				}
			}
		} else {
			line_id_type line_id;
			line_id.proc = range.proc;
			part_int const start_line = (range.range.first / line_size) * line_size;
			part_int const stop_line = ((range.range.second - 1) / line_size) * line_size;
			part_int dest_index = offset;
			for (part_int line = start_line; line <= stop_line; line += line_size) {
				line_id.index = line;
				auto const* ptr = sph_particles_fcold_cache_read_line(line_id);
				auto const begin = std::max(line_id.index, range.range.first);
				auto const end = std::min(line_id.index + line_size, range.range.second);
				for (part_int i = begin; i < end; i++) {
					part_int const src_index = i - line_id.index;
					pair<float, char> const& part = ptr[src_index];
					if (cfrac) {
						cfrac[dest_index] = part.first;
					}
					if (isstar) {
						isstar[dest_index] = part.second;
					}
					dest_index++;
				}
			}
		}
	}
}

static pair<float, char> const* sph_particles_fcold_cache_read_line(line_id_type line_id) {
	return readSphCacheLine<sph_particles_fetch_fcold_cache_line_action>(line_id, fcold_part_cache, fcold_mutexes);
}

static vector<pair<float, char>> sph_particles_fetch_fcold_cache_line(part_int index) {
	part_int const line_size = get_options().part_cache_line_size;
	vector<pair<float, char>> line(line_size);
	part_int const begin = (index / line_size) * line_size;
	part_int const end = std::min(sph_particles_size(), begin + line_size);
	for (part_int i = begin; i < end; i++) {
		line[i - begin].first = sph_particles_cold_mass(i);
		line[i - begin].second = sph_particles_isstar(i);
	}
	return line;
}

void sph_particles_global_read_energy_and_smoothlen(particle_global_range range, float* ent, float* h, part_int offset) {
	part_int const line_size = get_options().part_cache_line_size;
	int const sz = offset + range.range.second - range.range.first;
	if (range.range.first != range.range.second) {
		if (range.proc == hpx_rank()) {
			part_int const dif = offset - range.range.first;
			part_int const sz = range.range.second - range.range.first;
			for (part_int i = range.range.first; i < range.range.second; i++) {
				int const j = offset + i - range.range.first;
				if (ent) {
#ifdef ENTROPY
					ent[j] = sph_particles_entr(i);
#else
					ent[j] = sph_particles_eint(i);
#endif
				}
				if (h) {
					h[j] = sph_particles_smooth_len(i);
				}
			}
		} else {
			line_id_type line_id;
			line_id.proc = range.proc;
			part_int const start_line = (range.range.first / line_size) * line_size;
			part_int const stop_line = ((range.range.second - 1) / line_size) * line_size;
			part_int dest_index = offset;
			for (part_int line = start_line; line <= stop_line; line += line_size) {
				line_id.index = line;
				auto const* ptr = sph_particles_energy_and_smoothlen_cache_read_line(line_id);
				auto const begin = std::max(line_id.index, range.range.first);
				auto const end = std::min(line_id.index + line_size, range.range.second);
				for (part_int i = begin; i < end; i++) {
					part_int const src_index = i - line_id.index;
					if (ent) {
						ent[dest_index] = ptr[src_index].first;
					}
					if (h) {
						h[dest_index] = ptr[src_index].second;
					}
					dest_index++;
				}
			}
		}
	}
}

static pair<float> const* sph_particles_energy_and_smoothlen_cache_read_line(line_id_type line_id) {
	return readSphCacheLine<sph_particles_fetch_energy_and_smoothlen_cache_line_action>(line_id, entr_part_cache, entr_mutexes);
}

static vector<pair<float>> sph_particles_fetch_energy_and_smoothlen_cache_line(part_int index) {
	part_int const line_size = get_options().part_cache_line_size;
	vector<pair<float>> line(line_size);
	part_int const begin = (index / line_size) * line_size;
	part_int const end = std::min(sph_particles_size(), begin + line_size);
	for (part_int i = begin; i < end; i++) {
#ifdef ENTROPY
		line[i - begin].first = sph_particles_entr(i);
#else
		line[i - begin].first = sph_particles_eint(i);
#endif
		line[i - begin].second = sph_particles_smooth_len(i);
	}
	return line;
}

void sph_particles_global_read_rungs(particle_global_range range, char* rungs, part_int offset) {
	part_int const line_size = get_options().part_cache_line_size;
	if (range.range.first != range.range.second) {
		if (range.proc == hpx_rank()) {
			part_int const dif = offset - range.range.first;
			part_int const sz = range.range.second - range.range.first;
			for (part_int i = range.range.first; i < range.range.second; i++) {
				int const j = offset + i - range.range.first;
				rungs[j] = sph_particles_rung(i);
			}
		} else {
			line_id_type line_id;
			line_id.proc = range.proc;
			part_int const start_line = (range.range.first / line_size) * line_size;
			part_int const stop_line = ((range.range.second - 1) / line_size) * line_size;
			part_int dest_index = offset;
			for (part_int line = start_line; line <= stop_line; line += line_size) {
				line_id.index = line;
				auto const* ptr = sph_particles_rung_cache_read_line(line_id);
				auto const begin = std::max(line_id.index, range.range.first);
				auto const end = std::min(line_id.index + line_size, range.range.second);
				for (part_int i = begin; i < end; i++) {
					part_int const src_index = i - line_id.index;
					rungs[dest_index] = ptr[src_index];
					dest_index++;
				}
			}
		}
	}
}

static char const* sph_particles_rung_cache_read_line(line_id_type line_id) {
	return readSphCacheLine<sph_particles_fetch_rung_cache_line_action>(line_id, rung_part_cache, rung_mutexes);
}

static vector<char> sph_particles_fetch_rung_cache_line(part_int index) {
	part_int const line_size = get_options().part_cache_line_size;
	vector<char> line(line_size);
	part_int const begin = (index / line_size) * line_size;
	part_int const end = std::min(sph_particles_size(), begin + line_size);
	for (part_int i = begin; i < end; i++) {
		line[i - begin] = sph_particles_rung(i);
	}
	return line;
}

void sph_particles_global_read_vels(particle_global_range range, float* vx, float* vy, float* vz, part_int offset) {
	part_int const line_size = get_options().part_cache_line_size;
	if (range.range.first != range.range.second) {
		if (range.proc == hpx_rank()) {
			part_int const dif = offset - range.range.first;
			part_int const sz = range.range.second - range.range.first;
			for (part_int i = range.range.first; i < range.range.second; i++) {
				int const j = offset + i - range.range.first;
				int const k = sph_particles_dm_index(i);
				vx[j] = particles_vel(XDIM, k);
				vy[j] = particles_vel(YDIM, k);
				vz[j] = particles_vel(ZDIM, k);
			}
		} else {
			line_id_type line_id;
			line_id.proc = range.proc;
			part_int const start_line = (range.range.first / line_size) * line_size;
			part_int const stop_line = ((range.range.second - 1) / line_size) * line_size;
			part_int dest_index = offset;
			for (part_int line = start_line; line <= stop_line; line += line_size) {
				line_id.index = line;
				auto const* ptr = sph_particles_vels_cache_read_line(line_id);
				auto const begin = std::max(line_id.index, range.range.first);
				auto const end = std::min(line_id.index + line_size, range.range.second);
				for (part_int i = begin; i < end; i++) {
					part_int const src_index = i - line_id.index;
					vx[dest_index] = ptr[src_index][XDIM];
					vy[dest_index] = ptr[src_index][YDIM];
					vz[dest_index] = ptr[src_index][ZDIM];
					dest_index++;
				}
			}
		}
	}
}

static array<float, NDIM> const* sph_particles_vels_cache_read_line(line_id_type line_id) {
	return readSphCacheLine<sph_particles_fetch_vels_cache_line_action>(line_id, vels_part_cache, vels_mutexes);
}

static vector<array<float, NDIM>> sph_particles_fetch_vels_cache_line(part_int index) {
	part_int const line_size = get_options().part_cache_line_size;
	vector<array<float, NDIM>> line(line_size);
	part_int const begin = (index / line_size) * line_size;
	part_int const end = std::min(sph_particles_size(), begin + line_size);
	for (part_int i = begin; i < end; i++) {
		int const k = sph_particles_dm_index(i);
		line[i - begin][XDIM] = particles_vel(k, XDIM);
		line[i - begin][YDIM] = particles_vel(k, YDIM);
		line[i - begin][ZDIM] = particles_vel(k, ZDIM);
	}
	return line;
}

void sph_particles_global_read_kappas(particle_global_range range, float* kappas, part_int offset) {
	part_int const line_size = get_options().part_cache_line_size;
	if (range.range.first != range.range.second) {
		if (range.proc == hpx_rank()) {
			part_int const dif = offset - range.range.first;
			part_int const sz = range.range.second - range.range.first;
			for (part_int i = range.range.first; i < range.range.second; i++) {
				int const j = offset + i - range.range.first;
				kappas[j] = sph_particles_kappa(i);
			}
		} else {
			line_id_type line_id;
			line_id.proc = range.proc;
			part_int const start_line = (range.range.first / line_size) * line_size;
			part_int const stop_line = ((range.range.second - 1) / line_size) * line_size;
			part_int dest_index = offset;
			for (part_int line = start_line; line <= stop_line; line += line_size) {
				line_id.index = line;
				auto const* ptr = sph_particles_kappas_cache_read_line(line_id);
				auto const begin = std::max(line_id.index, range.range.first);
				auto const end = std::min(line_id.index + line_size, range.range.second);
				for (part_int i = begin; i < end; i++) {
					part_int const src_index = i - line_id.index;
					kappas[dest_index] = ptr[src_index];
					dest_index++;
				}
			}
		}
	}
}

static float const* sph_particles_kappas_cache_read_line(line_id_type line_id) {
	return readSphCacheLine<sph_particles_fetch_kappas_cache_line_action>(line_id, kappas_part_cache, kappas_mutexes);
}

static vector<float> sph_particles_fetch_kappas_cache_line(part_int index) {
	part_int const line_size = get_options().part_cache_line_size;
	vector<float> line(line_size);
	part_int const begin = (index / line_size) * line_size;
	part_int const end = std::min(sph_particles_size(), begin + line_size);
	for (part_int i = begin; i < end; i++) {
		line[i - begin] = sph_particles_kappa(i);
	}
	return line;
}

void sph_particles_global_read_rho(particle_global_range range, float* rho, part_int offset) {
	part_int const line_size = get_options().part_cache_line_size;
	if (range.range.first != range.range.second) {
		if (range.proc == hpx_rank()) {
			part_int const dif = offset - range.range.first;
			part_int const sz = range.range.second - range.range.first;
			for (part_int i = range.range.first; i < range.range.second; i++) {
				int const j = offset + i - range.range.first;
#ifdef HOPKINS
				rho[j] = sph_particles_rho_rho(i);
#else
				rho[j] = sph_particles_rho(i);
#endif
			}
		} else {
			line_id_type line_id;
			line_id.proc = range.proc;
			part_int const start_line = (range.range.first / line_size) * line_size;
			part_int const stop_line = ((range.range.second - 1) / line_size) * line_size;
			part_int dest_index = offset;
			for (part_int line = start_line; line <= stop_line; line += line_size) {
				line_id.index = line;
				auto const* ptr = sph_particles_rho_cache_read_line(line_id);
				auto const begin = std::max(line_id.index, range.range.first);
				auto const end = std::min(line_id.index + line_size, range.range.second);
				for (part_int i = begin; i < end; i++) {
					part_int const src_index = i - line_id.index;
					rho[dest_index] = ptr[src_index];
					dest_index++;
				}
			}
		}
	}
}

static float const* sph_particles_rho_cache_read_line(line_id_type line_id) {
	return readSphCacheLine<sph_particles_fetch_rho_cache_line_action>(line_id, rho_part_cache, rho_mutexes);
}

static vector<float> sph_particles_fetch_rho_cache_line(part_int index) {
	part_int const line_size = get_options().part_cache_line_size;
	vector<float> line(line_size);
	part_int const begin = (index / line_size) * line_size;
	part_int const end = std::min(sph_particles_size(), begin + line_size);
	for (part_int i = begin; i < end; i++) {
#ifdef HOPKINS
		line[i - begin] = sph_particles_rho_rho(i);
#else
		line[i - begin] = sph_particles_rho(i);
#endif
	}
	return line;
}

void sph_particles_global_read_aux(particle_global_range range, float* alpha, float* omega, float* omegaP, float* pre, float* shearv,
								   array<float, NCHEMFRACS>* fracs, part_int offset) {
	part_int const line_size = get_options().part_cache_line_size;
	if (range.range.first == range.range.second) {
		return;
	}
	if (range.proc == hpx_rank()) {
		part_int const dif = offset - range.range.first;
		part_int const sz = range.range.second - range.range.first;
		for (part_int i = range.range.first; i < range.range.second; i++) {
			int const j = offset + i - range.range.first;
			if (alpha) {
				alpha[j] = sph_particles_alpha(i);
			}
			if (omega) {
				omega[j] = sph_particles_omega(i);
			}
			if (omegaP) {
				omegaP[j] = sph_particles_omegaP(i);
			}
			if (pre) {
				pre[j] = sph_particles_pressure(i);
			}
			if (shearv) {
				shearv[j] = sph_particles_shear(i);
			}
			if (fracs) {
				fracs[j] = sph_particles_chem(i);
			}
		}
	} else {
		line_id_type line_id;
		line_id.proc = range.proc;
		part_int const start_line = (range.range.first / line_size) * line_size;
		part_int const stop_line = ((range.range.second - 1) / line_size) * line_size;
		part_int dest_index = offset;
		for (part_int line = start_line; line <= stop_line; line += line_size) {
			line_id.index = line;
			auto const* ptr = sph_particles_aux_cache_read_line(line_id);
			auto const begin = std::max(line_id.index, range.range.first);
			auto const end = std::min(line_id.index + line_size, range.range.second);
			for (part_int i = begin; i < end; i++) {
				part_int const src_index = i - line_id.index;
				int const j = dest_index;
				if (alpha) {
					alpha[j] = ptr[src_index].alpha;
				}
				if (omega) {
					omega[j] = ptr[src_index].omega;
				}
				if (omegaP) {
					omegaP[j] = ptr[src_index].omegaP;
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
				dest_index++;
			}
		}
	}
}

static aux_quantities const* sph_particles_aux_cache_read_line(line_id_type line_id) {
	return readSphCacheLine<sph_particles_fetch_aux_cache_line_action>(line_id, aux_cache, aux_mutexes);
}

static vector<aux_quantities> sph_particles_fetch_aux_cache_line(part_int index) {
	part_int const line_size = get_options().part_cache_line_size;
	vector<aux_quantities> line(line_size);
	part_int const begin = (index / line_size) * line_size;
	part_int const end = std::min(sph_particles_size(), begin + line_size);
	for (part_int i = begin; i < end; i++) {
		line[i - begin] = sph_particles_aux_quantities(i);
	}
	return line;
}

void sph_particles_cache_free1() {
	profiler_enter(__FUNCTION__);
	static auto const stars = get_options().stars;
	vector<hpx::future<void>> futs;
	for (auto const& c : hpx_children()) {
		futs.push_back(hpx::async<sph_particles_cache_free1_action>(c));
	}
	int const nthreads = hpx_hardware_concurrency();
	entr_part_cache = decltype(entr_part_cache)();
	vels_part_cache = decltype(vels_part_cache)();
	aux_cache = decltype(aux_cache)();
	hpx::wait_all(futs.begin(), futs.end());
	profiler_exit();
}

#ifdef ENTROPY
void sph_particles_cache_free_entr() {
	profiler_enter(__FUNCTION__);
	static auto const stars = get_options().stars;
	vector<hpx::future<void>> futs;
	for (auto const& c : hpx_children()) {
		futs.push_back(hpx::async<sph_particles_cache_free_entr_action>(c));
	}
	int const nthreads = hpx_hardware_concurrency();
	entr_part_cache = decltype(entr_part_cache)();
	hpx::wait_all(futs.begin(), futs.end());
	profiler_exit();
}
#else
void sph_particles_cache_free_eint() {
	profiler_enter(__FUNCTION__);
	static auto const stars = get_options().stars;
	vector<hpx::future<void>> futs;
	for (auto const& c : hpx_children()) {
		futs.push_back(hpx::async<sph_particles_cache_free_eint_action>(c));
	}
	int const nthreads = hpx_hardware_concurrency();
	entr_part_cache = decltype(entr_part_cache)();
	hpx::wait_all(futs.begin(), futs.end());
	profiler_exit();
}
#endif

void sph_particles_cache_free2() {
	profiler_enter(__FUNCTION__);
	static auto const stars = get_options().stars;
	vector<hpx::future<void>> futs;
	for (auto const& c : hpx_children()) {
		futs.push_back(hpx::async<sph_particles_cache_free2_action>(c));
	}
	int const nthreads = hpx_hardware_concurrency();
	part_cache = decltype(part_cache)();
	entr_part_cache = decltype(entr_part_cache)();
	rung_part_cache = decltype(rung_part_cache)();
	fcold_part_cache = decltype(fcold_part_cache)();
	vels_part_cache = decltype(vels_part_cache)();
	kappas_part_cache = decltype(kappas_part_cache)();
	rho_part_cache = decltype(rho_part_cache)();
	aux_cache = decltype(aux_cache)();
	hpx::wait_all(futs.begin(), futs.end());
	profiler_exit();
}

void sph_particles_load(FILE* fp) {
	uint32_t header[7];
	part_int savedCount;
	FREAD(header, sizeof(uint32_t), 7, fp);
	FREAD(&savedCount, sizeof(part_int), 1, fp);
	if (header[0] != 0x43545350U || header[1] != 1 || header[2] != sizeof(part_int) || header[3] != sizeof(sph_record1) || header[4] != sizeof(sph_record2) ||
		header[5] != NDIM || header[6] != sphCheckpointMode() || savedCount != sph_particles_size()) {
		THROW_ERROR("Incompatible SPH checkpoint payload\n");
	}
	visitSphCheckpointArrays([fp](auto* values, size_t count) {
		if (count != 0) {
			FREAD(values, sizeof(*values), count, fp);
		}
	});
}

void sph_particles_save(FILE* fp) {
	uint32_t header[7] = {0x43545350U, 1, sizeof(part_int), sizeof(sph_record1), sizeof(sph_record2), NDIM, sphCheckpointMode()};
	part_int count = sph_particles_size();
	FWRITE(header, sizeof(uint32_t), 7, fp);
	FWRITE(&count, sizeof(part_int), 1, fp);
	visitSphCheckpointArrays([fp](auto* values, size_t count) {
		if (count != 0) {
			FWRITE(values, sizeof(*values), count, fp);
		}
	});
}

void sphCheckpointTest() {
	vector<unsigned char> expected;
	FILE* file = std::tmpfile();
	if (file == nullptr) {
		THROW_ERROR("Unable to create SPH checkpoint test file\n");
	}
	sph_particles_save(file);
	visitSphCheckpointArrays([&expected](auto* values, size_t count) {
		size_t const bytes = sizeof(*values) * count;
		if (bytes != 0) {
			auto* data = reinterpret_cast<unsigned char*>(values);
			expected.insert(expected.end(), data, data + bytes);
			std::memset(data, 0x5a, bytes);
		}
	});
	std::rewind(file);
	sph_particles_load(file);
	size_t offset = 0;
	visitSphCheckpointArrays([&expected, &offset](auto* values, size_t count) {
		size_t const bytes = sizeof(*values) * count;
		if (bytes != 0) {
			if (std::memcmp(values, expected.data() + offset, bytes) != 0) {
				THROW_ERROR("SPH checkpoint did not restore every gas field\n");
			}
			offset += bytes;
		}
	});
	if (sph_particles_size() >= 2) {
		visitSphCheckpointArrays([](auto* values, size_t) {
			std::memset(values, 0x11, sizeof(*values));
			std::memset(values + 1, 0x22, sizeof(*values));
		});
		auto checkOrder = [](unsigned char first, unsigned char second) {
			visitSphCheckpointArrays([first, second](auto* values, size_t) {
				auto const* bytes = reinterpret_cast<unsigned char const*>(values);
				for (size_t i = 0; i < sizeof(*values); i++) {
					if (bytes[i] != first || bytes[sizeof(*values) + i] != second) {
						THROW_ERROR("SPH reorder did not preserve every gas field\n");
					}
				}
			});
		};
		sph_particles_swap(0, 1);
		checkOrder(0x22, 0x11);
		sph_particles_swap2(0, 1);
		checkOrder(0x11, 0x22);
		std::rewind(file);
		sph_particles_load(file);
		PRINT("SPH reorder passed (all fields, gravity and predictor stages)\n");
	}
	part_int const savedSize = size;
	size = 0;
	std::rewind(file);
	sph_particles_save(file);
	std::rewind(file);
	sph_particles_load(file);
	size = savedSize;
	std::fclose(file);
	PRINT("SPH checkpoint roundtrip passed (all fields and empty gas payload)\n");
}

HPX_PLAIN_ACTION(sph_particles_reset_converged);

void sph_particles_reset_converged() {
	int const nthread = hpx_hardware_concurrency();
	std::vector<hpx::future<void>> futs;
	for (auto& c : hpx_children()) {
		futs.push_back(hpx::async<sph_particles_reset_converged_action>(c));
	}
	for (int proc = 0; proc < nthread; proc++) {
		futs.push_back(hpx::async([proc, nthread] {
			part_int const b = (size_t)proc * sph_particles_size() / nthread;
			part_int const e = (size_t)(proc + 1) * sph_particles_size() / nthread;
			for (part_int i = b; i < e; i++) {
				sph_particles_converged(i) = false;
			}
		}));
	}
	hpx::wait_all(futs.begin(), futs.end());
}

HPX_PLAIN_ACTION(sph_particles_reset_semiactive);

void sph_particles_reset_semiactive() {
	int const nthread = hpx_hardware_concurrency();
	std::vector<hpx::future<void>> futs;
	for (auto& c : hpx_children()) {
		futs.push_back(hpx::async<sph_particles_reset_semiactive_action>(c));
	}
	for (int proc = 0; proc < nthread; proc++) {
		futs.push_back(hpx::async([proc, nthread] {
			part_int const b = (size_t)proc * sph_particles_size() / nthread;
			part_int const e = (size_t)(proc + 1) * sph_particles_size() / nthread;
			for (part_int i = b; i < e; i++) {
				sph_particles_semiactive(i) = false;
			}
		}));
	}
	hpx::wait_all(futs.begin(), futs.end());
}
