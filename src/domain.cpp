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

#include <shared_mutex>
#include <unordered_map>

#include <cosmictiger/containers.hpp>
#include <cosmictiger/domain.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/particles.hpp>
#include <cosmictiger/safe_io.hpp>
#include <cosmictiger/sph_particles.hpp>

constexpr bool verbose = true;

void domains_transmit_particles(vector<particle>);

void domains_init_rebounds();
void domains_transmit_boxes(std::unordered_map<size_t, domain_t> boxes);
void domains_rebound_sort(vector<double> bounds, int depth);
vector<size_t> domains_count_below(vector<double> bounds, int depth);
vector<size_t> domains_get_loads();

HPX_PLAIN_ACTION(domains_count_below);
HPX_PLAIN_ACTION(domains_get_loads);
HPX_PLAIN_ACTION(domains_rebound_sort);
HPX_PLAIN_ACTION(domains_begin);
HPX_PLAIN_ACTION(domains_end);
HPX_PLAIN_ACTION(domains_init_rebounds);
HPX_PLAIN_ACTION(domains_transmit_particles);
HPX_PLAIN_ACTION(domains_transmit_boxes);

static vector<part_int> free_indices;
static vector<vector<particle>> trans_particles;
static std::unordered_map<size_t, domain_t> boxes_by_key;

static int find_particle_domain(array<double, NDIM> const& x, size_t key = 1, int depth = 0);
static void domains_check();
static vector<domain_local> local_domains;

double domains_get_load_imbalance() {
	profiler_enter(__FUNCTION__);
	auto const loads = domains_get_loads();
	size_t max = 0;
	double avg = 0;
	for (auto const load : loads) {
		max = std::max(load, max);
		avg += load;
	}
	avg /= loads.size();
	double const imbalance = max / avg - 1.0;
	profiler_exit();
	return imbalance;
}

vector<size_t> domains_get_loads() {
	vector<hpx::future<vector<size_t>>> futs;
	for (auto const& c : hpx_children()) {
		futs.push_back(hpx::async<domains_get_loads_action>(c));
	}
	vector<size_t> loads;
	loads.push_back(particles_size());
	for (auto& f : futs) {
		auto these_loads = f.get();
		loads.insert(loads.end(), these_loads.begin(), these_loads.end());
	}
	return loads;
}

void domains_save(FILE* fp) {
	size_t size = boxes_by_key.size();
	fwrite((char const*)&size, sizeof(size_t), 1, fp);
	for (auto i = boxes_by_key.begin(); i != boxes_by_key.end(); i++) {
		fwrite(&(*i), sizeof(std::pair<size_t, domain_t>), 1, fp);
	}
	fwrite(local_domains.data(), sizeof(domain_local), hpx_size(), fp);
}

void domains_load(FILE* fp) {
	boxes_by_key = decltype(boxes_by_key)();
	size_t size;
	FREAD(&size, sizeof(size_t), 1, fp);
	for (int i = 0; i < size; i++) {
		std::pair<size_t, domain_t> entry;
		FREAD(&entry, sizeof(entry), 1, fp);
		boxes_by_key[entry.first] = entry.second;
	}
	local_domains.resize(hpx_size());
	FREAD(local_domains.data(), sizeof(domain_local), hpx_size(), fp);
}

range<double> domains_range(size_t key) { return boxes_by_key[key].box; }

void domains_transmit_boxes(std::unordered_map<size_t, domain_t> boxes) {
	vector<hpx::future<void>> futs;
	for (auto const& c : hpx_children()) {
		futs.push_back(hpx::async<domains_transmit_boxes_action>(c, boxes));
	}
	boxes_by_key = std::move(boxes);
	hpx::wait_all(futs.begin(), futs.end());
}

void domains_init_rebounds() {
	vector<hpx::future<void>> futs;
	for (auto const& c : hpx_children()) {
		futs.push_back(hpx::async<domains_init_rebounds_action>(c));
	}
	local_domains.resize(1);
	local_domains[0].box = unit_box<double>();
	local_domains[0].proc_range.first = 0;
	local_domains[0].proc_range.second = hpx_size();
	local_domains[0].part_range.first = 0;
	local_domains[0].part_range.second = particles_size();
	local_domains[0].depth = 0;
	//	PRINT( "Setting init rebound on %i %i %i\n", hpx_rank(), 0, particles_size());
	hpx::wait_all(futs.begin(), futs.end());
}

vector<size_t> domains_count_below(vector<double> bounds, int depth) {
	assert(bounds.size() == local_domains.size());
	vector<size_t> counts(bounds.size());
	vector<hpx::future<void>> futs;
	for (int i = 0; i < local_domains.size(); i++) {
		if (local_domains[i].depth == depth) {
			futs.push_back(hpx::async([i, &bounds, depth, &counts]() {
				//							PRINT("???? %i %i %i %i \n", hpx_rank(), i, local_domains[i].part_range.first, local_domains[i].part_range.second);
				auto const rng = local_domains[i].part_range;
				int const nthreads = std::max(2 * (size_t)(rng.second - rng.first) * hpx_hardware_concurrency() / (size_t)particles_size(), (size_t)1);
				vector<hpx::future<void>> futs;
				int const xdim = depth % NDIM;
				fixed32 const xmid = bounds[i];
				std::atomic<size_t> count(0);
				auto const func = [nthreads, rng, xdim, xmid, &count, &counts](int proc) {
					size_t const begin = rng.first + (size_t)proc * (rng.second - rng.first) / nthreads;
					size_t const end = rng.first + (size_t)(proc + 1) * (rng.second - rng.first) / nthreads;
					size_t this_count = 0;
					for (size_t i = begin; i < end; i++) {
						if (particles_pos(xdim, i) < xmid) {
							this_count++;
						}
					}
					count += this_count;
				};

				for (int proc = 1; proc < nthreads; proc++) {
					futs.push_back(hpx::async(func, proc));
				}
				func(0);
				hpx::wait_all(futs.begin(), futs.end());
				counts[i] = count;
			}));
		}
	}
	hpx::wait_all(futs.begin(), futs.end());
	return counts;
}

void domains_rebound_sort(vector<double> bounds, int depth) {
	assert(bounds.size() == local_domains.size());
	vector<size_t> mids(bounds.size());
	vector<hpx::future<void>> futs;
	for (int i = 0; i < local_domains.size(); i++) {
		if (local_domains[i].depth == depth) {
			futs.push_back(hpx::async([i, &bounds, depth, &mids]() {
				mids[i] = particles_sort(local_domains[i].part_range, bounds[i], depth % NDIM);
				///	PRINT( "!!!!!!!! %i %i %e %i %i\n", hpx_rank(), mids[i], bounds[i], local_domains[i].part_range.first, local_domains[i].part_range.second);
			}));
		}
	}
	hpx::wait_all(futs.begin(), futs.end());
	vector<domain_local> new_domains;
	for (int i = 0; i < local_domains.size(); i++) {
		if (local_domains[i].proc_range.second - local_domains[i].proc_range.first > 1) {
			domain_local left;
			domain_local right;
			left = local_domains[i];
			right = local_domains[i];
			left.box.end[depth % NDIM] = right.box.begin[depth % NDIM] = bounds[i];
			left.part_range.second = right.part_range.first = mids[i];
			left.proc_range.second = right.proc_range.first = (local_domains[i].proc_range.first + local_domains[i].proc_range.second) / 2;
			left.depth = right.depth = local_domains[i].depth + 1;
			new_domains.push_back(left);
			new_domains.push_back(right);
		} else {
			new_domains.push_back(local_domains[i]);
		}
	}
	local_domains = std::move(new_domains);
}

void domains_rebound() {
	profiler_enter(__FUNCTION__);
	domains_init_rebounds();
	//	PRINT( "Starting rebound\n");
	vector<domain_global> domains(1);
	boxes_by_key = decltype(boxes_by_key)();
	domains[0].box = unit_box<double>();
	domains[0].proc_range.first = 0;
	domains[0].proc_range.second = hpx_size();
	domains[0].total_count = get_options().nparts;
	domains[0].midhi = 1.0;
	domains[0].midlo = 0.0;
	domains[0].key = 1;
	boxes_by_key[domains[0].key].box = domains[0].box;
	boxes_by_key[domains[0].key].rank = domains[0].proc_range.second - domains[0].proc_range.first > 1 ? -1 : domains[0].proc_range.first;
	vector<domain_global> new_domains;
	std::swap(domains, new_domains);
	int depth = 0;
	auto const all_leaves = [](vector<domain_global> const& domains) {
		bool all = true;
		for (int i = 0; i < domains.size(); i++) {
			if (domains[i].proc_range.second - domains[i].proc_range.first > 1) {
				all = false;
				break;
			}
		}
		return all;
	};
	//	PRINT( "Cycling rebound\n");
	while (!all_leaves(new_domains)) {
		//		PRINT( "Cycle\n");
		domains = std::move(new_domains);
		vector<size_t> counts(domains.size());
		for (int iter = 0; iter < DOMAIN_REBOUND_ITERS; iter++) {
			for (auto& c : counts) {
				c = 0;
			}
			vector<double> bounds;
			for (int i = 0; i < domains.size(); i++) {
				double const midx = (domains[i].midhi + domains[i].midlo) * 0.5;
				bounds.push_back(midx);
			}
			//			PRINT( "Counting\n");
			vector<hpx::future<vector<size_t>>> count_futs;
			for (int i = 0; i < hpx_size(); i++) {
				count_futs.push_back(hpx::async<domains_count_below_action>(hpx_localities()[i], bounds, depth));
			}
			for (auto& f : count_futs) {
				auto const count = f.get();
				for (int i = 0; i < count.size(); i++) {
					counts[i] += count[i];
				}
			}
			//			PRINT( "Done counting\n");
			for (int i = 0; i < domains.size(); i++) {
				//				PRINT( "%i\n", i);
				int const first_proc = domains[i].proc_range.first;
				int const last_proc = domains[i].proc_range.second;
				int const mid_proc = (first_proc + last_proc) / 2;
				ALWAYS_ASSERT(last_proc - first_proc >= 1);
				size_t const target_count = (size_t)(mid_proc - first_proc) * domains[i].total_count / (last_proc - first_proc);
				double const midx = (domains[i].midhi + domains[i].midlo) * 0.5;
				if (counts[i] < target_count) {
					domains[i].midlo = midx;
				} else {
					domains[i].midhi = midx;
				}
				//				PRINT("----- %i %li %e %e\n", i, counts[i], domains[i].midlo, domains[i].midhi);
			}
		}
		vector<double> bounds;
		for (int i = 0; i < domains.size(); i++) {
			if (domains[i].proc_range.second - domains[i].proc_range.first) {
				double const midx = (domains[i].midhi + domains[i].midlo) * 0.5;
				boxes_by_key[domains[i].key].midx = midx;
				bounds.push_back(midx);
			} else {
				bounds.push_back(0.0);
			}
		}
		vector<hpx::future<void>> sort_futs;
		for (int i = 0; i < hpx_size(); i++) {
			sort_futs.push_back(hpx::async<domains_rebound_sort_action>(hpx_localities()[i], bounds, depth));
		}
		hpx::wait_all(sort_futs.begin(), sort_futs.end());
		new_domains.resize(0);
		for (int i = 0; i < domains.size(); i++) {
			if (domains[i].proc_range.second - domains[i].proc_range.first > 1) {
				domain_global left;
				domain_global right;
				left = domains[i];
				right = domains[i];
				left.box.end[depth % NDIM] = right.box.begin[depth % NDIM] = bounds[i];
				left.proc_range.second = right.proc_range.first = (domains[i].proc_range.first + domains[i].proc_range.second) / 2;
				int const leftdim = (depth + 1) % NDIM;
				int const rightdim = (depth + 1) % NDIM;
				left.total_count = counts[i];
				right.total_count = domains[i].total_count - counts[i];
				left.midhi = left.box.end[leftdim];
				left.midlo = left.box.begin[leftdim];
				right.midhi = right.box.end[rightdim];
				right.midlo = right.box.begin[rightdim];
				left.key = domains[i].key << 1;
				right.key = (domains[i].key << 1) + 1;
				boxes_by_key[left.key].box = left.box;
				boxes_by_key[right.key].box = right.box;
				boxes_by_key[left.key].rank = left.proc_range.second - left.proc_range.first > 1 ? -1 : left.proc_range.first;
				boxes_by_key[right.key].rank = right.proc_range.second - right.proc_range.first > 1 ? -1 : right.proc_range.first;
				new_domains.push_back(left);
				new_domains.push_back(right);
			} else {
				new_domains.push_back(domains[i]);
			}
		}
		depth++;
	}
	domains_transmit_boxes(boxes_by_key);
	profiler_exit();
}

void domains_transmit_particles(vector<particle> parts) {
	static mutex_type mutex;
	std::lock_guard<mutex_type> lock(mutex);
	trans_particles.push_back(parts);
}

// Ordinary SPH timesteps migrate the complete gas state together with its
// gravity particle. Category indices are local and are rebuilt at the receiver.
struct GasMigrationState {
	array<float, NCHEMFRACS> frac, dfrac;
	array<float, NDIM> dvel, predictedVelocity, gravity;
	float alpha, h, energy, dfcold;
	float dvValue, kapValue, e0Value, de3Value, de2Value, fp2Value, preValue, fp1Value, s2Value, fcValue, rhValue, tcValue;
	char orValue, saValue, stValue;
	template <class Archive> void serialize(Archive& arc, unsigned) {
		arc & frac;
		arc & dfrac;
		arc & dvel;
		arc & predictedVelocity;
		arc & gravity;
		arc & alpha;
		arc & h;
		arc & energy;
		arc & dfcold;
		arc & dvValue;
		arc & kapValue;
		arc & e0Value;
		arc & de3Value;
		arc & de2Value;
		arc & fp2Value;
		arc & preValue;
		arc & fp1Value;
		arc & s2Value;
		arc & fcValue;
		arc & rhValue;
		arc & tcValue;
		arc & orValue;
		arc & saValue;
		arc & stValue;
	}
};
struct SphMigrationParticle {
	particle part;
	GasMigrationState gas{};
	array<float, 4> auxiliary{};
	array<char, 2> flags{};
	template <class Archive> void serialize(Archive& arc, unsigned) {
		arc & part;
		arc & auxiliary;
		arc & flags;
		if (part.type == SPH_TYPE) {
			arc & gas;
		}
	}
};
static vector<SphMigrationParticle> retainedSphParticles;
static vector<vector<SphMigrationParticle>> receivedSphParticles;
void receiveSphParticles(vector<SphMigrationParticle> parts) {
	static mutex_type mutex;
	std::lock_guard<mutex_type> lock(mutex);
	receivedSphParticles.push_back(std::move(parts));
}
HPX_PLAIN_ACTION(receiveSphParticles);
static SphMigrationParticle packSphParticle(part_int index) {
	SphMigrationParticle result;
	result.part = particles_get_particle(index);
	ALWAYS_ASSERT(result.part.type != STAR_TYPE);
	result.auxiliary[0] = particles_softlen(index);
	result.auxiliary[1] = particles_zeta(index);
	result.auxiliary[2] = particles_rho(index);
	result.auxiliary[3] = particles_divv(index);
	result.flags[0] = particles_converged(index);
	result.flags[1] = particles_semiactive(index);
	if (result.part.type == SPH_TYPE) {
		auto const i = particles_cat_index(index);
		auto& gas = result.gas;
		gas.frac = sph_particles_r1[i].frac;
		gas.alpha = sph_particles_r1[i].alpha;
		gas.h = sph_particles_r2[i].h;
		gas.dfrac = sph_particles_r5[i].dfrac;
		gas.dfcold = sph_particles_r5[i].dfcold;
		gas.dvel = sph_particles_r6[i].dvel;
		gas.dvValue = sph_particles_dv[i];
		gas.kapValue = sph_particles_kap[i];
		gas.e0Value = sph_particles_e0[i];
		gas.de3Value = sph_particles_de3[i];
		gas.de2Value = sph_particles_de2[i];
		gas.fp2Value = sph_particles_fp2[i];
		gas.preValue = sph_particles_pre[i];
		gas.fp1Value = sph_particles_fp1[i];
		gas.s2Value = sph_particles_s2[i];
		gas.fcValue = sph_particles_fc[i];
		gas.rhValue = sph_particles_rh[i];
		gas.tcValue = sph_particles_tc[i];
		gas.orValue = sph_particles_or[i];
		gas.saValue = sph_particles_sa[i];
		gas.stValue = sph_particles_st[i];
#ifdef ENTROPY
		gas.energy = sph_particles_r2[i].A;
#else
		gas.energy = sph_particles_r2[i].eint;
#endif
		for (int dim = 0; dim < NDIM; dim++) {
			gas.predictedVelocity[dim] = sph_particles_dv2[dim][i];
			gas.gravity[dim] = sph_particles_g[dim][i];
		}
	}
	return result;
}
static void unpackSphParticle(SphMigrationParticle record, part_int index, part_int gasIndex) {
	if (record.part.type == SPH_TYPE) {
		record.part.cat_index = gasIndex;
	}
	particles_set_particle(record.part, index);
	particles_softlen(index) = record.auxiliary[0];
	particles_zeta(index) = record.auxiliary[1];
	particles_rho(index) = record.auxiliary[2];
	particles_divv(index) = record.auxiliary[3];
	particles_converged(index) = record.flags[0];
	particles_semiactive(index) = record.flags[1];
	if (record.part.type == SPH_TYPE) {
		auto const i = gasIndex;
		auto const& gas = record.gas;
		sph_particles_dm_index(i) = index;
		sph_particles_r1[i].frac = gas.frac;
		sph_particles_r1[i].alpha = gas.alpha;
		sph_particles_r2[i].h = gas.h;
		sph_particles_r5[i].dfrac = gas.dfrac;
		sph_particles_r5[i].dfcold = gas.dfcold;
		sph_particles_r6[i].dvel = gas.dvel;
		sph_particles_dv[i] = gas.dvValue;
		sph_particles_kap[i] = gas.kapValue;
		sph_particles_e0[i] = gas.e0Value;
		sph_particles_de3[i] = gas.de3Value;
		sph_particles_de2[i] = gas.de2Value;
		sph_particles_fp2[i] = gas.fp2Value;
		sph_particles_pre[i] = gas.preValue;
		sph_particles_fp1[i] = gas.fp1Value;
		sph_particles_s2[i] = gas.s2Value;
		sph_particles_fc[i] = gas.fcValue;
		sph_particles_rh[i] = gas.rhValue;
		sph_particles_tc[i] = gas.tcValue;
		sph_particles_or[i] = gas.orValue;
		sph_particles_sa[i] = gas.saValue;
		sph_particles_st[i] = gas.stValue;
#ifdef ENTROPY
		sph_particles_r2[i].A = gas.energy;
#else
		sph_particles_r2[i].eint = gas.energy;
#endif
		for (int dim = 0; dim < NDIM; dim++) {
			sph_particles_dv2[dim][i] = gas.predictedVelocity[dim];
			sph_particles_g[dim][i] = gas.gravity[dim];
		}
	}
}
static void beginSphMigration(int rung) {
	vector<hpx::future<void>> pending;
	for (auto const& child : hpx_children()) {
		pending.push_back(hpx::async<domains_begin_action>(child, rung));
	}
	retainedSphParticles.clear();
	auto const box = domains_find_my_box();
	std::unordered_map<int, vector<SphMigrationParticle>> sends;
	// The ordinary integrator retains every rung in the same particle range.
	// Moving every out-of-domain particle also handles domain rebalancing.
	for (part_int i = 0; i < particles_size(); i++) {
		array<double, NDIM> x;
		for (int dim = 0; dim < NDIM; dim++) {
			x[dim] = particles_pos(dim, i).to_double();
		}
		auto record = packSphParticle(i);
		if (box.contains(x)) {
			retainedSphParticles.push_back(std::move(record));
		} else {
			auto const rank = find_particle_domain(x);
			auto& send = sends[rank];
			send.push_back(std::move(record));
			if (send.size() >= MAX_PARTICLES_PER_PARCEL) {
				pending.push_back(hpx::async<receiveSphParticles_action>(hpx_localities()[rank], std::move(send)));
				send.clear();
			}
		}
	}
	for (auto& entry : sends) {
		if (!entry.second.empty()) {
			pending.push_back(hpx::async<receiveSphParticles_action>(hpx_localities()[entry.first], std::move(entry.second)));
		}
	}
	for (auto& future : pending) {
		future.get();
	}
}
static void endSphMigration() {
	vector<hpx::future<void>> pending;
	for (auto const& child : hpx_children()) {
		pending.push_back(hpx::async<domains_end_action>(child));
	}
	for (auto& parcel : receivedSphParticles) {
		for (auto& record : parcel) {
			retainedSphParticles.push_back(std::move(record));
		}
	}
	receivedSphParticles.clear();
	part_int gasCount = 0;
	for (auto const& record : retainedSphParticles) {
		gasCount += record.part.type == SPH_TYPE;
	}
	particles_resize(retainedSphParticles.size());
	sph_particles_resize(gasCount, false);
	part_int gasIndex = 0;
	for (part_int i = 0; i < retainedSphParticles.size(); i++) {
		unpackSphParticle(retainedSphParticles[i], i, gasIndex);
		gasIndex += retainedSphParticles[i].part.type == SPH_TYPE;
	}
	retainedSphParticles.clear();
	for (auto& future : pending) {
		future.get();
	}
}

void domains_begin(int rung) {
	if (get_options().sph) {
		beginSphMigration(rung);
		return;
	}
	profiler_enter(__FUNCTION__);
	vector<hpx::future<void>> futs;
	auto children = hpx_children();
	for (auto& c : children) {
		futs.push_back(hpx::async<domains_begin_action>(c, rung));
	}
	auto const my_domain = domains_find_my_box();
	free_indices.resize(0);
	mutex_type mutex;
	int const nthreads = hpx_hardware_concurrency();
	auto const current_range = particles_current_range();
	std::atomic<part_int> next_begin(0);
	constexpr int chunk_size = 1024;
	for (int proc = 0; proc < nthreads; proc++) {
		futs.push_back(hpx::async([proc, nthreads, &mutex, &next_begin, my_domain, current_range, rung]() {
			std::unordered_map<int, vector<particle>> sends;
			vector<hpx::future<void>> futs;
			vector<part_int> my_free_indices;
			part_int begin = next_begin++ * chunk_size + current_range.first;
			while (begin < current_range.second) {
				part_int const end = std::min(begin + chunk_size, current_range.second);
				for (part_int i = begin; i < end; i++) {
					if (particles_rung(i) == rung) {
						array<double, NDIM> x;
						for (part_int dim = 0; dim < NDIM; dim++) {
							x[dim] = particles_pos(dim, i).to_double();
						}
						if (!my_domain.contains(x)) {
							part_int const rank = find_particle_domain(x);
							auto& send = sends[rank];
							send.push_back(particles_get_particle(i));
							if (send.size() >= MAX_PARTICLES_PER_PARCEL) {
								//				PRINT( "Sending %i from %i to %i\n", send.size(), hpx_rank(), rank);
								futs.push_back(hpx::async<domains_transmit_particles_action>(hpx_localities()[rank], std::move(send)));
							}
							my_free_indices.push_back(i);
						}
					}
				}
				begin = next_begin++ * chunk_size + current_range.first;
			}
			for (auto i = sends.begin(); i != sends.end(); i++) {
				if (i->second.size()) {
					//			PRINT( "Sending %i from %i to %i\n", i->second.size(), hpx_rank(),i->first);
					futs.push_back(hpx::async<domains_transmit_particles_action>(hpx_localities()[i->first], std::move(i->second)));
				}
			}
			{
				std::lock_guard<mutex_type> lock(mutex);
				free_indices.insert(free_indices.end(), my_free_indices.begin(), my_free_indices.end());
			}
			hpx::wait_all(futs.begin(), futs.end());
		}));
	}

	hpx::wait_all(futs.begin(), futs.end());
	profiler_exit();
}

void domains_end() {
	if (get_options().sph) {
		endSphMigration();
		return;
	}
	profiler_enter(__FUNCTION__);
	vector<hpx::future<void>> futs;
	auto children = hpx_children();
	for (auto& c : children) {
		futs.push_back(hpx::async<domains_end_action>(c));
	}
	auto const particle_compare = [](particle a, particle b) {
		for (part_int dim = 0; dim < NDIM; dim++) {
			if (a.x[dim] < b.x[dim]) {
				return true;
			} else if (a.x[dim] > b.x[dim]) {
				return false;
			}
		}
		return false;
	};
	hpx::future<void> fut1;
	hpx::future<void> fut2;
	if (free_indices.size()) {
		fut1 = hpx::sort(PAR_EXECUTION_POLICY, free_indices.begin(), free_indices.end());
	} else {
		fut1 = hpx::make_ready_future();
	}
	vector<particle> all_trans_particles;
	vector<part_int> offsets(trans_particles.size());
	part_int sz = 0;
	for (int i = 0; i < trans_particles.size(); i++) {
		offsets[i] = sz;
		sz += trans_particles[i].size();
	}
	all_trans_particles.resize(sz);
	vector<hpx::future<void>> futs1;
	int const nthreads = hpx_hardware_concurrency();
	std::atomic<int> index(0);
	for (int proc = 0; proc < nthreads; proc++) {
		futs1.push_back(hpx::async([&index, &all_trans_particles, &offsets]() {
			int i;
			while ((i = index++) < trans_particles.size()) {
				for (part_int j = 0; j < trans_particles[i].size(); j++) {
					all_trans_particles[offsets[i] + j] = trans_particles[i][j];
				}
			}
		}));
	}
	hpx::wait_all(futs1.begin(), futs1.end());
	trans_particles = decltype(trans_particles)();
	if (all_trans_particles.size()) {
		fut2 = hpx::sort(PAR_EXECUTION_POLICY, all_trans_particles.begin(), all_trans_particles.end(), particle_compare);
	} else {
		fut2 = hpx::make_ready_future();
	}
	fut1.get();
	fut2.get();
	if (free_indices.size() < all_trans_particles.size()) {
		part_int const diff = all_trans_particles.size() - free_indices.size();
		for (part_int i = 0; i < diff; i++) {
			free_indices.push_back(particles_size() + i);
		}
		particles_resize(particles_size() + diff);
	} else {
		while (free_indices.size() > all_trans_particles.size()) {
			auto const p = particles_get_particle(particles_size() - 1);
			particles_set_particle(p, free_indices.back());
			free_indices.pop_back();
			particles_resize(particles_size() - 1);
		}
	}
	// PRINT("unloading particles on %i\n", hpx_rank());
	for (int proc = 0; proc < nthreads; proc++) {
		futs.push_back(hpx::async([nthreads, proc, &all_trans_particles]() {
			part_int const begin = (size_t)proc * free_indices.size() / nthreads;
			part_int const end = (size_t)(proc + 1) * free_indices.size() / nthreads;
			for (part_int i = begin; i < end; i++) {
				particles_set_particle(all_trans_particles[i], free_indices[i]);
			}
		}));
	}
	hpx::wait_all(futs.begin(), futs.end());
	// PRINT("Done on %i\n", hpx_rank());
	free_indices = decltype(free_indices)();
#ifdef DOMAINS_CHECK
	domains_check();
#endif
	profiler_exit();
}

range<double> domains_find_my_box() { return local_domains[hpx_rank()].box; }

static void domains_check() {
	std::atomic<int> fail(0);
	auto const my_domain = domains_find_my_box();
	int const nthreads = hpx_hardware_concurrency();
	vector<hpx::future<void>> futs;
	for (int proc = 0; proc < nthreads; proc++) {
		futs.push_back(hpx::async([proc, nthreads, &fail, my_domain]() {
			auto const rng = particles_current_range();
			part_int const begin = size_t(proc) * (rng.second - rng.first) / nthreads + rng.first;
			part_int const end = size_t(proc + 1) * (rng.second - rng.first) / nthreads + rng.first;
			for (part_int i = begin; i < end; i++) {
				array<double, NDIM> x;
				for (int dim = 0; dim < NDIM; dim++) {
					x[dim] = particles_pos(dim, i).to_double();
				}
				if (!my_domain.contains(x)) {
					PRINT("particle out of range %li of %li %e %e %e rung %i\n", (long long)i, (long long)particles_size(), x[0], x[1], x[2],
						  particles_rung(i));
					PRINT("%s\n", domains_find_my_box().to_string().c_str());
					ALWAYS_ASSERT(false);
					fail++;
				}
			}
		}));
	}
	hpx::wait_all(futs.begin(), futs.end());
	if (int(fail)) {
		THROW_ERROR("particle out of range!\n");
	}
}

/*
 pair<int, double> domains_find_ewald_level(double theta, size_t key, range<double> box, int depth, double rb) {
 pair<int, double> rc;
 double rmax = 0.0;
 bool root = key == 1;
 double xc = 0.5 * (box.begin[XDIM] + box.end[XDIM]);
 double yc = 0.5 * (box.begin[YDIM] + box.end[YDIM]);
 double zc = 0.5 * (box.begin[ZDIM] + box.end[ZDIM]);
 const double r = sqrtf(sqr(0.5 * (box.end[XDIM] - box.begin[XDIM]), 0.5 * (box.end[YDIM] - box.begin[YDIM]), 0.5 * (box.end[ZDIM] - box.begin[ZDIM])));
 const double h = get_options().hsoft;
 bool ewald_satisfied = false;
 if (rb <= 0.0) {
 ewald_satisfied = (r < 0.25 * (theta / (theta + 1)) && r < 0.125 - 0.5 * h);
 } else {
 ewald_satisfied = (r < 0.5 * (theta / (1.0 + theta)) - rb && r < 0.5 - rb - h);
 }
 if (ewald_satisfied) {
 rc.first = depth;
 rc.second = r;
 } else {
 const auto& entry = boxes_by_key[key];
 key <<= 1;
 pair<int, double> rcl, rcr;
 if (entry.rank == -1) {
 const int xdim = depth % NDIM;
 range<double> cboxr = box;
 range<double> cboxl = box;
 double midx = (box.begin[xdim] + box.end[xdim]) * 0.5;
 cboxl.end[xdim] = cboxr.begin[xdim] = midx;
 rcl = domains_find_ewald_level(theta, key, cboxl, depth + 1, rb);
 rcr = domains_find_ewald_level(theta, key | 1, cboxl, depth + 1, rb);
 } else {
 double max_span = 0.0;
 int xdim;
 for (int dim = 0; dim < NDIM; dim++) {
 double span = box.end[dim] - box.begin[dim];
 if (span > max_span) {
 max_span = span;
 xdim = dim;
 }
 }
 range<double> cboxr = box;
 range<double> cboxl = box;
 double midx = (box.begin[xdim] + box.end[xdim]) * 0.5;
 cboxl.end[xdim] = cboxr.begin[xdim] = midx;
 rcl = domains_find_ewald_level(theta, key, cboxl, depth + 1, rb);
 rcr = domains_find_ewald_level(theta, key | 1, cboxl, depth + 1, rb);
 }
 if (rb <= 0.0) {
 if (rcr.second > rcl.first) {
 rc = rcr;
 } else {
 rc = rcl;
 }
 } else {
 if (rcl.first > rcr.first) {
 rc = rcl;
 } else {
 rc = rcr;
 }
 }
 }
 if (root && rb <= 0.0) {
 auto rcb = domains_find_ewald_level(theta, key, box, depth, rc.second);
 if (rcb.first > rc.first) {
 rc = rcb;
 }
 }
 return rc;
 }

 */

static int find_particle_domain(array<double, NDIM> const& x, size_t key, int depth) {
	auto const& entry = boxes_by_key[key];
	if (entry.rank == -1) {
		int const xdim = depth % NDIM;
		key <<= 1;
		if (x[xdim] >= entry.midx) {
			key += 1;
		}
		return find_particle_domain(x, key, depth + 1);
	} else {
		return entry.rank;
	}
}
