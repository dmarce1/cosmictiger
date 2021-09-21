constexpr bool verbose = true;

#include <cosmictiger/domain.hpp>
#include <cosmictiger/containers.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/particles.hpp>
#include <cosmictiger/safe_io.hpp>
#include <cosmictiger/options.hpp>

#include <unordered_map>
#include <shared_mutex>

void domains_transmit_particles(vector<particle>);
void domains_init_rebounds();
void domains_transmit_boxes(std::unordered_map<size_t, domain_t> boxes);
void domains_rebound_sort(vector<double> bounds, vector<int> dims, int depth);
vector<size_t> domains_count_below(vector<double> bounds, vector<int> dims, int depth);
vector<size_t> domains_get_loads();

HPX_PLAIN_ACTION (domains_count_below);
HPX_PLAIN_ACTION (domains_get_loads);
HPX_PLAIN_ACTION (domains_rebound_sort);
HPX_PLAIN_ACTION (domains_begin);
HPX_PLAIN_ACTION (domains_end);
HPX_PLAIN_ACTION (domains_init_rebounds);
HPX_PLAIN_ACTION (domains_transmit_particles);
HPX_PLAIN_ACTION (domains_transmit_boxes);

static vector<part_int> free_indices;
static vector<particle> trans_particles;
static shared_mutex_type mutex;
static std::unordered_map<size_t, domain_t> boxes_by_key;

static int find_particle_domain(const array<double, NDIM>& x, size_t key = 1);
static void domains_check();
static vector<domain_local> local_domains;

double domains_get_load_imbalance() {
	const auto loads = domains_get_loads();
	size_t max = 0;
	double avg = 0;
	for (const auto load : loads) {
		max = std::max(load, max);
		avg += load;
	}
	avg /= loads.size();
	const double imbalance = max / avg - 1.0;
	return imbalance;
}

vector<size_t> domains_get_loads() {
	vector<hpx::future<vector<size_t>>>futs;
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async < domains_get_loads_action > (c));
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
	fwrite((const char*) &size, sizeof(size_t), 1, fp);
	for (auto i = boxes_by_key.begin(); i != boxes_by_key.end(); i++) {
		fwrite(&(*i), sizeof(std::pair<size_t, domain_t>), 1, fp);
	}
	fwrite(local_domains.data(), sizeof(domain_local), hpx_size(), fp);
}

void domains_load(FILE *fp) {
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

range<double> domains_range(size_t key) {
	return boxes_by_key[key].box;
}

void domains_transmit_boxes(std::unordered_map<size_t, domain_t> boxes) {
	vector<hpx::future<void>> futs;
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async < domains_transmit_boxes_action > (c, boxes));
	}
	boxes_by_key = std::move(boxes);
	hpx::wait_all(futs.begin(), futs.end());
}

void domains_init_rebounds() {
	vector<hpx::future<void>> futs;
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async < domains_init_rebounds_action > (c));
	}
	local_domains.resize(1);
	local_domains[0].box = unit_box<double>();
	local_domains[0].proc_range.first = 0;
	local_domains[0].proc_range.second = hpx_size();
	local_domains[0].part_range.first = 0;
	local_domains[0].part_range.second = particles_size();
	local_domains[0].depth = 0;
	hpx::wait_all(futs.begin(), futs.end());
}

vector<size_t> domains_count_below(vector<double> bounds, vector<int> dims, int depth) {
	assert(bounds.size() == local_domains.size());
	vector<size_t> counts(bounds.size());
	vector<hpx::future<void>> futs;
	for (int i = 0; i < local_domains.size(); i++) {
		if (local_domains[i].depth == depth) {
			futs.push_back(hpx::async([i, &bounds, &dims, &counts]() {
				//	PRINT("----%i %i %i %i \n", hpx_rank(), i, local_domains[i].part_range.first, local_domains[i].part_range.second);
					const auto rng = local_domains[i].part_range;
					const int nthreads = std::max(2 * (size_t) (rng.second - rng.first) * hpx::thread::hardware_concurrency() / particles_size(), (size_t) 1);
					vector<hpx::future<void>> futs;
					const int xdim = dims[i];
					const fixed32 xmid = bounds[i];
					std::atomic<size_t> count(0);
					const auto func = [nthreads,rng, xdim, xmid, &count, &counts](int proc) {
						const size_t begin = rng.first + (size_t) proc * (rng.second-rng.first) / nthreads;
						const size_t end = rng.first + (size_t) (proc+1) * (rng.second-rng.first) / nthreads;
						size_t this_count = 0;
						for( size_t i = begin; i < end; i++) {
							if( particles_pos(xdim,i) < xmid) {
								this_count++;
							}
						}
						count += this_count;
					};

					for( int proc = 1; proc < nthreads; proc++) {
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

void domains_rebound_sort(vector<double> bounds, vector<int> dims, int depth) {
	assert(bounds.size() == local_domains.size());
	vector<size_t> mids(bounds.size());
	vector<hpx::future<void>> futs;
	for (int i = 0; i < local_domains.size(); i++) {
		if (local_domains[i].depth == depth) {
			futs.push_back(hpx::async([i, &bounds, &dims, &mids]() {
				mids[i] = particles_sort(local_domains[i].part_range, bounds[i], dims[i]);
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
			left.box.end[dims[i]] = right.box.begin[dims[i]] = bounds[i];
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
	domains_init_rebounds();
	vector<domain_global> domains(1);
	boxes_by_key = decltype(boxes_by_key)();
	domains[0].box = unit_box<double>();
	domains[0].proc_range.first = 0;
	domains[0].proc_range.second = hpx_size();
	domains[0].total_count = std::pow((size_t) get_options().parts_dim, NDIM);
	domains[0].midhi = 1.0;
	domains[0].midlo = 0.0;
	domains[0].key = 1;
	boxes_by_key[domains[0].key].box = domains[0].box;
	boxes_by_key[domains[0].key].rank = domains[0].proc_range.second - domains[0].proc_range.first > 1 ? -1 : domains[0].proc_range.first;
	vector<domain_global> new_domains;
	std::swap(domains, new_domains);
	int depth = 0;
	const auto all_leaves = [](const vector<domain_global>& domains) {
		bool all = true;
		for( int i = 0; i < domains.size(); i++) {
			if( domains[i].proc_range.second - domains[i].proc_range.first > 1 ) {
				all = false;
				break;
			}
		}
		return all;
	};
	while (!all_leaves(new_domains)) {
		domains = std::move(new_domains);
		vector<size_t> counts(domains.size());
		for (int iter = 0; iter < DOMAIN_REBOUND_ITERS; iter++) {
			for (auto& c : counts) {
				c = 0;
			}
			vector<double> bounds;
			vector<int> dims;
			for (int i = 0; i < domains.size(); i++) {
				const double midx = (domains[i].midhi + domains[i].midlo) * 0.5;
				const int xdim = domains[i].box.longest_dim();
				bounds.push_back(midx);
				dims.push_back(xdim);
			}
			vector<hpx::future<vector<size_t>>>count_futs;
			for (int i = 0; i < hpx_size(); i++) {
				count_futs.push_back(hpx::async < domains_count_below_action > (hpx_localities()[i], bounds, dims, depth));
			}
			for (auto& f : count_futs) {
				const auto count = f.get();
				for (int i = 0; i < count.size(); i++) {
					counts[i] += count[i];
				}
			}
			for (int i = 0; i < domains.size(); i++) {
				const int first_proc = domains[i].proc_range.first;
				const int last_proc = domains[i].proc_range.second;
				const int mid_proc = (first_proc + last_proc) / 2;
				const size_t target_count = (size_t)(mid_proc - first_proc) * domains[i].total_count / (last_proc - first_proc);
				const double midx = (domains[i].midhi + domains[i].midlo) * 0.5;
				if (counts[i] < target_count) {
					domains[i].midlo = midx;
				} else {
					domains[i].midhi = midx;
				}
//				PRINT("%i %li %e %e\n", i, counts[i], domains[i].midlo, domains[i].midhi);
			}
		}
		vector<double> bounds;
		vector<int> dims;
		for (int i = 0; i < domains.size(); i++) {
			if (domains[i].proc_range.second - domains[i].proc_range.first) {
				const double midx = (domains[i].midhi + domains[i].midlo) * 0.5;
				const int xdim = domains[i].box.longest_dim();
				boxes_by_key[domains[i].key].midx = midx;
				bounds.push_back(midx);
				dims.push_back(xdim);
			} else {
				bounds.push_back(0.0);
				dims.push_back(-1);
			}
		}
		vector<hpx::future<void>> sort_futs;
		for (int i = 0; i < hpx_size(); i++) {
			sort_futs.push_back(hpx::async < domains_rebound_sort_action > (hpx_localities()[i], bounds, dims, depth));
		}
		hpx::wait_all(sort_futs.begin(), sort_futs.end());
		new_domains.resize(0);
		for (int i = 0; i < domains.size(); i++) {
			if (domains[i].proc_range.second - domains[i].proc_range.first > 1) {
				domain_global left;
				domain_global right;
				left = domains[i];
				right = domains[i];
				left.box.end[dims[i]] = right.box.begin[dims[i]] = bounds[i];
				left.proc_range.second = right.proc_range.first = (domains[i].proc_range.first + domains[i].proc_range.second) / 2;
				const int leftdim = left.box.longest_dim();
				const int rightdim = right.box.longest_dim();
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
}

void domains_transmit_particles(vector<particle> parts) {
//	PRINT("Receiving %li particles on %i\n", parts.size(), hpx_rank());
	std::unique_lock<shared_mutex_type> ulock(mutex);
	const part_int start = trans_particles.size();
	const part_int stop = start + parts.size();
	trans_particles.resize(stop);
	ulock.unlock();
	constexpr int chunk_size = 1024;
	std::shared_lock<shared_mutex_type> slock(mutex);
	for (int i = 0; i < parts.size(); i++) {
		if (i % chunk_size == chunk_size - 1) {
			slock.unlock();
			slock.lock();
		}
		trans_particles[start + i] = parts[i];
	}
}

void domains_begin() {
	vector<hpx::future<void>> futs;
	auto children = hpx_children();
	for (auto& c : children) {
		futs.push_back(hpx::async < domains_begin_action > (c));
	}
	const auto my_domain = domains_find_my_box();
	free_indices.resize(0);
	mutex_type mutex;
	const int nthreads = hpx::thread::hardware_concurrency();
	std::atomic<part_int> next_begin(0);
	constexpr int chunk_size = 1024;
	for (int proc = 0; proc < nthreads; proc++) {
		futs.push_back(hpx::async([proc,nthreads,&mutex,&next_begin,my_domain]() {
			std::unordered_map<int,vector<particle>> sends;
			vector<hpx::future<void>> futs;
			vector<part_int> my_free_indices;
			part_int begin = next_begin++ * chunk_size;
			while( begin < particles_size()) {
				const part_int end = std::min(begin + chunk_size, particles_size());
				for( part_int i = begin; i < end; i++) {
					array<double, NDIM> x;
					for( part_int dim = 0; dim < NDIM; dim++) {
						x[dim] = particles_pos(dim,i).to_double();
					}
					if( !my_domain.contains(x)) {
						const part_int rank = find_particle_domain(x);
						auto& send = sends[rank];
						send.push_back(particles_get_particle(i));
						if( send.size() >= MAX_PARTICLES_PER_PARCEL) {
							futs.push_back(hpx::async<domains_transmit_particles_action>(hpx_localities()[rank], std::move(send)));
						}
						my_free_indices.push_back(i);
					}
					begin = next_begin++ * chunk_size;
				}
			}
			for( auto i = sends.begin(); i != sends.end(); i++) {
				if( i->second.size()) {
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
}

void domains_end() {
	vector<hpx::future<void>> futs;
	auto children = hpx_children();
	for (auto& c : children) {
		futs.push_back(hpx::async < domains_end_action > (c));
	}
	const auto particle_compare = [](particle a, particle b) {
		for( part_int dim = 0; dim < NDIM; dim++) {
			if( a.x[dim] < b.x[dim]) {
				return true;
			} else if( a.x[dim] > b.x[dim]) {
				return false;
			}
		}
		return false;
	};
	/** THIS SORT IS REQUIRED FOR DETERMINISM!!!*/
	//PRINT("Processing %li particles on %i\n", trans_particles.size(), hpx_rank());
	hpx::future<void> fut1;
	hpx::future<void> fut2;
	if (free_indices.size()) {
		fut1 = hpx::parallel::sort(PAR_EXECUTION_POLICY, free_indices.begin(), free_indices.end());
	} else {
		fut1 = hpx::make_ready_future();
	}
	/********************************************/
	if (trans_particles.size()) {
		fut2 = hpx::parallel::sort(PAR_EXECUTION_POLICY, trans_particles.begin(), trans_particles.end(), particle_compare);
	} else {
		fut2 = hpx::make_ready_future();
	}
	fut1.get();
	fut2.get();
	if (free_indices.size() < trans_particles.size()) {
		const part_int diff = trans_particles.size() - free_indices.size();
		for (part_int i = 0; i < diff; i++) {
			free_indices.push_back(particles_size() + i);
		}
		particles_resize(particles_size() + diff);
	} else {
		while (free_indices.size() > trans_particles.size()) {
			const auto p = particles_get_particle(particles_size() - 1);
			particles_set_particle(p, free_indices.back());
			free_indices.pop_back();
			particles_resize(particles_size() - 1);
		}

	}
	//PRINT("unloading particles on %i\n", hpx_rank());
	const int nthreads = hpx::thread::hardware_concurrency();
	for (int proc = 0; proc < nthreads; proc++) {
		futs.push_back(hpx::async([nthreads, proc]() {
			const part_int begin = (size_t) proc * free_indices.size() / nthreads;
			const part_int end = (size_t) (proc+1) * free_indices.size() / nthreads;
			for( part_int i = begin; i < end; i++) {
				particles_set_particle(trans_particles[i],free_indices[i]);
			}
		}));
	}
	hpx::wait_all(futs.begin(), futs.end());
	//PRINT("Done on %i\n", hpx_rank());
	trans_particles = decltype(trans_particles)();
	free_indices = decltype(free_indices)();
#ifdef DOMAINS_CHECK
	domains_check();
#endif
}

range<double> domains_find_my_box() {
	return local_domains[hpx_rank()].box;
}

static void domains_check() {
	std::atomic<int> fail(0);
	const auto my_domain = domains_find_my_box();
	const int nthreads = hpx_hardware_concurrency();
	vector<hpx::future<void>> futs;
	for (int proc = 0; proc < nthreads; proc++) {
		futs.push_back(hpx::async([proc,nthreads,&fail,my_domain]() {
			const int begin = size_t(proc) * particles_size() / nthreads;
			const int end = size_t(proc + 1) * particles_size() / nthreads;
			for (part_int i = begin; i < end; i++) {
				array<double, NDIM> x;
				for (int dim = 0; dim < NDIM; dim++) {
					x[dim] = particles_pos(dim, i).to_double();
				}
				if (!my_domain.contains(x)) {
					PRINT("particle out of range %li of %li %e %e %e\n", (long long ) i, (long long ) particles_size(), x[0], x[1], x[2]);
					PRINT("%s\n", domains_find_my_box().to_string().c_str());
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

static int find_particle_domain(const array<double, NDIM>& x, size_t key) {
	const auto& entry = boxes_by_key[key];
	if (entry.rank == -1) {
		const int xdim = entry.box.longest_dim();
		key <<= 1;
		if (x[xdim] >= entry.midx) {
			key += 1;
		}
		return find_particle_domain(x, key);
	} else {
		return entry.rank;
	}
}

