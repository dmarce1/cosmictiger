#include <cosmictiger/containers.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/lightcone.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/range.hpp>

#include <atomic>
#include <memory>
#include <shared_mutex>
#include <unordered_set>
#include <unordered_map>

#include <healpix_cxx/healpix_base.h>

using healpix_type = T_Healpix_Base< int >;

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

#define LC_NO_GROUP (0x7FFFFFFFFFFFFFFFLL)
#define LC_EDGE_GROUP (0x0LL)

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

struct lc_group_data {
	vector<lc_particle> parts;
};

struct lc_tree_id {
	int pix;
	int index;
	bool operator!=(const lc_tree_id& other) const {
		return pix != other.pix || index != other.index;
	}
};

struct lc_tree_node {
	range<lc_real> box;
	array<lc_tree_id, NCHILD> children;
	pair<int> part_range;
	bool active;
	bool last_active;
	int pix;
};

static std::shared_ptr<healpix_type> healpix;
static pair<int> my_pix_range;
static std::unordered_set<int> bnd_pix;
static std::unordered_map<int, vector<lc_particle>> part_map;
static std::unordered_map<int, vector<lc_tree_node>> tree_map;
static std::unordered_map<int, std::shared_ptr<spinlock_type>> mutex_map;
static std::atomic<long long> group_id_counter(0);
static vector<lc_particle> part_buffer;
static shared_mutex_type mutex;
static double tau_max;
static int Nside;
static int Npix;

static vector<lc_particle> lc_get_particles(int pix);
static void lc_send_particles(vector<lc_particle>);
static void lc_send_buffer_particles(vector<lc_particle>);
static int vec2pix(double x, double y, double z);
static int pix2rank(int pix);
static vector<int> pix_neighbors(int pix);

HPX_PLAIN_ACTION (lc_parts_waiting);
HPX_PLAIN_ACTION (lc_parts2groups);
HPX_PLAIN_ACTION (lc_init);
HPX_PLAIN_ACTION (lc_get_particles);
HPX_PLAIN_ACTION (lc_form_trees);
HPX_PLAIN_ACTION (lc_buffer2homes);
HPX_PLAIN_ACTION (lc_groups2homes);
HPX_PLAIN_ACTION (lc_send_buffer_particles);
HPX_PLAIN_ACTION (lc_send_particles);
HPX_PLAIN_ACTION (lc_find_groups);
HPX_PLAIN_ACTION (lc_particle_boundaries);

size_t lc_parts_waiting() {
	vector < hpx::future < size_t >> futs;
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async < lc_parts_waiting_action > (c));
	}
	size_t nparts = 0;
	for (int pix = my_pix_range.first; pix < my_pix_range.second; pix++) {
		nparts += part_map[pix].size();
	}
	for (auto& f : futs) {
		nparts += f.get();
	}
	return nparts;
}

void lc_parts2groups() {
	vector<hpx::future<void>> futs;
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async < lc_parts2groups_action > (c));
	}
	std::unordered_map<long long, lc_group_data> groups;
	int i = 0;
	while (i < part_buffer.size()) {
		const auto grp = part_buffer[i].group;
		if (grp != LC_EDGE_GROUP) {
			groups[grp].parts.push_back(part_buffer[i]);
			part_buffer[i] = part_buffer.back();
			part_buffer.pop_back();
		} else {
			i++;
		}
	}
	auto groups_old = std::move(groups);
	for (auto i = groups_old.begin(); i != groups_old.end(); i++) {
		auto tmp = std::move(i->second);
		if (i->second.parts.size() >= get_options().lc_min_group) {
			groups[i->first] = std::move(tmp);
		}
	}
	groups_old = decltype(groups_old)();
	hpx::wait_all(futs.begin(), futs.end());
}

static long long next_group_id() {
	return group_id_counter++ * hpx_size() + hpx_rank() + 1;
}

static int rank_from_group_id(long long id) {
	if (id != LC_NO_GROUP && id != LC_EDGE_GROUP) {
		return (id - 1) % hpx_size();
	} else {
		return -1;
	}
}

static void lc_send_particles(vector<lc_particle> parts) {
	for (const auto& part : parts) {
		const int pix = vec2pix(part.pos[XDIM], part.pos[YDIM], part.pos[ZDIM]);
		std::lock_guard<spinlock_type> lock(*mutex_map[pix]);
		part_map[pix].push_back(part);
	}
}

static void lc_send_buffer_particles(vector<lc_particle> parts) {
	int start, stop;
	std::unique_lock<shared_mutex_type> ulock(mutex);
	start = part_buffer.size();
	part_buffer.resize(start + parts.size());
	stop = part_buffer.size();
	ulock.unlock();
	constexpr int chunk_size = 1024;
	for (int i = 0; i < parts.size(); i += chunk_size) {
		const int end = std::min((int) parts.size(), i + chunk_size);
		std::shared_lock<shared_mutex_type> lock(mutex);
		for (int j = i; j < end; j++) {
			part_buffer[j + start] = parts[j];
		}
	}
}

void lc_buffer2homes() {
	vector<hpx::future<void>> futs1;
	vector<hpx::future<void>> futs2;
	for (const auto& c : hpx_children()) {
		futs1.push_back(hpx::async < lc_buffer2homes_action > (c));
	}
	const int nthreads = hpx_hardware_concurrency();
	for (int proc = 0; proc < nthreads; proc++) {
		futs2.push_back(hpx::async([proc,nthreads]() {
			const int begin = (size_t) proc * part_buffer.size() / nthreads;
			const int end = (size_t) (proc+1) * part_buffer.size() / nthreads;
			std::unordered_map<int,vector<lc_particle>> sends;
			for( int i = begin; i < end; i++) {
				const auto& part = part_buffer[i];
				const int pix = vec2pix(part.pos[XDIM],part.pos[YDIM],part.pos[ZDIM]);
				const int rank = pix2rank(pix);
				sends[rank].push_back(part);
			}
			vector<hpx::future<void>> futs;
			for( auto i = sends.begin(); i != sends.end(); i++) {
				futs.push_back(hpx::async<lc_send_particles_action>(hpx_localities()[i->first], std::move(i->second)));
			}
			hpx::wait_all(futs.begin(), futs.end());
		}));
	}

	hpx::wait_all(futs2.begin(), futs2.end());
	part_buffer = decltype(part_buffer)();
	hpx::wait_all(futs1.begin(), futs1.end());
}

void lc_groups2homes() {
	vector<hpx::future<void>> futs;
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async < lc_groups2homes_action > (c));
	}
	for (int pix = my_pix_range.first; pix < my_pix_range.second; pix++) {
		futs.push_back(hpx::async([pix]() {
			auto& parts = part_map[pix];
			std::unordered_map<int,vector<lc_particle>> sends;
			for( int i = 0; i < parts.size(); i++) {
				int rank = rank_from_group_id(parts[i].group);
				if( rank == -1 ) {
					if( parts[i].group == LC_EDGE_GROUP ) {
						rank = hpx_rank();
					}
				}
				if( rank != -1 ) {
					sends[rank].push_back(parts[i]);
				}
			}
			parts = vector<lc_particle>();
			vector<hpx::future<void>> futs;
			for( auto i = sends.begin(); i != sends.end(); i++) {
				futs.push_back(hpx::async<lc_send_buffer_particles_action>(hpx_localities()[i->first],std::move(i->second)));
			}
			hpx::wait_all(futs.begin(), futs.end());
		}));
	}
	hpx::wait_all(futs.begin(), futs.end());
}

static int lc_particles_sort(int pix, pair<int> rng, double xmid, int xdim) {
	int begin = rng.first;
	int end = rng.second;
	int lo = begin;
	int hi = end;
	auto& parts = part_map[pix];
	while (lo < hi) {
		if (parts[lo].pos[xdim] >= xmid) {
			while (lo != hi) {
				hi--;
				if (parts[hi].pos[xdim] < xmid) {
					auto tmp = parts[hi];
					parts[hi] = parts[lo];
					parts[lo] = tmp;
					break;
				}
			}
		}
		lo++;
	}
	return hi;

}

std::pair<int, range<double>> lc_tree_create(int pix, range<double> box, pair<int> part_range) {
	lc_tree_node this_node;
	vector<lc_tree_node>& nodes = tree_map[pix];
	vector<lc_particle>& parts = part_map[pix];
	range<double> part_box;
	if (part_range.second - part_range.first > GROUP_BUCKET_SIZE) {
		const int xdim = box.longest_dim();
		const double xmid = 0.5 * (box.begin[xdim] + box.end[xdim]);
		const int mid = lc_particles_sort(pix, part_range, xmid, xdim);
		auto box_left = box;
		auto box_right = box;
		auto parts_left = part_range;
		auto parts_right = part_range;
		box_left.end[xdim] = box_right.begin[xdim] = xmid;
		parts_left.second = parts_right.first = mid;
		const auto rcl = lc_tree_create(pix, box_left, parts_left);
		const auto rcr = lc_tree_create(pix, box_right, parts_right);
		for (int dim = 0; dim < NDIM; dim++) {
			part_box.begin[dim] = std::min(rcl.second.begin[dim], rcr.second.begin[dim]);
			part_box.end[dim] = std::max(rcl.second.end[dim], rcr.second.end[dim]);
		}
		this_node.children[LEFT].pix = this_node.children[RIGHT].pix = pix;
		this_node.children[LEFT].index = rcl.first;
		this_node.children[RIGHT].index = rcr.first;
	} else {
		for (int dim = 0; dim < NDIM; dim++) {
			part_box.begin[dim] = std::numeric_limits<double>::max();
			part_box.end[dim] = -std::numeric_limits<double>::max();
		}
		for (int i = part_range.first; i < part_range.second; i++) {
			parts[i].group = LC_NO_GROUP;
			for (int dim = 0; dim < NDIM; dim++) {
				const lc_real x = parts[i].pos[dim];
				part_box.begin[dim] = std::min(part_box.begin[dim], x);
				part_box.end[dim] = std::max(part_box.end[dim], x);
			}
		}
		this_node.children[LEFT].index = this_node.children[RIGHT].index = -1;
		this_node.children[LEFT].pix = this_node.children[RIGHT].pix = -1;
	}
	this_node.part_range = part_range;
	this_node.box = part_box;
	this_node.pix = pix;
	this_node.active = true;
	this_node.last_active = true;
	int index = nodes.size();
	nodes.push_back(this_node);
	std::pair<int, range<double>> rc;
	rc.first = index;
	rc.second = part_box;
	return rc;
}

size_t lc_find_groups_local(lc_tree_id self_id, vector<lc_tree_id> checklist, double link_len) {
	vector<lc_tree_id> nextlist;
	vector<lc_tree_id> leaflist;
	auto& self = tree_map[self_id.pix][self_id.index];
	const bool iamleaf = self.children[LEFT].index == -1;
	auto mybox = self.box.pad(link_len * 1.0001);
	do {
		for (int ci = 0; ci < checklist.size(); ci++) {
			const auto check = checklist[ci];
			const auto& other = tree_map[check.pix][check.index];
			if (other.last_active) {
				if (mybox.intersection(other.box).volume() > 0) {
					if (other.children[LEFT].index == -1) {
						leaflist.push_back(check);
					} else {
						nextlist.push_back(other.children[LEFT]);
						nextlist.push_back(other.children[RIGHT]);
					}
				}
			}
		}
		std::swap(nextlist, checklist);
		nextlist.resize(0);
	} while (iamleaf && checklist.size());
	if (iamleaf) {
		bool found_any_link = false;
		const double link_len2 = sqr(link_len);
		for (int li = 0; li < leaflist.size(); li++) {
			if (self_id != leaflist[li]) {
				const auto leafi = leaflist[li];
				const auto other = tree_map[leafi.pix][leafi.index];
				for (int pi = other.part_range.first; pi < other.part_range.second; pi++) {
					const auto& part = part_map[other.pix][pi];
					if (mybox.contains(part.pos)) {
						for (int pj = self.part_range.first; pj < self.part_range.second; pj++) {
							auto& mypart = part_map[self.pix][pj];
							const double dx = mypart.pos[XDIM] - part.pos[XDIM];
							const double dy = mypart.pos[YDIM] - part.pos[YDIM];
							const double dz = mypart.pos[ZDIM] - part.pos[ZDIM];
							const double R2 = sqr(dx, dy, dz);
							if (R2 <= link_len2) {
								if (mypart.group == LC_NO_GROUP) {
									mypart.group = next_group_id();
									found_any_link = true;
								}
								if (mypart.group > part.group) {
									mypart.group = part.group;
									found_any_link = true;
								}
							}
						}
					}
				}
			}
		}
		bool found_link;
		do {
			found_link = false;
			for (int pi = self.part_range.first; pi < self.part_range.second; pi++) {
				auto A = part_map[self.pix][pi];
				for (int pj = self.part_range.first; pj < self.part_range.second; pj++) {
					if (pi != pj) {
						auto B = part_map[self.pix][pj];
						const double dx = A.pos[XDIM] - B.pos[XDIM];
						const double dy = A.pos[YDIM] - B.pos[YDIM];
						const double dz = A.pos[ZDIM] - B.pos[ZDIM];
						const double R2 = sqr(dx, dy, dz);
						if (R2 <= link_len2) {
							if (A.group == LC_NO_GROUP) {
								A.group = next_group_id();
								found_any_link = true;
								found_link = true;
							}
							if (B.group == LC_NO_GROUP) {
								B.group = next_group_id();
								found_any_link = true;
								found_link = true;
							}
							if (A.group != B.group) {
								if (A.group < B.group) {
									B.group = A.group;
								} else {
									A.group = B.group;
								}
								found_any_link = true;
								found_link = true;
							}
						}

					}
				}
			}
		} while (found_link);
		self.active = found_any_link;
		return int(found_any_link);
	} else {
		int nactive = lc_find_groups_local(self.children[LEFT], checklist, link_len);
		nactive += lc_find_groups_local(self.children[RIGHT], std::move(checklist), link_len);
		self.active = nactive;
		return nactive;
	}

}

size_t lc_find_groups() {
	vector < hpx::future < size_t >> futs;
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async < lc_find_groups_action > (c));
	}
	size_t rc = 0;
	const double link_len = get_options().lc_b / (double) get_options().parts_dim;
	vector<hpx::future<void>> futs2;
	for (int pix = my_pix_range.first; pix < my_pix_range.second; pix++) {
		futs2.push_back(hpx::async([pix]() {
			auto& nodes = tree_map[pix];
			for( int i = 0; i < nodes.size(); i++) {
				nodes[i].last_active = nodes[i].active;
			}
		}));
	}
	hpx::wait_all(futs2.begin(), futs2.end());
	for (int pix = my_pix_range.first; pix < my_pix_range.second; pix++) {
		futs.push_back(hpx::async([pix, link_len]() {
			auto check_pix = pix_neighbors(pix);
			check_pix.push_back(pix);
			vector<lc_tree_id> checklist(check_pix.size());
			for (int i = 0; i < check_pix.size(); i++) {
				checklist[i].pix = check_pix[i];
				checklist[i].index = tree_map[check_pix[i]].size() - 1;
			}
			return lc_find_groups_local(checklist.back(), checklist, link_len);
		}));
	}
	for (auto& f : futs) {
		rc += f.get();
	}
	return rc;
}

static vector<lc_particle> lc_get_particles(int pix) {
	return part_map[pix];
}

static int vec2pix(double x, double y, double z) {
	vec3 vec;
	vec.x = x;
	vec.y = y;
	vec.z = z;
	return healpix->vec2pix(vec);
}

static int pix2rank(int pix) {
	int n = (size_t) pix * hpx_size() / Npix;
	while (pix < (size_t) n * Npix / hpx_size()) {
		n--;
	}
	while (pix >= (size_t)(n + 1) * Npix / hpx_size()) {
		n++;
	}
	return n;
}

void lc_particle_boundaries() {
	vector<hpx::future<void>> futs;
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async < lc_particle_boundaries_action > (c));
	}
	vector<hpx::future<vector<lc_particle>>>pfuts;
	pfuts.reserve(bnd_pix.size());
	for (auto pix : bnd_pix) {
		pfuts.push_back(hpx::async < lc_get_particles_action > (hpx_localities()[pix2rank(pix)], pix));
	}
	int i = 0;
	for (auto pix : bnd_pix) {
		part_map[pix] = pfuts[i].get();
		i++;
	}

	hpx::wait_all(futs.begin(), futs.end());

}

void lc_form_trees(double tau, double link_len) {
	vector<hpx::future<void>> futs;
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async < lc_form_trees_action > (c, tau, link_len));
	}
	for (auto iter = part_map.begin(); iter != part_map.end(); iter++) {
		const int pix = iter->first;
		auto& parts = part_map[pix];
		futs.push_back(hpx::async([pix,link_len,tau](vector<lc_particle>* parts_ptr) {
			auto& parts = *parts_ptr;
			range<double> box;
			pair<int> part_range;
			part_range.first = 0;
			part_range.second = parts.size();
			for (int dim = 0; dim < NDIM; dim++) {
				box.begin[dim] = std::numeric_limits<double>::max();
				box.end[dim] = -std::numeric_limits<double>::max();
			}
			for (int i = 0; i < parts.size(); i++) {
				for (int dim = 0; dim < NDIM; dim++) {
					const auto x = parts[i].pos[dim];
					box.begin[dim] = std::min(box.begin[dim], x);
					box.end[dim] = std::max(box.end[dim], x);
				}
			}
			lc_tree_create(pix, box, part_range);
			for( int i = 0; i < parts.size(); i++) {
				const double x = parts[i].pos[XDIM];
				const double y = parts[i].pos[YDIM];
				const double z = parts[i].pos[ZDIM];
				const double R2 = sqr(x, y, z);
				const double R = sqrt(R2);
				if( R < (tau_max - tau) + link_len * 1.0001) {
					parts[i].group = LC_EDGE_GROUP;
				} else {
					parts[i].group = LC_NO_GROUP;
				}
				ASSERT(R2 >= tau_max - tau);
			}
		}, &parts));
	}

	hpx::wait_all(futs.begin(), futs.end());
}

static vector<int> pix_neighbors(int pix) {
	vector<int> neighbors;
	neighbors.reserve(8);
	fix_arr<int, 8> result;
	healpix->neighbors(pix, result);
	for (int i = 0; i < 8; i++) {
		if (result[i] != -1) {
			neighbors.push_back(result[i]);
		}
	}
	return neighbors;
}

void lc_init(double tau_max_) {
	vector<hpx::future<void>> futs;
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async < lc_init_action > (c, tau_max_));
	}

	Nside = std::ceil(std::sqrt(2.0 * hpx::thread::hardware_concurrency() * hpx_size() / 12.0));
	Nside = std::min(Nside, (int) (get_options().parts_dim * std::sqrt(3) / get_options().lc_b / 100.0 / 2.0));
	PRINT("Nside = %i\n", Nside);
	Npix = 12 * sqr(Nside);
	healpix = std::make_shared < healpix_type > (Nside, NEST, SET_NSIDE);
	tau_max = tau_max_;
	my_pix_range.first = (size_t) hpx_rank() * Npix / hpx_size();
	my_pix_range.second = (size_t)(hpx_rank() + 1) * Npix / hpx_size();
	for (int pix = my_pix_range.first; pix < my_pix_range.second; pix++) {
		part_map[pix].resize(0);
		tree_map[pix].resize(0);
		mutex_map[pix] = std::make_shared<spinlock_type>();
		const auto neighbors = pix_neighbors(pix);
		for (const auto& n : neighbors) {
			if (n < my_pix_range.first || n >= my_pix_range.second) {
				bnd_pix.insert(n);
				part_map[n].resize(0);
				tree_map[n].resize(0);
			}
		}
	}

	hpx::wait_all(futs.begin(), futs.end());
}

int lc_add_particle(lc_real x0, lc_real y0, lc_real z0, lc_real x1, lc_real y1, lc_real z1, float vx, float vy, float vz, float t, float dt) {
	static simd_float8 images[NDIM] =
			{ simd_float8(0, -1, 0, -1, 0, -1, 0, -1), simd_float8(0, 0, -1, -1, 0, 0, -1, -1), simd_float8(0, 0, 0, 0, -1, -1, -1, -1) };
	const float tau_max_inv = 1.0 / tau_max;
	const simd_float8 simd_c0 = simd_float8(tau_max_inv);
	array<simd_float8, NDIM> X0;
	array<simd_float8, NDIM> X1;
	const simd_float8 simd_tau0 = simd_float8(t);
	const simd_float8 simd_tau1 = simd_float8(t + dt);
	simd_float8 dist0;
	simd_float8 dist1;
	int rc = 0;
	X0[XDIM] = simd_float8(x0) + images[XDIM];
	X0[YDIM] = simd_float8(y0) + images[YDIM];
	X0[ZDIM] = simd_float8(z0) + images[ZDIM];
	X1[XDIM] = simd_float8(x1) + images[XDIM];
	X1[YDIM] = simd_float8(y1) + images[YDIM];
	X1[ZDIM] = simd_float8(z1) + images[ZDIM];
	dist0 = sqrt(sqr(X0[0], X0[1], X0[2]));
	dist1 = sqrt(sqr(X1[0], X1[1], X1[2]));
	simd_float8 tau0 = simd_tau0 + dist0;
	simd_float8 tau1 = simd_tau1 + dist1;
	simd_int8 I0 = tau0 * simd_c0;
	simd_int8 I1 = tau1 * simd_c0;
	for (int ci = 0; ci < SIMD_FLOAT8_SIZE; ci++) {
		if (dist1[ci] <= 1.0 || dist0[ci] <= 1.0) {
			const int i0 = I0[ci];
			const int i1 = I1[ci];
			if (i0 != i1) {
				x0 = X0[XDIM][ci];
				y0 = X0[YDIM][ci];
				z0 = X0[ZDIM][ci];
				const double ti = (i0 + 1) * tau_max;
				const double sqrtauimtau0 = sqr(ti - t);
				const double tau0mtaui = t - ti;
				const double u2 = sqr(vx, vy, vz);                                    // 5
				const double x2 = sqr(x0, y0, z0);                                       // 5
				const double udotx = vx * x0 + vy * y0 + vz * z0;               // 5
				const double A = 1.f - u2;                                                     // 1
				const double B = 2.0 * (tau0mtaui - udotx);                                    // 2
				const double C = sqrtauimtau0 - x2;                                            // 1
				const double t = -(B + sqrt(B * B - 4.f * A * C)) / (2.f * A);                // 15
				const double x1 = x0 + vx * t;                                            // 2
				const double y1 = y0 + vy * t;                                            // 2
				const double z1 = z0 + vz * t;                                            // 2
				long int ipix;
				if (sqr(x1, y1, z1) <= 1.f) {                                                 // 6
					rc++;
					const auto pix = vec2pix(x1, y1, z1);
					lc_particle part;
					part.pos[XDIM] = x1;
					part.pos[XDIM] = y1;
					part.pos[XDIM] = z1;
					part.vel[XDIM] = vx;
					part.vel[YDIM] = vy;
					part.vel[ZDIM] = vz;
		//			std::unique_lock<shared_mutex_type> lock(mutex);
		//			part_buffer.push_back(part);
				}
			}
		}
	}
	return rc;
}
