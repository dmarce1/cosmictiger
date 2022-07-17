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

#include <cosmictiger/bh.hpp>
#include <cosmictiger/constants.hpp>
#include <cosmictiger/containers.hpp>
#include <cosmictiger/cosmology.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/lightcone.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/range.hpp>
#include <cosmictiger/fixed.hpp>
#include <cosmictiger/safe_io.hpp>
#include <cosmictiger/group_entry.hpp>
#include <cosmictiger/device_vector.hpp>
#include <cosmictiger/timer.hpp>
#include <cosmictiger/compress.hpp>
#include <cosmictiger/rockstar.hpp>

#include <atomic>
#include <memory>
#include <shared_mutex>
#include <unordered_set>
#include <unordered_map>

#include <healpix/healpix_base.h>
#include <chealpix.h>

using healpix_type = T_Healpix_Base< int >;

struct lc_group_data {
	vector<lc_particle> parts;
	group_entry arc;
};

static std::shared_ptr<healpix_type> healpix;
static pair<int> my_pix_range;
static std::unordered_set<int> bnd_pix;
static lc_part_map_type* part_map_ptr = nullptr;
static lc_tree_map_type* tree_map_ptr = nullptr;
static std::unordered_map<int, std::shared_ptr<spinlock_type>> mutex_map;
static lc_group group_id_counter = 0;
//static vector<group_entry> saved_groups;
static vector<lc_particle> part_buffer;
static shared_mutex_type shared_mutex;
static spinlock_type unique_mutex;
static spinlock_type leaf_mutex;
static device_vector<lc_tree_id> leaf_nodes;
static double tau_max;
static int Nside;
static int Npix;

static lc_particles lc_get_particles1(int pix);
static pair<vector<long long>, vector<char>> lc_get_particles2(int pix);
static void lc_send_particles(vector<lc_particle>);
static void lc_send_buffer_particles(vector<lc_particle>);
static int vec2pix(double x, double y, double z);
static int pix2rank(int pix);
static int compute_nside(double tau);
static vector<int> pix_neighbors(int pix);
static int epoch = 0;

HPX_PLAIN_ACTION (lc_time_to_flush);
HPX_PLAIN_ACTION (lc_parts2groups);
HPX_PLAIN_ACTION (lc_init);
HPX_PLAIN_ACTION (lc_get_particles1);
HPX_PLAIN_ACTION (lc_get_particles2);
HPX_PLAIN_ACTION (lc_form_trees);
HPX_PLAIN_ACTION (lc_buffer2homes);
HPX_PLAIN_ACTION (lc_groups2homes);
HPX_PLAIN_ACTION (lc_send_buffer_particles);
HPX_PLAIN_ACTION (lc_send_particles);
HPX_PLAIN_ACTION (lc_find_groups);
HPX_PLAIN_ACTION (lc_find_neighbors);
HPX_PLAIN_ACTION (lc_particle_boundaries1);
HPX_PLAIN_ACTION (lc_particle_boundaries2);

static std::atomic<int> next_filenum = 0;

static int lc_next_filenum();

HPX_PLAIN_ACTION (lc_next_filenum);

static int lc_next_filenum() {
	if (hpx_rank() == 0) {
		return next_filenum++;
	} else {
		lc_next_filenum_action act;
		static const auto root = hpx_localities()[0];
		return act(root);
	}
}

size_t lc_add_parts(const device_vector<device_vector<lc_entry>>& entries, double scale, double tau) {

	const int start = part_buffer.size();
	vector<size_t> counts(entries.size() + 1);
	size_t count = 0;
	for (int i = 0; i < entries.size(); i++) {
		counts[i] = count;
		count += entries[i].size();
	}
	counts.back() = count;
	part_buffer.resize(start + count);
	const int nthreads = hpx_hardware_concurrency();
	vector<hpx::future<void>> futs;
	std::atomic<int> index(0);
	for (int proc = 0; proc < nthreads; proc++) {
		futs.push_back(hpx::async([scale, tau, &index,start,nthreads,proc,&entries,&counts]() {
			//	const auto a0 = 1.0 / (get_options().z0 + 1.0);
				int k = index++;
				double m = 0.0;
				while( k < entries.size()) {
					auto& entry = entries[k];
					const int b = start + counts[k];
					const int e = start + counts[k+1];
					for (int i = b; i < e; i++) {
						const int j = i - b;
						auto& pi = part_buffer[i];
						const auto& pj = entry[j];
						pi.pos[XDIM] = pj.x;
						pi.pos[YDIM] = pj.y;
						pi.pos[ZDIM] = pj.z;
						const double tau1 = sqr(pj.x.to_double(), pj.y.to_double(), pj.z.to_double());
						const auto a = scale + cosmos_dadtau(scale) * (tau1 - tau);
						pi.vel[XDIM] = pj.vx / a;
						pi.vel[YDIM] = pj.vy / a;
						pi.vel[ZDIM] = pj.vz / a;
						m = std::max(pj.vx, (float) fabs(m));
					}
					k = index++;
				}
			}));
	}
	hpx::wait_all(futs.begin(), futs.end());
	return 0;
}

int lc_nside() {
	return Nside;
}

void lc_save(FILE* fp) {

	int dummy;
	size_t sz = part_buffer.size();
	fwrite(&sz, sizeof(size_t), 1, fp);
	fwrite(part_buffer.data(), sizeof(lc_particle), part_buffer.size(), fp);
	fwrite(&epoch, sizeof(int), 1, fp);
	fwrite(&sz, sizeof(size_t), 1, fp);
	fwrite(&group_id_counter, sizeof(lc_group), 1, fp);
}

void lc_load(FILE* fp) {
	size_t sz;
	int dummy;
	FREAD(&sz, sizeof(size_t), 1, fp);
	part_buffer.resize(sz);
	FREAD(part_buffer.data(), sizeof(lc_particle), part_buffer.size(), fp);
	FREAD(&epoch, sizeof(int), 1, fp);
	FREAD(&sz, sizeof(size_t), 1, fp);
	FREAD(&group_id_counter, sizeof(lc_group), 1, fp);
}

size_t lc_time_to_flush(double tau, double tau_max_) {
	vector < hpx::future < size_t >> futs;
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async<lc_time_to_flush_action>(c, tau, tau_max_));
	}
	size_t nparts = part_buffer.size();
	for (auto& f : futs) {
		nparts = std::max(nparts, f.get());
	}
	if (hpx_rank() == 0) {
		tau_max = tau_max_;
		const int nside = compute_nside(tau);
		const int npix = nside == 0 ? 1 : 12 * sqr(nside);
		const size_t ranks = std::min(npix, hpx_size());
		const size_t parts_per_rank = std::pow(get_options().parts_dim, NDIM) / hpx_size();
		/*		double factor = 1.0 / 8.0;
		 double tm = tau / tau_max * 64;
		 if (tm > 63.0) {
		 factor = (64.0 - tm) / 8.0;
		 }*/
		if (nparts >= (M_PI / 10.0) * parts_per_rank) {
			return 1;
		} else {
			return 0;
		}
	} else {
		return nparts;
	}
}

void lc_parts2groups(double a, double link_len, int ti) {
	vector<hpx::future<void>> futs;
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async<lc_parts2groups_action>(c, a, link_len, ti));
	}
	std::unordered_map<lc_group, vector<lc_entry>> groups_map;
	int i = 0;
	while (i < part_buffer.size()) {
		const auto grp = part_buffer[i].group;
		if (grp != LC_EDGE_GROUP) {
			lc_entry entry;
			const auto& p = part_buffer[i];
			entry.x = p.pos[XDIM];
			entry.y = p.pos[YDIM];
			entry.z = p.pos[ZDIM];
			entry.vx = p.vel[XDIM];
			entry.vy = p.vel[YDIM];
			entry.vz = p.vel[ZDIM];
			groups_map[grp].push_back(entry);
			part_buffer[i] = part_buffer.back();
			part_buffer.pop_back();
		} else {
			i++;
		}
	}
	static int total = 0;
	const auto min_sz = get_options().lc_min_group;
	auto j = groups_map.begin();
	while (j != groups_map.end()) {
		if (j->second.size() < min_sz && j->first != LC_NO_GROUP) {
			groups_map[LC_NO_GROUP].insert(groups_map[LC_NO_GROUP].end(), j->second.begin(), j->second.end());
			j = groups_map.erase(j);
		} else {
			j++;
		}
	}
	vector<pair<lc_group, vector<lc_entry>>> groups;
	vector<pair<lc_group, compressed_particles>> group_arcs;
	for (auto i = groups_map.begin(); i != groups_map.end(); i++) {
		pair<lc_group, vector<lc_entry>> entry;
		entry.first = i->first;
		total += i->second.size();
		entry.second = std::move(i->second);
		groups.push_back(std::move(entry));
	}
	std::sort(groups.begin(), groups.end(), [](const pair<lc_group, vector<lc_entry>>& a, const pair<lc_group, vector<lc_entry>>& b){
		return a.second.size() >= b.second.size();

	});
	std::atomic<int> next_index(0);
	const int nthreads = 2 * hpx_hardware_concurrency();
	for (int proc = 0; proc < nthreads; proc++) {
		futs.push_back(hpx::async([&next_index, &groups]() {
			int index = next_index++;
			while( index < groups.size()) {
				if( groups[index].first != LC_NO_GROUP) {
					auto subgroups = rockstar_find_subgroups(groups[index].second, false);
				}
				index = next_index++;
			}
		}));
	}
	hpx::wait_all(futs.begin(), futs.end());
	epoch++;
}

static int rank_from_group_id(long long id) {
	if (id != LC_NO_GROUP && id != LC_EDGE_GROUP) {
		return (id - 1) % hpx_size();
	} else {
		return -1;
	}
}

static void lc_send_particles(vector<lc_particle> parts) {
	timer tm;
	auto& part_map = *part_map_ptr;
	for (const auto& part : parts) {
		const int pix = vec2pix(part.pos[XDIM].to_double(), part.pos[YDIM].to_double(), part.pos[ZDIM].to_double());
		std::unique_lock<spinlock_type> lock(*mutex_map[pix]);
		for (int dim = 0; dim < NDIM; dim++) {
			part_map[pix].pos[dim].push_back(part.pos[dim]);
			part_map[pix].vel[dim].push_back(part.vel[dim]);
		}
		part_map[pix].group.push_back(part.group);
	}
}

static void lc_send_buffer_particles(vector<lc_particle> parts) {
	int start, stop;
	std::unique_lock<shared_mutex_type> ulock(shared_mutex);
	start = part_buffer.size();
	part_buffer.resize(start + parts.size());
	stop = part_buffer.size();
	ulock.unlock();
	constexpr int chunk_size = 1024;
	for (int i = 0; i < parts.size(); i += chunk_size) {
		const int end = std::min((int) parts.size(), i + chunk_size);
		std::shared_lock<shared_mutex_type> lock(shared_mutex);
		for (int j = i; j < end; j++) {
			part_buffer[j + start] = parts[j];
		}
	}
}

void lc_buffer2homes() {
	vector<hpx::future<void>> futs1;
	vector<hpx::future<void>> futs2;
	for (const auto& c : hpx_children()) {
		futs1.push_back(hpx::async<lc_buffer2homes_action>(c));
	}
	const int nthreads = hpx_hardware_concurrency();
	static mutex_type map_mutex;
	for (int proc = 0; proc < nthreads; proc++) {
		futs2.push_back(hpx::async([proc,nthreads]() {
			const int begin = (size_t) proc * part_buffer.size() / nthreads;
			const int end = (size_t) (proc+1) * part_buffer.size() / nthreads;
			std::unordered_map<int,vector<lc_particle>> sends;
			for( int i = begin; i < end; i++) {
				const auto& part = part_buffer[i];
				const int pix = vec2pix(part.pos[XDIM].to_double(),part.pos[YDIM].to_double(),part.pos[ZDIM].to_double());
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
	auto& part_map = *part_map_ptr;
	vector<hpx::future<void>> futs;
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async<lc_groups2homes_action>(c));
	}
	for (int pix = my_pix_range.first; pix < my_pix_range.second; pix++) {
		futs.push_back(hpx::async([pix,&part_map]() {
			vector<hpx::future<void>> futs;
			auto& parts = part_map[pix];
			std::unordered_map<int,vector<lc_particle>> sends;
			for( int i = 0; i < parts.group.size(); i++) {
				int rank = rank_from_group_id(parts.group[i]);
				if( rank == -1 ) {
					if( parts.group[i] == LC_EDGE_GROUP ) {
						rank = hpx_rank();
					}
				}
				if( rank != -1 ) {
					lc_particle part;
					for( int dim = 0; dim < NDIM; dim++) {
						part.pos[dim] = parts.pos[dim][i];
						part.vel[dim] = parts.vel[dim][i];
					}
					part.group = parts.group[i];
					sends[rank].push_back(part);
				}
			}
			parts = lc_particles();
			for( auto i = sends.begin(); i != sends.end(); i++) {
				futs.push_back(hpx::async<lc_send_buffer_particles_action>(hpx_localities()[i->first],std::move(i->second)));
			}
			hpx::wait_all(futs.begin(), futs.end());
		}));
	}
	hpx::wait_all(futs.begin(), futs.end());
}

static int lc_particles_sort(int pix, pair<int> rng, double xm, int xdim) {
	int begin = rng.first;
	int end = rng.second;
	int lo = begin;
	int hi = end;
	auto& part_map = *part_map_ptr;
	auto& parts = part_map[pix];
	lc_real xmid = xm;
	while (lo < hi) {
		if (parts.pos[xdim][lo] >= xmid) {
			while (lo != hi) {
				hi--;
				if (parts.pos[xdim][hi] < xmid) {
					for (int dim = 0; dim < NDIM; dim++) {
						std::swap(parts.pos[dim][hi], parts.pos[dim][lo]);
						std::swap(parts.vel[dim][hi], parts.vel[dim][lo]);
					}
					std::swap(parts.group[hi], parts.group[lo]);
					break;
				}
			}
		}
		lo++;
	}
	return hi;

}

std::pair<int, range<double>> lc_tree_create(int pix, range<double> box, pair<int> part_range) {
	constexpr int bucket_size = GROUP_BUCKET_SIZE;
	lc_tree_node this_node;
	auto& part_map = *part_map_ptr;
	auto& tree_map = *tree_map_ptr;
	auto& nodes = tree_map[pix];
	auto& parts = part_map[pix];
	range<double> part_box;
	bool leaf;
	if (part_range.second - part_range.first > bucket_size) {
		leaf = false;
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
		leaf = true;
		for (int dim = 0; dim < NDIM; dim++) {
			part_box.begin[dim] = 1.0;
			part_box.end[dim] = -1.0;
		}
		for (int i = part_range.first; i < part_range.second; i++) {
			parts.group[i] = LC_NO_GROUP;
			for (int dim = 0; dim < NDIM; dim++) {
				const double x = parts.pos[dim][i].to_double();
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
	this_node.active = 1;
	this_node.last_active = 0;
	int index = nodes.size();
	nodes.push_back(this_node);
	if (leaf) {
		lc_tree_id selfid;
		selfid.index = index;
		selfid.pix = pix;
		std::lock_guard<spinlock_type> lock(leaf_mutex);
		leaf_nodes.push_back(selfid);
	}
	std::pair<int, range<double>> rc;
	rc.first = index;
	rc.second = part_box;
	return rc;
}

void lc_find_neighbors_local(lc_tree_id self_id, vector<lc_tree_id> checklist, double link_len) {
	static thread_local vector<lc_tree_id> nextlist;
	static thread_local vector<lc_tree_id> leaflist;
	nextlist.resize(0);
	leaflist.resize(0);
	auto& tree_map = *tree_map_ptr;
	auto& self = tree_map[self_id.pix][self_id.index];
	const bool iamleaf = self.children[LEFT].index == -1;
	auto mybox = self.box.pad(link_len * 1.01);
	do {
		for (int ci = 0; ci < checklist.size(); ci++) {
			const auto check = checklist[ci];
			const auto& other = tree_map[check.pix][check.index];
			if (mybox.intersection(other.box).volume() > 0) {
				if (other.children[LEFT].index == -1) {
					leaflist.push_back(check);
				} else {
					nextlist.push_back(other.children[LEFT]);
					nextlist.push_back(other.children[RIGHT]);
				}
			}
		}
		std::swap(nextlist, checklist);
		nextlist.resize(0);
	} while (iamleaf && checklist.size());
	if (iamleaf) {
		self.neighbors.resize(leaflist.size());
		memcpy(self.neighbors.data(), leaflist.data(), sizeof(lc_tree_id) * leaflist.size());
	} else {
		checklist.insert(checklist.end(), leaflist.begin(), leaflist.end());
		if (checklist.size()) {
			lc_find_neighbors_local(self.children[LEFT], checklist, link_len);
			lc_find_neighbors_local(self.children[RIGHT], std::move(checklist), link_len);
		}
	}
}

size_t lc_find_neighbors() {
	auto& tree_map = *tree_map_ptr;
	vector<hpx::future<void>> futs;
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async<lc_find_neighbors_action>(c));
	}
	size_t rc = 0;
	const double link_len = get_options().lc_b / (double) get_options().parts_dim;
	for (int pix = my_pix_range.first; pix < my_pix_range.second; pix++) {
		futs.push_back(hpx::async([pix, link_len, &tree_map]() {
			vector<int> check_pix;
			check_pix = pix_neighbors(pix);
			check_pix.push_back(pix);
			vector<lc_tree_id> checklist(check_pix.size());
			for (int i = 0; i < check_pix.size(); i++) {
				checklist[i].pix = check_pix[i];
				checklist[i].index = tree_map[check_pix[i]].size() - 1;
			}
			lc_find_neighbors_local(checklist.back(), checklist, link_len);
		}));
	}
	for (auto& f : futs) {
		f.get();
	}
	return rc;
}

size_t lc_find_groups() {
	auto& tree_map = *tree_map_ptr;
	vector < hpx::future < size_t >> futs;
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async<lc_find_groups_action>(c));
	}
	size_t rc = 0;
	const double link_len = get_options().lc_b / (double) get_options().parts_dim;
	vector<hpx::future<void>> futs2;
	for (int pix = my_pix_range.first; pix < my_pix_range.second; pix++) {
		futs2.push_back(hpx::async([pix, &tree_map]() {
			auto& nodes = tree_map[pix];
			for( int i = 0; i < nodes.size(); i++) {
				nodes[i].last_active = nodes[i].active;
				nodes[i].active = 0;
			}
		}));
	}
	hpx::wait_all(futs2.begin(), futs2.end());

	return cuda_lightcone(leaf_nodes, part_map_ptr, tree_map_ptr, &group_id_counter);
}

static lc_particles lc_get_particles1(int pix) {
	auto& part_map = *part_map_ptr;
	return part_map[pix];
}

static pair<vector<long long>, vector<char>> lc_get_particles2(int pix) {
	auto& part_map = *part_map_ptr;
	auto& tree_map = *tree_map_ptr;
	pair<vector<long long>, vector<char>> rc;
	const auto& parts = part_map[pix];
	for (int i = 0; i < parts.group.size(); i++) {
		rc.first.push_back((long long) parts.group[i]);
	}
	const auto& nodes = tree_map[pix];
	for (int i = 0; i < nodes.size(); i++) {
		rc.second.push_back(nodes[i].active);
	}
	return rc;
}

static int vec2pix(double x, double y, double z) {
	if (Nside == 0) {
		return 0;
	} else {
		vec3 vec;
		vec.x = x;
		vec.y = y;
		vec.z = z;
		return healpix->vec2pix(vec);
	}
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

void lc_particle_boundaries1() {
	auto& part_map = *part_map_ptr;
	auto& tree_map = *tree_map_ptr;
	vector<hpx::future<void>> futs;
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async<lc_particle_boundaries1_action>(c));
	}
	vector<hpx::future<lc_particles>> pfuts;
	pfuts.reserve(bnd_pix.size());
	for (auto pix : bnd_pix) {
		const int rank = pix2rank(pix);
		pfuts.push_back(hpx::async<lc_get_particles1_action>(hpx_localities()[rank], pix));
	}
	int i = 0;
	for (auto pix : bnd_pix) {
		part_map[pix] = pfuts[i].get();
		i++;
	}

}

void lc_particle_boundaries2() {
	auto& part_map = *part_map_ptr;
	auto& tree_map = *tree_map_ptr;
	vector<hpx::future<void>> futs;
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async<lc_particle_boundaries2_action>(c));
	}
	vector<hpx::future<pair<vector<long long>, vector<char>>> >pfuts;
	pfuts.reserve(bnd_pix.size());
	for (auto pix : bnd_pix) {
		const int rank = pix2rank(pix);
		pfuts.push_back(hpx::async<lc_get_particles2_action>(hpx_localities()[rank], pix));
	}
	int i = 0;
	for (auto pix : bnd_pix) {
		const auto p = pfuts[i].get();
		const int nthreads = hpx_hardware_concurrency();
		vector<hpx::future<void>> futs;
		auto& parts = part_map[pix];
		for (int thread = 0; thread < nthreads; thread++) {
			futs.push_back(hpx::async([thread,nthreads,&p,&parts]() {
				const int jmin = thread * parts.group.size() / nthreads;
				const int jmax = (thread+1) * parts.group.size() / nthreads;
				for (int j = jmin; j < jmax; j++) {
					parts.group[j] = p.first[j];
				}
			}));
		}
		hpx::wait_all(futs.begin(), futs.end());
		auto& nodes = tree_map[pix];
		for (int thread = 0; thread < nthreads; thread++) {
			futs.push_back(hpx::async([thread,nthreads,&p,&nodes]() {
				const int jmin = thread * nodes.size() / nthreads;
				const int jmax = (thread+1) * nodes.size() / nthreads;
				for (int j = jmin; j < jmax; j++) {
					nodes[j].last_active = p.second[j];
				}
			}));
		}
		hpx::wait_all(futs.begin(), futs.end());
		i++;
	}

	hpx::wait_all(futs.begin(), futs.end());

}

void lc_form_trees(double tau, double link_len) {
	auto& part_map = *part_map_ptr;
	auto& tree_map = *tree_map_ptr;
	vector<hpx::future<void>> futs;
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async<lc_form_trees_action>(c, tau, link_len));
	}
	for (auto iter = part_map.begin(); iter != part_map.end(); iter++) {
		const int pix = iter->first;
		futs.push_back(hpx::async([pix,link_len,tau,&part_map]() {
			auto& parts = part_map[pix];
			range<double> box;
			pair<int> part_range;
			part_range.first = 0;
			part_range.second = parts.group.size();
			for (int dim = 0; dim < NDIM; dim++) {
				box.begin[dim] = 1.0;
				box.end[dim] = -1.0;
			}
			for (int i = 0; i < parts.group.size(); i++) {
				for (int dim = 0; dim < NDIM; dim++) {
					const auto x = parts.pos[dim][i].to_double();
					box.begin[dim] = std::min(box.begin[dim], x);
					box.end[dim] = std::max(box.end[dim], x);
				}
			}
			lc_tree_create(pix, box, part_range);
			for( int i = 0; i < parts.group.size(); i++) {
				const double x = parts.pos[XDIM][i].to_double();
				const double y = parts.pos[YDIM][i].to_double();
				const double z = parts.pos[ZDIM][i].to_double();
				const double R2 = sqr(x, y, z);
				const double R = sqrt(R2);
				if( sqr((float)(R - (tau_max - tau))) < (float)sqr(link_len) && tau < tau_max) {
					parts.group[i] = LC_EDGE_GROUP;
				} else {
					parts.group[i] = LC_NO_GROUP;
				}
				if( R < tau_max - tau) {
					PRINT( "-----> low R %e %e\n", R, tau_max - tau);
			//		ASSERT(R >= tau_max - tau);
				}
			}

		}));
	}
	std::sort(leaf_nodes.begin(), leaf_nodes.end(), [&tree_map](lc_tree_id a, lc_tree_id b) {
		if( a.pix < b.pix ) {
			return true;
		} else if( a.pix > b.pix ) {
			return false;
		} else {
			const auto A = tree_map[a.pix][a.index];
			const auto B = tree_map[b.pix][b.index];
			return A.part_range.first < B.part_range.first;
		}
	});
	hpx::wait_all(futs.begin(), futs.end());
}

static vector<int> pix_neighbors(int pix) {
	vector<int> neighbors;
	if (Nside != 0) {
		neighbors.reserve(8);
		fix_arr<int, 8> result;
		healpix->neighbors(pix, result);
		for (int i = 0; i < 8; i++) {
			if (result[i] != -1) {
				neighbors.push_back(result[i]);
			}
		}
	}
	return neighbors;
}

static int compute_nside(double tau) {
	double nside = std::sqrt(8.0 * hpx_hardware_concurrency() * hpx_size() / 12.0);
	const double w0 = std::max((tau_max - tau) / tau_max, 0.0);
	nside = std::min(nside, (get_options().parts_dim * std::sqrt(3) / get_options().lc_b / 2.0 * w0));
	Nside = 1;
	while (Nside < nside) {
		Nside *= 2;
	}
	Nside /= 2;
//	PRINT("compute_nside = %i\n", Nside);
	return Nside;
}

void lc_init(double tau, double tau_max_) {
	vector<hpx::future<void>> futs;
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async<lc_init_action>(c, tau, tau_max_));
	}
	if (part_map_ptr == nullptr) {
		CUDA_CHECK(cudaMallocManaged(&part_map_ptr, sizeof(lc_part_map_type)));
		CUDA_CHECK(cudaMallocManaged(&tree_map_ptr, sizeof(lc_tree_map_type)));
		new (part_map_ptr) lc_part_map_type;
		new (tree_map_ptr) lc_tree_map_type;
	}
	auto& part_map = *part_map_ptr;
	auto& tree_map = *tree_map_ptr;
	mutex_map = decltype(mutex_map)();
	bnd_pix = decltype(bnd_pix)();
	leaf_nodes = decltype(leaf_nodes)();
	tau_max = tau_max_;
	Nside = compute_nside(tau);
	Npix = Nside == 0 ? 1 : 12 * sqr(Nside);
	if (Nside > 0) {
		healpix = std::make_shared < healpix_type > (Nside, NEST, SET_NSIDE);
	} else {
		healpix = nullptr;
	}
	my_pix_range.first = (size_t) hpx_rank() * Npix / hpx_size();
	my_pix_range.second = (size_t)(hpx_rank() + 1) * Npix / hpx_size();
	for (int pix = my_pix_range.first; pix < my_pix_range.second; pix++) {
		part_map[pix].group.resize(0);
		for (int dim = 0; dim < NDIM; dim++) {
			part_map[pix].pos[dim].resize(0);
			part_map[pix].vel[dim].resize(0);
		}
		tree_map[pix].resize(0);
		mutex_map[pix] = std::make_shared<spinlock_type>();
		const auto neighbors = pix_neighbors(pix);
		for (const auto& n : neighbors) {
			if (n < my_pix_range.first || n >= my_pix_range.second) {
				bnd_pix.insert(n);
				part_map[n].group.resize(0);
				for (int dim = 0; dim < NDIM; dim++) {
					part_map[n].pos[dim].resize(0);
					part_map[n].vel[dim].resize(0);
				}
				tree_map[n].resize(0);
			}
		}
	}
	hpx::wait_all(futs.begin(), futs.end());
}

int lc_add_particle(double x0, double y0, double z0, double x1, double y1, double z1, float vx, float vy, float vz, float t0, float t1,
		vector<lc_particle>& this_part_buffer) {
	int rc = 0;
	/*static simd_float8 images[NDIM] =
	 { simd_float8(0, -1, 0, -1, 0, -1, 0, -1), simd_float8(0, 0, -1, -1, 0, 0, -1, -1), simd_float8(0, 0, 0, 0, -1, -1, -1, -1) };
	 const float tau_max_inv = 1.0 / tau_max;
	 const simd_float8 simd_c0 = simd_float8(tau_max_inv);
	 array<simd_float8, NDIM> X0;
	 array<simd_float8, NDIM> X1;
	 const simd_float8 simd_tau0 = simd_float8(t0);
	 const simd_float8 simd_tau1 = simd_float8(t1);
	 simd_float8 dist0;
	 simd_float8 dist1;
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
	 static const int map_nside = get_options().lc_map_size;
	 const int i0 = I0[ci];
	 const int i1 = I1[ci];
	 if (i0 != i1) {
	 x0 = X0[XDIM][ci];
	 y0 = X0[YDIM][ci];
	 z0 = X0[ZDIM][ci];
	 const double ti = (i0 + 1) * tau_max;
	 const double sqrtauimtau0 = sqr(ti - t0);
	 const double tau0mtaui = t0 - ti;
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
	 part.pos[YDIM] = y1;
	 part.pos[ZDIM] = z1;
	 part.vel[XDIM] = vx;
	 part.vel[YDIM] = vy;
	 part.vel[ZDIM] = vz;
	 this_part_buffer.push_back(part);
	 }
	 }
	 }
	 }*/
	return rc;
}

void lc_add_parts(vector<lc_particle> && these_parts) {
	std::unique_lock<shared_mutex_type> ulock(shared_mutex);
	const int start = part_buffer.size();
	part_buffer.resize(start + these_parts.size());
	const int stop = part_buffer.size();
	ulock.unlock();
	const int chunk_size = 64;
	for (int i = 0; i < these_parts.size(); i += chunk_size) {
		std::shared_lock<shared_mutex_type> slock(shared_mutex);
		const int jend = std::min(i + chunk_size, (int) these_parts.size());
		for (int j = i; j < jend; j++) {
			part_buffer[j + start] = these_parts[j];
		}
	}

}
