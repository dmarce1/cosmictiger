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

#include <atomic>
#include <memory>
#include <shared_mutex>
#include <unordered_set>
#include <unordered_map>

#include <healpix/healpix_base.h>
#include <chealpix.h>

using healpix_type = T_Healpix_Base< int >;

#define LC_NO_GROUP (0x7FFFFFFFFFFFFFFFLL)
#define LC_EDGE_GROUP (0x0LL)

struct lc_group_data {
	vector<lc_particle> parts;
	group_entry arc;
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

struct pixel {
	float pix;
	pixel() {
		pix = 0.0;
	}
	pixel(const pixel& other) {
		pix = (float) other.pix;
	}
	pixel(pixel&& other) {
		pix = (float) other.pix;
	}
	pixel& operator=(const pixel& other) {
		pix = (float) other.pix;
		return *this;
	}
	pixel& operator=(pixel&& other) {
		pix = (float) other.pix;
		return *this;
	}
	pixel& operator=(float other) {
		pix = other;
		return *this;
	}

};

static std::shared_ptr<healpix_type> healpix;
static pair<int> my_pix_range;
static std::unordered_set<int> bnd_pix;
static std::unordered_map<int, vector<lc_particle>> part_map;
static std::unordered_map<int, vector<lc_tree_node>> tree_map;
static std::unordered_map<int, std::shared_ptr<spinlock_type>> mutex_map;
static std::unordered_map<int, pixel> healpix_map;
static std::atomic<long long> group_id_counter(0);
static vector<group_entry> saved_groups;
static vector<lc_particle> part_buffer;
static shared_mutex_type mutex;
static double tau_max;
static int Nside;
static int Npix;

static vector<lc_particle> lc_get_particles1(int pix);
static pair<vector<long long>, vector<char>> lc_get_particles2(int pix);
static void lc_send_particles(vector<lc_particle>);
static void lc_send_buffer_particles(vector<lc_particle>);
static int vec2pix(double x, double y, double z);
static int pix2rank(int pix);
static int compute_nside(double tau);
static vector<int> pix_neighbors(int pix);

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
HPX_PLAIN_ACTION (lc_particle_boundaries1);
HPX_PLAIN_ACTION (lc_particle_boundaries2);
HPX_PLAIN_ACTION (lc_flush_final);

vector<float> lc_flush_final() {
	if (hpx_rank() == 0) {
		if (system("mkdir -p lc") != 0) {
			THROW_ERROR("Unable to make directory lc\n");
		}
	}
	vector<hpx::future<vector<float>>>futs;
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async<lc_flush_final_action>(HPX_PRIORITY_HI, c));
	}

	std::string filename = "./lc/lc." + std::to_string(hpx_rank()) + ".dat";
	FILE* fp = fopen(filename.c_str(), "wb");
	if (fp == NULL) {
		THROW_ERROR("Unable to open %s for writing\n", filename.c_str());
	}
	for (int i = 0; i < saved_groups.size(); i++) {
		saved_groups[i].write(fp);
	}
	fclose(fp);

	const int nside = get_options().lc_map_size;
	const int npix = 12 * nside * nside;
	vector<float> pix(npix, 0.0f);
	for (auto i = healpix_map.begin(); i != healpix_map.end(); i++) {
		pix[i->first] += i->second.pix;
	}
	for (auto& f : futs) {
		const auto v = f.get();
		for (int i = 0; i < npix; i++) {
			pix[i] += v[i];
		}
	}
	if (hpx_rank() == 0) {
		FILE* fp = fopen("lc_map.dat", "wb");
		if (fp == NULL) {
			THROW_ERROR("unable to open lc_map.dat for writing\n");
		}
		fwrite(&nside, sizeof(int), 1, fp);
		fwrite(&npix, sizeof(int), 1, fp);
		fwrite(pix.data(), sizeof(float), npix, fp);
		int number = 0;
		double time = cosmos_time(1.0e-6, 1.0) * get_options().code_to_s / constants::spyr;
		fwrite(&number, sizeof(int), 1, fp);
		fwrite(&time, sizeof(double), 1, fp);
		fclose(fp);

	}
	return std::move(pix);
}

void lc_save(FILE* fp) {
	int dummy;
	size_t sz = part_buffer.size();
	fwrite(&sz, sizeof(size_t), 1, fp);
	fwrite(part_buffer.data(), sizeof(lc_particle), part_buffer.size(), fp);
	fwrite(&dummy, sizeof(int), 1, fp);
	sz = healpix_map.size();
	fwrite(&sz, sizeof(size_t), 1, fp);
	for (auto iter = healpix_map.begin(); iter != healpix_map.end(); iter++) {
		int pix = iter->first;
		float value = iter->second.pix;
		fwrite(&pix, sizeof(int), 1, fp);
		fwrite(&value, sizeof(float), 1, fp);
	}
	sz = saved_groups.size();
	fwrite(&sz, sizeof(size_t), 1, fp);
	for (int i = 0; i < sz; i++) {
		saved_groups[i].write(fp);
	}
}

void lc_load(FILE* fp) {
	size_t sz;
	int dummy;
	FREAD(&sz, sizeof(size_t), 1, fp);
	part_buffer.resize(sz);
	FREAD(part_buffer.data(), sizeof(lc_particle), part_buffer.size(), fp);
	FREAD(&dummy, sizeof(int), 1, fp);
	FREAD(&sz, sizeof(size_t), 1, fp);
	for (int i = 0; i < sz; i++) {
		int pix;
		float value;
		FREAD(&pix, sizeof(int), 1, fp);
		FREAD(&value, sizeof(float), 1, fp);
		healpix_map[pix].pix = value;
	}
	FREAD(&sz, sizeof(size_t), 1, fp);
	saved_groups.resize(sz);
	for (int i = 0; i < sz; i++) {
		saved_groups[i].read(fp);
	}
}

size_t lc_time_to_flush(double tau, double tau_max_) {
	vector < hpx::future < size_t >> futs;
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async<lc_time_to_flush_action>(HPX_PRIORITY_HI, c, tau, tau_max_));
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
		if (8 * nparts >= parts_per_rank) {
			return 1;
		} else {
			return 0;
		}
	} else {
		return nparts;
	}
}

void lc_parts2groups(double a, double link_len) {
	vector<hpx::future<void>> futs;
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async<lc_parts2groups_action>(HPX_PRIORITY_HI, c, a, link_len));
	}
	std::unordered_map<long long, lc_group_data> groups_map;
	int i = 0;
	while (i < part_buffer.size()) {
		const auto grp = part_buffer[i].group;
		if (grp != LC_EDGE_GROUP) {
			groups_map[grp].parts.push_back(part_buffer[i]);
			part_buffer[i] = part_buffer.back();
			part_buffer.pop_back();
		} else {
			i++;
		}
	}
	PRINT("%li particles remaining in buffer\n", part_buffer.size());
	vector<lc_group_data> groups;
	for (auto i = groups_map.begin(); i != groups_map.end(); i++) {
		auto tmp = std::move(i->second);
		if (tmp.parts.size() >= get_options().lc_min_group) {
			tmp.arc.id = i->first;
			groups.push_back(std::move(tmp));
		}
	}
	groups_map = decltype(groups_map)();
	vector<hpx::future<void>> futs2;
	const int nthreads = hpx_hardware_concurrency();
	for (int proc = 0; proc < nthreads; proc++) {
		futs2.push_back(hpx::async([proc,nthreads,&groups,a,link_len]() {
			const double ainv = 1.0 / a;
			const int begin = (size_t) (proc) * groups.size() / nthreads;
			const int end = (size_t) (proc+1) * groups.size() / nthreads;
			for( int i = begin; i < end; i++) {
				auto& parts = groups[i].parts;
				bool incomplete = false;
				for( int i = 0; i < parts.size(); i++) {
					const double x = parts[i].pos[XDIM];
					const double y = parts[i].pos[YDIM];
					const double z = parts[i].pos[ZDIM];
					const double r = sqrt(sqr(x,y,z));
					if( r + link_len > 1.0 ) {
						incomplete = true;
						break;
					}
				}
				vector<array<fixed32, NDIM>> bh_x(parts.size());
				for( int j = 0; j < parts.size(); j++) {
					for( int dim = 0; dim < NDIM; dim++) {
						bh_x[j][dim] = 0.5 + parts[j].pos[dim] - parts[0].pos[dim];
					}
				}
				auto pot = bh_evaluate_potential(bh_x);
				for( auto& phi : pot) {
					phi *= ainv;
				}
				array<double, NDIM> xcom;
				array<float, NDIM> vcom;
				array<float, NDIM> J;
				for( int dim = 0; dim < NDIM; dim++) {
					xcom[dim] = 0.0;
					vcom[dim] = 0.0;
					J[dim] = 0.0;
				}
				for( int i = 0; i < parts.size(); i++) {
					for( int dim = 0; dim < NDIM; dim++) {
						xcom[dim] += parts[i].pos[dim];
						vcom[dim] += parts[i].vel[dim];
					}
				}
				for( int dim = 0; dim < NDIM; dim++) {
					xcom[dim] /= parts.size();
					vcom[dim] /= parts.size();
				}
				for( int i = 0; i < parts.size(); i++) {
					for( int dim = 0; dim < NDIM; dim++) {
						parts[i].pos[dim] -= xcom[dim];
						parts[i].vel[dim] -= vcom[dim];
					}
				}
				double ekin = 0.0;
				double epot = 0.0;
				double vxdisp = 0.0;
				double vydisp = 0.0;
				double vzdisp = 0.0;
				double rmax = 0.0;
				double ravg = 0.0;
				double Ixx = 0.0;
				double Iyy = 0.0;
				double Izz = 0.0;
				double Ixy = 0.0;
				double Ixz = 0.0;
				double Iyz = 0.0;
				vector<double> radii;
				for( int i = 0; i < parts.size(); i++) {
					const double vx = parts[i].vel[XDIM];
					const double vy = parts[i].vel[YDIM];
					const double vz = parts[i].vel[ZDIM];
					const double x = parts[i].pos[XDIM];
					const double y = parts[i].pos[YDIM];
					const double z = parts[i].pos[ZDIM];
					const double r = sqrt(sqr(x, y, z));
					J[XDIM] += y * vz - z * vy;
					J[YDIM] -= x * vz - z * vx;
					J[ZDIM] += x * vy - y * vx;
					Ixx += sqr(y) + sqr(z);
					Iyy += sqr(x) + sqr(z);
					Izz += sqr(x) + sqr(y);
					Ixy -= x * y;
					Ixz -= x * z;
					Iyz -= y * z;
					rmax = std::max(r, rmax);
					ravg += r;
					radii.push_back(r);
					vxdisp += sqr(vx);
					vydisp += sqr(vy);
					vzdisp += sqr(vz);
					ekin += sqr(vx, vy, vz);
					epot += pot[i];
				}
				std::sort(radii.begin(), radii.end());
				const double countinv = 1.0 / parts.size();
				ekin *= countinv;
				epot *= countinv;
				vxdisp *= countinv;
				vydisp *= countinv;
				vzdisp *= countinv;
				vxdisp = sqrt(vxdisp);
				vydisp = sqrt(vydisp);
				vzdisp = sqrt(vzdisp);
				ravg *= countinv;
				Ixx *= countinv;
				Iyy *= countinv;
				Izz *= countinv;
				Ixy *= countinv;
				Iyz *= countinv;
				Ixz *= countinv;
				for( int dim = 0; dim < NDIM; dim++) {
					J[dim] *= countinv;
				}
				const auto radial_percentile = [&radii](double r) {
					const double dr = 1.0 / (radii.size() - 1);
					double r0 = r / dr;
					int n0 = r0;
					int n1 = n0 + 1;
					double w1 = r0 - n0;
					double w0 = 1.0 - w1;
					return w0*radii[n0] + w1*radii[n1];
				};
				const double code_to_mpc = get_options().code_to_cm / constants::mpc_to_cm;
				const double code_to_mpc2 = sqr(code_to_mpc);
				groups[i].arc.r25 = radial_percentile(0.25) * code_to_mpc;
				groups[i].arc.r50 = radial_percentile(0.5) * code_to_mpc;
				groups[i].arc.r75 = radial_percentile(0.75) * code_to_mpc;
				groups[i].arc.r90 = radial_percentile(0.9) * code_to_mpc;
				groups[i].arc.rmax = rmax * code_to_mpc;
				groups[i].arc.ravg = ravg * code_to_mpc;
				groups[i].arc.Ixx = Ixx * code_to_mpc2;
				groups[i].arc.Iyy = Iyy * code_to_mpc2;
				groups[i].arc.Izz = Izz * code_to_mpc2;
				groups[i].arc.Ixy = Ixy * code_to_mpc2;
				groups[i].arc.Iyz = Iyz * code_to_mpc2;
				groups[i].arc.Ixz = Ixz * code_to_mpc2;
				for( int dim = 0; dim < NDIM; dim++) {
					groups[i].arc.lang[dim] = J[dim];
					groups[i].arc.vel[dim] = vcom[dim];
					groups[i].arc.com[dim] = xcom[dim] * code_to_mpc;
				}
				groups[i].arc.mass = parts.size() * get_options().code_to_g / constants::M0;
				groups[i].arc.vxdisp = vxdisp;
				groups[i].arc.vydisp = vydisp;
				groups[i].arc.vzdisp = vzdisp;
				groups[i].arc.incomplete = incomplete;
				groups[i].arc.epot = epot;
				groups[i].arc.ekin = ekin;
			}
		}));
	}
	hpx::wait_all(futs2.begin(), futs2.end());
	for (int i = 0; i < groups.size(); i++) {
		saved_groups.push_back(std::move(groups[i].arc));
	}
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
		futs1.push_back(hpx::async<lc_buffer2homes_action>(HPX_PRIORITY_HI, c));
	}
	const int nthreads = hpx_hardware_concurrency();
	static mutex_type map_mutex;
	for (int proc = 0; proc < nthreads; proc++) {
		futs2.push_back(hpx::async([proc,nthreads]() {
			const int begin = (size_t) proc * part_buffer.size() / nthreads;
			const int end = (size_t) (proc+1) * part_buffer.size() / nthreads;
			std::unordered_map<int,vector<lc_particle>> sends;
			std::unordered_map<int,float> my_healpix;
			for( int i = begin; i < end; i++) {
				const auto& part = part_buffer[i];
				const int pix = vec2pix(part.pos[XDIM],part.pos[YDIM],part.pos[ZDIM]);
				const int rank = pix2rank(pix);
				sends[rank].push_back(part);
				double vec[NDIM];
				vec[XDIM] = part.pos[XDIM];
				vec[YDIM] = part.pos[YDIM];
				vec[ZDIM] = part.pos[ZDIM];
				long int ipix;
				vec2pix_ring(get_options().lc_map_size, vec, &ipix);
				auto iter = my_healpix.find(ipix);
				if( iter == my_healpix.end()) {
					iter = my_healpix.insert(std::make_pair(ipix,0.0)).first;
				}
				iter->second += 1.0 / sqr(vec[XDIM], vec[YDIM], vec[ZDIM]);
			}
			vector<hpx::future<void>> futs;
			for( auto i = sends.begin(); i != sends.end(); i++) {
				futs.push_back(hpx::async<lc_send_particles_action>(hpx_localities()[i->first], std::move(i->second)));
			}
			std::unique_lock<mutex_type> lock(map_mutex);
			for( auto i = my_healpix.begin(); i != my_healpix.end(); i++) {
				healpix_map[i->first].pix += i->second;
			}
			lock.unlock();
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
		futs.push_back(hpx::async<lc_groups2homes_action>(HPX_PRIORITY_HI, c));
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
				auto& A = part_map[self.pix][pi];
				for (int pj = self.part_range.first; pj < self.part_range.second; pj++) {
					if (pi != pj) {
						auto& B = part_map[self.pix][pj];
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
								//					PRINT( "%li %li\n", (long long) A.group, (long long) B.group);
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
		checklist.insert(checklist.end(), leaflist.begin(), leaflist.end());
		int nactive = 0;
		if (checklist.size()) {
			nactive += lc_find_groups_local(self.children[LEFT], checklist, link_len);
			nactive += lc_find_groups_local(self.children[RIGHT], std::move(checklist), link_len);
		}
		self.active = nactive;
		return nactive;
	}

}

size_t lc_find_groups() {
	vector < hpx::future < size_t >> futs;
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async<lc_find_groups_action>(HPX_PRIORITY_HI, c));
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

static vector<lc_particle> lc_get_particles1(int pix) {
	return part_map[pix];
}

static pair<vector<long long>, vector<char>> lc_get_particles2(int pix) {
	pair<vector<long long>, vector<char>> rc;
	const auto& parts = part_map[pix];
	for (int i = 0; i < parts.size(); i++) {
		rc.first.push_back((long long) parts[i].group);
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
	vector<hpx::future<void>> futs;
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async<lc_particle_boundaries1_action>(HPX_PRIORITY_HI, c));
	}
	vector < hpx::future<vector<lc_particle>>>  pfuts;
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
	vector<hpx::future<void>> futs;
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async<lc_particle_boundaries2_action>(HPX_PRIORITY_HI, c));
	}
	vector < hpx::future < pair<vector<long long>, vector<char>>> > pfuts;
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
				const int jmin = thread * parts.size() / nthreads;
				const int jmax = (thread+1) * parts.size() / nthreads;
				for (int j = jmin; j < jmax; j++) {
					parts[j].group = p.first[j];
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
	vector<hpx::future<void>> futs;
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async<lc_form_trees_action>(HPX_PRIORITY_HI, c, tau, link_len));
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
				if( R < (tau_max - tau) + link_len * 1.0001 && tau < tau_max) {
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
	double nside = std::sqrt(2.0 * hpx::thread::hardware_concurrency() * hpx_size() / 12.0);
	const double w0 = std::max((tau_max - tau) / tau_max, 0.0);
	nside = std::min(nside, (get_options().parts_dim * std::sqrt(3) / get_options().lc_b / 2.0 * w0));
	Nside = 1;
	while (Nside <= nside) {
		Nside *= 2;
	}
	Nside /= 2;
	return Nside;
}

void lc_init(double tau, double tau_max_) {
	vector<hpx::future<void>> futs;
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async<lc_init_action>(HPX_PRIORITY_HI, c, tau, tau_max_));
	}
	tree_map = decltype(tree_map)();
	part_map = decltype(part_map)();
	mutex_map = decltype(mutex_map)();
	bnd_pix = decltype(bnd_pix)();
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

int lc_add_particle(lc_real x0, lc_real y0, lc_real z0, lc_real x1, lc_real y1, lc_real z1, float vx, float vy, float vz, float t0, float t1,
		vector<lc_particle>& this_part_buffer) {
	static simd_float8 images[NDIM] =
			{ simd_float8(0, -1, 0, -1, 0, -1, 0, -1), simd_float8(0, 0, -1, -1, 0, 0, -1, -1), simd_float8(0, 0, 0, 0, -1, -1, -1, -1) };
	const float tau_max_inv = 1.0 / tau_max;
	const simd_float8 simd_c0 = simd_float8(tau_max_inv);
	array<simd_float8, NDIM> X0;
	array<simd_float8, NDIM> X1;
	const simd_float8 simd_tau0 = simd_float8(t0);
	const simd_float8 simd_tau1 = simd_float8(t1);
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
	}
	return rc;
}

void lc_add_parts(vector<lc_particle> && these_parts) {
	std::unique_lock<shared_mutex_type> ulock(mutex);
	const int start = part_buffer.size();
	part_buffer.resize(start + these_parts.size());
	const int stop = part_buffer.size();
	ulock.unlock();
	const int chunk_size = 64;
	for (int i = 0; i < these_parts.size(); i += chunk_size) {
		std::shared_lock<shared_mutex_type> slock(mutex);
		const int jend = std::min(i + chunk_size, (int) these_parts.size());
		for (int j = i; j < jend; j++) {
			part_buffer[j + start] = these_parts[j];
		}
	}

}
