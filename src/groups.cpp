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
#include <cosmictiger/containers.hpp>
#include <cosmictiger/constants.hpp>
#include <cosmictiger/groups.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/particles.hpp>

#include <unordered_set>

void groups_transmit_particles(vector<std::pair<group_int, vector<particle_data>>>entries);
vector<group_int> groups_exist(vector<group_int>);
static void groups_remove_indexes(vector<part_int> indexes);

HPX_PLAIN_ACTION (groups_add_particles);
HPX_PLAIN_ACTION (groups_transmit_particles);
HPX_PLAIN_ACTION (groups_reduce);
HPX_PLAIN_ACTION (groups_save);
HPX_PLAIN_ACTION (groups_exist);
HPX_PLAIN_ACTION (groups_cull);
HPX_PLAIN_ACTION (groups_remove_indexes);

#define GROUP_TABLE_SIZE 1024

struct group_hash_hi {
	size_t operator()(group_int grp) const {
		return grp / (GROUP_TABLE_SIZE * GROUP_WAVES);
	}
};

static array<spinlock_type, GROUP_TABLE_SIZE> mutexes;
static array<std::unordered_map<group_int, vector<particle_data>, group_hash_hi>, GROUP_TABLE_SIZE> group_data;
static vector<group_entry> groups;
static std::unordered_set<group_int> existing_groups;
static vector<group_int> local_existing_groups;
static std::atomic<size_t> group_candidates;

void groups_remove_indexes(vector<part_int> indexes) {
	for (int i = 0; i < indexes.size(); i++) {
		particles_group(indexes[i]) = NO_GROUP;
	}
}

void groups_cull() {
	vector<hpx::future<void>> futs1;
	vector<hpx::future<void>> futs2;
	for (const auto& c : hpx_children()) {
		futs1.push_back(hpx::async<groups_cull_action>(HPX_PRIORITY_HI, c));
	}
	std::unordered_set<group_int> final_existing;
	mutex_type mutex;
	const int nthreads = hpx::thread::hardware_concurrency();
	for (int proc = 0; proc < nthreads; proc++) {
		futs2.push_back(hpx::async([proc,nthreads,&mutex,&final_existing]() {
			std::unordered_map<int,vector<group_int>> map;
			for( int i = proc; i < local_existing_groups.size(); i+= nthreads) {
				const auto& grp = local_existing_groups[i];
				map[particles_group_home(grp)].push_back(grp);
			}
			vector<hpx::future<vector<group_int>>> futs;
			for( auto i = map.begin(); i != map.end(); i++) {
				futs.push_back(hpx::async<groups_exist_action>(hpx_localities()[i->first], std::move(i->second)));
			}
			for( auto& f : futs) {
				auto v = f.get();
				std::lock_guard<mutex_type> lock(mutex);
				for( const auto& grp : v) {
					final_existing.insert(grp);
				}
			}
		}));
	}
	hpx::wait_all(futs2.begin(), futs2.end());
	local_existing_groups = decltype(local_existing_groups)();
	for (int proc = 0; proc < nthreads; proc++) {
		futs1.push_back(hpx::async([proc,nthreads,&final_existing]() {
			const part_int begin = (size_t)proc * particles_size() / nthreads;
			const part_int end = (size_t)proc * particles_size() / nthreads;
			for( part_int i = begin; i < end; i++) {
				if( particles_group(i) != NO_GROUP) {
					if( final_existing.find(particles_group(i)) == final_existing.end()) {
						particles_group(i) = NO_GROUP;
					}
				}
			}
		}));
	}
	hpx::wait_all(futs1.begin(), futs1.end());
}

vector<group_int> groups_exist(vector<group_int> grps) {
	int i = 0;
	while (i < grps.size()) {
		if (existing_groups.find(grps[i]) != existing_groups.end()) {
			i++;
		} else {
			grps[i] = grps.back();
			grps.pop_back();
		}
	}
	return std::move(grps);
}

std::pair<size_t, size_t> groups_save(int number, double scale, double time) {
	if (hpx_rank() == 0) {
		PRINT("Writing group database\n");
		const std::string command = std::string("mkdir -p groups.") + std::to_string(number) + "\n";
		if (system(command.c_str()) != 0) {
			THROW_ERROR("Unable to execute : %s\n", command.c_str());
		}
		const std::string fname = std::string("groups.") + std::to_string(number) + std::string("/groups.dat");
		FILE* fp = fopen(fname.c_str(), "wb");
		if (fp == NULL) {
			THROW_ERROR("Unable to open %s for writing\n", fname.c_str());
		}
		group_header header;
		header.box_len = get_options().code_to_cm / constants::mpc_to_cm;
		header.time = time;
		header.number = number;
		header.scale = scale;
		fwrite(&header, sizeof(group_header), 1, fp);
		fclose(fp);
	}

	vector<hpx::future<std::pair<size_t, size_t>>>futs;
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async<groups_save_action>(HPX_PRIORITY_HI, c, number, scale, time));
	}

	const std::string fname = std::string("groups.") + std::to_string(number) + std::string("/groups.") + std::to_string(number) + "."
			+ std::to_string(hpx_rank()) + std::string(".dat");
	FILE* fp = fopen(fname.c_str(), "wb");
	if (fp == NULL) {
		THROW_ERROR("Unable to open %s\n", fname.c_str());
	}
	if (groups.size()) {
		hpx::parallel::sort(PAR_EXECUTION_POLICY, groups.begin(), groups.end(), [](const group_entry& a, const group_entry& b) {
			return a.id < b.id;
		}).get();
	}
	for (int i = 0; i < groups.size(); i++) {
		groups[i].write(fp);
	}
	fclose(fp);
	size_t count = groups.size();
	groups = decltype(groups)();
	existing_groups = decltype(existing_groups)();
	for (auto& f : futs) {
		const auto tmp = f.get();
		count += tmp.first;
		group_candidates += tmp.second;
	}
	return std::make_pair(count, (size_t) group_candidates);;
}

#define GROUPS_REDUCE_OVERSUBSCRIBE  2

void groups_reduce(double scale_factor) {
	vector<hpx::future<void>> futs;
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async<groups_reduce_action>(HPX_PRIORITY_HI, c, scale_factor));
	}
	const int nthreads = GROUPS_REDUCE_OVERSUBSCRIBE * hpx::thread::hardware_concurrency();
	spinlock_type mutex;
	const float ainv = 1.0 / scale_factor;
	for (int proc = 0; proc < nthreads; proc++) {
		futs.push_back(hpx::async([proc,nthreads,&mutex,&ainv]() {
			std::unordered_map<int,vector<part_int>> remove_indexes;
			for (int bin = proc; bin < GROUP_TABLE_SIZE; bin += nthreads) {
				for (auto iter = group_data[bin].begin(); iter != group_data[bin].end(); iter++) {
					group_candidates++;
					vector<particle_data>& parts = iter->second;
					vector<float> phi;
					if( parts.size() >= get_options().min_group) {
						bool removed_one;
						vector<array<fixed32,NDIM>> X(parts.size());
						for( int i = 0; i < parts.size(); i++) {
							for( int dim = 0; dim < NDIM; dim++) {
								const double x = parts[i].x[dim].to_double();
								X[i][dim] = x;
							}
						}
						phi = bh_evaluate_potential_fixed(X);
						for( auto& p : phi ) {
							p *= ainv;
						}
						if( parts.size() < get_options().min_group) {
							for( int i = 0; i < parts.size(); i++) {
								remove_indexes[parts[i].rank].push_back(parts[i].index);
							}
						} else {
							group_entry entry;
							entry.id = iter->first;
							int count = 0;
							float ekin = 0.0;
							array<double, NDIM> xcom;
							array<float, NDIM> vel;
							array<float, NDIM> lang;
							for( int dim = 0; dim < NDIM; dim++) {
								xcom[dim] = 0.0;
								vel[dim] = 0.0;
								lang[dim] = 0.0;
							}
							const auto constrain_range_half = [](double& x) {
								if( x < -0.5 ) {
									x += 1.0;
								} else if( x > 0.5) {
									x -= 1.0;
								}
							};
							for( int i = 0; i < parts.size(); i++) {
								for( int dim = 0; dim < NDIM; dim++) {
									const double x = parts[i].x[dim].to_double();
									const float v = parts[i].v[dim];
									vel[dim] += v;
									double this_dx = x - xcom[dim];
									if( i > 0 ) {
										constrain_range_half(this_dx);
									}
									const double xsum = (count + 1) * xcom[dim] + this_dx;
									xcom[dim] = xsum / (count + 1);
								}
								count++;
							}
							float countinv = 1.0 / count;
							for( int dim = 0; dim < NDIM; dim++) {
								vel[dim] *= countinv;
								constrain_range(xcom[dim]);
							}
							for( int i = 0; i < parts.size(); i++) {
								double x = parts[i].x[XDIM].to_double() - xcom[XDIM];
								double y = parts[i].x[YDIM].to_double() - xcom[YDIM];
								double z = parts[i].x[ZDIM].to_double() - xcom[ZDIM];
								constrain_range_half(x);
								constrain_range_half(y);
								constrain_range_half(z);
								const double vx = parts[i].v[XDIM] - vel[XDIM];
								const double vy = parts[i].v[YDIM] - vel[YDIM];
								const double vz = parts[i].v[ZDIM] - vel[ZDIM];
								ekin += sqr(vx, vy, vz) * 0.5;
								lang[XDIM] = (y * vz - z * vy)*countinv;
								lang[YDIM] = (-x * vz + z * vx)*countinv;
								lang[ZDIM] = (x * vy - y * vx)*countinv;
							}
							float pot = 0.0;
							for( const auto& p : phi) {
								pot += p;
							}
							pot *= countinv;
							pot *= 0.5;
							ekin *= countinv;
							//		PRINT( "%i %e %e %e\n", parts.size(), ekin, pot, (2*ekin + pot) / (ekin - pot));
				vector<float> radii;
				float ravg = 0.0;
				radii.reserve(parts.size());
				float vxdisp = 0.0;
				float vydisp = 0.0;
				float vzdisp = 0.0;
				float Ixx = 0.0, Ixy = 0.0, Ixz = 0.0, Iyy = 0.0, Iyz = 0.0, Izz = 0.0;
				for( int i = 0; i < parts.size(); i++) {
					double dx = parts[i].x[XDIM].to_double() - xcom[XDIM];
					double dy = parts[i].x[YDIM].to_double() - xcom[YDIM];
					double dz = parts[i].x[ZDIM].to_double() - xcom[ZDIM];
					constrain_range_half(dx);
					constrain_range_half(dy);
					constrain_range_half(dz);
					const double vx = parts[i].v[XDIM] - vel[XDIM];
					const double vy = parts[i].v[YDIM] - vel[YDIM];
					const double vz = parts[i].v[ZDIM] - vel[ZDIM];
					Ixx += dy * dy + dz * dz;
					Iyy += dx * dx + dz * dz;
					Izz += dy * dy + dx * dx;
					Ixy += dx * dy;
					Iyz += dy * dz;
					Ixz += dx * dz;
					vxdisp += vx * vx;
					vydisp += vy * vy;
					vzdisp += vz * vz;
					const double r = sqrt(sqr(dx,dy,dz));
					radii.push_back(r);
					ravg += r;
				}
				vxdisp = sqrt(vxdisp*countinv);
				vydisp = sqrt(vydisp*countinv);
				vzdisp = sqrt(vzdisp*countinv);
				std::sort(radii.begin(),radii.end());
				const double dr = 1.0 / (radii.size() - 1);
				const auto radial_percentile = [&radii,dr](double r) {
					double r0 = r / dr;
					int n0 = r0;
					int n1 = n0 + 1;
					double w1 = r0 - n0;
					double w0 = 1.0 - w1;
					return w0*radii[n0] + w1*radii[n1];
				};
				ravg / count;
				for( int i = 0; i <parts.size(); i++) {
					const auto lgrp = parts[i].last_group;
					if( lgrp != NO_GROUP) {
						if( entry.parents.find(lgrp) == entry.parents.end()) {
							entry.parents[lgrp] = 1;
						} else {
							entry.parents[lgrp]++;
						}
					}
				}
				entry.lang = lang;
				entry.Ixx = Ixx;
				entry.Ixy = Ixy;
				entry.Ixz = Ixz;
				entry.Iyy = Iyy;
				entry.Iyz = Iyz;
				entry.Izz = Izz;
				entry.vxdisp = vxdisp;
				entry.vydisp = vydisp;
				entry.vzdisp = vzdisp;
				entry.incomplete = false;
				entry.ekin = ekin;
				entry.epot = pot;
				entry.r25 = radial_percentile(0.25);
				entry.r50 = radial_percentile(0.50);
				entry.r75 = radial_percentile(0.75);
				entry.r90 = radial_percentile(0.90);
				entry.rmax = radii.back();
				entry.ravg = ravg;
				entry.com = xcom;
				entry.mass = count * get_options().code_to_g / constants::M0;
				entry.vel = vel;
				std::lock_guard<spinlock_type> lock(mutex);
				groups.push_back(std::move(entry));
				existing_groups.insert(entry.id);
			}
		}
	}
	vector<hpx::future<void>> futs;
	for( auto iter = remove_indexes.begin(); iter != remove_indexes.end(); iter++) {
		futs.push_back(hpx::async<groups_remove_indexes_action>(hpx_localities()[iter->first], std::move(iter->second)));
	}
	hpx::wait_all(futs.begin(),futs.end());
}
}));
	}
	hpx::wait_all(futs.begin(), futs.end());
	group_data = decltype(group_data)();
}

void groups_transmit_particles(vector<std::pair<group_int, vector<particle_data>>>entries) {
	vector<hpx::future<void>> futs;
	const int nthreads = hpx::thread::hardware_concurrency();
	for (int proc = 0; proc < nthreads; proc++) {
		futs.push_back(hpx::async([nthreads,proc,&entries]() {
					for( int i = proc; i < entries.size(); i+= nthreads) {
						const auto grp = entries[i].first;
						const int bin = (grp / GROUP_WAVES) % GROUP_TABLE_SIZE;
						std::lock_guard<spinlock_type> lock(mutexes[bin]);
						auto& entry = group_data[bin][grp];
						entry.insert(entry.end(), entries[i].second.begin(), entries[i].second.end());
					}
				}));
	}
	hpx::wait_all(futs.begin(), futs.end());
}

void groups_add_particles(int wave, double scale) {
	vector<hpx::future<void>> futs;
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async<groups_add_particles_action>(HPX_PRIORITY_HI, c, wave, scale));
	}
	const int nthreads = hpx::thread::hardware_concurrency();
	spinlock_type mutex;
	const float ainv = 1.0 / scale;
	for (int proc = 0; proc < nthreads; proc++) {
		futs.push_back(hpx::async([proc,nthreads,wave,&mutex,ainv]() {
			vector<hpx::future<void>> futs;
			std::unordered_map<int,std::unordered_map<group_int, vector<particle_data>>> local_groups;
			const part_int begin = (size_t) proc * particles_size() / nthreads;
			const part_int end = (size_t) (proc+1) * particles_size() / nthreads;
			for( part_int i = begin; i < end; i++) {
				const auto grp = (group_int) particles_group(i);
				if( grp != NO_GROUP) {
					const int home = particles_group_home(grp);
					if( grp % GROUP_WAVES == wave) {
						if( local_groups[home].find(grp) == local_groups[home].end()) {
							std::lock_guard<spinlock_type> lock(mutex);
							local_existing_groups.push_back(grp);
						}
						auto& entry = local_groups[home][grp];
						particle_data part;
						for( int dim = 0; dim < NDIM; dim++) {
							part.x[dim] = particles_pos(dim,i);
							part.v[dim] = particles_vel(dim,i) * ainv;
							part.index = i;
							part.last_group = particles_lastgroup(i);
							part.rank = hpx_rank();
						}
						entry.push_back(part);
					}
				}
			}
			for( auto iter = local_groups.begin(); iter != local_groups.end(); iter++) {
				vector<std::pair<group_int,vector<particle_data>>> data;
				data.reserve(iter->second.size());
				for( auto j = iter->second.begin(); j != iter->second.end(); j++) {
					data.push_back(std::move(*j));
				}
				futs.push_back(hpx::async<groups_transmit_particles_action>(hpx_localities()[iter->first],std::move(data)));
			}
			hpx::wait_all(futs.begin(), futs.end());
		}));
	}
	if (wave == 0) {
		group_candidates = 0;
	}
	hpx::wait_all(futs.begin(), futs.end());
}

