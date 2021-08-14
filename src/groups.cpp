#include <cosmictiger/bh.hpp>
#include <cosmictiger/containers.hpp>
#include <cosmictiger/groups.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/particles.hpp>

#include <unordered_set>

struct group_entry {
	group_int id;
	int count;
	float ekin;
	float epot;
	array<double, NDIM> com;
	array<float, NDIM> vel;
	float r25;
	float r50;
	float r75;
	float r90;
	float rmax;
	float ravg;
	float vxdisp;
	float vydisp;
	float vzdisp;
	float xdisp;
	float ydisp;
	float zdisp;
};

struct particle_data {
	array<fixed32, NDIM> x;
	array<float, NDIM> v;
	part_int index;
	int rank;
	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & x;
		arc & v;
		arc & index;
		arc & rank;
	}
};

void groups_transmit_particles(vector<std::pair<group_int, vector<particle_data>>>entries);
void groups_reduce();
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

void groups_remove_indexes(vector<part_int> indexes) {
	for (int i = 0; i < indexes.size(); i++) {
		particles_group(indexes[i]) = NO_GROUP;
	}
}

void groups_cull() {
	vector<hpx::future<void>> futs1;
	vector<hpx::future<void>> futs2;
	for (const auto& c : hpx_children()) {
		futs1.push_back(hpx::async < groups_cull_action > (c));
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

void groups_save(int number) {
	if (hpx_rank() == 0) {
		PRINT("Writing group database\n");
		const std::string command = std::string("mkdir -p groups.") + std::to_string(number) + "\n";
		if (system(command.c_str()) != 0) {
			THROW_ERROR("Unable to execute : %s\n", command.c_str());
		}
	}

	vector<hpx::future<void>> futs;
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async < groups_save_action > (c, number));
	}

	const std::string fname = std::string("groups.") + std::to_string(number) + std::string("/groups.") + std::to_string(number) + "."
			+ std::to_string(hpx_rank()) + std::string(".dat");
	FILE* fp = fopen(fname.c_str(), "wb");
	if (fp == NULL) {
		THROW_ERROR("Unable to open %s\n", fname.c_str());
	}
	fwrite(groups.data(), sizeof(group_entry), groups.size(), fp);
	fclose(fp);
	groups = decltype(groups)();
	existing_groups = decltype(existing_groups)();
	hpx::wait_all(futs.begin(), futs.end());
}

#define GROUPS_REDUCE_OVERSUBSCRIBE  2

void groups_reduce() {
	vector<hpx::future<void>> futs;
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async < groups_reduce_action > (c));
	}
	const int nthreads = GROUPS_REDUCE_OVERSUBSCRIBE * hpx::threads::hardware_concurrency();
	spinlock_type mutex;
	for (int proc = 0; proc < nthreads; proc++) {
		futs.push_back(hpx::async([proc,nthreads,&mutex]() {
			std::unordered_map<int,vector<part_int>> remove_indexes;
			for (int bin = proc; bin < GROUP_TABLE_SIZE; bin += nthreads) {
				for (auto iter = group_data[bin].begin(); iter != group_data[bin].end(); iter++) {
					vector<particle_data>& parts = iter->second;
					if( parts.size() >= get_options().min_group) {
						vector<array<fixed32,NDIM>> X(parts.size());
						for( int i = 0; i < parts.size(); i++) {
							for( int dim = 0; dim < NDIM; dim++) {
								const double x = parts[i].x[dim].to_double();
								X[i][dim] = x;
							}
						}
						auto phi = bh_evaluate_potential(X);
						int i = 0;
						while( i < parts.size() ) {
							float kin = 0.0;
							for( int dim = 0; dim < NDIM; dim++) {
								kin += sqr(parts[i].v[dim]);
							}
							kin *= 0.5;
							if( kin + phi[i] > 0.0 ) {
								remove_indexes[parts[i].rank].push_back(parts[i].index);
								parts[i] = parts.back();
								phi[i] = phi.back();
								parts.pop_back();
								phi.pop_back();
							} else {
								i++;
							}
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
							for( int dim = 0; dim < NDIM; dim++) {
								xcom[dim] = 0.0;
								vel[dim] = 0.0;
							}
							for( int i = 0; i < parts.size(); i++) {
								for( int dim = 0; dim < NDIM; dim++) {
									const double x = parts[i].x[dim].to_double();
									const float v = parts[i].v[dim];
									vel[dim] += v;
									ekin += v * v * 0.5;
									double this_dx = x - xcom[dim];
									if( this_dx > 0.5 ) {
										this_dx -= 1.0;
									} else if( this_dx < -0.5 ) {
										this_dx += 1.0;
									}
									const double xsum = (count + 1) * xcom[dim] + this_dx;
									xcom[dim] = xsum / (count + 1);
								}
								count++;
							}
							float countinv = 1.0 / count;
							float pot = 0.0;
							for( const auto& p : phi) {
								pot += p;
							}
							pot *= countinv;
							pot *= 0.5;
							ekin *= countinv;
							for( int dim = 0; dim < NDIM; dim++) {
								vel[dim] /= count;
								constrain_range(xcom[dim]);
							}
							vector<float> radii;
							float ravg = 0.0;
							radii.reserve(parts.size());
							float vxdisp = 0.0;
							float vydisp = 0.0;
							float vzdisp = 0.0;
							float xdisp = 0.0;
							float ydisp = 0.0;
							float zdisp = 0.0;
							for( int i = 0; i < parts.size(); i++) {
								const double dx = parts[i].x[XDIM].to_double() - xcom[XDIM];
								const double dy = parts[i].x[YDIM].to_double() - xcom[YDIM];
								const double dz = parts[i].x[ZDIM].to_double() - xcom[ZDIM];
								const double vx = parts[i].v[XDIM] - vel[XDIM];
								const double vy = parts[i].v[YDIM] - vel[YDIM];
								const double vz = parts[i].v[ZDIM] - vel[ZDIM];
								xdisp += dx * dx;
								ydisp += dy * dy;
								zdisp += dz * dz;
								vxdisp += vx * vx;
								vydisp += vy * vy;
								vzdisp += vz * vz;
								const double r = sqrt(sqr(dx,dy,dz));
								radii.push_back(r);
								ravg += r;
							}
							xdisp = sqrt(xdisp*countinv);
							ydisp = sqrt(xdisp*countinv);
							zdisp = sqrt(xdisp*countinv);
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
							entry.vxdisp = vxdisp;
							entry.vydisp = vydisp;
							entry.vzdisp = vzdisp;
							entry.xdisp = xdisp;
							entry.ydisp = ydisp;
							entry.zdisp = zdisp;
							entry.ekin = ekin;
							entry.epot = pot;
							entry.r25 = radial_percentile(0.25);
							entry.r50 = radial_percentile(0.50);
							entry.r75 = radial_percentile(0.75);
							entry.r90 = radial_percentile(0.90);
							entry.rmax = radii.back();
							entry.ravg = ravg;
							entry.com = xcom;
							entry.count = count;
							entry.vel = vel;
							std::lock_guard<spinlock_type> lock(mutex);
							groups.push_back(entry);
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
	const int nthreads = hpx::threads::hardware_concurrency();
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

void groups_add_particles(int wave) {
	vector<hpx::future<void>> futs;
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async < groups_add_particles_action > (c, wave));
	}
	const int nthreads = hpx::threads::hardware_concurrency();
	spinlock_type mutex;
	for (int proc = 0; proc < nthreads; proc++) {
		futs.push_back(hpx::async([proc,nthreads,wave,&mutex]() {
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
							part.v[dim] = particles_vel(dim,i);
							part.index = i;
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
	hpx::wait_all(futs.begin(), futs.end());
}

