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

#include <unordered_set>

#include <cosmictiger/bh.hpp>
#include <cosmictiger/constants.hpp>
#include <cosmictiger/containers.hpp>
#include <cosmictiger/groups.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/particles.hpp>
#include <cosmictiger/rockstar.hpp>

void groupsTransmitParticles(Vector<std::pair<GroupInteger, Vector<ParticleData>>> entries);
Vector<GroupInteger> groupsExist(Vector<GroupInteger>);
static void groupsRemoveIndexes(Vector<ParticleInteger> indexes);

HPX_PLAIN_ACTION(groupsAddParticles, GroupsAddParticlesAction);
HPX_PLAIN_ACTION(groupsTransmitParticles, GroupsTransmitParticlesAction);
HPX_PLAIN_ACTION(groupsReduce, GroupsReduceAction);
HPX_PLAIN_ACTION(groupsSave, GroupsSaveAction);
HPX_PLAIN_ACTION(groupsExist, GroupsExistAction);
HPX_PLAIN_ACTION(groupsCull, GroupsCullAction);
HPX_PLAIN_ACTION(groupsRemoveIndexes, GroupsRemoveIndexesAction);

#define GROUP_TABLE_SIZE 1024

struct GroupHashHi {
    size_t operator()(GroupInteger grp) const {
        return grp / (GROUP_TABLE_SIZE * GROUP_WAVES);
    }
};

static Array<SpinlockType, GROUP_TABLE_SIZE> mutexes;
static Array<std::unordered_map<GroupInteger, Vector<ParticleData>, GroupHashHi>, GROUP_TABLE_SIZE>
    groupData;
static Vector<GroupEntry> groups;
static std::unordered_set<GroupInteger> existingGroups;
static Vector<GroupInteger> localExistingGroups;
static std::atomic<size_t> groupCandidates;

void groupsRemoveIndexes(Vector<ParticleInteger> indexes) {
    for (int i = 0; i < indexes.size(); i++) {
        particlesGroup(indexes[i]) = NO_GROUP;
    }
}

void groupsCull() {
    Vector<hpx::future<void>> futs1;
    Vector<hpx::future<void>> futs2;
    for (auto const& c : hpxChildren()) {
        futs1.push_back(hpx::async<GroupsCullAction>(c));
    }
    std::unordered_set<GroupInteger> finalExisting;
    MutexType mutex;
    int const numberThreads = hpxHardwareConcurrency();
    for (int proc = 0; proc < numberThreads; proc++) {
        futs2.push_back(hpx::async([proc, numberThreads, &mutex, &finalExisting]() {
            std::unordered_map<int, Vector<GroupInteger>> map;
            for (int i = proc; i < localExistingGroups.size(); i += numberThreads) {
                auto const& grp = localExistingGroups[i];
                map[particlesGroupHome(grp)].push_back(grp);
            }
            Vector<hpx::future<Vector<GroupInteger>>> futs;
            for (auto i = map.begin(); i != map.end(); i++) {
                futs.push_back(
                    hpx::async<GroupsExistAction>(hpxLocalities()[i->first], std::move(i->second)));
            }
            for (auto& f : futs) {
                auto v = f.get();
                std::lock_guard<MutexType> lock(mutex);
                for (auto const& grp : v) {
                    finalExisting.insert(grp);
                }
            }
        }));
    }
    hpx::wait_all(futs2.begin(), futs2.end());
    localExistingGroups = decltype(localExistingGroups)();
    for (int proc = 0; proc < numberThreads; proc++) {
        futs1.push_back(hpx::async([proc, numberThreads, &finalExisting]() {
            ParticleInteger const begin = (size_t) proc * particlesSize() / numberThreads;
            ParticleInteger const end = (size_t) proc * particlesSize() / numberThreads;
            for (ParticleInteger i = begin; i < end; i++) {
                if (particlesGroup(i) != NO_GROUP) {
                    if (finalExisting.find(particlesGroup(i)) == finalExisting.end()) {
                        particlesGroup(i) = NO_GROUP;
                    }
                }
            }
        }));
    }
    hpx::wait_all(futs1.begin(), futs1.end());
}

Vector<GroupInteger> groupsExist(Vector<GroupInteger> grps) {
    int i = 0;
    while (i < grps.size()) {
        if (existingGroups.find(grps[i]) != existingGroups.end()) {
            i++;
        } else {
            grps[i] = grps.back();
            grps.pop_back();
        }
    }
    return std::move(grps);
}

std::pair<size_t, size_t> groupsSave(int number, double scale, double time) {
    if (hpxRank() == 0) {
        PRINT("Writing group database\n");
        std::string const command = std::string("mkdir -p groups.") + std::to_string(number) + "\n";
        if (system(command.c_str()) != 0) {
            THROW_ERROR("Unable to execute : %s\n", command.c_str());
        }
        std::string const fname =
            std::string("groups.") + std::to_string(number) + std::string("/groups.dat");
        FILE* fp = fopen(fname.c_str(), "wb");
        if (fp == NULL) {
            THROW_ERROR("Unable to open %s for writing\n", fname.c_str());
        }
        GroupHeader header;
        header.boxLen = getOptions().codeToCm / constants::kMpcToCm;
        header.time = time;
        header.number = number;
        header.scale = scale;
        fwrite(&header, sizeof(GroupHeader), 1, fp);
        fclose(fp);
    }

    Vector<hpx::future<std::pair<size_t, size_t>>> futs;
    for (auto const& c : hpxChildren()) {
        futs.push_back(hpx::async<GroupsSaveAction>(c, number, scale, time));
    }

    std::string const fname = std::string("groups.") + std::to_string(number) +
        std::string("/groups.") + std::to_string(number) + "." + std::to_string(hpxRank()) +
        std::string(".dat");
    FILE* fp = fopen(fname.c_str(), "wb");
    if (fp == NULL) {
        THROW_ERROR("Unable to open %s\n", fname.c_str());
    }
    if (groups.size()) {
        hpx::sort(PAR_EXECUTION_POLICY, groups.begin(), groups.end(),
            [](GroupEntry const& a, GroupEntry const& b) { return a.id < b.id; })
            .get();
    }
    for (int i = 0; i < groups.size(); i++) {
        groups[i].write(fp);
    }
    fclose(fp);
    size_t count = groups.size();
    groups = decltype(groups)();
    existingGroups = decltype(existingGroups)();
    for (auto& f : futs) {
        auto const tmp = f.get();
        count += tmp.first;
        groupCandidates += tmp.second;
    }
    return std::make_pair(count, (size_t) groupCandidates);
    ;
}

#define GROUPS_REDUCE_OVERSUBSCRIBE 2

void groupsReduce(double scaleFactor) {
    Vector<hpx::future<void>> futs;
    for (auto const& c : hpxChildren()) {
        futs.push_back(hpx::async<GroupsReduceAction>(c, scaleFactor));
    }
    int const numberThreads = GROUPS_REDUCE_OVERSUBSCRIBE * hpxHardwareConcurrency();
    SpinlockType mutex;
    float const ainv = 1.0 / scaleFactor;
    for (int proc = 0; proc < numberThreads; proc++) {
        futs.push_back(hpx::async([proc, numberThreads, &mutex, &ainv, scaleFactor]() {
            std::unordered_map<int, Vector<ParticleInteger>> removeIndexes;
            for (int bin = proc; bin < GROUP_TABLE_SIZE; bin += numberThreads) {
                for (auto iter = groupData[bin].begin(); iter != groupData[bin].end(); iter++) {
                    groupCandidates++;
                    Vector<ParticleData>& parts = iter->second;

                    Vector<float> phi;
                    if (parts.size() >= getOptions().minGroup) {
                        auto rockstarGroups = rockstarFindSubgroups(parts, scaleFactor);
                        int haloCount = 0;
                        for (int i = 0; i < rockstarGroups.size(); i++) {
                            if (rockstarGroups[i].parent == ROCKSTAR_NO_GROUP &&
                                rockstarGroups[i].id != ROCKSTAR_NO_GROUP) {
                                haloCount++;
                            }
                        }
                        if (haloCount > 1) {
                            PRINT("%i halos and %i subhalos found in group of %i\n", haloCount,
                                rockstarGroups.size() - haloCount - 1, parts.size());
                        }
                        bool removedOne;
                        DeviceVector<Array<Fixed32, NDIM>> xValue(parts.size());
                        for (int i = 0; i < parts.size(); i++) {
                            for (int dim = 0; dim < NDIM; dim++) {
                                double const x = parts[i].x[dim].toDouble();
                                xValue[i][dim] = x;
                            }
                        }
                        /*	phi = bh_evaluate_potential_fixed(X);
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
                         //		PRINT( "%i %e %e %e\n", parts.size(), ekin, pot, (2*ekin + pot) /
                         (ekin - pot)); vector<float> radii; float ravg = 0.0;
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
                         }*/
                    }
                }
                Vector<hpx::future<void>> futs;
                for (auto iter = removeIndexes.begin(); iter != removeIndexes.end(); iter++) {
                    futs.push_back(hpx::async<GroupsRemoveIndexesAction>(
                        hpxLocalities()[iter->first], std::move(iter->second)));
                }
                hpx::wait_all(futs.begin(), futs.end());
            }
        }));
    }
    hpx::wait_all(futs.begin(), futs.end());
    groupData = decltype(groupData)();
}

void groupsTransmitParticles(Vector<std::pair<GroupInteger, Vector<ParticleData>>> entries) {
    Vector<hpx::future<void>> futs;
    int const numberThreads = hpxHardwareConcurrency();
    for (int proc = 0; proc < numberThreads; proc++) {
        futs.push_back(hpx::async([numberThreads, proc, &entries]() {
            for (int i = proc; i < entries.size(); i += numberThreads) {
                auto const grp = entries[i].first;
                int const bin = (grp / GROUP_WAVES) % GROUP_TABLE_SIZE;
                std::lock_guard<SpinlockType> lock(mutexes[bin]);
                auto& entry = groupData[bin][grp];
                entry.insert(entry.end(), entries[i].second.begin(), entries[i].second.end());
            }
        }));
    }
    hpx::wait_all(futs.begin(), futs.end());
}

void groupsAddParticles(int wave, double scale) {
    Vector<hpx::future<void>> futs;
    for (auto const& c : hpxChildren()) {
        futs.push_back(hpx::async<GroupsAddParticlesAction>(c, wave, scale));
    }
    int const numberThreads = hpxHardwareConcurrency();
    SpinlockType mutex;
    float const ainv = 1.0 / scale;
    for (int proc = 0; proc < numberThreads; proc++) {
        futs.push_back(hpx::async([proc, numberThreads, wave, &mutex, ainv]() {
            Vector<hpx::future<void>> futs;
            std::unordered_map<int, std::unordered_map<GroupInteger, Vector<ParticleData>>>
                localGroups;
            ParticleInteger const begin = (size_t) proc * particlesSize() / numberThreads;
            ParticleInteger const end = (size_t) (proc + 1) * particlesSize() / numberThreads;
            for (ParticleInteger i = begin; i < end; i++) {
                auto const grp = (GroupInteger) particlesGroup(i);
                if (grp != NO_GROUP) {
                    int const home = particlesGroupHome(grp);
                    if (grp % GROUP_WAVES == wave) {
                        if (localGroups[home].find(grp) == localGroups[home].end()) {
                            std::lock_guard<SpinlockType> lock(mutex);
                            localExistingGroups.push_back(grp);
                        }
                        auto& entry = localGroups[home][grp];
                        ParticleData part;
                        for (int dim = 0; dim < NDIM; dim++) {
                            part.x[dim] = particlesPos(dim, i);
                            part.v[dim] = particlesVel(dim, i) * ainv;
                            part.index = i;
                            part.lastGroup = particlesLastgroup(i);
                            part.rank = hpxRank();
                        }
                        entry.push_back(part);
                    }
                }
            }
            for (auto iter = localGroups.begin(); iter != localGroups.end(); iter++) {
                Vector<std::pair<GroupInteger, Vector<ParticleData>>> data;
                data.reserve(iter->second.size());
                for (auto j = iter->second.begin(); j != iter->second.end(); j++) {
                    data.push_back(std::move(*j));
                }
                futs.push_back(hpx::async<GroupsTransmitParticlesAction>(
                    hpxLocalities()[iter->first], std::move(data)));
            }
            hpx::wait_all(futs.begin(), futs.end());
        }));
    }
    if (wave == 0) {
        groupCandidates = 0;
    }
    hpx::wait_all(futs.begin(), futs.end());
}
