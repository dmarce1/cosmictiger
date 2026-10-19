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

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <cosmictiger/analysis/groups.hpp>
#include <cosmictiger/analysis/rockstar.hpp>
#include <cosmictiger/core/containers.hpp>
#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/gravity/bh.hpp>
#include <cosmictiger/math/constants.hpp>
#include <cosmictiger/math/math.hpp>
#include <cosmictiger/particles/particles.hpp>

void groupsTransmitParticles(Vector<std::pair<GroupInteger, Vector<ParticleData>>> entries);
Vector<GroupInteger> groupsExist(Vector<GroupInteger>);
void groupsQueueAssignments(Vector<Pair<ParticleInteger, GroupInteger>> assignments);
void groupsInstallHistory(Vector<Pair<GroupInteger>> entries);

HPX_PLAIN_ACTION(groupsAddParticles, GroupsAddParticlesAction);
HPX_PLAIN_ACTION(groupsTransmitParticles, GroupsTransmitParticlesAction);
HPX_PLAIN_ACTION(groupsReduce, GroupsReduceAction);
HPX_PLAIN_ACTION(groupsSave, GroupsSaveAction);
HPX_PLAIN_ACTION(groupsExist, GroupsExistAction);
HPX_PLAIN_ACTION(groupsCull, GroupsCullAction);
HPX_PLAIN_ACTION(groupsQueueAssignments, GroupsQueueAssignmentsAction);
HPX_PLAIN_ACTION(groupsInstallHistory, GroupsInstallHistoryAction);

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
static Vector<RockstarRecord> rockstarRecords;
static Vector<Pair<GroupInteger>> haloHosts;
static std::unordered_map<GroupInteger, GroupInteger> previousHaloHosts;
static Vector<Pair<ParticleInteger, GroupInteger>> pendingAssignments;
static SpinlockType assignmentMutex;
static std::unordered_set<GroupInteger> existingGroups;
static Vector<GroupInteger> localExistingGroups;
static std::atomic<size_t> groupCandidates;

void groupsQueueAssignments(Vector<Pair<ParticleInteger, GroupInteger>> assignments) {
    std::lock_guard<SpinlockType> lock(assignmentMutex);
    pendingAssignments.insert(pendingAssignments.end(), assignments.begin(), assignments.end());
}

void groupsInstallHistory(Vector<Pair<GroupInteger>> entries) {
    Vector<hpx::future<void>> futures;
    for (auto const& child : hpxChildren()) {
        futures.push_back(hpx::async<GroupsInstallHistoryAction>(child, entries));
    }
    previousHaloHosts.clear();
    for (auto const& entry : entries) {
        previousHaloHosts[entry.first] = entry.second;
    }
    pendingAssignments.clear();
    hpx::wait_all(futures.begin(), futures.end());
    for (auto& future : futures) {
        future.get();
    }
}

void groupsPrepareHistory(int number) {
    // ROCKSTAR Sec. III.4: prior halo/core membership supplies temporal hosts.
    // Particle previous-halo tags move with particles. The sidecars supply the
    // corresponding previous host even after a restart or rank redistribution.
    Vector<Pair<GroupInteger>> entries;
    int latest = -1;
    std::filesystem::path latestDirectory;
    for (auto const& item : std::filesystem::directory_iterator(".")) {
        if (!item.is_directory()) {
            continue;
        }
        std::string const name = item.path().filename().string();
        if (name.rfind("groups.", 0) != 0) {
            continue;
        }
        std::string const suffix = name.substr(7);
        if (suffix.empty() || suffix.find_first_not_of("0123456789") != std::string::npos) {
            continue;
        }
        try {
            int const candidate = std::stoi(suffix);
            if (candidate < number && candidate > latest &&
                std::filesystem::is_regular_file(item.path() / "rockstar_hosts.complete")) {
                latest = candidate;
                latestDirectory = item.path();
            }
        } catch (std::out_of_range const&) {
            continue;
        }
    }
    if (latest >= 0) {
        std::ifstream manifest(latestDirectory / "rockstar_hosts.complete");
        int version;
        int rankCount;
        size_t haloCount;
        if (!(manifest >> version >> rankCount >> haloCount) || version != 1 || rankCount <= 0) {
            throw std::runtime_error("Malformed ROCKSTAR host-history completion manifest");
        }
        std::unordered_set<GroupInteger> identifiers;
        for (int rank = 0; rank < rankCount; ++rank) {
            auto const path = latestDirectory / ("rockstar_hosts." + std::to_string(rank) + ".dat");
            std::ifstream input(path);
            if (!input) {
                throw std::runtime_error("Cannot read ROCKSTAR host history: " + path.string());
            }
            GroupInteger halo;
            GroupInteger host;
            while (input >> halo) {
                if (!(input >> host) || halo < 0 || host < 0 || halo == NO_GROUP || host == halo ||
                    !identifiers.insert(halo).second) {
                    throw std::runtime_error("Malformed ROCKSTAR host history: " + path.string());
                }
                entries.push_back({halo, host});
            }
            if (!input.eof()) {
                throw std::runtime_error("Malformed ROCKSTAR host history: " + path.string());
            }
        }
        if (entries.size() != haloCount) {
            throw std::runtime_error("Incomplete ROCKSTAR host history");
        }
        for (auto const& entry : entries) {
            if (entry.second != NO_GROUP && identifiers.count(entry.second) == 0) {
                throw std::runtime_error("ROCKSTAR history refers to an unwritten host halo");
            }
        }
    }
    groupsInstallHistory(std::move(entries));
}

void groupsCull() {
    Vector<hpx::future<void>> futs1;
    Vector<hpx::future<void>> futs2;
    for (auto const& c : hpxChildren()) {
        futs1.push_back(hpx::async<GroupsCullAction>(c));
    }
    std::unordered_set<GroupInteger> finalExisting;
    MutexType mutex;
    int const numberOfThreads = hpxHardwareConcurrency();
    for (int proc = 0; proc < numberOfThreads; proc++) {
        futs2.push_back(hpx::async([proc, numberOfThreads, &mutex, &finalExisting]() {
            std::unordered_map<int, Vector<GroupInteger>> map;
            for (int i = proc; i < localExistingGroups.size(); i += numberOfThreads) {
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
    for (auto& future : futs2) {
        future.get();
    }
    localExistingGroups = decltype(localExistingGroups)();
    for (int proc = 0; proc < numberOfThreads; proc++) {
        futs1.push_back(hpx::async([proc, numberOfThreads, &finalExisting]() {
            ParticleInteger const begin = (size_t) proc * particlesSize() / numberOfThreads;
            ParticleInteger const end = (size_t) (proc + 1) * particlesSize() / numberOfThreads;
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
    for (auto& future : futs1) {
        future.get();
    }
    // FOF validity checks above must see the original FOF tags. Install the
    // exclusive ROCKSTAR membership only after those checks have completed.
    for (auto const& assignment : pendingAssignments) {
        particlesGroup(assignment.first) = assignment.second;
    }
    pendingAssignments.clear();
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
        std::filesystem::create_directories("groups." + std::to_string(number));
        // Invalidate a previous attempt before touching its per-rank files.
        std::filesystem::remove("groups." + std::to_string(number) + "/rockstar_hosts.complete");
        std::string const fname =
            std::string("groups.") + std::to_string(number) + std::string("/groups.dat");
        FILE* fp = fopen(fname.c_str(), "wb");
        if (fp == NULL) {
            THROW_ERROR("Unable to open %s for writing\n", fname.c_str());
        }
        GroupHeader header;
        header.boxLength = getOptions().units.lengthCm / constants::kMpcToCm;
        header.time = time;
        header.number = number;
        header.scale = scale;
        size_t const headerWritten = fwrite(&header, sizeof(GroupHeader), 1, fp);
        int const headerClose = fclose(fp);
        if (headerWritten != 1 || headerClose != 0) {
            throw std::runtime_error("Incomplete group catalogue header: " + fname);
        }
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
    bool const groupWriteFailed = ferror(fp) != 0;
    int const groupClose = fclose(fp);
    if (groupWriteFailed || groupClose != 0) {
        throw std::runtime_error("Incomplete group catalogue: " + fname);
    }
    std::string const directory = "groups." + std::to_string(number) + "/";
    std::string const rank = std::to_string(hpxRank());
    std::string const recordName = directory + "rockstar." + rank + ".dat";
    rockstarWriteCatalogue(recordName, rockstarRecords.data(), rockstarRecords.size());
    std::string const historyName = directory + "rockstar_hosts." + rank + ".dat";
    std::ofstream history(historyName);
    for (auto const& entry : haloHosts) {
        history << entry.first << ' ' << entry.second << '\n';
    }
    history.close();
    if (!history) {
        throw std::runtime_error("Cannot write ROCKSTAR host history: " + historyName);
    }
    rockstarRecords.clear();
    haloHosts.clear();
    size_t count = groups.size();
    groups = decltype(groups)();
    existingGroups = decltype(existingGroups)();
    for (auto& f : futs) {
        auto const tmp = f.get();
        count += tmp.first;
        groupCandidates += tmp.second;
    }
    if (hpxRank() == 0) {
        // ROCKSTAR Sec. III.5.4: descendant = maximum exclusive particle overlap.
        // Particle-based descendants use exclusive overlap counts, independent
        // of inclusive spherical-overdensity masses. A missing descendant is
        // represented explicitly for a previously catalogued halo.
        std::unordered_map<GroupInteger, std::pair<GroupInteger, int>> descendants;
        for (auto const& previous : previousHaloHosts) {
            descendants.emplace(previous.first, std::make_pair(NO_GROUP, 0));
        }
        for (int sourceRank = 0; sourceRank < hpxSize(); ++sourceRank) {
            std::string const groupName = directory + "groups." + std::to_string(number) + "." +
                std::to_string(sourceRank) + ".dat";
            FILE* groupFile = fopen(groupName.c_str(), "rb");
            if (groupFile == nullptr) {
                throw std::runtime_error("Cannot read catalogue for descendants: " + groupName);
            }
            while (true) {
                GroupEntry entry;
                if (!entry.read(groupFile)) {
                    break;
                }
                for (auto const& parent : entry.parents) {
                    auto [position, inserted] =
                        descendants.emplace(parent.first, std::make_pair(entry.id, parent.second));
                    auto& best = position->second;
                    if (!inserted &&
                        (parent.second > best.second ||
                            (parent.second == best.second && entry.id < best.first))) {
                        best = {entry.id, parent.second};
                    }
                }
            }
            fclose(groupFile);
        }
        std::vector<GroupInteger> previousIds;
        for (auto const& descendant : descendants) {
            previousIds.push_back(descendant.first);
        }
        std::sort(previousIds.begin(), previousIds.end());
        std::ofstream descendantFile(directory + "rockstar_descendants.dat");
        for (auto const previous : previousIds) {
            auto const& descendant = descendants.at(previous);
            descendantFile << previous << ' ' << descendant.first << ' ' << descendant.second
                           << '\n';
        }
        descendantFile.close();
        if (!descendantFile) {
            throw std::runtime_error("Cannot write ROCKSTAR descendants");
        }
        // Publishing this marker only after every locality and descendant
        // writer succeeds makes interrupted output ineligible as history.
        std::string const temporaryManifest = directory + "rockstar_hosts.complete.tmp";
        std::ofstream manifest(temporaryManifest);
        manifest << 1 << ' ' << hpxSize() << ' ' << count << '\n';
        manifest.close();
        if (!manifest) {
            throw std::runtime_error("Cannot write ROCKSTAR history completion manifest");
        }
        std::filesystem::rename(temporaryManifest, directory + "rockstar_hosts.complete");
    }
    return std::make_pair(count, (size_t) groupCandidates);
}

#define GROUPS_REDUCE_OVERSUBSCRIBE 2

void groupsReduce(double scaleFactor) {
    Vector<hpx::future<void>> futures;
    for (auto const& child : hpxChildren()) {
        futures.push_back(hpx::async<GroupsReduceAction>(child, scaleFactor));
    }
    int const numberOfThreads = GROUPS_REDUCE_OVERSUBSCRIBE * hpxHardwareConcurrency();
    SpinlockType mutex;
    for (int proc = 0; proc < numberOfThreads; ++proc) {
        futures.push_back(hpx::async([proc, numberOfThreads, &mutex, scaleFactor]() {
            std::unordered_map<int, Vector<Pair<ParticleInteger, GroupInteger>>> assignments;
            for (int bin = proc; bin < GROUP_TABLE_SIZE; bin += numberOfThreads) {
                for (auto const& candidate : groupData[bin]) {
                    ++groupCandidates;
                    auto const& particles = candidate.second;
                    // Start with a rejected assignment for every input particle;
                    // overwrite only particles retained in an exclusive halo.
                    std::unordered_map<int, std::unordered_map<ParticleInteger, GroupInteger>> tags;
                    for (auto const& particle : particles) {
                        tags[particle.rank][particle.index] = NO_GROUP;
                    }
                    if (particles.size() >= getOptions().analysis.minGroupSize) {
                        auto const halos = rockstarFindSubgroups(particles, scaleFactor);
                        if (halos.size() > 32767) {
                            throw std::runtime_error("ROCKSTAR record halo ID capacity exceeded");
                        }
                        Array<double, NDIM> origin;
                        for (int dim = 0; dim < NDIM; ++dim) {
                            origin[dim] = particles.front().x[dim].toDouble();
                        }
                        for (int i = 0; i < halos.size(); ++i) {
                            auto const& halo = halos[i];
                            auto entry = rockstarMakeGroupEntry(halos, i, scaleFactor, origin);
                            entry.id = halo.catalogueId;
                            for (int particleIndex = 0; particleIndex < halo.parts.size();
                                ++particleIndex) {
                                auto const& particle = halo.parts[particleIndex];
                                tags[particle.rank][particle.index] = halo.catalogueId;
                            }
                            auto record = rockstarMakeRecord(halos, i, scaleFactor, origin);
                            record.id.fofId = candidate.first;
                            record.id.haloId = i;
                            record.pid = halo.parent;
                            GroupInteger const host =
                                halo.parent >= 0 ? halos[halo.parent].catalogueId : NO_GROUP;
                            std::lock_guard<SpinlockType> lock(mutex);
                            groups.push_back(std::move(entry));
                            rockstarRecords.push_back(record);
                            haloHosts.push_back({halo.catalogueId, host});
                            existingGroups.insert(candidate.first);
                        }
                    }
                    for (auto const& rank : tags) {
                        for (auto const& assignment : rank.second) {
                            assignments[rank.first].push_back(
                                {assignment.first, assignment.second});
                        }
                    }
                }
            }
            Vector<hpx::future<void>> transmissions;
            for (auto& rank : assignments) {
                transmissions.push_back(hpx::async<GroupsQueueAssignmentsAction>(
                    hpxLocalities()[rank.first], std::move(rank.second)));
            }
            hpx::wait_all(transmissions.begin(), transmissions.end());
            for (auto& transmission : transmissions) {
                transmission.get();
            }
        }));
    }
    hpx::wait_all(futures.begin(), futures.end());
    for (auto& future : futures) {
        future.get();
    }
    groupData = decltype(groupData)();
}

void groupsTransmitParticles(Vector<std::pair<GroupInteger, Vector<ParticleData>>> entries) {
    Vector<hpx::future<void>> futs;
    int const numberOfThreads = hpxHardwareConcurrency();
    for (int proc = 0; proc < numberOfThreads; proc++) {
        futs.push_back(hpx::async([numberOfThreads, proc, &entries]() {
            for (int i = proc; i < entries.size(); i += numberOfThreads) {
                auto const grp = entries[i].first;
                int const bin = (grp / GROUP_WAVES) % GROUP_TABLE_SIZE;
                std::lock_guard<SpinlockType> lock(mutexes[bin]);
                auto& entry = groupData[bin][grp];
                entry.insert(entry.end(), entries[i].second.begin(), entries[i].second.end());
            }
        }));
    }
    hpx::wait_all(futs.begin(), futs.end());
    for (auto& future : futs) {
        future.get();
    }
}

void groupsAddParticles(int wave, double scale) {
    Vector<hpx::future<void>> futs;
    for (auto const& c : hpxChildren()) {
        futs.push_back(hpx::async<GroupsAddParticlesAction>(c, wave, scale));
    }
    int const numberOfThreads = hpxHardwareConcurrency();
    SpinlockType mutex;
    float const ainv = 1.0 / scale;
    for (int proc = 0; proc < numberOfThreads; proc++) {
        futs.push_back(hpx::async([proc, numberOfThreads, wave, &mutex, ainv]() {
            Vector<hpx::future<void>> futs;
            std::unordered_map<int, std::unordered_map<GroupInteger, Vector<ParticleData>>>
                localGroups;
            ParticleInteger const begin = (size_t) proc * particlesSize() / numberOfThreads;
            ParticleInteger const end = (size_t) (proc + 1) * particlesSize() / numberOfThreads;
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
                        }
                        part.index = i;
                        part.lastGroup = particlesLastgroup(i);
                        part.rank = hpxRank();
                        part.identity = particlesGroupInit(i);
                        auto const previousHost = previousHaloHosts.find(part.lastGroup);
                        part.lastHost = previousHost == previousHaloHosts.end() ?
                            NO_GROUP :
                            previousHost->second;
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
            for (auto& future : futs) {
                future.get();
            }
        }));
    }
    if (wave == 0) {
        groupCandidates = 0;
    }
    hpx::wait_all(futs.begin(), futs.end());
    for (auto& future : futs) {
        future.get();
    }
}

namespace detail {

Vector<GroupEntry> const& groupsLocalEntries() {
    return groups;
}

Vector<Pair<ParticleInteger, GroupInteger>> const& groupsPendingAssignments() {
    return pendingAssignments;
}

std::unordered_map<GroupInteger, GroupInteger> const& groupsPreviousHaloHosts() {
    return previousHaloHosts;
}

void groupsResetHistory() {
    previousHaloHosts.clear();
}

void groupsForEachCandidate(
    std::function<void(GroupInteger, Vector<ParticleData> const&)> const& visit) {
    for (auto const& bin : groupData) {
        for (auto const& candidate : bin) {
            visit(candidate.first, candidate.second);
        }
    }
}

}    // namespace detail
