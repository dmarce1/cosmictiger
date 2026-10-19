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
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

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

void groupsCatalogueTest() {
    if (hpxSize() != 1) {
        throw std::runtime_error("groupsCatalogue test requires one locality");
    }
    auto const require = [](bool condition, char const* message) {
        if (!condition) {
            throw std::runtime_error(message);
        }
    };
    require(getOptions().analysis.minGroupSize <= 128,
        "groupsCatalogue test requires analysis.minGroupSize<=128");
    require(getOptions().analysis.groups,
        "groupsCatalogue test requires analysis.groups=true for persistent particle tags");
    auto const originalDirectory = std::filesystem::current_path();
    auto const stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    auto const directory = originalDirectory / ("groups_catalogue_test_" + std::to_string(stamp));
    require(std::filesystem::create_directory(directory), "Cannot create catalogue test directory");
    std::filesystem::current_path(directory);
    try {
        constexpr int kHaloParticles = 128;
        constexpr int kParticleCount = kHaloParticles + 2;
        particlesResize(kParticleCount);
        std::mt19937 generator(141018);
        std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
        float const support = getOptions().gravity.supportRadius;
        float const speed = 0.03f * std::sqrt(getOptions().gravity.gm / support);
        for (int i = 0; i < kParticleCount; ++i) {
            for (int dim = 0; dim < NDIM; ++dim) {
                particlesPos(dim, i) = 0.5 + 0.2 * support * distribution(generator);
                particlesVel(dim, i) = speed * distribution(generator);
            }
            particlesLastgroup(i) = NO_GROUP;
        }
        Vector<GroupInteger> firstTags(kParticleCount, NO_GROUP);
        size_t firstHaloCount = 0;
        for (int snapshot = 1; snapshot <= 2; ++snapshot) {
            int const snapshotNumber = snapshot == 1 ? 1 : 3;
            if (snapshot == 2) {
                // An interrupted newer output must not hide the last complete
                // host history when the simulation restarts.
                std::filesystem::create_directory("groups.2");
                std::ofstream incomplete("groups.2/rockstar_hosts.0.dat");
                incomplete << "12345";
            }
            previousHaloHosts.clear();
            groupsPrepareHistory(snapshotNumber);
            if (snapshot == 2) {
                require(previousHaloHosts.size() == firstHaloCount,
                    "Saved halo host map was not restored");
            }
            particlesGroupsInit();
            for (int i = 0; i < kHaloParticles; ++i) {
                particlesGroup(i) = particlesGroupInit(0);
            }
            particlesGroup(kHaloParticles) = particlesGroupInit(kHaloParticles);
            for (int wave = 0; wave < GROUP_WAVES; ++wave) {
                groupsAddParticles(wave, 1.0);
                if (snapshot == 2) {
                    for (auto const& bin : groupData) {
                        for (auto const& candidate : bin) {
                            for (auto const& particle : candidate.second) {
                                require(particle.lastGroup == firstTags[particle.index],
                                    "Previous halo tags did not reach the finder");
                                require(particle.identity == particlesGroupInit(particle.index),
                                    "Particle identity missing from distributed group input");
                                auto const host = previousHaloHosts.find(particle.lastGroup);
                                require(particle.lastHost ==
                                        (host == previousHaloHosts.end() ? NO_GROUP : host->second),
                                    "Previous host missing from finder input");
                            }
                        }
                    }
                }
                groupsReduce(1.0);
            }
            // This invalid original FOF tag has no queued halo assignment; it
            // specifically verifies that culling visits the final partition.
            particlesGroup(kParticleCount - 1) = particlesGroupInit(kParticleCount - 1);
            groupsCull();
            require(!groups.empty(), "Bound synthetic group produced an empty catalogue");
            require(particlesGroup(kHaloParticles) == NO_GROUP,
                "Undersized FOF group retained a particle tag");
            require(particlesGroup(kParticleCount - 1) == NO_GROUP,
                "FOF culling skipped the last particle partition");
            std::unordered_map<GroupInteger, std::unordered_map<GroupInteger, int>>
                expectedOverlaps;
            for (int i = 0; i < kParticleCount; ++i) {
                GroupInteger const current = particlesGroup(i);
                if (current != NO_GROUP && firstTags[i] != NO_GROUP) {
                    ++expectedOverlaps[current][firstTags[i]];
                }
            }
            std::unordered_map<GroupInteger, std::pair<GroupInteger, int>> expectedDescendants;
            for (auto const& previous : previousHaloHosts) {
                expectedDescendants.emplace(previous.first, std::make_pair(NO_GROUP, 0));
            }
            std::unordered_set<GroupInteger> identifiers;
            for (auto const& group : groups) {
                require(identifiers.insert(group.id).second, "Duplicate halo catalogue identifier");
                require(group.parents == expectedOverlaps[group.id],
                    "Particle overlap ancestry counts are not exact");
                for (auto const& parent : group.parents) {
                    auto [position, inserted] = expectedDescendants.emplace(
                        parent.first, std::make_pair(group.id, parent.second));
                    auto& best = position->second;
                    if (!inserted &&
                        (parent.second > best.second ||
                            (parent.second == best.second && group.id < best.first))) {
                        best = {group.id, parent.second};
                    }
                }
            }
            size_t retained = 0;
            for (int i = 0; i < kParticleCount; ++i) {
                GroupInteger const tag = particlesGroup(i);
                if (tag != NO_GROUP) {
                    require(identifiers.count(tag) != 0, "Particle points to an unwritten halo");
                    ++retained;
                }
                firstTags[i] = tag;
            }
            require(retained >= ROCKSTAR_MIN_GROUP, "Synthetic halo lost all particle memberships");
            auto const written = groupsSave(snapshotNumber, 1.0, 0.0);
            require(written.first == identifiers.size(), "Saved halo count does not match records");
            auto const name = "groups." + std::to_string(snapshotNumber) + "/groups." +
                std::to_string(snapshotNumber) + ".0.dat";
            FILE* input = fopen(name.c_str(), "rb");
            require(input != nullptr, "Cannot read saved halo catalogue");
            size_t readCount = 0;
            while (true) {
                GroupEntry entry;
                if (!entry.read(input)) {
                    break;
                }
                require(
                    identifiers.count(entry.id) != 0, "Saved catalogue ID differs from membership");
                ++readCount;
            }
            fclose(input);
            require(
                readCount == written.first, "Saved GroupEntry catalogue is empty or incomplete");
            auto const recordName = "groups." + std::to_string(snapshotNumber) + "/rockstar.0.dat";
            require(
                std::filesystem::file_size(recordName) == written.first * sizeof(RockstarRecord),
                "Saved ROCKSTAR property catalogue is empty or incomplete");
            std::ifstream descendants(
                "groups." + std::to_string(snapshotNumber) + "/rockstar_descendants.dat");
            GroupInteger previous;
            GroupInteger descendant;
            int overlap;
            size_t descendantCount = 0;
            while (descendants >> previous >> descendant >> overlap) {
                auto const expected = expectedDescendants.find(previous);
                require(expected != expectedDescendants.end() &&
                        expected->second == std::make_pair(descendant, overlap),
                    "Saved particle-based descendant or overlap count is wrong");
                expectedDescendants.erase(expected);
                ++descendantCount;
            }
            require(descendants.eof() && expectedDescendants.empty(),
                "Saved particle-based descendant table is incomplete");
            if (snapshot == 2) {
                require(descendantCount != 0, "Second snapshot has no particle-based descendants");
            }
            particlesGroupsDestroy();
            for (int i = 0; i < kParticleCount; ++i) {
                require(particlesLastgroup(i) == firstTags[i],
                    "Halo membership was not carried to the next snapshot");
            }
            firstHaloCount = written.first;
        }
        std::filesystem::current_path(originalDirectory);
        std::filesystem::remove_all(directory);
        PRINT("Snapshot halo catalogue and temporal membership tests passed\n");
    } catch (...) {
        std::filesystem::current_path(originalDirectory);
        throw;
    }
}

namespace {
constexpr int kDistributedHaloHalf = 64;
constexpr int kDistributedLocalCount = 2 * kDistributedHaloHalf + 2;
Vector<GroupInteger> distributedPreviousTags;

void requireDistributedCatalogue(bool condition, char const* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

GroupInteger distributedFofId(int snapshot, int localIndex) {
    bool const firstHalo = localIndex < kDistributedHaloHalf;
    bool const firstOwner = snapshot == 1 ? firstHalo : !firstHalo;
    return firstOwner ? 0 : kDistributedLocalCount + kDistributedHaloHalf;
}
}    // namespace

void groupsCatalogueTestDirectory(std::string directory) {
    std::filesystem::current_path(directory);
}
HPX_PLAIN_ACTION(groupsCatalogueTestDirectory, GroupsCatalogueTestDirectoryAction);

void groupsCatalogueDistributedSetup(std::string directory) {
    std::filesystem::current_path(directory);
    particlesResize(kDistributedLocalCount);
    std::mt19937 generator(141018 + hpxRank());
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    float const support = getOptions().gravity.supportRadius;
    float const speed = 0.03f * std::sqrt(getOptions().gravity.gm / support);
    for (int i = 0; i < kDistributedLocalCount; ++i) {
        double const center = i < kDistributedHaloHalf ? 0.25 : 0.75;
        for (int dim = 0; dim < NDIM; ++dim) {
            particlesPos(dim, i) = center + 0.2 * support * distribution(generator);
            particlesVel(dim, i) = speed * distribution(generator);
        }
        particlesLastgroup(i) = NO_GROUP;
    }
}
HPX_PLAIN_ACTION(groupsCatalogueDistributedSetup, GroupsCatalogueDistributedSetupAction);

void groupsCatalogueDistributedBegin(
    int snapshot, Vector<GroupInteger> previousTags, Vector<Pair<GroupInteger>> expectedHosts) {
    requireDistributedCatalogue(previousHaloHosts.size() == expectedHosts.size(),
        "Distributed host history was not installed on every locality");
    for (auto const& expected : expectedHosts) {
        auto const actual = previousHaloHosts.find(expected.first);
        requireDistributedCatalogue(
            actual != previousHaloHosts.end() && actual->second == expected.second,
            "Distributed host-history value differs from the written catalogue");
    }
    distributedPreviousTags = std::move(previousTags);
    for (int i = 0; i < kDistributedLocalCount; ++i) {
        requireDistributedCatalogue(particlesGroupInit(i) == hpxRank() * kDistributedLocalCount + i,
            "Distributed particle offsets do not match the fixture");
        requireDistributedCatalogue(particlesLastgroup(i) ==
                distributedPreviousTags[hpxRank() * kDistributedLocalCount + i],
            "Distributed particle history did not persist between snapshots");
        particlesGroup(i) = i < 2 * kDistributedHaloHalf ?
            distributedFofId(snapshot, i) :
            (i == 2 * kDistributedHaloHalf ? particlesGroupInit(i) : NO_GROUP);
    }
}
HPX_PLAIN_ACTION(groupsCatalogueDistributedBegin, GroupsCatalogueDistributedBeginAction);

int groupsCatalogueDistributedInspect(int snapshot) {
    int realInputs = 0;
    for (auto const& bin : groupData) {
        for (auto const& candidate : bin) {
            requireDistributedCatalogue(particlesGroupHome(candidate.first) == hpxRank(),
                "FOF inputs reached the wrong analysis locality");
            bool const realHalo = candidate.first == 0 ||
                candidate.first == kDistributedLocalCount + kDistributedHaloHalf;
            requireDistributedCatalogue(candidate.second.size() == (realHalo ? 128 : 1),
                "Cross-rank FOF particle collection is incomplete");
            int sourceCounts[2] = {0, 0};
            std::unordered_set<GroupInteger> identities;
            for (auto const& particle : candidate.second) {
                requireDistributedCatalogue(particle.rank >= 0 && particle.rank < 2 &&
                        particle.index >= 0 && particle.index < kDistributedLocalCount,
                    "Distributed finder input lost its source address");
                GroupInteger const identity =
                    particle.rank * kDistributedLocalCount + particle.index;
                requireDistributedCatalogue(
                    particle.identity == identity && identities.insert(identity).second,
                    "Distributed finder input duplicated or changed a particle identity");
                requireDistributedCatalogue(particle.lastGroup == distributedPreviousTags[identity],
                    "Previous halo membership did not reach the remote finder");
                auto const host = previousHaloHosts.find(particle.lastGroup);
                requireDistributedCatalogue(particle.lastHost ==
                        (host == previousHaloHosts.end() ? NO_GROUP : host->second),
                    "Previous host did not reach the remote finder");
                if (realHalo) {
                    requireDistributedCatalogue(particle.index < 2 * kDistributedHaloHalf &&
                            candidate.first == distributedFofId(snapshot, particle.index),
                        "Distributed FOF contains particles from the other halo");
                    ++sourceCounts[particle.rank];
                    ++realInputs;
                }
            }
            if (realHalo) {
                requireDistributedCatalogue(sourceCounts[0] == kDistributedHaloHalf &&
                        sourceCounts[1] == kDistributedHaloHalf,
                    "FOF input did not include both MPI ranks");
            }
        }
    }
    return realInputs;
}
HPX_PLAIN_ACTION(groupsCatalogueDistributedInspect, GroupsCatalogueDistributedInspectAction);

Vector<GroupInteger> groupsCatalogueDistributedBeforeCull(int snapshot) {
    for (int i = 0; i < 2 * kDistributedHaloHalf; ++i) {
        requireDistributedCatalogue(particlesGroup(i) == distributedFofId(snapshot, i),
            "Halo assignments overwrote the FOF tags before distributed culling");
    }
    requireDistributedCatalogue(pendingAssignments.size() == kDistributedLocalCount - 1,
        "Remote halo assignments were not delivered to every source particle");
    std::unordered_set<ParticleInteger> indices;
    for (auto const& assignment : pendingAssignments) {
        requireDistributedCatalogue(assignment.first >= 0 &&
                assignment.first < kDistributedLocalCount - 1 &&
                indices.insert(assignment.first).second,
            "Remote halo assignments contain invalid or duplicate particle indices");
    }
    Vector<GroupInteger> identifiers;
    for (auto const& group : groups) {
        identifiers.push_back(group.id);
    }
    requireDistributedCatalogue(
        !identifiers.empty(), "One analysis locality failed to produce a bound halo");
    // No finder assignment exists for this particle. Culling must visit the
    // final local partition independently of the queued membership updates.
    particlesGroup(kDistributedLocalCount - 1) = particlesGroupInit(kDistributedLocalCount - 1);
    return identifiers;
}
HPX_PLAIN_ACTION(groupsCatalogueDistributedBeforeCull, GroupsCatalogueDistributedBeforeCullAction);

Vector<GroupInteger> groupsCatalogueDistributedAfterCull(Vector<GroupInteger> identifiers) {
    requireDistributedCatalogue(
        pendingAssignments.empty(), "Distributed membership updates were not consumed by culling");
    std::unordered_set<GroupInteger> valid(identifiers.begin(), identifiers.end());
    Vector<GroupInteger> tags;
    int retained[2] = {0, 0};
    for (int i = 0; i < kDistributedLocalCount; ++i) {
        GroupInteger const tag = particlesGroup(i);
        if (i >= 2 * kDistributedHaloHalf) {
            requireDistributedCatalogue(tag == NO_GROUP,
                "Distributed culling retained a rejected or unprocessed FOF particle");
        } else if (tag != NO_GROUP) {
            requireDistributedCatalogue(valid.count(tag) != 0,
                "Distributed particle membership refers to an unwritten halo");
            ++retained[i / kDistributedHaloHalf];
        }
        tags.push_back(tag);
    }
    requireDistributedCatalogue(
        retained[0] >= ROCKSTAR_MIN_GROUP && retained[1] >= ROCKSTAR_MIN_GROUP,
        "Remote membership delivery lost a synthetic halo on a source locality");
    return tags;
}
HPX_PLAIN_ACTION(groupsCatalogueDistributedAfterCull, GroupsCatalogueDistributedAfterCullAction);

void groupsCatalogueDistributedCheckHistory(Vector<GroupInteger> tags) {
    for (int i = 0; i < kDistributedLocalCount; ++i) {
        requireDistributedCatalogue(
            particlesLastgroup(i) == tags[hpxRank() * kDistributedLocalCount + i],
            "Final distributed halo membership was not retained for the next snapshot");
    }
}
HPX_PLAIN_ACTION(
    groupsCatalogueDistributedCheckHistory, GroupsCatalogueDistributedCheckHistoryAction);

void groupsCatalogueDistributedTest() {
    requireDistributedCatalogue(
        hpxSize() == 2, "groupsCatalogueDistributed test requires exactly two localities");
    requireDistributedCatalogue(
        getOptions().analysis.groups && getOptions().analysis.minGroupSize <= 128,
        "Distributed catalogue fixture requires groups enabled and minimum group size <=128");
    auto const originalDirectory = std::filesystem::current_path();
    auto const stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    auto const directory =
        originalDirectory / ("groups_catalogue_mpi_test_" + std::to_string(stamp));
    requireDistributedCatalogue(std::filesystem::create_directory(directory),
        "Cannot create distributed catalogue test directory");
    auto const restoreDirectories = [&]() {
        for (auto const& locality : hpxLocalities()) {
            hpx::async<GroupsCatalogueTestDirectoryAction>(locality, originalDirectory.string())
                .get();
        }
    };
    try {
        for (auto const& locality : hpxLocalities()) {
            hpx::async<GroupsCatalogueDistributedSetupAction>(locality, directory.string()).get();
        }
        Vector<GroupInteger> previousTags(2 * kDistributedLocalCount, NO_GROUP);
        Vector<Pair<GroupInteger>> previousHosts;
        for (int snapshot = 1; snapshot <= 2; ++snapshot) {
            groupsPrepareHistory(snapshot);
            particlesGroupsInit();
            for (auto const& locality : hpxLocalities()) {
                hpx::async<GroupsCatalogueDistributedBeginAction>(
                    locality, snapshot, previousTags, previousHosts)
                    .get();
            }
            int inputCount = 0;
            for (int wave = 0; wave < GROUP_WAVES; ++wave) {
                groupsAddParticles(wave, 1.0);
                for (auto const& locality : hpxLocalities()) {
                    inputCount +=
                        hpx::async<GroupsCatalogueDistributedInspectAction>(locality, snapshot)
                            .get();
                }
                groupsReduce(1.0);
            }
            requireDistributedCatalogue(inputCount == 4 * kDistributedHaloHalf,
                "Distributed finder did not consume both complete FOF groups");
            Vector<GroupInteger> identifiers;
            Vector<Vector<GroupInteger>> rankIdentifiers;
            for (auto const& locality : hpxLocalities()) {
                auto localIds =
                    hpx::async<GroupsCatalogueDistributedBeforeCullAction>(locality, snapshot)
                        .get();
                identifiers.insert(identifiers.end(), localIds.begin(), localIds.end());
                rankIdentifiers.push_back(std::move(localIds));
            }
            std::unordered_set<GroupInteger> uniqueIds(identifiers.begin(), identifiers.end());
            requireDistributedCatalogue(uniqueIds.size() == identifiers.size(),
                "Different MPI ranks assigned the same persistent halo identifier");
            groupsCull();
            Vector<GroupInteger> tags;
            for (auto const& locality : hpxLocalities()) {
                auto const localTags =
                    hpx::async<GroupsCatalogueDistributedAfterCullAction>(locality, identifiers)
                        .get();
                tags.insert(tags.end(), localTags.begin(), localTags.end());
            }
            std::unordered_map<GroupInteger, std::unordered_map<GroupInteger, int>>
                expectedOverlaps;
            for (size_t i = 0; i < tags.size(); ++i) {
                if (tags[i] != NO_GROUP && previousTags[i] != NO_GROUP) {
                    ++expectedOverlaps[tags[i]][previousTags[i]];
                }
            }
            std::unordered_map<GroupInteger, std::pair<GroupInteger, int>> expectedDescendants;
            for (auto const& previous : previousHosts) {
                expectedDescendants.emplace(previous.first, std::make_pair(NO_GROUP, 0));
            }
            for (auto const& descendant : expectedOverlaps) {
                for (auto const& parent : descendant.second) {
                    auto& best = expectedDescendants.at(parent.first);
                    if (parent.second > best.second ||
                        (parent.second == best.second && descendant.first < best.first)) {
                        best = {descendant.first, parent.second};
                    }
                }
            }
            auto const written = groupsSave(snapshot, 1.0, 0.0);
            requireDistributedCatalogue(written.first == identifiers.size() && written.second == 4,
                "Distributed catalogue or FOF candidate count is incomplete");
            std::string const prefix = "groups." + std::to_string(snapshot) + "/";
            Vector<Pair<GroupInteger>> currentHosts;
            std::unordered_set<GroupInteger> writtenIds;
            for (int rank = 0; rank < 2; ++rank) {
                std::unordered_set<GroupInteger> rankIds(
                    rankIdentifiers[rank].begin(), rankIdentifiers[rank].end());
                auto const name = prefix + "groups." + std::to_string(snapshot) + "." +
                    std::to_string(rank) + ".dat";
                FILE* input = fopen(name.c_str(), "rb");
                requireDistributedCatalogue(
                    input != nullptr, "Cannot read per-rank halo catalogue");
                size_t readCount = 0;
                while (true) {
                    GroupEntry entry;
                    if (!entry.read(input)) {
                        break;
                    }
                    requireDistributedCatalogue(
                        rankIds.count(entry.id) != 0 && writtenIds.insert(entry.id).second,
                        "Written per-rank halo identifier differs from its particle membership");
                    requireDistributedCatalogue(entry.parents == expectedOverlaps[entry.id],
                        "Distributed catalogue ancestry differs from exact particle overlaps");
                    ++readCount;
                }
                fclose(input);
                requireDistributedCatalogue(readCount == rankIds.size(),
                    "Written per-rank GroupEntry catalogue is incomplete");
                auto const recordName = prefix + "rockstar." + std::to_string(rank) + ".dat";
                requireDistributedCatalogue(std::filesystem::file_size(recordName) ==
                        rankIds.size() * sizeof(RockstarRecord),
                    "Written per-rank ROCKSTAR property catalogue is incomplete");
                std::ifstream records(recordName, std::ios::binary);
                RockstarRecord record;
                size_t recordCount = 0;
                while (records.read(reinterpret_cast<char*>(&record), sizeof(record))) {
                    requireDistributedCatalogue(record.id.fofId ==
                            (rank == 0 ? 0 : kDistributedLocalCount + kDistributedHaloHalf),
                        "ROCKSTAR record was written by the wrong FOF analysis locality");
                    ++recordCount;
                }
                requireDistributedCatalogue(records.eof() && recordCount == rankIds.size(),
                    "Cannot read complete per-rank ROCKSTAR records");
                std::ifstream hosts(prefix + "rockstar_hosts." + std::to_string(rank) + ".dat");
                GroupInteger halo;
                GroupInteger host;
                size_t hostCount = 0;
                while (hosts >> halo >> host) {
                    requireDistributedCatalogue(rankIds.erase(halo) == 1 &&
                            (host == NO_GROUP || uniqueIds.count(host) != 0),
                        "Written distributed host history is missing or refers to an unwritten "
                        "halo");
                    currentHosts.push_back({halo, host});
                    ++hostCount;
                }
                requireDistributedCatalogue(
                    hosts.eof() && rankIds.empty() && hostCount == readCount,
                    "Written per-rank host history is incomplete");
            }
            std::ifstream manifest(prefix + "rockstar_hosts.complete");
            int version = 0;
            int rankCount = 0;
            size_t haloCount = 0;
            requireDistributedCatalogue(bool(manifest >> version >> rankCount >> haloCount) &&
                    version == 1 && rankCount == 2 && haloCount == written.first,
                "Distributed completion manifest does not describe all written ranks");
            std::ifstream descendants(prefix + "rockstar_descendants.dat");
            GroupInteger previous;
            GroupInteger descendant;
            int overlap;
            while (descendants >> previous >> descendant >> overlap) {
                auto const expected = expectedDescendants.find(previous);
                requireDistributedCatalogue(expected != expectedDescendants.end() &&
                        expected->second == std::make_pair(descendant, overlap),
                    "Distributed descendant or overlap count differs from particle memberships");
                expectedDescendants.erase(expected);
            }
            requireDistributedCatalogue(descendants.eof() && expectedDescendants.empty(),
                "Distributed descendant catalogue is incomplete");
            particlesGroupsDestroy();
            for (auto const& locality : hpxLocalities()) {
                hpx::async<GroupsCatalogueDistributedCheckHistoryAction>(locality, tags).get();
            }
            previousTags = std::move(tags);
            previousHosts = std::move(currentHosts);
        }
        restoreDirectories();
        std::filesystem::remove_all(directory);
        PRINT("Two-locality snapshot catalogue, remote membership, and temporal history tests "
              "passed\n");
    } catch (...) {
        restoreDirectories();
        throw;
    }
}
