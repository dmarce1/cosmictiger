/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2021  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <hpx/threading_base/annotated_function.hpp>

#include <cosmictiger/analysis/bh.hpp>
#include <cosmictiger/analysis/groups.hpp>
#include <cosmictiger/analysis/rockstar.hpp>
#include <cosmictiger/core/containers.hpp>
#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/core/profiler.hpp>
#include <cosmictiger/io/output.hpp>
#include <cosmictiger/math/constants.hpp>
#include <cosmictiger/math/math.hpp>
#include <cosmictiger/particles/particles.hpp>

void groupsTransmitParticles(Vector<std::pair<GroupInteger, Vector<ParticleData>>> entries);
Vector<GroupInteger> groupsExist(Vector<GroupInteger>);
void groupsQueueAssignments(Vector<Pair<ParticleInteger, GroupInteger>> assignments);
void groupsInstallHistory(Vector<Pair<GroupInteger>> entries);
Vector<HaloAperture> groupsGatherApertures();
Vector<Pair<double>> groupsSumBaryons(Vector<HaloAperture> apertures);
void groupsInstallBaryons(Vector<Pair<double>> masses);

HPX_PLAIN_ACTION(groupsAddParticles, GroupsAddParticlesAction);
HPX_PLAIN_ACTION(groupsTransmitParticles, GroupsTransmitParticlesAction);
HPX_PLAIN_ACTION(groupsReduce, GroupsReduceAction);
HPX_PLAIN_ACTION(groupsSave, GroupsSaveAction);
HPX_PLAIN_ACTION(groupsExist, GroupsExistAction);
HPX_PLAIN_ACTION(groupsCull, GroupsCullAction);
HPX_PLAIN_ACTION(groupsQueueAssignments, GroupsQueueAssignmentsAction);
HPX_PLAIN_ACTION(groupsInstallHistory, GroupsInstallHistoryAction);
HPX_PLAIN_ACTION(groupsGatherApertures, GroupsGatherAperturesAction);
HPX_PLAIN_ACTION(groupsSumBaryons, GroupsSumBaryonsAction);
HPX_PLAIN_ACTION(groupsInstallBaryons, GroupsInstallBaryonsAction);

constexpr int kGroupTableSize = 1024;

struct GroupHashHi {
    size_t operator()(GroupInteger grp) const {
        return grp / (kGroupTableSize * kGroupWaves);
    }
};

static Array<SpinlockType, kGroupTableSize> mutexes;
static Array<std::unordered_map<GroupInteger, Vector<ParticleData>, GroupHashHi>, kGroupTableSize>
    groupData;
static Vector<GroupEntry> groups;
static Vector<RockstarRecord> rockstarRecords;
// One aperture per rockstarRecords entry, in the same order.
static Vector<HaloAperture> haloApertures;
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
    profile::ScopedPhase profilePhase("groups.history");
    // ROCKSTAR Sec. III.4: prior halo/core membership supplies temporal hosts.
    // Particle previous-halo tags move with particles. The sidecars supply the
    // corresponding previous host even after a restart or rank redistribution.
    Vector<Pair<GroupInteger>> entries;
    int latest = -1;
    std::filesystem::path latestDirectory;
    for (auto const& item : std::filesystem::directory_iterator(outputPath("."))) {
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
                if (!(input >> host) || halo < 0 || host < 0 || halo == kNoParticleGroup ||
                    host == halo || !identifiers.insert(halo).second) {
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
            if (entry.second != kNoParticleGroup && identifiers.count(entry.second) == 0) {
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
        futs2.push_back(hpx::async(hpx::annotated_function(
            [proc, numberOfThreads, &mutex, &finalExisting]() {
                std::unordered_map<int, Vector<GroupInteger>> map;
                for (int i = proc; i < localExistingGroups.size(); i += numberOfThreads) {
                    auto const& grp = localExistingGroups[i];
                    map[particlesGroupHome(grp)].push_back(grp);
                }
                Vector<hpx::future<Vector<GroupInteger>>> futs;
                for (auto i = map.begin(); i != map.end(); i++) {
                    futs.push_back(hpx::async<GroupsExistAction>(
                        hpxLocalities()[i->first], std::move(i->second)));
                }
                for (auto& f : futs) {
                    auto v = f.get();
                    std::lock_guard<MutexType> lock(mutex);
                    for (auto const& grp : v) {
                        finalExisting.insert(grp);
                    }
                }
            },
            "analysis.groups.cull.queryExisting")));
    }
    hpx::wait_all(futs2.begin(), futs2.end());
    for (auto& future : futs2) {
        future.get();
    }
    localExistingGroups = decltype(localExistingGroups)();
    for (int proc = 0; proc < numberOfThreads; proc++) {
        futs1.push_back(hpx::async(hpx::annotated_function(
            [proc, numberOfThreads, &finalExisting]() {
                ParticleInteger const begin = (size_t) proc * particlesSize() / numberOfThreads;
                ParticleInteger const end = (size_t) (proc + 1) * particlesSize() / numberOfThreads;
                for (ParticleInteger i = begin; i < end; i++) {
                    if (particlesGroup(i) != kNoParticleGroup) {
                        if (finalExisting.find(particlesGroup(i)) == finalExisting.end()) {
                            particlesGroup(i) = kNoParticleGroup;
                        }
                    }
                }
            },
            "analysis.groups.cull.removeStale")));
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
        std::filesystem::create_directories(outputPath("groups." + std::to_string(number)));
        // Invalidate a previous attempt before touching its per-rank files.
        std::filesystem::remove(
            outputPath("groups." + std::to_string(number) + "/rockstar_hosts.complete"));
        std::string const fname = outputPath("groups." + std::to_string(number) + "/groups.dat");
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

    std::string const fname = outputPath("groups." + std::to_string(number) + "/groups." +
        std::to_string(number) + "." + std::to_string(hpxRank()) + ".dat");
    FILE* fp = fopen(fname.c_str(), "wb");
    if (fp == NULL) {
        THROW_ERROR("Unable to open %s\n", fname.c_str());
    }
    if (groups.size()) {
        hpx::sort(annotatedParallelPolicy("analysis.groups.sortCatalogue"), groups.begin(),
            groups.end(), [](GroupEntry const& a, GroupEntry const& b) { return a.id < b.id; })
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
    std::string const directory = outputPath("groups." + std::to_string(number)) + "/";
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
    haloApertures.clear();
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
            descendants.emplace(previous.first, std::make_pair(kNoParticleGroup, 0));
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

constexpr int kGroupsReduceOversubscribe = 2;

void groupsReduce(double scaleFactor) {
    Vector<hpx::future<void>> futures;
    for (auto const& child : hpxChildren()) {
        futures.push_back(hpx::async<GroupsReduceAction>(child, scaleFactor));
    }
    int const numberOfThreads = kGroupsReduceOversubscribe * hpxHardwareConcurrency();
    SpinlockType mutex;
    for (int proc = 0; proc < numberOfThreads; ++proc) {
        futures.push_back(hpx::async(hpx::annotated_function(
            [proc, numberOfThreads, &mutex, scaleFactor]() {
                std::unordered_map<int, Vector<Pair<ParticleInteger, GroupInteger>>> assignments;
                for (int bin = proc; bin < kGroupTableSize; bin += numberOfThreads) {
                    for (auto const& candidate : groupData[bin]) {
                        ++groupCandidates;
                        auto const& particles = candidate.second;
                        // Start with a rejected assignment for every input particle;
                        // overwrite only particles retained in an exclusive halo.
                        std::unordered_map<int, std::unordered_map<ParticleInteger, GroupInteger>>
                            tags;
                        for (auto const& particle : particles) {
                            tags[particle.rank][particle.index] = kNoParticleGroup;
                        }
                        if (particles.size() >= getOptions().analysis.minGroupSize) {
                            auto const halos = rockstarFindSubgroups(particles, scaleFactor);
                            size_t const haloCount = rockstarCatalogueSize(halos.size());
                            Vector3<double> origin;
                            for (int dim = 0; dim < kDimensions; ++dim) {
                                origin[dim] = particles.front().position[dim].toDouble();
                            }
                            for (int i = 0; i < haloCount; ++i) {
                                auto const& halo = halos[i];
                                auto entry = rockstarMakeGroupEntry(halos, i, scaleFactor, origin);
                                entry.id = halo.catalogueId;
                                for (int particleIndex = 0; particleIndex < halo.parts.size();
                                    ++particleIndex) {
                                    auto const& particle = halo.parts[particleIndex];
                                    tags[particle.rank][particle.index] = halo.catalogueId;
                                }
                                double virialRadius = 0.0;
                                auto record = rockstarMakeRecord(
                                    halos, i, scaleFactor, origin, true, true, &virialRadius);
                                HaloAperture aperture;
                                aperture.radius = virialRadius;
                                aperture.rank = hpxRank();
                                Vector3<double> peak;
                                peak[0] = halo.x;
                                peak[1] = halo.y;
                                peak[2] = halo.z;
                                for (int dim = 0; dim < kDimensions; ++dim) {
                                    double const position = origin[dim] + peak[dim];
                                    aperture.center[dim] = position - std::floor(position);
                                }
                                record.id.fofId = candidate.first;
                                record.id.haloId = i;
                                int const parent =
                                    halo.parent >= 0 && size_t(halo.parent) < haloCount ?
                                    halo.parent :
                                    -1;
                                record.pid = parent;
                                GroupInteger const host =
                                    parent >= 0 ? halos[halo.parent].catalogueId : kNoParticleGroup;
                                std::lock_guard<SpinlockType> lock(mutex);
                                groups.push_back(std::move(entry));
                                rockstarRecords.push_back(record);
                                haloApertures.push_back(aperture);
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
            },
            "analysis.groups.reduceHalos")));
    }
    hpx::wait_all(futures.begin(), futures.end());
    for (auto& future : futures) {
        future.get();
    }
    groupData = decltype(groupData)();
}

Vector<HaloAperture> groupsGatherApertures() {
    Vector<hpx::future<Vector<HaloAperture>>> futures;
    for (auto const& child : hpxChildren()) {
        futures.push_back(hpx::async<GroupsGatherAperturesAction>(child));
    }
    // Each rank's apertures stay contiguous and in record order, so the owner
    // can install the reduced masses by position.
    Vector<HaloAperture> apertures = haloApertures;
    for (auto& future : futures) {
        auto const part = future.get();
        apertures.insert(apertures.end(), part.begin(), part.end());
    }
    return apertures;
}

Vector<Pair<double>> groupsSumBaryons(Vector<HaloAperture> apertures) {
    Vector<hpx::future<Vector<Pair<double>>>> futures;
    for (auto const& child : hpxChildren()) {
        futures.push_back(hpx::async<GroupsSumBaryonsAction>(child, apertures));
    }
    size_t const count = apertures.size();
    // Equal per-species particle masses make these sums order independent.
    std::vector<std::atomic<double>> gas(count);
    std::vector<std::atomic<double>> star(count);
    double maxRadius = 0.0;
    for (auto const& aperture : apertures) {
        maxRadius = std::max(maxRadius, aperture.radius);
    }
    if (getOptions().hydro.enabled && maxRadius > 0.0) {
        // Bucket halo centres on a periodic grid whose cells are at least one
        // maximum radius wide, so a particle's 27 neighbouring cells hold every
        // aperture that can contain it. Fewer than three cells per dimension
        // would alias the neighbourhood; use a single cell instead.
        int cells = std::clamp(int(1.0 / maxRadius), 1, 256);
        if (cells < 3) {
            cells = 1;
        }
        auto const cellOf = [cells](double coordinate) {
            return std::clamp(int(coordinate * cells), 0, cells - 1);
        };
        auto const cellIndex = [cells](int i, int j, int k) { return (i * cells + j) * cells + k; };
        std::vector<Vector<int>> grid(size_t(cells) * cells * cells);
        for (size_t i = 0; i < count; ++i) {
            auto const& center = apertures[i].center;
            grid[cellIndex(cellOf(center[0]), cellOf(center[1]), cellOf(center[2]))].push_back(i);
        }
        int const neighbours = cells == 1 ? 0 : 1;
        int const numberOfThreads = hpxHardwareConcurrency();
        Vector<hpx::future<void>> workers;
        for (int proc = 0; proc < numberOfThreads; ++proc) {
            workers.push_back(hpx::async(hpx::annotated_function(
                [proc, numberOfThreads, cells, neighbours, &cellOf, &cellIndex, &grid, &apertures,
                    &gas, &star]() {
                    ParticleInteger const begin = (size_t) proc * particlesSize() / numberOfThreads;
                    ParticleInteger const end =
                        (size_t) (proc + 1) * particlesSize() / numberOfThreads;
                    for (ParticleInteger i = begin; i < end; ++i) {
                        int const type = particlesType(i);
                        if (type == kDarkMatterType) {
                            continue;
                        }
                        auto& species = type == kStarType ? star : gas;
                        double const mass = particlesMass(i);
                        Vector3<double> position;
                        Array<int, kDimensions> cell;
                        for (int dim = 0; dim < kDimensions; ++dim) {
                            position[dim] = particlesPos(dim, i).toDouble();
                            cell[dim] = cellOf(position[dim]);
                        }
                        for (int di = -neighbours; di <= neighbours; ++di) {
                            for (int dj = -neighbours; dj <= neighbours; ++dj) {
                                for (int dk = -neighbours; dk <= neighbours; ++dk) {
                                    auto const& bucket =
                                        grid[cellIndex((cell[0] + di + cells) % cells,
                                            (cell[1] + dj + cells) % cells,
                                            (cell[2] + dk + cells) % cells)];
                                    for (int const index : bucket) {
                                        auto const& aperture = apertures[index];
                                        double r2 = 0.0;
                                        for (int dim = 0; dim < kDimensions; ++dim) {
                                            // Minimum image in the unit periodic box.
                                            double dx = position[dim] - aperture.center[dim];
                                            dx -= std::round(dx);
                                            r2 += dx * dx;
                                        }
                                        if (r2 <= aperture.radius * aperture.radius) {
                                            species[index].fetch_add(mass);
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                "analysis.groups.sumBaryons")));
        }
        hpx::wait_all(workers.begin(), workers.end());
        for (auto& worker : workers) {
            worker.get();
        }
    }
    Vector<Pair<double>> masses(count);
    for (size_t i = 0; i < count; ++i) {
        masses[i] = {gas[i].load(), star[i].load()};
    }
    for (auto& future : futures) {
        auto const part = future.get();
        ALWAYS_ASSERT(part.size() == count);
        for (size_t i = 0; i < count; ++i) {
            masses[i].first += part[i].first;
            masses[i].second += part[i].second;
        }
    }
    return masses;
}

void groupsInstallBaryons(Vector<Pair<double>> masses) {
    ALWAYS_ASSERT(masses.size() == haloApertures.size());
    ALWAYS_ASSERT(rockstarRecords.size() == haloApertures.size());
    ALWAYS_ASSERT(groups.size() == haloApertures.size());
    // Records are in solar masses/h like mvir; GroupEntry masses are solar masses.
    double const recordConversion = rockstarMassConversion();
    double const entryConversion = getOptions().units.massG / constants::kM0;
    for (size_t i = 0; i < masses.size(); ++i) {
        rockstarRecords[i].mGas = masses[i].first * recordConversion;
        rockstarRecords[i].mStar = masses[i].second * recordConversion;
        groups[i].mGas = masses[i].first * entryConversion;
        groups[i].mStar = masses[i].second * entryConversion;
    }
}

void groupsMeasureBaryons() {
    // SOAP-style aperture masses (Mondal et al. 2025, MNRAS 543, 1339): every
    // gas and star particle within rvir counts, bound or not, while the FOF,
    // 6D linking and unbinding above stay dark-matter-only.
    if (!getOptions().hydro.enabled) {
        return;
    }
    ALWAYS_ASSERT(hpxRank() == 0);
    auto const apertures = []() {
        profile::ScopedPhase profilePhase("groups.baryons.gather");
        return groupsGatherApertures();
    }();
    auto const masses = [&]() {
        profile::ScopedPhase profilePhase("groups.baryons.measure");
        return groupsSumBaryons(apertures);
    }();
    profile::ScopedPhase profilePhase("groups.baryons.install");
    Vector<Vector<Pair<double>>> perRank(hpxSize());
    for (size_t i = 0; i < apertures.size(); ++i) {
        perRank[apertures[i].rank].push_back(masses[i]);
    }
    Vector<hpx::future<void>> futures;
    for (int rank = 0; rank < hpxSize(); ++rank) {
        futures.push_back(hpx::async<GroupsInstallBaryonsAction>(
            hpxLocalities()[rank], std::move(perRank[rank])));
    }
    hpx::wait_all(futures.begin(), futures.end());
    for (auto& future : futures) {
        future.get();
    }
}

void groupsTransmitParticles(Vector<std::pair<GroupInteger, Vector<ParticleData>>> entries) {
    Vector<hpx::future<void>> futs;
    int const numberOfThreads = hpxHardwareConcurrency();
    for (int proc = 0; proc < numberOfThreads; proc++) {
        futs.push_back(hpx::async(hpx::annotated_function(
            [numberOfThreads, proc, &entries]() {
                for (int i = proc; i < entries.size(); i += numberOfThreads) {
                    auto const grp = entries[i].first;
                    int const bin = (grp / kGroupWaves) % kGroupTableSize;
                    std::lock_guard<SpinlockType> lock(mutexes[bin]);
                    auto& entry = groupData[bin][grp];
                    entry.insert(entry.end(), entries[i].second.begin(), entries[i].second.end());
                }
            },
            "analysis.groups.receiveParticles")));
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
        futs.push_back(hpx::async(hpx::annotated_function(
            [proc, numberOfThreads, wave, &mutex, ainv]() {
                Vector<hpx::future<void>> futs;
                std::unordered_map<int, std::unordered_map<GroupInteger, Vector<ParticleData>>>
                    localGroups;
                ParticleInteger const begin = (size_t) proc * particlesSize() / numberOfThreads;
                ParticleInteger const end = (size_t) (proc + 1) * particlesSize() / numberOfThreads;
                for (ParticleInteger i = begin; i < end; i++) {
                    auto const grp = (GroupInteger) particlesGroup(i);
                    // The group tree excludes gas and stars; keep the catalogue
                    // dark-matter-only even for tags set outside the tree.
                    if (grp != kNoParticleGroup && particlesType(i) == kDarkMatterType) {
                        int const home = particlesGroupHome(grp);
                        if (grp % kGroupWaves == wave) {
                            if (localGroups[home].find(grp) == localGroups[home].end()) {
                                std::lock_guard<SpinlockType> lock(mutex);
                                localExistingGroups.push_back(grp);
                            }
                            auto& entry = localGroups[home][grp];
                            ParticleData part;
                            for (int dim = 0; dim < kDimensions; dim++) {
                                part.position[dim] = particlesPos(dim, i);
                                part.velocity[dim] = particlesVel(dim, i) * ainv;
                            }
                            part.index = i;
                            part.lastGroup = particlesLastgroup(i);
                            part.rank = hpxRank();
                            part.identity = particlesGroupInit(i);
                            auto const previousHost = previousHaloHosts.find(part.lastGroup);
                            part.lastHost = previousHost == previousHaloHosts.end() ?
                                kNoParticleGroup :
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
            },
            "analysis.groups.collectParticles")));
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

Vector<RockstarRecord> const& groupsLocalRecords() {
    return rockstarRecords;
}

Vector<HaloAperture> const& groupsLocalApertures() {
    return haloApertures;
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
