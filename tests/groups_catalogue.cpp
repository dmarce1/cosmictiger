/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2021  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

// Catalogue tests for the group finder, run from the main binary with
// --run.test=groupsCatalogue and --run.test=groupsCatalogueDistributed.

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <cosmictiger/analysis/groups.hpp>
#include <cosmictiger/analysis/rockstar.hpp>
#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/particles/particles.hpp>

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
            for (int dim = 0; dim < kDimensions; ++dim) {
                particlesPos(dim, i) = 0.5 + 0.2 * support * distribution(generator);
                particlesVel(dim, i) = speed * distribution(generator);
            }
            particlesLastgroup(i) = kNoParticleGroup;
        }
        Vector<GroupInteger> firstTags(kParticleCount, kNoParticleGroup);
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
            detail::groupsResetHistory();
            groupsPrepareHistory(snapshotNumber);
            if (snapshot == 2) {
                require(detail::groupsPreviousHaloHosts().size() == firstHaloCount,
                    "Saved halo host map was not restored");
            }
            particlesGroupsInit();
            for (int i = 0; i < kHaloParticles; ++i) {
                particlesGroup(i) = particlesGroupInit(0);
            }
            particlesGroup(kHaloParticles) = particlesGroupInit(kHaloParticles);
            for (int wave = 0; wave < kGroupWaves; ++wave) {
                groupsAddParticles(wave, 1.0);
                if (snapshot == 2) {
                    detail::groupsForEachCandidate(
                        [&](GroupInteger, Vector<ParticleData> const& members) {
                            auto const& hosts = detail::groupsPreviousHaloHosts();
                            for (auto const& particle : members) {
                                require(particle.lastGroup == firstTags[particle.index],
                                    "Previous halo tags did not reach the finder");
                                require(particle.identity == particlesGroupInit(particle.index),
                                    "Particle identity missing from distributed group input");
                                auto const host = hosts.find(particle.lastGroup);
                                require(particle.lastHost ==
                                        (host == hosts.end() ? kNoParticleGroup : host->second),
                                    "Previous host missing from finder input");
                            }
                        });
                }
                groupsReduce(1.0);
            }
            // This invalid original FOF tag has no queued halo assignment; it
            // specifically verifies that culling visits the final partition.
            particlesGroup(kParticleCount - 1) = particlesGroupInit(kParticleCount - 1);
            groupsCull();
            require(!detail::groupsLocalEntries().empty(),
                "Bound synthetic group produced an empty catalogue");
            require(particlesGroup(kHaloParticles) == kNoParticleGroup,
                "Undersized FOF group retained a particle tag");
            require(particlesGroup(kParticleCount - 1) == kNoParticleGroup,
                "FOF culling skipped the last particle partition");
            std::unordered_map<GroupInteger, std::unordered_map<GroupInteger, int>>
                expectedOverlaps;
            for (int i = 0; i < kParticleCount; ++i) {
                GroupInteger const current = particlesGroup(i);
                if (current != kNoParticleGroup && firstTags[i] != kNoParticleGroup) {
                    ++expectedOverlaps[current][firstTags[i]];
                }
            }
            std::unordered_map<GroupInteger, std::pair<GroupInteger, int>> expectedDescendants;
            for (auto const& previous : detail::groupsPreviousHaloHosts()) {
                expectedDescendants.emplace(previous.first, std::make_pair(kNoParticleGroup, 0));
            }
            std::unordered_set<GroupInteger> identifiers;
            for (auto const& group : detail::groupsLocalEntries()) {
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
                if (tag != kNoParticleGroup) {
                    require(identifiers.count(tag) != 0, "Particle points to an unwritten halo");
                    ++retained;
                }
                firstTags[i] = tag;
            }
            require(retained >= kRockstarMinGroup, "Synthetic halo lost all particle memberships");
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
        for (int dim = 0; dim < kDimensions; ++dim) {
            particlesPos(dim, i) = center + 0.2 * support * distribution(generator);
            particlesVel(dim, i) = speed * distribution(generator);
        }
        particlesLastgroup(i) = kNoParticleGroup;
    }
}
HPX_PLAIN_ACTION(groupsCatalogueDistributedSetup, GroupsCatalogueDistributedSetupAction);

void groupsCatalogueDistributedBegin(
    int snapshot, Vector<GroupInteger> previousTags, Vector<Pair<GroupInteger>> expectedHosts) {
    requireDistributedCatalogue(detail::groupsPreviousHaloHosts().size() == expectedHosts.size(),
        "Distributed host history was not installed on every locality");
    for (auto const& expected : expectedHosts) {
        auto const actual = detail::groupsPreviousHaloHosts().find(expected.first);
        requireDistributedCatalogue(
            actual != detail::groupsPreviousHaloHosts().end() && actual->second == expected.second,
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
            (i == 2 * kDistributedHaloHalf ? particlesGroupInit(i) : kNoParticleGroup);
    }
}
HPX_PLAIN_ACTION(groupsCatalogueDistributedBegin, GroupsCatalogueDistributedBeginAction);

int groupsCatalogueDistributedInspect(int snapshot) {
    int realInputs = 0;
    auto const& hosts = detail::groupsPreviousHaloHosts();
    detail::groupsForEachCandidate([&](GroupInteger const id, Vector<ParticleData> const& members) {
        requireDistributedCatalogue(
            particlesGroupHome(id) == hpxRank(), "FOF inputs reached the wrong analysis locality");
        bool const realHalo = id == 0 || id == kDistributedLocalCount + kDistributedHaloHalf;
        requireDistributedCatalogue(members.size() == (realHalo ? 128 : 1),
            "Cross-rank FOF particle collection is incomplete");
        int sourceCounts[2] = {0, 0};
        std::unordered_set<GroupInteger> identities;
        for (auto const& particle : members) {
            requireDistributedCatalogue(particle.rank >= 0 && particle.rank < 2 &&
                    particle.index >= 0 && particle.index < kDistributedLocalCount,
                "Distributed finder input lost its source address");
            GroupInteger const identity = particle.rank * kDistributedLocalCount + particle.index;
            requireDistributedCatalogue(
                particle.identity == identity && identities.insert(identity).second,
                "Distributed finder input duplicated or changed a particle identity");
            requireDistributedCatalogue(particle.lastGroup == distributedPreviousTags[identity],
                "Previous halo membership did not reach the remote finder");
            auto const host = hosts.find(particle.lastGroup);
            requireDistributedCatalogue(
                particle.lastHost == (host == hosts.end() ? kNoParticleGroup : host->second),
                "Previous host did not reach the remote finder");
            if (realHalo) {
                requireDistributedCatalogue(particle.index < 2 * kDistributedHaloHalf &&
                        id == distributedFofId(snapshot, particle.index),
                    "Distributed FOF contains particles from the other halo");
                ++sourceCounts[particle.rank];
                ++realInputs;
            }
        }
        if (realHalo) {
            requireDistributedCatalogue(
                sourceCounts[0] == kDistributedHaloHalf && sourceCounts[1] == kDistributedHaloHalf,
                "FOF input did not include both MPI ranks");
        }
    });
    return realInputs;
}
HPX_PLAIN_ACTION(groupsCatalogueDistributedInspect, GroupsCatalogueDistributedInspectAction);

Vector<GroupInteger> groupsCatalogueDistributedBeforeCull(int snapshot) {
    for (int i = 0; i < 2 * kDistributedHaloHalf; ++i) {
        requireDistributedCatalogue(particlesGroup(i) == distributedFofId(snapshot, i),
            "Halo assignments overwrote the FOF tags before distributed culling");
    }
    requireDistributedCatalogue(
        detail::groupsPendingAssignments().size() == kDistributedLocalCount - 1,
        "Remote halo assignments were not delivered to every source particle");
    std::unordered_set<ParticleInteger> indices;
    for (auto const& assignment : detail::groupsPendingAssignments()) {
        requireDistributedCatalogue(assignment.first >= 0 &&
                assignment.first < kDistributedLocalCount - 1 &&
                indices.insert(assignment.first).second,
            "Remote halo assignments contain invalid or duplicate particle indices");
    }
    Vector<GroupInteger> identifiers;
    for (auto const& group : detail::groupsLocalEntries()) {
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
    requireDistributedCatalogue(detail::groupsPendingAssignments().empty(),
        "Distributed membership updates were not consumed by culling");
    std::unordered_set<GroupInteger> valid(identifiers.begin(), identifiers.end());
    Vector<GroupInteger> tags;
    int retained[2] = {0, 0};
    for (int i = 0; i < kDistributedLocalCount; ++i) {
        GroupInteger const tag = particlesGroup(i);
        if (i >= 2 * kDistributedHaloHalf) {
            requireDistributedCatalogue(tag == kNoParticleGroup,
                "Distributed culling retained a rejected or unprocessed FOF particle");
        } else if (tag != kNoParticleGroup) {
            requireDistributedCatalogue(valid.count(tag) != 0,
                "Distributed particle membership refers to an unwritten halo");
            ++retained[i / kDistributedHaloHalf];
        }
        tags.push_back(tag);
    }
    requireDistributedCatalogue(
        retained[0] >= kRockstarMinGroup && retained[1] >= kRockstarMinGroup,
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
        Vector<GroupInteger> previousTags(2 * kDistributedLocalCount, kNoParticleGroup);
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
            for (int wave = 0; wave < kGroupWaves; ++wave) {
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
                if (tags[i] != kNoParticleGroup && previousTags[i] != kNoParticleGroup) {
                    ++expectedOverlaps[tags[i]][previousTags[i]];
                }
            }
            std::unordered_map<GroupInteger, std::pair<GroupInteger, int>> expectedDescendants;
            for (auto const& previous : previousHosts) {
                expectedDescendants.emplace(previous.first, std::make_pair(kNoParticleGroup, 0));
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
                            (host == kNoParticleGroup || uniqueIds.count(host) != 0),
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
