/* CosmicTiger - lightcone CPU connectivity regressions; BSL-1.0. */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <cosmictiger/analysis/lightcone.hpp>

void cosmictigerAssert(char const* expression, bool condition, char const* file, int line) {
    if (!condition) {
        throw std::runtime_error(
            std::string(file) + ":" + std::to_string(line) + ": " + expression);
    }
}

void* cudaAllocate(size_t bytes) {
    void* pointer = std::malloc(bytes);
    if (pointer == nullptr && bytes != 0) {
        throw std::bad_alloc();
    }
    return pointer;
}

void cudaRelease(void* pointer) {
    std::free(pointer);
}

void copyCudaMemory(void* destination, void* source, int bytes) {
    if (bytes != 0) {
        std::memcpy(destination, source, bytes);
    }
}

namespace detail {

using Position = std::array<double, NDIM>;

void require(bool condition, char const* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

struct Fixture {
    LightconeParticleMapType particles;
    LightconeTreeMapType trees;
    DeviceVector<LightconeTreeId> leaves;
    std::vector<LightconeTreeId> allLeaves;
    LightconeGroup nextGroup = 0;
    int rank = 0;
    int ranks = 1;
    Pair<int> ownedPixels;

    Fixture() {
        ownedPixels.first = 0;
        ownedPixels.second = 100;
    }

    LightconeTreeId addLeaf(int pixel, std::vector<Position> const& positions,
        std::vector<LightconeGroup> const& labels = {}, bool owned = true) {
        auto& parts = particles[pixel];
        LightconeTreeNode node;
        node.particleRange.first = parts.group.size();
        node.particleRange.second = node.particleRange.first + positions.size();
        node.pix = pixel;
        node.active = 1;
        node.lastActive = 0;
        for (int dim = 0; dim < NDIM; dim++) {
            node.box.begin[dim] = std::numeric_limits<double>::max();
            node.box.end[dim] = -std::numeric_limits<double>::max();
        }
        for (size_t index = 0; index < positions.size(); index++) {
            for (int dim = 0; dim < NDIM; dim++) {
                LightconeReal const position = positions[index][dim];
                parts.pos[dim].pushBack(position);
                node.box.begin[dim] = std::min(node.box.begin[dim], position.toDouble());
                node.box.end[dim] = std::max(node.box.end[dim], position.toDouble());
            }
            parts.group.pushBack(labels.empty() ? LC_NO_GROUP : labels[index]);
        }
        LightconeTreeId const leaf{pixel, trees[pixel].size()};
        trees[pixel].pushBack(std::move(node));
        allLeaves.push_back(leaf);
        if (owned) {
            leaves.pushBack(leaf);
        }
        return leaf;
    }

    void connectLeaves() {
        for (auto const& leaf : allLeaves) {
            auto& neighbors = trees[leaf.pix][leaf.index].neighbors;
            for (auto const& neighbor : allLeaves) {
                neighbors.pushBack(neighbor);
            }
        }
    }

    size_t step(double linkingLength) {
        prepareLightconeGroupIteration(leaves, trees, ownedPixels);
        return cpuLightcone(leaves, particles, trees, nextGroup, linkingLength, rank, ranks);
    }

    void converge(double linkingLength) {
        for (int iteration = 0; iteration < 100; iteration++) {
            if (step(linkingLength) == 0) {
                return;
            }
        }
        throw std::runtime_error("lightcone labels failed to converge");
    }

    std::vector<LightconeGroup> labels() {
        std::vector<LightconeGroup> result;
        for (auto const& leaf : allLeaves) {
            auto const& node = trees[leaf.pix][leaf.index];
            for (int index = node.particleRange.first; index < node.particleRange.second; index++) {
                result.push_back(particles[leaf.pix].group[index]);
            }
        }
        return result;
    }

    std::vector<Position> positions() {
        std::vector<Position> result;
        for (auto const& leaf : allLeaves) {
            auto const& node = trees[leaf.pix][leaf.index];
            for (int index = node.particleRange.first; index < node.particleRange.second; index++) {
                Position position;
                for (int dim = 0; dim < NDIM; dim++) {
                    position[dim] = particles[leaf.pix].pos[dim][index].toDouble();
                }
                result.push_back(position);
            }
        }
        return result;
    }
};

void checkOracle(Fixture& fixture, double linkingLength) {
    auto const positions = fixture.positions();
    auto const original = fixture.labels();
    size_t const count = positions.size();
    // Independent dense reachability oracle, using double Cartesian distances.
    std::vector<std::vector<bool>> connected(count, std::vector<bool>(count, false));
    for (size_t first = 0; first < count; first++) {
        connected[first][first] = true;
        for (size_t second = first + 1; second < count; second++) {
            double distanceSquared = 0.0;
            for (int dim = 0; dim < NDIM; dim++) {
                double const delta = positions[first][dim] - positions[second][dim];
                distanceSquared += delta * delta;
            }
            connected[first][second] = connected[second][first] =
                distanceSquared < linkingLength * linkingLength;
        }
    }
    for (size_t via = 0; via < count; via++) {
        for (size_t first = 0; first < count; first++) {
            for (size_t second = 0; second < count; second++) {
                connected[first][second] =
                    connected[first][second] || (connected[first][via] && connected[via][second]);
            }
        }
    }
    fixture.converge(linkingLength);
    auto const actual = fixture.labels();
    for (size_t first = 0; first < count; first++) {
        LightconeGroup expected = LC_NO_GROUP;
        size_t members = 0;
        for (size_t second = 0; second < count; second++) {
            if (connected[first][second]) {
                members++;
                expected = std::min(expected, original[second]);
                require(actual[first] == actual[second], "connected particles share a label");
            } else if (actual[first] != LC_NO_GROUP) {
                require(actual[first] != actual[second],
                    "disconnected components have distinct labels");
            }
        }
        if (members == 1 || expected != LC_NO_GROUP) {
            require(actual[first] == expected, "singleton or pre-existing minimum label preserved");
        } else {
            require(actual[first] != LC_NO_GROUP && actual[first] != LC_EDGE_GROUP,
                "unlabelled linked components receive a group");
            require((actual[first] - 1) % fixture.ranks == fixture.rank,
                "new IDs retain the locality residue");
        }
    }
    require(fixture.step(linkingLength) == 0, "converged iteration is idle");
}

void testComponents() {
    Fixture fixture;
    fixture.rank = 2;
    fixture.ranks = 5;
    fixture.addLeaf(0, {{0.15, 0.0, 0.0}, {0.05, 0.0, 0.0}, {0.10, 0.0, 0.0}});
    fixture.addLeaf(0, {{0.20, 0.0, 0.0}, {0.25, 0.0, 0.0}});
    fixture.addLeaf(1, {{0.30, 0.0, 0.0}, {0.35, 0.0, 0.0}});
    fixture.addLeaf(1, {{-0.5, 0.0, 0.0}, {-0.45, 0.0, 0.0}});
    fixture.addLeaf(7, {{0.75, 0.75, 0.75}});
    fixture.addLeaf(7, {});
    fixture.connectLeaves();
    checkOracle(fixture, 0.0625);
}

void testExistingLabels() {
    Fixture fixture;
    fixture.nextGroup = 1000;
    fixture.addLeaf(0, {{0.0, 0.0, 0.0}, {0.05, 0.0, 0.0}}, {91, LC_NO_GROUP});
    fixture.addLeaf(1, {{0.10, 0.0, 0.0}, {0.15, 0.0, 0.0}}, {17, LC_NO_GROUP});
    fixture.addLeaf(1, {{-0.5, 0.0, 0.0}, {-0.45, 0.0, 0.0}}, {LC_EDGE_GROUP, 14});
    fixture.addLeaf(2, {{-0.40, 0.0, 0.0}}, {LC_NO_GROUP});
    fixture.addLeaf(2, {{0.75, 0.75, 0.75}}, {129});
    fixture.connectLeaves();
    checkOracle(fixture, 0.0625);
}

void testThresholdAndDuplicates() {
    Fixture fixture;
    fixture.addLeaf(0, {{0.0, 0.0, 0.0}, {0.125, 0.0, 0.0}});
    fixture.addLeaf(1, {{0.5, 0.0, 0.0}, {0.5, 0.0, 0.0}, {0.5625, 0.0, 0.0}});
    fixture.connectLeaves();
    checkOracle(fixture, 0.125);
    require(fixture.particles[0].group[0] == LC_NO_GROUP &&
            fixture.particles[0].group[1] == LC_NO_GROUP,
        "exact linking-length separation is excluded");
    Fixture zero;
    zero.addLeaf(0, {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}});
    zero.connectLeaves();
    checkOracle(zero, 0.0);
}

void testRemoteActivity() {
    Fixture fixture;
    fixture.ownedPixels.first = 0;
    fixture.ownedPixels.second = 1;
    fixture.rank = 0;
    fixture.ranks = 2;
    fixture.nextGroup = 100;
    fixture.addLeaf(0, {{0.0, 0.0, 0.0}}, {101});
    fixture.addLeaf(1, {{0.05, 0.0, 0.0}}, {2}, false);
    fixture.connectLeaves();
    fixture.trees[0][0].active = 0;
    fixture.trees[1][0].lastActive = 0;
    require(fixture.step(0.0625) == 0, "inactive leaves skip unchanged boundaries");
    fixture.trees[1][0].lastActive = 1;
    require(fixture.step(0.0625) == 1, "remote activity wakes an inactive local leaf");
    require(fixture.particles[0].group[0] == 2, "remote minimum label propagates locally");
    require(fixture.particles[1].group[0] == 2, "ghost labels remain read-only");
}

void testDistributedChain() {
    constexpr int kRanks = 3;
    std::array<Fixture, kRanks> fixtures;
    for (int rank = 0; rank < kRanks; rank++) {
        auto& fixture = fixtures[rank];
        fixture.rank = rank;
        fixture.ranks = kRanks;
        fixture.nextGroup = 100;
        fixture.ownedPixels.first = rank;
        fixture.ownedPixels.second = rank + 1;
        for (int pixel = 0; pixel < kRanks; pixel++) {
            fixture.addLeaf(pixel, {{pixel * 0.05, 0.0, 0.0}},
                {pixel == 2 ? LC_EDGE_GROUP : LC_NO_GROUP}, pixel == rank);
        }
        fixture.connectLeaves();
    }
    bool converged = false;
    for (int iteration = 0; iteration < 20; iteration++) {
        // Exchange one consistent snapshot before any locality starts its sweep.
        for (int rank = 0; rank < kRanks; rank++) {
            for (int remote = 0; remote < kRanks; remote++) {
                if (remote != rank) {
                    fixtures[rank].particles[remote].group[0] =
                        fixtures[remote].particles[remote].group[0];
                    fixtures[rank].trees[remote][0].lastActive =
                        fixtures[remote].trees[remote][0].active;
                }
            }
        }
        size_t changed = 0;
        for (auto& fixture : fixtures) {
            changed += fixture.step(0.0625);
        }
        if (changed == 0) {
            converged = true;
            break;
        }
    }
    require(converged, "distributed chain converges");
    for (int rank = 0; rank < kRanks; rank++) {
        require(fixtures[rank].particles[rank].group[0] == LC_EDGE_GROUP,
            "edge label crosses multiple locality boundaries");
    }
}

void testDeterministicCloud() {
    Fixture fixture;
    unsigned state = 7;
    for (int leaf = 0; leaf < 8; leaf++) {
        std::vector<Position> positions;
        for (int index = 0; index < 6; index++) {
            Position position;
            for (int dim = 0; dim < NDIM; dim++) {
                state = state * 1664525u + 1013904223u;
                position[dim] = (static_cast<double>((state >> 16) % 81) - 40.0) / 128.0;
            }
            positions.push_back(position);
        }
        fixture.addLeaf(leaf / 2, positions);
    }
    fixture.connectLeaves();
    checkOracle(fixture, 0.15625);
}

}    // namespace detail

int main() {
    try {
        detail::testComponents();
        detail::testExistingLabels();
        detail::testThresholdAndDuplicates();
        detail::testRemoteActivity();
        detail::testDistributedChain();
        detail::testDeterministicCloud();
        std::puts("CPU lightcone grouping regressions passed");
    } catch (std::exception const& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    return 0;
}
