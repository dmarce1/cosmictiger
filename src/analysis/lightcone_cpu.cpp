/* CosmicTiger - CPU lightcone friends-of-friends; BSL-1.0. */

#include <algorithm>
#include <numeric>
#include <vector>

#include <cosmictiger/analysis/lightcone.hpp>

namespace detail {

bool lightconeParticlesLinked(LightconeParticles const& first, int firstIndex,
    LightconeParticles const& second, int secondIndex, float linkingLengthSquared) {
    float const dx = distance(first.pos[XDIM][firstIndex], second.pos[XDIM][secondIndex]);
    float const dy = distance(first.pos[YDIM][firstIndex], second.pos[YDIM][secondIndex]);
    float const dz = distance(first.pos[ZDIM][firstIndex], second.pos[ZDIM][secondIndex]);
    return sqr(dx, dy, dz) < linkingLengthSquared;
}

int lightconeComponentRoot(std::vector<int>& parents, int index) {
    while (parents[index] != index) {
        parents[index] = parents[parents[index]];
        index = parents[index];
    }
    return index;
}

}    // namespace detail

void prepareLightconeGroupIteration(DeviceVector<LightconeTreeId> const& leaves,
    LightconeTreeMapType& treeMap, Pair<int> ownedPixels) {
    for (auto const& leaf : leaves) {
        auto& self = treeMap[leaf.pix][leaf.index];
        self.lastActive = self.active;
        // Boundary exchange stores the owner's previous activity in ghost lastActive.
        // A remote label can change after this leaf has already become inactive.
        for (auto const& neighbor : self.neighbors) {
            if (neighbor.pix < ownedPixels.first || neighbor.pix >= ownedPixels.second) {
                self.lastActive |= treeMap[neighbor.pix][neighbor.index].lastActive;
            }
        }
        self.active = 0;
    }
}

size_t cpuLightcone(DeviceVector<LightconeTreeId> const& leaves,
    LightconeParticleMapType& particleMap, LightconeTreeMapType& treeMap, LightconeGroup& nextGroup,
    double linkingLength, int rank, int ranks) {
    float const linkingLengthSquared = sqr(linkingLength);
    auto const generateGroup = [&nextGroup, rank, ranks]() {
        LightconeGroup const group = nextGroup++ * ranks + rank + 1;
        ALWAYS_ASSERT(group != LC_EDGE_GROUP && group < LC_NO_GROUP);
        return group;
    };
    size_t changedLeaves = 0;
    // Leaves share neighbor labels and activity flags. A single host task owns
    // this sweep, avoiding unsynchronized reads while other leaves write labels.
    for (auto const& leaf : leaves) {
        auto& self = treeMap[leaf.pix][leaf.index];
        if (!self.lastActive) {
            continue;
        }
        auto& particles = particleMap[self.pix];
        bool changed = false;
        auto const box = self.box.pad(1.01 * linkingLength);
        for (auto const& neighbor : self.neighbors) {
            if (!(leaf != neighbor)) {
                continue;
            }
            auto const& other = treeMap[neighbor.pix][neighbor.index];
            auto const& otherParticles = particleMap[other.pix];
            for (int otherIndex = other.particleRange.first;
                 otherIndex < other.particleRange.second; otherIndex++) {
                Array<double, NDIM> position;
                for (int dim = 0; dim < NDIM; dim++) {
                    position[dim] = otherParticles.pos[dim][otherIndex].toDouble();
                }
                if (!box.contains(position)) {
                    continue;
                }
                LightconeGroup const otherGroup = otherParticles.group[otherIndex];
                for (int index = self.particleRange.first; index < self.particleRange.second;
                     index++) {
                    if (detail::lightconeParticlesLinked(
                            particles, index, otherParticles, otherIndex, linkingLengthSquared)) {
                        auto& group = particles.group[index];
                        if (group == LC_NO_GROUP) {
                            group = otherGroup == LC_NO_GROUP ? generateGroup() : otherGroup;
                            changed = true;
                        } else if (otherGroup < group) {
                            group = otherGroup;
                            changed = true;
                        }
                    }
                }
            }
        }

        // Close all links inside the leaf in one union-find pass. Components
        // without links retain LC_NO_GROUP unless they already carry a label.
        int const begin = self.particleRange.first;
        int const count = self.particleRange.second - begin;
        std::vector<int> parents(count);
        std::iota(parents.begin(), parents.end(), 0);
        for (int first = 0; first < count; first++) {
            for (int second = first + 1; second < count; second++) {
                if (detail::lightconeParticlesLinked(particles, begin + first, particles,
                        begin + second, linkingLengthSquared)) {
                    int const firstRoot = detail::lightconeComponentRoot(parents, first);
                    int const secondRoot = detail::lightconeComponentRoot(parents, second);
                    parents[secondRoot] = firstRoot;
                }
            }
        }
        std::vector<LightconeGroup> labels(count, LC_NO_GROUP);
        std::vector<int> sizes(count, 0);
        for (int index = 0; index < count; index++) {
            int const root = detail::lightconeComponentRoot(parents, index);
            labels[root] = std::min(labels[root], particles.group[begin + index]);
            sizes[root]++;
        }
        for (int index = 0; index < count; index++) {
            if (sizes[index] > 1 && labels[index] == LC_NO_GROUP) {
                labels[index] = generateGroup();
            }
        }
        for (int index = 0; index < count; index++) {
            auto& group = particles.group[begin + index];
            auto const label = labels[detail::lightconeComponentRoot(parents, index)];
            if (label != group) {
                group = label;
                changed = true;
            }
        }
        if (changed) {
            self.active = 1;
            for (auto const& neighbor : self.neighbors) {
                treeMap[neighbor.pix][neighbor.index].active = 1;
            }
            changedLeaves++;
        }
    }
    return changedLeaves;
}
