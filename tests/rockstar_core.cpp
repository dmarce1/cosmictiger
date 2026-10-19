#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include <cosmictiger/analysis/rockstar_core.hpp>

namespace {
struct Box {
    std::array<float, 6> begin{};
    std::array<float, 6> end{};
};

struct Particle {
    std::array<float, 6> xValue{};
    int subgroup;
};

void require(bool condition, char const* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void checkConnectivity(std::vector<Particle> particles, float linkingLength) {
    // Independent union-find oracle for the geometric connectivity graph.
    std::vector<int> parents(particles.size());
    std::iota(parents.begin(), parents.end(), 0);
    auto const root = [&parents](int index) {
        while (parents[index] != index) {
            index = parents[index];
        }
        return index;
    };
    for (int i = 0; i < int(particles.size()); i++) {
        particles[i].subgroup = i;
        for (int j = 0; j < i; j++) {
            float distanceSquared = 0.0f;
            for (int dim = 0; dim < 6; dim++) {
                float const delta = particles[i].xValue[dim] - particles[j].xValue[dim];
                distanceSquared += delta * delta;
            }
            if (distanceSquared <= linkingLength * linkingLength) {
                parents[root(i)] = root(j);
            }
        }
    }

    // Exercise the production CUDA geometry/label helper on the host. This
    // checks connectivity, including partial leaves, without claiming to
    // validate CUDA atomics or scheduling on a device.
    bool changed;
    int passes = 0;
    do {
        changed = false;
        for (int i = int(particles.size()) - 1; i >= 0; i--) {
            int label = particles[i].subgroup;
            for (int begin = 0; begin < int(particles.size()); begin += 32) {
                int const end = std::min(begin + 32, int(particles.size()));
                label = rockstar::detail::minimumLinkedLabel(particles.data(), i, begin, end,
                    linkingLength * linkingLength, label, [](int const* value) { return *value; });
            }
            changed |= label != particles[i].subgroup;
            particles[i].subgroup = label;
        }
        require(++passes <= int(particles.size()) + 1, "label propagation did not converge");
    } while (changed);

    for (int i = 0; i < int(particles.size()); i++) {
        for (int j = 0; j < int(particles.size()); j++) {
            require((root(i) == root(j)) == (particles[i].subgroup == particles[j].subgroup),
                "production label propagation disagrees with geometric connectivity");
        }
    }
}
}    // namespace

int main() {
    try {
        Box planar{{0, 0, 0, 0, 0, 0}, {1, 1, 0, 1, 1, 1}};
        require(rockstar::detail::boxesOverlap(planar, planar, 6), "zero-width box rejected");
        require(rockstar::detail::boxContains(planar, planar.end, 6), "closed upper face rejected");
        require(rockstar::detail::initialLinkingLength(planar, 20, 6) > 0.0f,
            "planar search radius did not advance");
        Box point;
        require(rockstar::detail::boxesOverlap(point, point, 6), "coincident points rejected");
        require(rockstar::detail::initialLinkingLength(point, 20, 6) == 0.0f,
            "coincident search should begin at zero distance");
        Box separatedVelocity = planar;
        separatedVelocity.begin[3] = 100.0f;
        separatedVelocity.end[3] = 101.0f;
        require(rockstar::detail::boxesOverlap(planar, separatedVelocity, 3),
            "spatial boxes incorrectly culled by velocity");
        require(!rockstar::detail::boxesOverlap(planar, separatedVelocity, 6),
            "phase-space boxes ignored velocity separation");
        require(rockstar::detail::boxContains(planar, separatedVelocity.begin, 3),
            "spatial point incorrectly culled by velocity");
        require(rockstar::detail::normalizationScale(0.0f) == 1.0f,
            "zero dispersion creates singular normalization");
        require(rockstar::detail::normalizationScale(4.0f) == 2.0f,
            "nonzero dispersion normalization changed");

        std::vector<Particle> laneRegression(16);
        laneRegression[0].xValue[0] = 10.0f;
        for (int i = 1; i < 16; i++) {
            laneRegression[i].xValue[0] = 0.01f * i;
        }
        checkConnectivity(laneRegression, 1.0f);

        std::vector<Particle> chain(67);
        for (int i = 0; i < int(chain.size()); i++) {
            chain[i].xValue[0] = i;
        }
        checkConnectivity(chain, 1.0f);
        checkConnectivity(std::vector<Particle>(65), 0.0f);

        std::mt19937 random(42);
        std::uniform_real_distribution<float> coordinate(-1.0f, 1.0f);
        std::vector<Particle> cloud(95);
        for (auto& particle : cloud) {
            for (auto& value : particle.xValue) {
                value = coordinate(random);
            }
        }
        checkConnectivity(cloud, 1.2f);
        std::puts("ROCKSTAR core geometry and connectivity checks passed");
    } catch (std::exception const& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
