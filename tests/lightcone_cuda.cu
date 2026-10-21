/* CosmicTiger - bounded CPU/CUDA lightcone parity; BSL-1.0. */

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

#include <cosmictiger/analysis/lightcone.hpp>
#include "cuda_test.hpp"

void cosmictigerAssert(char const* expression, bool condition, char const* file, int line) {
    if (!condition) {
        std::fprintf(stderr, "%s:%d: %s\n", file, line, expression);
        std::abort();
    }
}

CUDA_EXPORT void* cudaAllocate(size_t bytes) {
#ifdef __CUDA_ARCH__
    __trap();
    return nullptr;
#else
    void* pointer = nullptr;
    if (cudaMallocManaged(&pointer, bytes) != cudaSuccess) {
        throw std::bad_alloc();
    }
    return pointer;
#endif
}

CUDA_EXPORT void cudaRelease(void* pointer) {
#ifdef __CUDA_ARCH__
    __trap();
#else
    if (pointer != nullptr) {
        cudaFree(pointer);
    }
#endif
}

CUDA_EXPORT void copyCudaMemory(void* destination, void* source, int bytes) {
    if (bytes != 0) {
        memcpy(destination, source, bytes);
    }
}

int cudaSmpCount() {
    // The tested launcher always receives an explicit four-block limit.
    throw std::runtime_error("unbounded lightcone launch requested by parity test");
}

namespace detail {

using Position = std::array<double, NDIM>;

struct Fixture {
    LightconeParticleMapType particles;
    LightconeTreeMapType trees;
    DeviceVector<LightconeTreeId> leaves;
    LightconeGroup nextGroup = 100;

    void addLeaf(int pixel, std::vector<Position> const& positions,
        std::vector<LightconeGroup> const& labels = {}) {
        auto& parts = particles[pixel];
        LightconeTreeNode node;
        node.pix = pixel;
        node.active = 1;
        node.lastActive = 0;
        node.particleRange.first = parts.group.size();
        node.particleRange.second = node.particleRange.first + positions.size();
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
        leaves.pushBack(leaf);
    }

    void initialize() {
        addLeaf(0, {{0.0, 0.0, 0.0}, {0.05, 0.0, 0.0}, {0.10, 0.0, 0.0}});
        addLeaf(0, {{0.15, 0.0, 0.0}, {0.20, 0.0, 0.0}, {0.7, 0.0, 0.0}});
        addLeaf(1, {{0.25, 0.0, 0.0}, {0.30, 0.0, 0.0}, {0.35, 0.0, 0.0}},
            {LC_NO_GROUP, LC_NO_GROUP, LC_EDGE_GROUP});
        addLeaf(1, {{-0.5, 0.0, 0.0}, {-0.45, 0.0, 0.0}, {-0.40, 0.0, 0.0}},
            {LC_NO_GROUP, 23, LC_NO_GROUP});
        addLeaf(2, {{0.0, 0.0, 0.5}, {0.05, 0.0, 0.5}, {0.05, 0.0, 0.5}});
        addLeaf(2, {{-0.75, -0.5, 0.0}, {-0.6875, -0.5, 0.0}});
        for (auto const& leaf : leaves) {
            for (auto const& neighbor : leaves) {
                trees[leaf.pix][leaf.index].neighbors.pushBack(neighbor);
            }
        }
    }

    void converge(bool cuda) {
        Pair<int> ownedPixels;
        ownedPixels.first = 0;
        ownedPixels.second = 3;
        for (int iteration = 0; iteration < 100; iteration++) {
            prepareLightconeGroupIteration(leaves, trees, ownedPixels);
            size_t const changed = cuda ?
                cudaLightcone(leaves, &particles, &trees, &nextGroup, 0.0625, 1, 3, 4) :
                cpuLightcone(leaves, particles, trees, nextGroup, 0.0625, 1, 3);
            if (cuda && cudaGetLastError() != cudaSuccess) {
                throw std::runtime_error("lightcone CUDA kernel failed");
            }
            if (changed == 0) {
                return;
            }
        }
        throw std::runtime_error("lightcone labels failed to converge");
    }

    std::vector<LightconeGroup> labels() {
        std::vector<LightconeGroup> result;
        for (auto const& leaf : leaves) {
            auto const& node = trees[leaf.pix][leaf.index];
            for (int index = node.particleRange.first; index < node.particleRange.second; index++) {
                result.push_back(particles[leaf.pix].group[index]);
            }
        }
        return result;
    }
};

}    // namespace detail

using LightconeCuda = detail::CudaTest;

TEST_F(LightconeCuda, ConnectedComponentsMatchCpu) {
    try {
        auto* cpu = new (cudaAllocate(sizeof(detail::Fixture))) detail::Fixture;
        auto* gpu = new (cudaAllocate(sizeof(detail::Fixture))) detail::Fixture;
        cpu->initialize();
        gpu->initialize();
        cpu->converge(false);
        gpu->converge(true);
        auto const cpuLabels = cpu->labels();
        auto const gpuLabels = gpu->labels();
        for (size_t first = 0; first < cpuLabels.size(); first++) {
            if ((cpuLabels[first] == LC_NO_GROUP) != (gpuLabels[first] == LC_NO_GROUP) ||
                (cpuLabels[first] == LC_EDGE_GROUP) != (gpuLabels[first] == LC_EDGE_GROUP)) {
                throw std::runtime_error("CPU/CUDA singleton or edge labels differ");
            }
            for (size_t second = 0; second < cpuLabels.size(); second++) {
                if ((cpuLabels[first] == cpuLabels[second]) !=
                    (gpuLabels[first] == gpuLabels[second])) {
                    throw std::runtime_error("CPU/CUDA connected components differ");
                }
            }
        }
        cpu->~Fixture();
        gpu->~Fixture();
        cudaRelease(cpu);
        cudaRelease(gpu);
        std::puts("CPU/CUDA lightcone parity passed (four blocks of 32 threads)");
    } catch (std::exception const& error) {
        std::fprintf(stderr, "%s\n", error.what());
        FAIL() << error.what();
    }
}
