/* CosmicTiger - bounded production CUDA kick profiling regression; BSL-1.0. */

// Run the actual kick kernel without application initialization or an HPX runtime.
// Its isolated four-particle tree uses only direct softened pairs. The periodic
// policy below deliberately contributes zero; this is not an Ewald accuracy test.
// Every launch has at most two one-warp blocks, further bounded by SM_count / 6.
// No initialization, memset, or reduction kernels are launched by this fixture.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include <cosmictiger/core/device_profile.hpp>
#include <cosmictiger/gravity/kick_cuda_kernel.hpp>
#include <cosmictiger/math/kernel.hpp>

// These are the production direct-interaction diagnostic symbols, normally
// supplied by kick.cu. No particle, option, or force arithmetic is replaced.
__managed__ double iclose = 0.0;
__managed__ double idirect = 0.0;
__managed__ double numberOfProcessedSinkParticles = 0.0;
__managed__ bool useGravityCounters = false;

void cosmictigerAssert(char const* condition, bool value, char const* file, int line) {
    if (!value) {
        throw std::runtime_error(std::string(file) + ":" + std::to_string(line) + ": " + condition);
    }
}

#if COSMICTIGER_ENABLE_FMM
template <int order>
__device__ __forceinline__ FlopCount cudaApplyPeriodicPolicy(CudaKickData const&,
    KickParameters const&, cosmictiger::gravity::cartesian::tracefree::Expansion<float, order>&,
    TreeNode const&, StackVector<int>& checks, DeviceVector<int>&, DeviceVector<int>&,
    DeviceVector<int>&, DeviceVector<int>&) {
    // All positions occupy one small interior region, and the fixture asks for
    // isolated forces. Empty periodic checklists are an explicit precondition.
    ALWAYS_ASSERT(checks.size() == 0);
    return 0;
}
#endif

namespace detail {

constexpr int kOrder = kFmmDefaultOrder;
constexpr int kParticleCount = 4;
constexpr int kItemCount = 5;
constexpr int kOutputCount = kParticleCount * kItemCount;
constexpr float kSupport = 0.06f;
constexpr std::uint64_t kPoison = std::numeric_limits<std::uint64_t>::max();
using Output = std::array<std::array<float, kOutputCount>, kDimensions + 1>;
using PhaseCalls = std::array<std::uint64_t, profile::kDevicePhaseCount>;

void checkCuda(cudaError_t status, char const* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

void require(bool condition, char const* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <class Value>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count)
      : count_(count) {
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(Value)),
            "allocate fixture buffer");
    }

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    DeviceBuffer(DeviceBuffer const&) = delete;
    DeviceBuffer& operator=(DeviceBuffer const&) = delete;

    Value* data() const {
        return data_;
    }

    void upload(Value const* values) {
        checkCuda(cudaMemcpy(data_, values, count_ * sizeof(Value), cudaMemcpyHostToDevice),
            "upload fixture buffer");
    }

    void download(Value* values) const {
        checkCuda(cudaMemcpy(values, data_, count_ * sizeof(Value), cudaMemcpyDeviceToHost),
            "download fixture buffer");
    }

private:
    Value* data_ = nullptr;
    size_t count_;
};

constexpr int alignUp(int value, int alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

CudaKickArenaLayout makeLayout() {
    // The same region lifetimes as the production layout, with fixed small
    // capacities. Every vector and stack is backed by the reserved list slab;
    // no pool initialization or dynamic growth is needed for this fixture.
    CudaKickArenaLayout layout{};
    layout.order = kOrder;
    layout.controlOffset = 0;
    layout.expansionOffset = alignUp(int(sizeof(CudaKickControl<kOrder>)), 16);
    layout.arenaOffset = layout.expansionOffset +
        alignUp(alignUp(FmmOrder<kOrder>::kLocalExpansionBytes_, int(alignof(Position))) +
                kDimensions * int(sizeof(Position)),
            16);
    layout.forceOffset = 0;
    layout.scratchOffset = 0;
    layout.sinkCapacity = kWarpSize;
    layout.particleCapacity = kWarpSize;
    layout.multipoleCapacity = 4;
    layout.multipoleStride = FmmOrder<kOrder>::kPositionedMultipoleStride_;
    layout.sourceOffset = 4 * kWarpSize * int(sizeof(float));
    int const sourceBytes = std::max(kDimensions * kWarpSize * int(sizeof(Position)),
        layout.multipoleCapacity * layout.multipoleStride);
    layout.arenaBytes = alignUp(
        std::max(layout.sourceOffset + sourceBytes, FmmOrder<kOrder>::kLocalExpansionBytes_), 16);
    layout.totalBytes = layout.arenaOffset + layout.arenaBytes;
    layout.budgetBytes = layout.totalBytes;
    layout.residency = 1;
    layout.massPlane = false;
    layout.listReserve = 64;
    layout.forceBackingCapacity = 0;
    CudaKickListStorage<kOrder> storage;
    layout.listSlabBytes = kickCarveListStorage<kOrder>(layout, nullptr, storage);
    require(layout.totalBytes <= 48 * 1024, "fixture unexpectedly needs large shared memory");
    return layout;
}

std::array<TreeNode, 3> makeTree() {
    std::array<TreeNode, 3> nodes{};
    for (size_t i = 0; i < nodes.size(); ++i) {
        auto& node = nodes[i];
        node.particleRange.first = i == 2 ? 2 : 0;
        node.particleRange.second = i == 1 ? 2 : kParticleCount;
        node.processRange.first = 0;
        node.processRange.second = 1;
        node.depth = i == 0 ? 0 : 1;
        node.localRoot = i == 0;
        node.leaf = i != 0;
        node.radius = 0.5f;
        node.maxGravitySmoothingLength = kSupport;
        // Deliberately conservative overlapping boxes/radii prevent acceptance
        // of any far-field or hybrid pair. Every source reaches the PP list.
        for (int dim = 0; dim < kDimensions; ++dim) {
            node.pos[dim] = Position(0.5);
            node.box.begin[dim] = RangeFixed(0.25);
            node.box.end[dim] = RangeFixed(0.75);
        }
        for (int child = 0; child < kChildCount; ++child) {
            node.children[child].proc = 0;
            node.children[child].index = i == 0 ? child + 1 : -1;
        }
    }
    return nodes;
}

Output reference(std::array<std::array<Position, kParticleCount>, kDimensions> const& positions) {
    Output result{};
    double const inverseSupport = 1.0 / double(kSupport);
    auto const& kernel = selectedGravityKernel();
    for (int item = 0; item < kItemCount; ++item) {
        for (int sink = 0; sink < kParticleCount; ++sink) {
            std::array<double, kDimensions + 1> field{};
            for (int source = 0; source < kParticleCount; ++source) {
                if (sink == source) {
                    continue;
                }
                std::array<double, kDimensions> displacement{};
                double distanceSquared = 0.0;
                for (int dim = 0; dim < kDimensions; ++dim) {
                    displacement[dim] =
                        positions[dim][sink].toDouble() - positions[dim][source].toDouble();
                    distanceSquared += displacement[dim] * displacement[dim];
                }
                double const q = std::sqrt(distanceSquared) * inverseSupport;
                double const force = radial_kernel::forceFactor(kernel, q) * inverseSupport *
                    inverseSupport * inverseSupport;
                field[0] -= radial_kernel::potential(kernel, q) * inverseSupport;
                for (int dim = 0; dim < kDimensions; ++dim) {
                    field[dim + 1] -= displacement[dim] * force;
                }
            }
            for (int fieldIndex = 0; fieldIndex <= kDimensions; ++fieldIndex) {
                result[fieldIndex][item * kParticleCount + sink] = float(field[fieldIndex]);
            }
        }
    }
    return result;
}

void checkOutput(Output const& actual, Output const& expected) {
    for (int fieldIndex = 0; fieldIndex <= kDimensions; ++fieldIndex) {
        for (int i = 0; i < kOutputCount; ++i) {
            double const value = actual[fieldIndex][i];
            double const target = expected[fieldIndex][i];
            if (!std::isfinite(value) ||
                std::abs(value - target) > 5.e-5 * std::max(1.0, std::abs(target))) {
                throw std::runtime_error(
                    "production kick disagrees with direct reference at field " +
                    std::to_string(fieldIndex) + ", output " + std::to_string(i) + ": " +
                    std::to_string(value) + " versus " + std::to_string(target));
            }
        }
    }
}

PhaseCalls checkRecords(std::vector<profile::DeviceBlockRecord> const& records) {
    PhaseCalls calls{};
    std::array<std::uint64_t, profile::kDevicePhaseCount> cycles{};
    for (auto const& record : records) {
        for (size_t phase = 0; phase < profile::kDevicePhaseCount; ++phase) {
            auto const& counter = record.phases[phase];
            require(counter.calls != kPoison && counter.cycles != kPoison,
                "profiling failed to overwrite a block record");
            require(counter.calls != 0 || counter.cycles == 0,
                "unused device phase retained elapsed cycles");
            calls[phase] += counter.calls;
            cycles[phase] += counter.cycles;
        }
    }
    for (size_t phase = 0; phase < profile::kDevicePhaseCount; ++phase) {
#if !COSMICTIGER_ENABLE_FMM
        if (phase == static_cast<size_t>(profile::DevicePhase::KickPeriodic)) {
            require(calls[phase] == 0, "non-FMM build unexpectedly called the periodic phase");
            continue;
        }
#endif
        require(calls[phase] > 0 && cycles[phase] > 0,
            "production kick omitted a required device phase");
    }
    return calls;
}

void run(int blocks) {
    std::array<std::array<Position, kParticleCount>, kDimensions> positions{};
    double const coordinates[kDimensions][kParticleCount] = {
        {0.48, 0.49, 0.51, 0.52}, {0.49, 0.51, 0.48, 0.52}, {0.50, 0.48, 0.52, 0.51}};
    for (int dim = 0; dim < kDimensions; ++dim) {
        for (int i = 0; i < kParticleCount; ++i) {
            positions[dim][i] = Position(coordinates[dim][i]);
        }
    }
    auto const expected = reference(positions);
    auto const nodes = makeTree();
    auto const layout = makeLayout();
    DeviceBuffer<Position> x(kParticleCount), y(kParticleCount), z(kParticleCount);
    DeviceBuffer<TreeNode> tree(nodes.size());
    DeviceBuffer<int> checks(1), nextItem(1);
    DeviceBuffer<KickReturn> result(1);
    DeviceBuffer<CudaKickParameters<kOrder>> parameters(kItemCount);
    DeviceBuffer<char> slab(size_t(blocks) * size_t(layout.listSlabBytes));
    DeviceBuffer<float> potential(kOutputCount), gx(kOutputCount), gy(kOutputCount),
        gz(kOutputCount);
    DeviceBuffer<profile::DeviceBlockRecord> records{size_t(blocks)};
    x.upload(positions[0].data());
    y.upload(positions[1].data());
    z.upload(positions[2].data());
    tree.upload(nodes.data());
    int const root = 0;
    checks.upload(&root);
    std::array<CudaKickParameters<kOrder>, kItemCount> items{};
    for (int item = 0; item < kItemCount; ++item) {
        items[item].self = 0;
        items[item].outputOffset = item * kParticleCount;
        items[item].lpos = nodes[0].pos;
        items[item].lValue = 0.0f;
        items[item].dchecks = checks.data();
        items[item].dcount = 1;
        items[item].echecks = nullptr;
        items[item].ecount = 0;
    }
    parameters.upload(items.data());
    CudaKickData data{};
    data.treeNodes = tree.data();
    data.x = x.data();
    data.y = y.data();
    data.z = z.data();
    data.pot = potential.data();
    data.gx = gx.data();
    data.gy = gy.data();
    data.gz = gz.data();
    data.sph = false;
    KickParameters global{};
    global.forceOnly = true;
    global.gravitySmoothingLength = kSupport;
    global.theta = 0.5f;
    global.doPhi = true;
    global.openLargerCell = false;
    std::vector<profile::DeviceBlockRecord> hostRecords(static_cast<size_t>(blocks));
    for (auto& record : hostRecords) {
        for (auto& counter : record.phases) {
            counter.cycles = kPoison;
            counter.calls = kPoison;
        }
    }
    records.upload(hostRecords.data());
    Output baseline{};
    PhaseCalls firstCalls{};
    for (int repetition = 0; repetition < 3; ++repetition) {
        int const firstItem = 0;
        KickReturn const initialResult;
        nextItem.upload(&firstItem);
        result.upload(&initialResult);
        std::array<float, kOutputCount> poisonOutput{};
        poisonOutput.fill(std::numeric_limits<float>::quiet_NaN());
        potential.upload(poisonOutput.data());
        gx.upload(poisonOutput.data());
        gy.upload(poisonOutput.data());
        gz.upload(poisonOutput.data());
        cudaKickKernel<kOrder><<<blocks, kWarpSize, layout.totalBytes>>>(result.data(), global,
            data, parameters.data(), kItemCount, nextItem.data(), layout, slab.data()
#if COSMICTIGER_PROFILE && COSMICTIGER_DEVICE_PROFILE
                                                                              ,
            repetition == 0 ? nullptr : records.data()
#endif
        );
        checkCuda(cudaGetLastError(), "launch bounded production kick");
        checkCuda(cudaDeviceSynchronize(), "complete bounded production kick");
        Output actual{};
        potential.download(actual[0].data());
        gx.download(actual[1].data());
        gy.download(actual[2].data());
        gz.download(actual[3].data());
        checkOutput(actual, expected);
        if (repetition == 0) {
            baseline = actual;
        } else {
            require(actual == baseline, "device profiling changed force-only particle output");
        }
        records.download(hostRecords.data());
#if COSMICTIGER_PROFILE && COSMICTIGER_DEVICE_PROFILE
        if (repetition > 0) {
            auto const calls = checkRecords(hostRecords);
            if (repetition == 1) {
                firstCalls = calls;
            } else {
                // Do not reset records between recorded launches. Scheduling
                // may redistribute items among blocks, but aggregate calls
                // must describe one launch rather than accumulate two launches.
                require(calls == firstCalls, "repeated launch accumulated device profile calls");
            }
        } else
#endif
        {
            for (auto const& record : hostRecords) {
                for (auto const& counter : record.phases) {
                    require(counter.calls == kPoison && counter.cycles == kPoison,
                        "disabled device profiling modified its output records");
                }
            }
        }
    }
    std::printf("Production kick: %d blocks x %d threads, order %d, three launches; "
                "direct-force agreement and repeated-output checks passed.\n",
        blocks, kWarpSize, kOrder);
#if COSMICTIGER_PROFILE && COSMICTIGER_DEVICE_PROFILE
    std::puts("Device phase coverage and per-launch record overwrite passed "
              "(isolated periodic callback, empty M2L/P2L/M2P lists).");
#else
    std::puts("Device profiling compiled out; output records remained untouched.");
#endif
    for (int fieldIndex = 0; fieldIndex <= kDimensions; ++fieldIndex) {
        std::printf("Force reference field %d:", fieldIndex);
        for (int i = 0; i < kParticleCount; ++i) {
            std::printf(" %.9g", baseline[fieldIndex][i]);
        }
        std::puts("");
    }
}

}    // namespace detail

int main() {
    try {
        int deviceCount = 0;
        auto const status = cudaGetDeviceCount(&deviceCount);
        if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
            (status == cudaSuccess && deviceCount == 0)) {
            std::puts("SKIP: no usable CUDA device");
            return 77;
        }
        detail::checkCuda(status, "enumerate CUDA devices");
        int device = 0;
        detail::checkCuda(cudaGetDevice(&device), "read CUDA device");
        int multiprocessors = 0;
        detail::checkCuda(
            cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device),
            "read CUDA multiprocessor count");
        int const blocks = std::min(2, multiprocessors / 6);
        if (blocks == 0) {
            std::puts("SKIP: one block would exceed the one-sixth-SM fixture limit");
            return 77;
        }
        uploadKernelSelections();
        detail::run(blocks);
        return 0;
    } catch (std::exception const& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
