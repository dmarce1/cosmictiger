/* CosmicTiger - periodic PMMM force, distribution, and cache regression; BSL-1.0. */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cosmictiger/core/build_config.hpp>
#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/gravity/fmm_kernels/spherical_ewald_kernels.hpp>
#include <cosmictiger/gravity/fmm_order.hpp>
#include <cosmictiger/gravity/nitadori/cell_tree.hpp>
#include <cosmictiger/gravity/nitadori/ghost_storage.hpp>
#include <cosmictiger/gravity/nitadori/near_pp.hpp>
#include <cosmictiger/gravity/nitadori/solver.hpp>
#include <cosmictiger/gravity/sources.hpp>
#include <cosmictiger/gravity/tree.hpp>
#include <cosmictiger/math/kernel.hpp>
#include <cosmictiger/particles/domain.hpp>
#include <cosmictiger/particles/particle_mesh_sort.hpp>
#include <cosmictiger/particles/particles.hpp>
#if COSMICTIGER_WITH_SPH
#include <cosmictiger/sph/sph_particles.hpp>
#endif
#include <src/gravity/nitadori/state.hpp>

namespace detail {

size_t expectedPmmmParticleCount = 0;

int pmmmFixtureParticleCount(int fixture) {
    constexpr std::array<int, 9> kFixtureSizes{7, 1, 0, 4, 8, 320, 32, 7, 7};
    return kFixtureSizes.at(fixture);
}

struct PmmmSample {
    Array<double, kDimensions> position{};
    Array<PositionInteger, kDimensions> rawPosition{};
    Array<float, kDimensions> acceleration{};
    Array<float, kDimensions> velocity{};
    int rung{};
    float potential{};
    float mass{};
    int id{};
    int rank{};
    uint64_t cellKey{};

    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & position & rawPosition & acceleration & potential & mass & id & rank & cellKey &
            velocity & rung;
    }
};

void requirePmmm(bool condition, std::string const& message) {
    if (!condition) {
        throw std::runtime_error("PMMM regression: " + message);
    }
}

#if COSMICTIGER_PMMM_NEAR_FMM
void reservePmmmCudaFixture() {
    // Reserve the largest fixture on every rank before temporary CPU execution.
    // The application's retained particle/tree arenas assume a fixed backend.
    particlesResize(320);
#if COSMICTIGER_WITH_SPH
    sphParticlesResize(160, false);
#endif
    fmmGhostReserve(320);
    fmmGhostClear();
    particlesResize(0);
#if COSMICTIGER_WITH_SPH
    sphParticlesResize(0, false);
#endif
}

#endif

void resetPmmmSavedForces() {
    // Domain migration transfers physical particle state, not saved diagnostic
    // forces. Initialize these sentinels after migration, before testing solves.
    for (ParticleInteger i = 0; i < particlesSize(); ++i) {
        int const id = int(particlesVel(kXDimension, i)) - 1;
        particlesPot(i) = -0.25f * float(id + 1);
        for (int dim = 0; dim < kDimensions; ++dim) {
            particlesGravityAcceleration(dim, i) = float(4 * (id + 1) + dim);
        }
    }
}

#if !COSMICTIGER_PMMM_NEAR_FMM
void checkedPmmmNearField(std::vector<PmmmNearParticle> const& sinks,
    std::vector<PmmmNearParticle> const& sources, double cellWidth, bool doPotential,
    std::vector<std::array<double, 4>>& fields) {
    FlopCount expected = 0;
    if (!sinks.empty() && !sources.empty()) {
        expected = 3;
        double const inverseRadius = 1 / getOptions().gravity.supportRadius;
        double const inverseCubed = inverseRadius * inverseRadius * inverseRadius;
        for (auto const& sink : sinks) {
            std::array<double, 4> ignored{};
            for (auto const& source : sources) {
                expected += pmmm::detail::accumulateNearPair<true>(sink, source, cellWidth,
                    inverseRadius, inverseCubed, doPotential, selectedGravityKernel(), ignored);
            }
        }
    }
    // The arithmetic oracle is separately tested against CountingReal. This
    // check verifies that the production CPU/CUDA path submits it exactly once.
    resetFlops();
    pmmmNearField(sinks, sources, cellWidth, doPotential, fields);
    requirePmmm(readFlops() == expected, "PP backend lost or duplicated operation counts");
    bool const useCuda = getOptions().compute.cuda &&
        pmmm::detail::shouldUsePmmmNearCuda(
            sources.empty() ? 0 : sinks.size(), sinks.size() * sources.size());
    FlopCount const expectedGpu = useCuda && expected != 0 ? expected - 3 : 0;
    requirePmmm(getGpuFlops() == expectedGpu, "PP arithmetic was charged to the wrong backend");
}

void checkedPmmmNearFieldBatch(std::vector<PmmmNearParticle> const& sinks,
    std::vector<PmmmNearParticle> const& sources,
    std::vector<PmmmNearSourceRange> const& sourceRanges, bool doPotential,
    std::vector<std::array<double, 4>>& fields) {
    constexpr double kCellWidth = 0.25;
    double const inverseRadius = 1 / getOptions().gravity.supportRadius;
    double const inverseCubed = inverseRadius * inverseRadius * inverseRadius;
    FlopCount expected = 0;
    size_t usefulSinks = 0;
    size_t candidatePairs = 0;
    std::vector<std::array<double, 4>> reference(sinks.size());
    for (size_t sink = 0; sink < sinks.size(); ++sink) {
        auto const range = sourceRanges[sink];
        usefulSinks += range.begin != range.end;
        candidatePairs += range.end - range.begin;
        for (size_t source = range.begin; source < range.end; ++source) {
            expected += pmmm::detail::accumulateNearPair<true>(sinks[sink], sources[source],
                kCellWidth, inverseRadius, inverseCubed, doPotential, selectedGravityKernel(),
                reference[sink]);
        }
    }
    if (candidatePairs != 0) {
        expected += 3;
    }
    resetFlops();
    pmmmNearFieldBatch(sinks, sources, sourceRanges, kCellWidth, doPotential, fields);
    requirePmmm(readFlops() == expected, "PP batch lost or duplicated operation counts");
    bool const useCuda = getOptions().compute.cuda &&
        pmmm::detail::shouldUsePmmmNearCuda(usefulSinks, candidatePairs);
    FlopCount const expectedGpu = useCuda && expected != 0 ? expected - 3 : 0;
    requirePmmm(getGpuFlops() == expectedGpu, "PP batch arithmetic used the wrong backend");
    requirePmmm(fields.size() == reference.size(), "PP batch retained stale sink outputs");
    for (size_t sink = 0; sink < sinks.size(); ++sink) {
        for (int component = 0; component < 4; ++component) {
            requirePmmm(std::isfinite(fields[sink][component]) &&
                    std::abs(fields[sink][component] - reference[sink][component]) <
                        2.0e-11 * std::max(1.0, std::abs(reference[sink][component])),
                "PP batch mixed source ranges, identities, images, or retained stale fields");
        }
    }
}

void checkPmmmNearBatches() {
    auto const saved = getOptions();
    auto options = saved;
    options.gravity.supportRadius = 0.125;
    // Rectangles straddle both CUDA scheduling limits.
    // Later heterogeneous batches grow and shrink both reusable device buffers.
    struct BatchShape {
        size_t sinks;
        size_t sources;
        bool mixedRanges;
    };
    std::array<BatchShape, 13> const shapes{
        {{0, 129, false}, {1, 0, false}, {31, 129, false}, {127, 512, false}, {128, 511, false},
            {128, 512, false}, {129, 512, false}, {513, 257, true}, {257, 257, true},
            {513, 385, true}, {129, 1024, false}, {513, 257, true}, {33, 129, false}}};
    for (int backend = 0; backend < (saved.compute.cuda ? 2 : 1); ++backend) {
        options.compute.cuda = backend != 0;
        setOptions(options);
        bool usedCuda = false;
        std::vector<std::array<double, 4>> fields{{1.0, 2.0, 3.0, 4.0}};
        for (auto const shape : shapes) {
            std::vector<PmmmNearParticle> sources(shape.sources);
            for (size_t source = 0; source < sources.size(); ++source) {
                auto& particle = sources[source];
                // Offsets beyond +/-2 retain the specified periodic image;
                // repeated positions include coincident particles with distinct IDs.
                particle.offset = {0.75 * (int(source % 9) - 4), 0.0625 * int(source % 5),
                    -0.125 * int(source % 3)};
                particle.mass = 0.5 + 0.125 * double(source % 7);
                particle.rank = int(source % 2);
                particle.index = int64_t(source);
            }
            std::vector<PmmmNearParticle> sinks(shape.sinks);
            std::vector<PmmmNearSourceRange> sourceRanges(shape.sinks, {0, sources.size()});
            for (size_t sink = 0; sink < sinks.size(); ++sink) {
                if (!sources.empty()) {
                    sinks[sink] = sources[sink % sources.size()];
                    if (sink % 3 == 0) {
                        sinks[sink].index += int64_t(sources.size());
                    }
                }
                if (shape.mixedRanges) {
                    // Unequal cell populations and source spans share one batch.
                    sourceRanges[sink] = sink % 3 == 0 ? PmmmNearSourceRange{129, sources.size()} :
                                                         PmmmNearSourceRange{0, 129};
                }
            }
            if (shape.mixedRanges) {
                sourceRanges[0] = {sources.size(), sources.size()};
            }
            for (bool doPotential : {true, false}) {
                checkedPmmmNearFieldBatch(sinks, sources, sourceRanges, doPotential, fields);
                usedCuda |= getGpuFlops() != 0;
            }
            if (!sinks.empty()) {
                // A pooled buffer containing a previous field must not leak into
                // a call whose sources are all omitted by its ranges.
                std::fill(sourceRanges.begin(), sourceRanges.end(), PmmmNearSourceRange{});
                checkedPmmmNearFieldBatch(sinks, sources, sourceRanges, true, fields);
            }
        }
        requirePmmm(backend == 0 || usedCuda, "PP batch regression never exercised CUDA");
        std::vector<PmmmNearParticle> const invalidSinks(2);
        std::vector<PmmmNearParticle> const invalidSources(3);
        std::vector<std::vector<PmmmNearSourceRange>> const invalidRanges{
            {}, {{0, 3}}, {{0, 3}, {0, 3}, {0, 3}}, {{0, 3}, {2, 1}}, {{0, 3}, {0, 4}}};
        for (auto const& ranges : invalidRanges) {
            std::vector<std::array<double, 4>> const sentinel{{1.0, 2.0, 3.0, 4.0}};
            fields = sentinel;
            bool rejected = false;
            resetFlops();
            try {
                pmmmNearFieldBatch(invalidSinks, invalidSources, ranges, 0.25, true, fields);
            } catch (std::invalid_argument const&) {
                rejected = true;
            }
            requirePmmm(rejected, "PP batch accepted invalid source range bounds or cardinality");
            requirePmmm(fields == sentinel, "invalid PP batch corrupted its output fields");
            requirePmmm(readFlops() == 0 && getGpuFlops() == 0,
                "invalid PP batch submitted arithmetic counts");
        }
    }
    setOptions(saved);
}

void checkPmmmNearEdges() {
    auto options = getOptions();
    bool const useCuda = options.compute.cuda;
    double const support = 0.125;
    options.gravity.supportRadius = support;
    PmmmNearParticle const sink{{0.0, 0.0, 0.0}, 0.75, 0, 7};
    std::vector<PmmmNearParticle> const sinks{sink};
    std::vector<std::vector<PmmmNearParticle>> const cases{{sink}, {{{0.0, 0.0, 0.0}, 2.0, 0, 8}},
        {{{0.0, 0.0, 0.0}, 2.0, 1, 7}}, {{{support * 0.5 / 0.25, 0.0, 0.0}, 2.0, 0, 8}},
        {{{3.0, 0.0, 0.0}, 2.0, 0, 8}}, {}};
    std::vector<std::array<double, 4>> cpuResults;
    for (int backend = 0; backend < (useCuda ? 2 : 1); ++backend) {
        options.compute.cuda = backend != 0;
        setOptions(options);
        auto const& kernel = selectedGravityKernel();
        for (size_t example = 0; example < cases.size(); ++example) {
            std::vector<std::array<double, 4>> fields{{1.0, 2.0, 3.0, 4.0}};
            checkedPmmmNearField(sinks, cases[example], 0.25, true, fields);
            requirePmmm(fields.size() == 1, "near kernel changed sink output size");
            std::array<double, 4> expected{};
            if (example == 1 || example == 2) {
                expected[0] = -2.0 * radial_kernel::potential(kernel, 0.0) / support;
            } else if (example == 3) {
                expected[0] = -2.0 * radial_kernel::potential(kernel, 0.5) / support;
                expected[1] = radial_kernel::forceFactor(kernel, 0.5) / (support * support);
            } else if (example == 4) {
                // The requested image is +0.75 away. Wrapping to -0.25 would
                // reverse the force and use the wrong subtracted primary image.
                expected[0] = -2.0 / 0.75;
                expected[1] = 2.0 / (0.75 * 0.75);
            }
            for (int component = 0; component < 4; ++component) {
                requirePmmm(std::isfinite(fields[0][component]) &&
                        std::abs(fields[0][component] - expected[component]) <
                            2.0e-11 * std::max(1.0, std::abs(expected[component])),
                    "near kernel failed identity, coincidence, softening, or image selection");
                if (backend != 0) {
                    requirePmmm(std::abs(fields[0][component] - cpuResults[example][component]) <
                            2.0e-11 * std::max(1.0, std::abs(cpuResults[example][component])),
                        "CPU/CUDA near kernels disagree");
                }
            }
            if (backend == 0) {
                cpuResults.push_back(fields[0]);
            }
            auto const withPotential = fields[0];
            checkedPmmmNearField(sinks, cases[example], 0.25, false, fields);
            requirePmmm(fields[0][0] == 0.0, "force-only near evaluation computed a potential");
            for (int component = 1; component < 4; ++component) {
                requirePmmm(fields[0][component] == withPotential[component],
                    "potential switch changed a near acceleration");
            }
        }
        std::vector<std::array<double, 4>> emptyFields{{1.0, 2.0, 3.0, 4.0}};
        checkedPmmmNearField({}, cases[1], 0.25, true, emptyFields);
        requirePmmm(emptyFields.empty(), "empty near sink set retained stale outputs");
        // Cross warp/block boundaries with both self-skipped and active lanes.
        for (size_t count : {size_t(31), size_t(33), size_t(129)}) {
            std::vector<PmmmNearParticle> manySinks(count, sink);
            for (size_t i = 0; i < count; ++i) {
                manySinks[i].index += i;
            }
            for (bool doPotential : {false, true}) {
                checkedPmmmNearField(manySinks, sinks, 0.25, doPotential, emptyFields);
            }
        }
    }
    options.compute.cuda = useCuda;
    options.gravity.supportRadius = 0.001;
    setOptions(options);
}

#endif

void initializePmmmFixture(int fixture, int ownership) {
    // The first two and the next pair share cells on the n=8 mesh. Their
    // different offsets generate dipoles and the independent radial moment.
    std::array<std::array<double, kDimensions>, 7> const points{
        {{0.069, 0.161, 0.326}, {0.104, 0.202, 0.295}, {0.454, 0.166, 0.329}, {0.487, 0.206, 0.287},
            {0.989, 0.171, 0.317}, {0.671, 0.839, 0.582}, {0.011, 0.986, 0.971}}};
    int const total = pmmmFixtureParticleCount(fixture);
    auto owner = [ownership](int id) { return ownership == 0 ? id % hpxSize() : 0; };
    int count = 0;
    int gasCount = 0;
    for (int id = 0; id < total; ++id) {
        if (owner(id) == hpxRank()) {
            ++count;
            gasCount += getOptions().hydro.enabled && fixture != 4 && id % 2 != 0;
        }
    }
    particlesResize(count);
#if COSMICTIGER_WITH_SPH
    if (getOptions().hydro.enabled) {
        sphParticlesResize(gasCount, false);
    }
#endif
    int local = 0;
    int gas = 0;
    for (int id = total - 1; id >= 0; --id) {
        if (owner(id) != hpxRank()) {
            continue;
        }
        for (int dim = 0; dim < kDimensions; ++dim) {
            double coordinate = 0.0;
            if (fixture == 8) {
                // Centered, distinct sources eliminate multipole truncation
                // from the stage test while retaining near and far pairs.
                constexpr std::array<std::array<int, kDimensions>, 7> kStageCells{
                    {{0, 1, 2}, {1, 1, 2}, {3, 1, 2}, {3, 2, 2}, {7, 1, 2}, {5, 6, 4}, {0, 7, 7}}};
                coordinate = (kStageCells[id][dim] + 0.5) / 8.0;
            } else if (fixture == 7) {
                // Every rank owns a fragment of the same mesh cell. Distinct
                // positions avoid softened pairs in the independent Ewald sum.
                int const multiplier = dim == kXDimension ? 1 : dim == kYDimension ? 3 : 5;
                double const offset = 0.2 + 0.1 * ((id * multiplier) % 7);
                coordinate = (1.0 + offset) / getOptions().gravity.meshSize;
            } else if (fixture == 6) {
                int const cell = dim == kXDimension ? id % 8 :
                    dim == kYDimension              ? 4 * ((id / 8) % 2) :
                                                      4 * (id / 16);
                coordinate = (cell + 0.5) / 8.0;
            } else if (fixture == 5) {
                int const side = getOptions().gravity.meshSize;
                int const group = id / 64;
                int const cell = dim == kXDimension ? (group == 1        ? side - 1 :
                                                              group == 3 ? 1 :
                                                                           0) :
                                                      1;
                std::array<std::array<double, 3>, 5> const centers{{{-0.22, -0.20, -0.18},
                    {0.14, 0.03, -0.06}, {0.22, 0.20, 0.18}, {0.32, 0.05, 0.09}, {0.0, 0.0, 0.0}}};
                int const multiplier = dim == kXDimension ? 17 : dim == kYDimension ? 29 : 43;
                double const jitter = (double((id * multiplier) % 64) + 0.5) / 64.0 - 0.5;
                coordinate = (cell + 0.5 + centers[group][dim] + 0.11 * jitter) / side;
            } else {
                coordinate = fixture == 4 ? 0.0625 + 0.5 * ((id >> dim) & 1) : points[id][dim];
            }
            if (fixture == 3 && id >= 2) {
                coordinate = (dim == kXDimension        ? 0.58 :
                                     dim == kYDimension ? 0.57 :
                                                          0.81) +
                    (id == 2 ? 0.013 : -0.019);
            }
            particlesPos(dim, local) = coordinate;
            if (fixture == 5 && id >= total - 16) {
                // Distinct coincident sources at the last representable point
                // outnumber the eight-particle leaf bucket, exercising robust
                // subdivision and a center conversion that rounds up to 1.
                particlesPos(dim, local).setInteger(std::numeric_limits<PositionInteger>::max());
            }
            particlesVel(dim, local) = dim == kXDimension ? float(id + 1) :
                dim == kYDimension                        ? 0.25f * float(id) :
                                                            -0.5f * float(id);
            particlesGravityAcceleration(dim, local) = float(4 * (id + 1) + dim);
        }
        particlesPot(local) = -0.25f * float(id + 1);
        particlesRung(local) = 1 + id % 4;
#if COSMICTIGER_WITH_SPH
        if (getOptions().hydro.enabled) {
            bool const isGas = fixture != 4 && id % 2 != 0;
            particlesType(local) = isGas ? kSphType : kDarkMatterType;
            particlesCatIndex(local) = isGas ? gas : kNoParticleIndex;
            if (isGas) {
                sphParticlesParticleIndex(gas) = local;
                sphParticlesSupportRadius(gas) = 0.01f;
                for (int dim = 0; dim < kDimensions; ++dim) {
                    sphParticlesGravityAcceleration(dim, gas) = 0.0f;
                }
                ++gas;
            }
        }
#endif
        ++local;
    }
}

Vector<PmmmSample> readPmmmFixture() {
    requirePmmm(particlesSize() == particlesOwnedSize(),
        "force solve retained ghost particles in the owned population");
    Vector<PmmmSample> samples(particlesSize());
    for (ParticleInteger i = 0; i < particlesSize(); ++i) {
        auto& sample = samples[i];
        for (int dim = 0; dim < kDimensions; ++dim) {
            sample.position[dim] = particlesPos(dim, i).toDouble();
            sample.rawPosition[dim] = particlesPos(dim, i).raw();
        }
        sample.acceleration = pmmmAcceleration(i);
        sample.potential = pmmmPotential(i);
        sample.mass = particlesMass(i);
        sample.id = int(particlesVel(kXDimension, i)) - 1;
        requirePmmm(particlesVel(kXDimension, i) == float(sample.id + 1) &&
                particlesVel(kYDimension, i) == 0.25f * float(sample.id) &&
                particlesVel(kZDimension, i) == -0.5f * float(sample.id) &&
                particlesRung(i) == 1 + sample.id % 4 &&
                particlesPot(i) == -0.25f * float(sample.id + 1),
            "force solve changed velocity, rung, or saved potential");
        sample.rank = hpxRank();
        sample.cellKey = 0;
        for (int dim = 0; dim < kDimensions; ++dim) {
            requirePmmm(particlesGravityAcceleration(dim, i) == float(4 * (sample.id + 1) + dim),
                "force solve changed saved acceleration");
            sample.cellKey = sample.cellKey * getOptions().gravity.meshSize +
                particleMeshCoordinate(particlesPos(dim, i).raw(), getOptions().gravity.meshSize);
        }
#if COSMICTIGER_WITH_SPH
        if (getOptions().hydro.enabled && particlesType(i) == kSphType) {
            requirePmmm(sphParticlesParticleIndex(particlesCatIndex(i)) == i,
                "gas reverse index was not repaired after sorting");
            for (int dim = 0; dim < kDimensions; ++dim) {
                requirePmmm(sphParticlesGravityAcceleration(dim, particlesCatIndex(i)) == 0.0f,
                    "force solve changed gas acceleration");
            }
        }
#endif
    }
    return samples;
}

Vector<PmmmSample> readPmmmKickFixture() {
    Vector<PmmmSample> samples(particlesSize());
    for (ParticleInteger i = 0; i < particlesSize(); ++i) {
        auto& sample = samples[i];
        for (int dim = 0; dim < kDimensions; ++dim) {
            sample.position[dim] = particlesPos(dim, i).toDouble();
            sample.rawPosition[dim] = particlesPos(dim, i).raw();
            sample.velocity[dim] = particlesVel(dim, i);
            if (saveParticleAcceleration(getOptions().analysis)) {
                sample.acceleration[dim] = particlesGravityAcceleration(dim, i);
            }
        }
        sample.mass = particlesMass(i);
        sample.rung = particlesRung(i);
        if (saveParticlePotential(getOptions().analysis)) {
            sample.potential = particlesPot(i);
        }
    }
    return samples;
}

struct PmmmOverlapGate {
    std::mutex mutex;
    hpx::promise<void> promise;
    hpx::shared_future<void> release = promise.get_future().share();
    hpx::promise<void> requestPromise;
    hpx::shared_future<void> requestStarted = requestPromise.get_future().share();
    uint64_t heldKey = 0;
    bool chosen = false;
    bool completedIndependent = false;
    bool completedDependentEarly = false;
    bool receivedHeld = false;
};

std::shared_ptr<PmmmOverlapGate> pmmmOverlapGate;

bool pmmmCellsNeighbor(uint64_t first, uint64_t second) {
    int const side = getOptions().gravity.meshSize;
    int const width = getOptions().gravity.pmmmNearWidth;
    for (int dim = 0; dim < kDimensions; ++dim) {
        int const difference = std::abs(int(first % side) - int(second % side));
        if (std::min(difference, side - difference) > width) {
            return false;
        }
        first /= side;
        second /= side;
    }
    return true;
}

void installPmmmOverlapGate(bool enable, uint64_t heldKey) {
    pmmmSetCellObserver({});
    pmmmOverlapGate.reset();
    if (!enable || hpxRank() != 0) {
        return;
    }
    auto gate = std::make_shared<PmmmOverlapGate>();
    gate->heldKey = heldKey;
    pmmmOverlapGate = gate;
    pmmmSetCellObserver([gate](PmmmCellEvent event, uint64_t key) {
        // Ensure a very fast interior cell cannot release the gate before the
        // chosen remote request exists. HPX suspension leaves workers free.
        if (event == PmmmCellEvent::NearCompleted) {
            gate->requestStarted.get();
        }
        bool wait = false;
        bool release = false;
        {
            std::lock_guard<std::mutex> lock(gate->mutex);
            if (event == PmmmCellEvent::GhostRequested && key == gate->heldKey && !gate->chosen) {
                gate->chosen = true;
                wait = true;
            } else if (event == PmmmCellEvent::NearCompleted && !gate->completedIndependent) {
                if (pmmmCellsNeighbor(key, gate->heldKey)) {
                    gate->completedDependentEarly = true;
                } else {
                    gate->completedIndependent = true;
                    release = true;
                }
            } else if (event == PmmmCellEvent::GhostReady && key == gate->heldKey) {
                gate->receivedHeld = true;
            }
        }
        if (release) {
            gate->promise.set_value();
        }
        if (wait) {
            gate->requestPromise.set_value();
            gate->release.get();
        }
    });
}

Vector<int> readPmmmOverlapGate() {
    auto const gate = pmmmOverlapGate;
    if (!gate) {
        return {};
    }
    std::lock_guard<std::mutex> lock(gate->mutex);
    return {int(gate->chosen), int(gate->completedIndependent), int(gate->completedDependentEarly),
        int(gate->receivedHeld)};
}

#if COSMICTIGER_PMMM_NEAR_FMM
size_t verifyPmmmTreeCell(TreeId id, uint64_t key, int meshSize) {
    auto const* node = treeGetNode(id);
    uint64_t positionKey = 0;
    uint64_t multipoleKey = 0;
    for (int dim = 0; dim < kDimensions; ++dim) {
        positionKey =
            positionKey * meshSize + particleMeshCoordinate(node->pos[dim].raw(), meshSize);
        multipoleKey =
            multipoleKey * meshSize + particleMeshCoordinate(node->mpos->pos[dim].raw(), meshSize);
    }
    requirePmmm(positionKey == key && multipoleKey == key,
        "forest node or multipole center escaped its exact mesh cell: expected=" +
            std::to_string(key) + " position=" + std::to_string(positionKey) +
            " multipole=" + std::to_string(multipoleKey) + " node=" + std::to_string(id.proc) +
            ":" + std::to_string(id.index));
    if (node->leaf) {
        // Every local leaf is checked by its owning locality before the roots
        // are combined. Compare exact integers so a Q0.64 coordinate just below
        // one cannot round onto an incorrectly inward bound during this test.
        if (node->isLocalHere()) {
            using WideInteger = __int128;
            WideInteger constexpr kBoundaryScale = WideInteger(1) << (kPositionBits - 29);
            for (ParticleInteger i = node->particleRange.first; i < node->particleRange.second;
                ++i) {
                long double radiusSquared = 0.0;
                for (int dim = 0; dim < kDimensions; ++dim) {
                    WideInteger const raw = particlesPos(dim, i).raw();
                    WideInteger const lower =
                        WideInteger(node->box.begin[dim].raw()) * kBoundaryScale;
                    WideInteger const upper =
                        WideInteger(node->box.end[dim].raw()) * kBoundaryScale;
                    requirePmmm(lower <= raw && raw <= upper,
                        "forest leaf box excludes a represented fixed-point particle");
                    long double const distance = std::ldexp(
                        static_cast<long double>(raw - WideInteger(node->pos[dim].raw())),
                        -kPositionBits);
                    radiusSquared += distance * distance;
                }
                requirePmmm(radiusSquared <= static_cast<long double>(node->radius) * node->radius,
                    "forest leaf radius excludes a represented fixed-point particle");
            }
        }
        return 0;
    }
    auto const children = node->children;
    size_t internalNodes = 1;
    for (auto const& child : children) {
        internalNodes += verifyPmmmTreeCell(child, key, meshSize);
    }
    return internalNodes;
}

Vector<TreeCellRoot> buildPmmmTestForest() {
    int const meshSize = getOptions().gravity.meshSize;
    auto const cells = particleMeshSort(meshSize);
    auto roots = treeCreateCellForest(cells, meshSize);
    for (auto const& root : roots) {
        verifyPmmmTreeCell(root.id, root.key, meshSize);
    }
    return roots;
}

#endif

// Independent direct Ewald summation, with no generated kernels, mesh, FFT,
// harmonic coefficients, or production image tables. The primary self term is
// excluded; its finite periodic images remain, matching the FMM convention.
std::array<double, 4> directPmmmReference(size_t sink, Vector<PmmmSample> const& samples) {
    double constexpr kPi = std::numbers::pi;
    double constexpr kAlpha = 2.0;
    int constexpr kCutoff = 4;
    double const inverseSqrtPi = 1.0 / std::sqrt(kPi);
    std::array<double, 4> result{};
    for (size_t source = 0; source < samples.size(); ++source) {
        double const mass = samples[source].mass * getOptions().gravity.gm;
        std::array<double, kDimensions> separation{};
        for (int dim = 0; dim < kDimensions; ++dim) {
            separation[dim] = samples[sink].position[dim] - samples[source].position[dim];
            separation[dim] -= std::nearbyint(separation[dim]);
        }
        result[0] += mass * kPi / (kAlpha * kAlpha);
        if (source == sink) {
            result[0] += mass * 2.0 * kAlpha * inverseSqrtPi;
        }
        for (int x = -kCutoff; x <= kCutoff; ++x) {
            for (int y = -kCutoff; y <= kCutoff; ++y) {
                for (int z = -kCutoff; z <= kCutoff; ++z) {
                    bool const origin = x == 0 && y == 0 && z == 0;
                    std::array<double, kDimensions> const displacement{
                        separation[0] + x, separation[1] + y, separation[2] + z};
                    double const radiusSquared = displacement[0] * displacement[0] +
                        displacement[1] * displacement[1] + displacement[2] * displacement[2];
                    if (!(source == sink && origin)) {
                        double const radius = std::sqrt(radiusSquared);
                        requirePmmm(!origin || radius > getOptions().gravity.supportRadius,
                            "direct Ewald fixture contains a softened pair");
                        double const screened = std::erfc(kAlpha * radius) / radius;
                        double const radial = (screened +
                                                  2.0 * kAlpha * inverseSqrtPi *
                                                      std::exp(-kAlpha * kAlpha * radiusSquared)) /
                            radiusSquared;
                        result[0] -= mass * screened;
                        for (int dim = 0; dim < kDimensions; ++dim) {
                            result[dim + 1] -= mass * radial * displacement[dim];
                        }
                    }
                    if (!origin) {
                        std::array<double, kDimensions> const wave{
                            2.0 * kPi * x, 2.0 * kPi * y, 2.0 * kPi * z};
                        double const waveSquared = x * x + y * y + z * z;
                        double const coefficient =
                            std::exp(-kPi * kPi * waveSquared / (kAlpha * kAlpha)) /
                            (kPi * waveSquared);
                        double const phase = wave[0] * separation[0] + wave[1] * separation[1] +
                            wave[2] * separation[2];
                        result[0] -= mass * coefficient * std::cos(phase);
                        for (int dim = 0; dim < kDimensions; ++dim) {
                            result[dim + 1] -= mass * coefficient * std::sin(phase) * wave[dim];
                        }
                    }
                }
            }
        }
    }
    return result;
}

// Evaluate the selected finite expansion directly in double precision. Physical
// offsets and generated M2L bypass mesh deposition, the contraction table, FFTs,
// and cell-width rescaling. This checks implementation error separately from
// truncation; directPmmmReference remains the independent full Ewald oracle.
std::array<double, 4> directPmmmExpansionReference(size_t sink, Vector<PmmmSample> const& samples) {
    namespace spherical = cosmictiger::gravity::spherical;
    constexpr int kSourceDegree = kFmmDefaultOrder - 2;
    constexpr int kLocalDegree = kFmmDefaultOrder - 1;
    auto const& options = getOptions();
    int const n = options.gravity.meshSize;
    auto const coordinates = [n](PmmmSample const& sample) {
        return std::array<int, kDimensions>{
            int(sample.cellKey / (n * n)), int(sample.cellKey / n % n), int(sample.cellKey % n)};
    };
    auto const sinkCell = coordinates(samples[sink]);
    std::array<double, kDimensions> sinkOffset{};
    for (int dim = 0; dim < kDimensions; ++dim) {
        sinkOffset[dim] = samples[sink].position[dim] - (sinkCell[dim] + 0.5) / n;
    }
    std::array<double, (kLocalDegree + 1) * (kLocalDegree + 1)> local{};
    std::array<double, 4> field{};
    double localTrace = 0.0;
    for (size_t source = 0; source < samples.size(); ++source) {
        auto const sourceCell = coordinates(samples[source]);
        std::array<double, kDimensions> sourceOffset{}, displacement{}, pair{};
        int cellDistance = 0;
        double sourceTrace = 0.0;
        for (int dim = 0; dim < kDimensions; ++dim) {
            sourceOffset[dim] = samples[source].position[dim] - (sourceCell[dim] + 0.5) / n;
            int delta = (sinkCell[dim] - sourceCell[dim] + n) % n;
            if (delta > n / 2) {
                delta -= n;
            }
            cellDistance = std::max(cellDistance, std::abs(delta));
            displacement[dim] = double(delta) / n;
            pair[dim] = displacement[dim] + sinkOffset[dim] - sourceOffset[dim];
            sourceTrace += samples[source].mass * sourceOffset[dim] * sourceOffset[dim];
        }
        bool const near = cellDistance <= options.gravity.pmmmNearWidth;
        std::array<double, (kSourceDegree + 1) * (kSourceDegree + 1)> moments{};
        spherical::detail::P2M<kSourceDegree>::add<double>(
            moments, samples[source].mass, sourceOffset[0], sourceOffset[1], sourceOffset[2]);
        std::array<double, (kLocalDegree + 1) * (kLocalDegree + 1)> green{};
        double greenTrace = 0.0;
        spherical::EwaldGreen<kLocalDegree>::evaluate<double>(
            green, greenTrace, displacement[0], displacement[1], displacement[2]);
        if (!near) {
            std::array<double, (kLocalDegree + 1) * (kLocalDegree + 1)> primary{};
            spherical::detail::Green<kLocalDegree>::evaluate<double>(
                primary, displacement[0], displacement[1], displacement[2]);
            for (size_t slot = 0; slot < green.size(); ++slot) {
                green[slot] += primary[slot];
            }
        }
        spherical::EwaldM2L<kLocalDegree>::add<double>(
            local, localTrace, moments, sourceTrace, green, greenTrace);
        if (near && source != sink) {
            double const radiusSquared = pair[0] * pair[0] + pair[1] * pair[1] + pair[2] * pair[2];
            double const radius = std::sqrt(radiusSquared);
            requirePmmm(radius > options.gravity.supportRadius,
                "direct expansion fixture contains a softened pair");
            field[0] -= samples[source].mass / radius;
            for (int dim = 0; dim < kDimensions; ++dim) {
                field[dim + 1] -= samples[source].mass * pair[dim] / (radius * radiusSquared);
            }
        }
    }
    spherical::EwaldL2P<kLocalDegree>::add<double>(
        field, local, localTrace, sinkOffset[0], sinkOffset[1], sinkOffset[2]);
    for (double& component : field) {
        component *= options.gravity.gm;
    }
    return field;
}

double verifyPmmmReference(Vector<PmmmSample> const& samples) {
    auto const& options = getOptions();
    std::string const context = " mesh=" + std::to_string(options.gravity.meshSize) +
        " order=" + std::to_string(options.gravity.pmmmOrder) +
        " width=" + std::to_string(options.gravity.pmmmNearWidth);
    // Potential/force truncation budgets for these off-center fixtures, measured
    // against the independent Ewald sum with a direct double expansion for both
    // unit and mixed masses. They are not universal PMMM accuracy guarantees.
    // In particular order 6 reaches 1.615e-3 force error, even without FFTs;
    // it cannot share the order-7 5e-4 budget. Numerical error is bounded below
    // independently and never absorbed into the truncation allowance.
    constexpr std::array<std::array<double, 2>, 7> kTruncationBudgets{
        {{3.0e-2, 6.0e-2}, {1.5e-3, 2.5e-2}, {3.5e-4, 7.5e-3}, {6.0e-5, 2.0e-3}, {3.0e-5, 5.0e-4},
            {8.0e-6, 1.25e-4}, {3.0e-6, 5.0e-5}}};
    auto const budget = kTruncationBudgets[kFmmDefaultOrder - 3];
    constexpr double kNumericalTolerance = 2.0e-5;
    double maximumError = 0.0;
    std::array<double, kDimensions> momentum{};
    double forceScale = 1.0;
    for (size_t sink = 0; sink < samples.size(); ++sink) {
        auto const expected = directPmmmReference(sink, samples);
        auto const expansion = directPmmmExpansionReference(sink, samples);
        double norm = 0.0;
        for (int dim = 0; dim < kDimensions; ++dim) {
            norm += expected[dim + 1] * expected[dim + 1];
        }
        norm = std::max(1.0, std::sqrt(norm));
        forceScale = std::max(forceScale, norm);
        double const potentialScale = std::max(1.0, std::abs(expected[0]));
        // Near FMM subtracts softened self potential in float; keep that
        // cancellation allowance separate from both expansion and FFT errors.
        double const selfCancellationAllowance = gravity::kBuildPmmmNearUsesFmm ?
            4.0 * std::numeric_limits<float>::epsilon() *
                std::abs(options.gravity.gm * samples[sink].mass *
                    selectedGravityKernel().centralPotential / options.gravity.supportRadius) :
            0.0;
        for (int component = 0; component < 4; ++component) {
            double const scale = component == 0 ? potentialScale : norm;
            double const actual = component == 0 ? samples[sink].potential :
                                                   samples[sink].acceleration[component - 1];
            double const numericalAllowance =
                kNumericalTolerance * scale + (component == 0 ? selfCancellationAllowance : 0.0);
            requirePmmm(std::abs(actual - expansion[component]) < numericalAllowance,
                "field differs from direct finite expansion for particle " + std::to_string(sink) +
                    " component=" + std::to_string(component) +
                    " actual=" + std::to_string(actual) +
                    " expected=" + std::to_string(expansion[component]) + context);
            requirePmmm(std::abs(expansion[component] - expected[component]) <
                    budget[component == 0 ? 0 : 1] * scale,
                "finite expansion exceeds the Ewald truncation budget for particle " +
                    std::to_string(sink) + " component=" + std::to_string(component) + context);
        }
        double const potentialError =
            std::abs(samples[sink].potential - expected[0]) / potentialScale;
        maximumError = std::max(maximumError, potentialError);
        requirePmmm(std::isfinite(samples[sink].potential) &&
                potentialError <
                    budget[0] + kNumericalTolerance + selfCancellationAllowance / potentialScale,
            "potential differs from direct periodic Ewald for particle " + std::to_string(sink) +
                ": scaled error " + std::to_string(potentialError) +
                " actual=" + std::to_string(samples[sink].potential) +
                " expected=" + std::to_string(expected[0]) + context);
        for (int dim = 0; dim < kDimensions; ++dim) {
            double const error =
                std::abs(samples[sink].acceleration[dim] - expected[dim + 1]) / norm;
            maximumError = std::max(maximumError, error);
            requirePmmm(std::isfinite(samples[sink].acceleration[dim]) &&
                    error < budget[1] + kNumericalTolerance,
                "force differs from direct periodic Ewald for particle " + std::to_string(sink) +
                    ": scaled error " + std::to_string(error) + " dim=" + std::to_string(dim) +
                    " actual=" + std::to_string(samples[sink].acceleration[dim]) +
                    " expected=" + std::to_string(expected[dim + 1]) + context);
            momentum[dim] += samples[sink].mass * samples[sink].acceleration[dim];
        }
    }
    for (double component : momentum) {
        requirePmmm(std::abs(component) < 3.0e-5 * forceScale,
            "unequal-mass periodic particles have nonzero total force");
    }
    return maximumError;
}

void comparePmmmSamples(
    Vector<PmmmSample> const& first, Vector<PmmmSample> const& second, bool potential = true) {
    auto const& options = getOptions();
    std::string const context = " near=" + options.gravity.pmmmNearMethod +
        " mesh=" + std::to_string(options.gravity.meshSize) +
        " fmmOrder=" + std::to_string(options.gravity.fmmOrder) +
        " pmmmOrder=" + std::to_string(options.gravity.pmmmOrder) +
        " width=" + std::to_string(options.gravity.pmmmNearWidth) +
        " theta=" + std::to_string(options.gravity.openingAngle) +
        " cuda=" + std::to_string(options.compute.cuda);
    requirePmmm(first.size() == second.size(), "solve changed the particle count");
    for (size_t i = 0; i < first.size(); ++i) {
        requirePmmm(first[i].id == second[i].id && first[i].mass == second[i].mass,
            "solve changed a particle's identity or mass");
        for (int dim = 0; dim < kDimensions; ++dim) {
            requirePmmm(first[i].rawPosition[dim] == second[i].rawPosition[dim],
                "solve changed a particle position");
            requirePmmm(std::abs(first[i].acceleration[dim] - second[i].acceleration[dim]) <
                    2.0e-5 * std::max(1.0f, std::abs(first[i].acceleration[dim])),
                "force changed for id=" + std::to_string(first[i].id) +
                    " dim=" + std::to_string(dim) +
                    " reference=" + std::to_string(first[i].acceleration[dim]) +
                    " actual=" + std::to_string(second[i].acceleration[dim]) + context);
        }
        if (potential) {
            // The existing FMM accumulates softened self potential and subtracts
            // it in float. Bound that cancellation's rounding scale separately
            // from the force/approximation limits; direct PMMM PP uses double.
            double const selfCancellationAllowance = options.gravity.pmmmNearMethod == "fmm" ?
                4.0 * std::numeric_limits<float>::epsilon() *
                    std::abs(options.gravity.gm * first[i].mass *
                        selectedGravityKernel().centralPotential / options.gravity.supportRadius) :
                0.0;
            requirePmmm(std::abs(first[i].potential - second[i].potential) <
                    2.0e-5 * std::max(1.0f, std::abs(first[i].potential)) +
                        selfCancellationAllowance,
                "potential changed for id=" + std::to_string(first[i].id) +
                    " reference=" + std::to_string(first[i].potential) +
                    " actual=" + std::to_string(second[i].potential) + context);
        }
    }
}

}    // namespace detail

HPX_PLAIN_ACTION(detail::initializePmmmFixture, InitializePmmmFixtureAction);
HPX_PLAIN_ACTION(detail::readPmmmFixture, ReadPmmmFixtureAction);
HPX_PLAIN_ACTION(detail::readPmmmKickFixture, ReadPmmmKickFixtureAction);
HPX_PLAIN_ACTION(detail::installPmmmOverlapGate, InstallPmmmOverlapGateAction);
HPX_PLAIN_ACTION(detail::readPmmmOverlapGate, ReadPmmmOverlapGateAction);
#if COSMICTIGER_PMMM_NEAR_FMM
HPX_PLAIN_ACTION(detail::buildPmmmTestForest, BuildPmmmTestForestAction);
HPX_PLAIN_ACTION(detail::reservePmmmCudaFixture, ReservePmmmCudaFixtureAction);
#endif
HPX_PLAIN_ACTION(detail::resetPmmmSavedForces, ResetPmmmSavedForcesAction);

namespace detail {

void preparePmmmFixture(int fixture, int ownership = 0) {
    expectedPmmmParticleCount = size_t(pmmmFixtureParticleCount(fixture));
    std::vector<hpx::future<void>> futures;
    for (auto const& locality : hpxLocalities()) {
        futures.push_back(hpx::async<InitializePmmmFixtureAction>(locality, fixture, ownership));
    }
    for (auto& future : futures) {
        future.get();
    }
}

void distributePmmmFixture(int fixture) {
    auto const saved = getOptions();
    auto options = saved;
    options.initial.numberOfParticles = pmmmFixtureParticleCount(fixture);
    // Fresh fixtures deliberately place every rung on arbitrary localities.
    // The mixed-mass fixture uses the all-particle hydro migration path before
    // testing rung-selected stages, independently of the compiled integrator.
    // Use the ordinary particle-balanced domains, including split mesh cells.
    setOptions(options);
    particlesSortByRung(0);
    domainsRebound();
    domainsBegin(0);
    domainsEnd();
    setOptions(saved);
    std::vector<hpx::future<void>> futures;
    for (auto const& locality : hpxLocalities()) {
        futures.push_back(hpx::async<ResetPmmmSavedForcesAction>(locality));
    }
    for (auto& future : futures) {
        future.get();
    }
}

#if COSMICTIGER_PMMM_NEAR_FMM
void checkPmmmForestGeometry() {
    treePrepareCellForest();
    std::vector<hpx::future<Vector<TreeCellRoot>>> futures;
    for (auto const& locality : hpxLocalities()) {
        futures.push_back(hpx::async<BuildPmmmTestForestAction>(locality));
    }
    hpx::wait_all(futures.begin(), futures.end());
    std::map<uint64_t, Vector<TreeId>> contributors;
    for (auto& future : futures) {
        for (auto const& root : future.get()) {
            contributors[root.key].push_back(root.id);
        }
    }
    Vector<TreeCellRoot> combinedRoots;
    for (auto const& [key, roots] : contributors) {
        auto const combined = treeCombineCellRoots(key, getOptions().gravity.meshSize, roots);
        combinedRoots.push_back({key, combined});
    }
    // Complete the forest before any remote node-cache reads: adding another
    // root can populate slots in a cache line fetched by an earlier traversal.
    size_t internalNodes = 0;
    for (auto const& root : combinedRoots) {
        internalNodes += verifyPmmmTreeCell(root.id, root.key, getOptions().gravity.meshSize);
    }
    requirePmmm(internalNodes > 0, "dense forest contained no internal tree nodes");
    treeDestroy();
    particlesCacheFree();
}

#endif

Vector<PmmmSample> gatherPmmmSamples() {
    std::vector<hpx::future<Vector<PmmmSample>>> futures;
    for (auto const& locality : hpxLocalities()) {
        futures.push_back(hpx::async<ReadPmmmFixtureAction>(locality));
    }
    Vector<PmmmSample> samples;
    for (auto& future : futures) {
        auto local = future.get();
        samples.insert(samples.end(), local.begin(), local.end());
    }
    requirePmmm(samples.size() == expectedPmmmParticleCount,
        "force solve changed the fixture's global owned particle count");
    std::sort(samples.begin(), samples.end(),
        [](auto const& first, auto const& second) { return first.id < second.id; });
    double mass = 0.0;
    for (size_t i = 0; i < samples.size(); ++i) {
        auto const& sample = samples[i];
        requirePmmm(sample.id == int(i), "force solve lost or duplicated an owned particle");
        mass += sample.mass;
    }
    auto const diagnostics = pmmmDiagnostics();
    requirePmmm(
        diagnostics.nearRemoteTreeFetches == 0 && diagnostics.nearRemoteParticleFetches == 0,
        "near FMM fetched remote tree nodes or particles during its local walk");
    if (hpxSize() == 1 || samples.empty()) {
        requirePmmm(diagnostics.ghostParticles == 0,
            "single-locality or empty solve retained remote ghost sources");
    }
    requirePmmm(std::abs(diagnostics.sourceMass - mass) < 1.0e-6 * std::max(1.0, mass),
        "source mass differs from the particle population");
    requirePmmm(std::abs(diagnostics.depositedMass - mass) < 2.0e-6 * std::max(1.0, mass),
        "cell moments lost mass");
    return samples;
}

double pmmmFieldDifference(Vector<PmmmSample> const& reference, Vector<PmmmSample> const& actual) {
    requirePmmm(reference.size() == actual.size(), "near solver changed the particle count");
    double maximumError = 0.0;
    for (size_t i = 0; i < reference.size(); ++i) {
        requirePmmm(reference[i].id == actual[i].id && reference[i].mass == actual[i].mass,
            "near solver changed particle identities or masses");
        double norm = 0.0;
        double error = 0.0;
        for (int axis = 0; axis < kDimensions; ++axis) {
            requirePmmm(reference[i].rawPosition[axis] == actual[i].rawPosition[axis],
                "near solver changed a particle position");
            requirePmmm(std::isfinite(actual[i].acceleration[axis]),
                "near solver produced a nonfinite acceleration");
            norm += double(reference[i].acceleration[axis]) * reference[i].acceleration[axis];
            double const delta =
                double(reference[i].acceleration[axis]) - actual[i].acceleration[axis];
            error += delta * delta;
        }
        maximumError = std::max(maximumError, std::sqrt(error) / std::max(1.0, std::sqrt(norm)));
        requirePmmm(
            std::isfinite(actual[i].potential), "near solver produced a nonfinite potential");
        maximumError = std::max(maximumError,
            std::abs(double(reference[i].potential) - actual[i].potential) /
                std::max(1.0, std::abs(double(reference[i].potential))));
    }
    return maximumError;
}

Vector<PmmmSample> gatherPmmmKickSamples() {
    Vector<PmmmSample> samples;
    std::vector<hpx::future<Vector<PmmmSample>>> futures;
    for (auto const& locality : hpxLocalities()) {
        futures.push_back(hpx::async<ReadPmmmKickFixtureAction>(locality));
    }
    for (auto& future : futures) {
        auto local = future.get();
        samples.insert(samples.end(), local.begin(), local.end());
    }
    std::sort(samples.begin(), samples.end(), [](auto const& first, auto const& second) {
        for (int dim = 0; dim < kDimensions; ++dim) {
            if (first.rawPosition[dim] != second.rawPosition[dim]) {
                return first.rawPosition[dim] < second.rawPosition[dim];
            }
        }
        return false;
    });
    return samples;
}

void checkPmmmPpBatchTargets() {
    if constexpr (gravity::kBuildPmmmNearUsesFmm) {
        return;
    }
    auto const saved = getOptions();
    auto options = saved;
    options.gravity.meshSize = 8;
    options.gravity.pmmmNearWidth = 1;
    options.gravity.pmmmBatchParticles = 1;
    pmmmDestroy();
    setOptions(options);
    preparePmmmFixture(0);
    distributePmmmFixture(0);
    pmmmSolve(true);
    auto const baseline = gatherPmmmSamples();
    auto const small = pmmmDiagnostics();
    std::map<std::pair<int, uint64_t>, size_t> cellPopulations;
    std::map<int, size_t> rankPopulations;
    size_t largestCell = 0;
    size_t largestRank = 0;
    for (auto const& sample : baseline) {
        largestCell = std::max(largestCell, ++cellPopulations[{sample.rank, sample.cellKey}]);
        largestRank = std::max(largestRank, ++rankPopulations[sample.rank]);
    }
    requirePmmm(
        small.nearBatches == cellPopulations.size() && small.peakBatchParticles == largestCell,
        "PP target one did not submit one complete occupied cell per batch");
    options.gravity.pmmmBatchParticles = 4096;
    setOptions(options);
    pmmmSolve(true);
    comparePmmmSamples(baseline, gatherPmmmSamples());
    auto const large = pmmmDiagnostics();
    requirePmmm(
        large.nearBatches == rankPopulations.size() && large.peakBatchParticles == largestRank,
        "PP large target did not combine the locality's occupied cells");
    requirePmmm(large.planBuilds == small.planBuilds && large.greenBuilds == small.greenBuilds,
        "changing the PP batch target rebuilt its FFT or Green cache");
    pmmmDestroy();
    setOptions(saved);
}

void checkPmmmStreamedStages() {
    auto const saved = getOptions();
    auto options = saved;
    // Exercise the hierarchical stage operator directly in either integrator
    // build, without changing the application's compiled evolution policy.
    options.gravity.meshSize = 8;
    options.gravity.pmmmNearWidth = 2;
    options.gravity.fmmOrder = kFmmDefaultOrder;
    options.gravity.pmmmOrder = options.gravity.fmmOrder;
    options.gravity.openingAngle = 0.02;
    options.analysis.saveForce = false;
    options.analysis.saveAcceleration = false;
    options.analysis.savePotential = false;
    pmmmDestroy();
    for (auto const& nearMethod : {std::string(gravity::kBuildPmmmNearMethod)}) {
        options.gravity.pmmmNearMethod = nearMethod;
        for (int batchParticles : {1, 4096}) {
            options.gravity.pmmmBatchParticles = batchParticles;
            setOptions(options);
            // At cell centers PMMM matches the independent Ewald field at every
            // supported order. Off-center force accuracy is checked separately;
            // its finite-order truncation must not set the kick tolerance.
            preparePmmmFixture(8);
            distributePmmmFixture(8);
            auto const before = gatherPmmmKickSamples();
            Vector<PmmmSample> active;
            double activeMass = 0.0;
            for (auto const& sample : before) {
                if (sample.rung >= 3) {
                    active.push_back(sample);
                    activeMass += sample.mass;
                }
            }
            requirePmmm(active.size() == 3 && before.size() == 7,
                "streaming fixture did not contain both active and inactive particles");
            particlesSortByRung(3);
            KickParameters parameters{};
            parameters.htime = true;
            parameters.ascending = true;
            parameters.descending = false;
            parameters.top = batchParticles == 1;
            parameters.sign = -1.0f;
            parameters.firstCall = false;
            parameters.saveForce = false;
            parameters.gpu = options.compute.cuda;
            parameters.minRung = 3;
            parameters.minLevel = 0;
            parameters.a = 1.0f;
            parameters.t0 = 1.0f;
            parameters.theta = options.gravity.openingAngle;
            parameters.gravitySupportRadius = options.gravity.supportRadius;
            parameters.eta = options.gravity.timestepEta;
            parameters.gm = options.gravity.gm;
            parameters.nodeLoad = 1.0f;
            parameters.doPhi = true;
            pmmmKick(parameters);
            auto const after = gatherPmmmKickSamples();
            auto const diagnostics = pmmmDiagnostics();
            requirePmmm(diagnostics.storedAccelerationParticles == 0 &&
                    diagnostics.storedPotentialParticles == 0,
                "streamed stage allocated retained full-particle result arrays");
            requirePmmm(std::abs(diagnostics.sourceMass - activeMass) < 1.0e-6 &&
                    std::abs(diagnostics.depositedMass - activeMass) < 1.0e-6,
                "hierarchical stage included inactive sources in the mesh moments");
            requirePmmm(diagnostics.nearRemoteTreeFetches == 0 &&
                    diagnostics.nearRemoteParticleFetches == 0,
                "streamed local walk performed a remote fetch");
            requirePmmm(before.size() == after.size(), "streamed stage changed owned count");
            size_t selected = 0;
            for (size_t i = 0; i < before.size(); ++i) {
                bool const isActive = before[i].rung >= 3;
                auto const reference =
                    isActive ? directPmmmReference(selected++, active) : std::array<double, 4>{};
                double const factor = parameters.sign * (parameters.top ? 1.0 : -1.0) / 16.0;
                requirePmmm(
                    before[i].rung == after[i].rung, "ascending-only kick changed a particle rung");
                for (int dim = 0; dim < kDimensions; ++dim) {
                    requirePmmm(before[i].rawPosition[dim] == after[i].rawPosition[dim],
                        "streamed kick changed an owned position");
                    double const delta = factor * reference[dim + 1];
                    double const expected = before[i].velocity[dim] + delta;
                    requirePmmm(isActive ? std::abs(after[i].velocity[dim] - expected) <
                                2.0e-6 * std::max(1.0, std::abs(expected)) +
                                    1.0e-3 * std::abs(delta) :
                                           after[i].velocity[dim] == before[i].velocity[dim],
                        "stage kick differs from its active-source Ewald field or changed an "
                        "inactive sink");
                }
            }
            // A descending stage must select the next rung from the combined
            // field while independently saving just the requested diagnostic.
            options.analysis.saveAcceleration = batchParticles == 1;
            options.analysis.savePotential = batchParticles != 1;
            setOptions(options);
            parameters.ascending = false;
            parameters.descending = true;
            parameters.maxDt = parameters.t0 / 16.0f;
            parameters.eta = 1.0e6f;
            pmmmKick(parameters);
            auto const descending = gatherPmmmKickSamples();
            requirePmmm(descending.size() == before.size(), "descending stage changed owned count");
            selected = 0;
            for (size_t i = 0; i < before.size(); ++i) {
                bool const isActive = before[i].rung >= 3;
                auto const reference =
                    isActive ? directPmmmReference(selected++, active) : std::array<double, 4>{};
                double const factor = parameters.sign * (parameters.top ? 1.0 : -1.0) / 16.0;
                requirePmmm(descending[i].rung == (isActive ? 4 : before[i].rung),
                    "descending stage selected the wrong next rung");
                for (int dim = 0; dim < kDimensions; ++dim) {
                    requirePmmm(descending[i].rawPosition[dim] == before[i].rawPosition[dim],
                        "descending kick changed an owned position");
                    double const delta = factor * reference[dim + 1];
                    double const expected = after[i].velocity[dim] + delta;
                    requirePmmm(isActive ? std::abs(descending[i].velocity[dim] - expected) <
                                2.0e-6 * std::max(1.0, std::abs(expected)) +
                                    1.0e-3 * std::abs(delta) :
                                           descending[i].velocity[dim] == before[i].velocity[dim],
                        "descending kick differs from its active-source Ewald field or changed an "
                        "inactive sink");
                    if (isActive && options.analysis.saveAcceleration) {
                        requirePmmm(std::abs(descending[i].acceleration[dim] - reference[dim + 1]) <
                                1.0e-3 * std::max(1.0, std::abs(reference[dim + 1])),
                            "acceleration-only diagnostics differ from the combined active-source "
                            "field");
                    }
                }
                if (isActive && options.analysis.savePotential) {
                    requirePmmm(std::abs(descending[i].potential - reference[0]) <
                            1.0e-3 * std::max(1.0, std::abs(reference[0])),
                        "potential-only diagnostics differ from the combined active-source field");
                }
            }
            options.analysis.saveAcceleration = false;
            options.analysis.savePotential = false;
            setOptions(options);
            particlesSortByRung(0);
        }
    }
    pmmmDestroy();
    setOptions(saved);
    PRINT("PMMM streamed active-source stages, signed kicks, optional fields, and batch-size "
          "checks passed\n");
}

void checkPmmmCellOverlap() {
    if (hpxSize() < 2 || !gravity::kBuildPmmmNearUsesFmm) {
        return;
    }
    auto const saved = getOptions();
    auto options = saved;
    options.gravity.meshSize = 8;
    options.gravity.pmmmNearWidth = 1;
    options.gravity.fmmOrder = kFmmDefaultOrder;
    options.gravity.pmmmOrder = options.gravity.fmmOrder;
    options.gravity.pmmmBatchParticles = 1;
    options.gravity.openingAngle = 0.02;
    pmmmDestroy();
    setOptions(options);
    preparePmmmFixture(6);
    distributePmmmFixture(6);
    pmmmSolve(true);
    auto const reference = gatherPmmmSamples();
    bool foundGate = false;
    uint64_t heldKey = 0;
    for (auto const& source : reference) {
        if (source.rank == 0) {
            continue;
        }
        bool dependent = false;
        bool independent = false;
        for (auto const& sink : reference) {
            if (sink.rank == 0) {
                dependent |= pmmmCellsNeighbor(source.cellKey, sink.cellKey);
                independent |= !pmmmCellsNeighbor(source.cellKey, sink.cellKey);
            }
        }
        if (dependent && independent) {
            heldKey = source.cellKey;
            foundGate = true;
            break;
        }
    }
    requirePmmm(
        foundGate, "overlap fixture lacks independent and remote-dependent rank-zero sinks");
    auto setGate = [heldKey](bool enabled) {
        std::vector<hpx::future<void>> futures;
        for (auto const& locality : hpxLocalities()) {
            futures.push_back(hpx::async<InstallPmmmOverlapGateAction>(locality, enabled, heldKey));
        }
        for (auto& future : futures) {
            future.get();
        }
    };
    setGate(true);
    try {
        pmmmSolve(true);
        auto const state = hpx::async<ReadPmmmOverlapGateAction>(hpxLocalities()[0]).get();
        requirePmmm(state.size() == 4 && state[0] && state[1] && !state[2] && state[3],
            "whole-cell gate did not prove independent completion before remote source arrival");
        comparePmmmSamples(reference, gatherPmmmSamples());
        requirePmmm(pmmmDiagnostics().completedWhileGhostsPending > 0,
            "cell pipeline did not report completion while the deterministic ghost gate was held");
    } catch (...) {
        setGate(false);
        throw;
    }
    setGate(false);
    pmmmDestroy();
    setOptions(saved);
    PRINT("PMMM deterministic interior/boundary whole-cell overlap checks passed\n");
}

#if COSMICTIGER_PMMM_NEAR_FMM
void checkPmmmFmmNear() {
    if constexpr (!gravity::kBuildPmmmNearUsesFmm) {
        return;
    }
    auto const saved = getOptions();
    auto options = saved;
    // Isolated force fixtures exercise unequal masses without enabling an SPH
    // evolution. Startup continues to reject PMMM gravity in SPH simulations.
    options.hydro.enabled = COSMICTIGER_WITH_SPH;
    options.compute.bucketSize = 8;
    options.gravity.fmmOrder = kFmmDefaultOrder;
    options.gravity.pmmmOrder = options.gravity.fmmOrder;
    options.gravity.pmmmNearWidth = 1;
    options.gravity.supportRadius = 0.001;
    if (saved.compute.cuda) {
        setOptions(options);
        std::vector<hpx::future<void>> futures;
        for (auto const& locality : hpxLocalities()) {
            futures.push_back(hpx::async<ReservePmmmCudaFixtureAction>(locality));
        }
        for (auto& future : futures) {
            future.get();
        }
    }
    for (int side : {3, 5, 8}) {
        pmmmDestroy();
        options.compute.cuda = saved.compute.cuda;
        options.gravity.meshSize = side;
        options.gravity.openingAngle = 1.0e-6;
        setOptions(options);
        preparePmmmFixture(5);
        distributePmmmFixture(5);
        // The tree arena must also be allocated under the requested
        // backend before temporarily running the CPU reference on the same data.
        checkPmmmForestGeometry();
        options.compute.cuda = false;
        setOptions(options);
        pmmmSolve(true);
        // A very tight angle supplies a near-direct reference; point-sized
        // cells can still use exact monopoles. The small fixtures separately
        // check the independent periodic Ewald sum.
        auto const direct = gatherPmmmSamples();
        auto const cache = pmmmDiagnostics();
        requirePmmm(direct.size() == 320, "dense near fixture lost particles");
        if (hpxSize() > 1) {
            bool splitCell = false;
            for (auto const& first : direct) {
                for (auto const& second : direct) {
                    splitCell |= first.cellKey == second.cellKey && first.rank != second.rank;
                }
            }
            requirePmmm(splitCell, "dense fixture did not split a cell across valid domains");
        }
        // Clustered forces can strongly cancel. These angles test
        // a useful multipole approximation and convergence toward direct PP.
        options.gravity.openingAngle = 0.4;
        setOptions(options);
        pmmmSolve(true);
        auto const loose = gatherPmmmSamples();
        double const looseError = pmmmFieldDifference(direct, loose);
        auto const looseCounts = pmmmDiagnostics();
        requirePmmm(looseCounts.nearCpuAcceptedMultipoles > 0,
            "dense FMM near fixture never accepted a multipole interaction");
        if (hpxSize() > 1) {
            requirePmmm(looseCounts.ghostParticles > 0,
                "distributed dense fixture did not exchange ghost source particles");
        }
        options.gravity.openingAngle = 0.02;
        setOptions(options);
        pmmmSolve(true);
        auto const strict = gatherPmmmSamples();
        double const strictError = pmmmFieldDifference(direct, strict);
        auto const strictCounts = pmmmDiagnostics();
        requirePmmm(
            looseError < 2.0e-2, "loose FMM near field differs excessively from direct pairs");
        requirePmmm(strictError < 5.0e-5 && strictError <= std::max(2.0e-5, 0.5 * looseError),
            "tightening the FMM near angle did not converge to direct pairs");
        requirePmmm(strictCounts.nearCpuDirectPairs > 0,
            "dense FMM near fixture did not reach direct softened/self interactions");
        pmmmSolve(false);
        auto const forceOnly = gatherPmmmSamples();
        comparePmmmSamples(strict, forceOnly, false);
        for (auto const& sample : forceOnly) {
            requirePmmm(sample.potential == 0.0f, "force-only FMM near solve computed a potential");
        }
        requirePmmm(strictCounts.planBuilds == cache.planBuilds &&
                strictCounts.greenBuilds == cache.greenBuilds,
            "changing the near angle rebuilt FFT plans/Green spectra");
        pmmmSolve(true);
        comparePmmmSamples(strict, gatherPmmmSamples());
        requirePmmm(pmmmDiagnostics().ghostParticles == strictCounts.ghostParticles,
            "repeated FMM solve accumulated or lost ghost sources");
        preparePmmmFixture(5, 1);
        distributePmmmFixture(5);
        pmmmSolve(true);
        auto const repartitioned = gatherPmmmSamples();
        double const repartitionError = pmmmFieldDifference(direct, repartitioned);
        // Changing the kd decomposition changes the accepted multipoles. Test
        // both approximations against PP using the same force-vector criterion;
        // a nearly cancelling component need not match to relative float error.
        requirePmmm(repartitionError < 5.0e-5,
            "repartitioned FMM near field differs excessively from direct pairs");
        if (saved.compute.cuda) {
            options.compute.cuda = true;
            setOptions(options);
            pmmmSolve(true);
            auto const cuda = gatherPmmmSamples();
            requirePmmm(pmmmFieldDifference(direct, cuda) < 5.0e-5 &&
                    pmmmFieldDifference(repartitioned, cuda) < 5.0e-5,
                "CPU/CUDA FMM near fields disagree on dense periodic cells");
            options.compute.cuda = false;
            setOptions(options);
        }
        PRINT("PMMM FMM-near n=%i order=%i loose=%e strict=%e repartitioned=%e CPU accepted=%llu "
              "direct=%llu\n",
            side, options.gravity.fmmOrder, looseError, strictError, repartitionError,
            static_cast<unsigned long long>(looseCounts.nearCpuAcceptedMultipoles),
            static_cast<unsigned long long>(strictCounts.nearCpuDirectPairs));
    }
    pmmmDestroy();
    setOptions(saved);
}

#endif

}    // namespace detail

void pmmmTest() {
    auto options = getOptions();
    // Allocate type/category storage before the first fixture. Particle arrays
    // keep a shared capacity and cannot gain the optional hydro arrays after
    // allocation. This isolated force test performs no SPH evolution.
    options.hydro.enabled = COSMICTIGER_WITH_SPH;
    options.gravity.gm = 0.125;
    options.gravity.supportRadius = 0.001;
    options.gravity.fmmOrder = kFmmDefaultOrder;
    options.gravity.pmmmOrder = kFmmDefaultOrder;
    options.gravity.openingAngle = 0.05;
    options.analysis.saveForce = true;
    options.analysis.tracers = false;
    options.analysis.groups = false;
    options.initial.darkMatterMass = 0.75;
    options.initial.gasMass = 0.25;
    options.initial.createGlass = false;
    options.run.hierarchicalGravity =
        integration::kBuildUsesHierarchical && options.gravity.enabled;
    setOptions(options);
#if !COSMICTIGER_PMMM_NEAR_FMM
    detail::checkPmmmNearEdges();
    detail::checkPmmmNearBatches();
#endif
    std::array<std::array<int, 2>, 5> const configurations{
        {{5, 2}, {8, 1}, {8, 2}, {8, 3}, {10, 4}}};
    for (auto const configuration : configurations) {
        pmmmDestroy();
        options.gravity.meshSize = configuration[0];
        options.gravity.pmmmNearWidth = configuration[1];
        setOptions(options);
        for (int fixture : {0, 1, 2, 3, 4, 7}) {
            detail::preparePmmmFixture(fixture);
            detail::distributePmmmFixture(fixture);
            resetFlops();
            pmmmSolve(true);
            FlopCount const firstFlops = readFlops();
            detail::requirePmmm(firstFlops > 0, "PMMM solve did not submit arithmetic counts");
            auto const baseline = detail::gatherPmmmSamples();
            if (fixture == 7 && hpxSize() > 1) {
                bool splitCell = false;
                for (auto const& sample : baseline) {
                    detail::requirePmmm(sample.cellKey == baseline.front().cellKey,
                        "split-cell Ewald fixture spans more than one mesh cell");
                    splitCell |= sample.rank != baseline.front().rank;
                }
                detail::requirePmmm(
                    splitCell, "ordinary PMMM domains did not split the Ewald fixture's mesh cell");
            }
            double const error = detail::verifyPmmmReference(baseline);
            detail::distributePmmmFixture(fixture);
            resetFlops();
            pmmmSolve(true);
            FlopCount const cachedFlops = readFlops();
            detail::requirePmmm(cachedFlops > 0, "cached PMMM solve did not count its work");
            if (fixture == 0) {
                detail::requirePmmm(firstFlops > cachedFlops,
                    "cached PMMM solve still charged Green initialization");
            }
            detail::comparePmmmSamples(baseline, detail::gatherPmmmSamples());
            auto const cached = pmmmDiagnostics();
            resetFlops();
            pmmmSolve(false);
            FlopCount const forceFlops = readFlops();
            detail::requirePmmm(forceFlops > 0, "force-only PMMM solve did not count its work");
            if (fixture == 2) {
                detail::requirePmmm(
                    forceFlops == cachedFlops, "empty PMMM solve invented potential work");
                // With no particles, every remaining operation is mesh work.
                // This also detects charging FFTW's global MPI count per rank.
                auto const& mesh = pmmm::detail::cache;
                FlopCount const n = options.gravity.meshSize;
                detail::requirePmmm(mesh.forwardFlops > 0 && mesh.inverseFlops > 0,
                    "FFTW plan arithmetic was not cached on the root locality");
                FlopCount const expected = mesh.forwardFlops + mesh.inverseFlops + 4 * n * n * n +
                    10 * n * n * (n / 2 + 1) * mesh.terms.size() + 5 * FlopCount(hpxSize() - 1) +
                    (gravity::kBuildPmmmNearUsesFmm ? 0 : 4 * FlopCount(hpxSize()));
                detail::requirePmmm(forceFlops == expected,
                    "empty PMMM mesh operation total is incomplete or counted more than once: " +
                        std::to_string(forceFlops) + " expected " + std::to_string(expected));
            }
            detail::comparePmmmSamples(baseline, detail::gatherPmmmSamples(), false);
            pmmmSolve(true);
            detail::comparePmmmSamples(baseline, detail::gatherPmmmSamples());
            auto const repeated = pmmmDiagnostics();
            detail::requirePmmm(repeated.planBuilds == cached.planBuilds &&
                    repeated.greenBuilds == cached.greenBuilds &&
                    repeated.solveCount == cached.solveCount + 2,
                "repeated force/potential solves rebuilt cached plans or Green spectra");
            if (fixture == 0) {
                detail::preparePmmmFixture(fixture, 1);
                detail::distributePmmmFixture(fixture);
                pmmmSolve(true);
                detail::comparePmmmSamples(baseline, detail::gatherPmmmSamples());
                auto const beforeDestroy = pmmmDiagnostics();
                pmmmDestroy();
                pmmmSolve(true);
                detail::comparePmmmSamples(baseline, detail::gatherPmmmSamples());
                auto const recreated = pmmmDiagnostics();
                detail::requirePmmm(recreated.planBuilds == beforeDestroy.planBuilds + 1 &&
                        recreated.greenBuilds == beforeDestroy.greenBuilds + 1,
                    "destroy/recreate did not rebuild exactly one cached plan/Green set");
            }
            PRINT("PMMM Ewald oracle n=%i order=%i near=%i fixture=%i error=%e\n", configuration[0],
                kFmmDefaultOrder, configuration[1], fixture, error);
        }
    }
    pmmmDestroy();
#if COSMICTIGER_PMMM_NEAR_FMM
    detail::checkPmmmFmmNear();
#endif
    detail::checkPmmmPpBatchTargets();
    detail::checkPmmmStreamedStages();
    detail::checkPmmmCellOverlap();
    PRINT("PMMM periodic force, mass, self, ownership, and cache checks passed at order %i "
          "with %s near interactions\n",
        kFmmDefaultOrder, gravity::kBuildPmmmNearMethod);
}
