/* CosmicTiger - optional complete-solve profiling fixture; BSL-1.0. */
// Link with the application's objects and -Wl,--wrap=_Z8pmmmTestv. This replaces
// only the pmmm test payload; it adds no production option or solver branch.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <vector>

#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/gravity/solver.hpp>
#include <cosmictiger/gravity/tree.hpp>
#include <cosmictiger/particles/particles.hpp>
#include <src/gravity/state.hpp>

namespace detail {
void preparePmmmFixture(int fixture, int ownership);
void distributePmmmFixture(int fixture);

std::atomic<size_t> profileDepth{0};
std::atomic<size_t> profileCells{0};
std::atomic<size_t> profileActiveCells{0};
std::atomic<size_t> profileEarlyCells{0};

size_t profileTreeDepth(TreeId id) {
    auto const* node = treeGetNode(id);
    return node->leaf ?
        1 :
        1 + std::max(profileTreeDepth(node->children[0]), profileTreeDepth(node->children[1]));
}

size_t prepareProfile(int denominator) {
    profileDepth = profileCells = profileActiveCells = profileEarlyCells = 0;
    size_t active = 0;
    for (ParticleInteger i = 0; i < particlesOwnedSize(); ++i) {
        int const id = int(particlesVel(0, i)) - 1;
        bool const selected = id % denominator == 0;
        particlesRung(i) = selected ? 2 : 0;
        active += selected;
    }
    pmmmSetCellObserver([](PmmmCellEvent event, uint64_t key) {
        if (event != PmmmCellEvent::NearCompleted) {
            return;
        }
        ++profileCells;
        bool const early = !pmmm::detail::state.farReady.is_ready();
        auto const index = pmmm::detail::state.sourceCellsByKey.at(key);
        auto const root = pmmm::detail::state.sourceCells[index]->ready.get();
        auto const range = treeGetNode(root.id)->particleRange;
        bool selected = false;
        for (ParticleInteger i = range.first; i < range.second; ++i) {
            if (particlesRung(i) >= pmmm::detail::state.request.parameters.minRung) {
                selected = true;
                break;
            }
        }
        profileActiveCells += selected;
        profileEarlyCells += selected && early;
        size_t const depth = profileTreeDepth(root.id);
        size_t previous = profileDepth.load();
        while (previous < depth && !profileDepth.compare_exchange_weak(previous, depth)) {
        }
    });
    return active;
}

Vector<uint64_t> finishProfile() {
    pmmmSetCellObserver({});
    return {profileDepth.load(), profileCells.load(), profileEarlyCells.load(),
        profileActiveCells.load()};
}
}    // namespace detail

HPX_PLAIN_ACTION(detail::prepareProfile, PrepareSphericalProfileAction);
HPX_PLAIN_ACTION(detail::finishProfile, FinishSphericalProfileAction);

extern "C" void __wrap__Z8pmmmTestv() {
    auto const saved = getOptions();
    auto options = saved;
    options.hydro.enabled = false;
    options.gravity.cuda = false;
    options.gravity.pmmmNearMethod = "fmm";
    options.gravity.pmmmNearWidth = 1;
    options.gravity.pmmmCoarseMeshSize = 0;
    options.gravity.openingAngle = .05;
    options.gravity.smoothingLength = .001;
    options.gravity.gm = .125;
    options.analysis.saveForce = true;
    options.initial.darkMatterMass = .75;
    options.initial.gasMass = .25;
    options.compute.bucketSize = 8;
    options.run.hierarchicalGravity = options.gravity.enabled;
    for (int mesh : {8, 16}) {
        pmmmDestroy();
        options.gravity.meshSize = mesh;
        setOptions(options);
        for (int fixture : {6, 5}) {
            for (int denominator : {1, 4, 16}) {
                for (bool potential : {false, true}) {
                    for (int repeat = -1; repeat < 3; ++repeat) {
                        detail::preparePmmmFixture(fixture, 0);
                        detail::distributePmmmFixture(fixture);
                        std::vector<hpx::future<size_t>> prepares;
                        for (auto const& locality : hpxLocalities()) {
                            prepares.push_back(
                                hpx::async<PrepareSphericalProfileAction>(locality, denominator));
                        }
                        size_t active = 0;
                        for (auto& future : prepares) {
                            active += future.get();
                        }
                        KickParameters parameters{};
                        parameters.htime = true;
                        parameters.minRung = 1;
                        parameters.a = parameters.t0 = parameters.maxDt = 1;
                        parameters.eta = 1e6f;
                        parameters.gravitySmoothingLength = options.gravity.smoothingLength;
                        parameters.gm = options.gravity.gm;
                        parameters.theta = options.gravity.openingAngle;
                        parameters.saveForce = true;
                        parameters.doPhi = potential;
                        auto const previousGreenBuilds = pmmmDiagnostics().greenBuilds;
                        auto const begin = std::chrono::steady_clock::now();
                        pmmmKick(parameters);
                        double const seconds =
                            std::chrono::duration<double>(std::chrono::steady_clock::now() - begin)
                                .count();
                        auto const diagnostics = pmmmDiagnostics();
                        if (diagnostics.evaluatedParticles != active ||
                            diagnostics.storedAccelerationParticles != 0 ||
                            diagnostics.storedPotentialParticles != 0) {
                            throw std::runtime_error(
                                "profile fixture lost its active selection or streaming storage");
                        }
                        size_t depth = 0, cells = 0, early = 0, activeCells = 0;
                        for (auto const& locality : hpxLocalities()) {
                            auto const metrics =
                                hpx::async<FinishSphericalProfileAction>(locality).get();
                            depth = std::max(depth, size_t(metrics[0]));
                            cells += metrics[1];
                            early += metrics[2];
                            activeCells += metrics[3];
                        }
                        if (early > activeCells || activeCells > cells ||
                            (active != 0 && activeCells == 0)) {
                            throw std::runtime_error(
                                "profile fixture lost its active cell mapping");
                        }
                        bool const cold = diagnostics.greenBuilds != previousGreenBuilds;
                        std::printf(
                            "PMMM_PROFILE {\"mesh\":%d,\"fixture\":%d,\"denominator\":%d,"
                            "\"potential\":%s,\"repeat\":%d,\"cold\":%s,\"seconds\":%.9g,"
                            "\"particles\":%llu,\"evaluated\":%llu,\"cells\":%zu,"
                            "\"source_depth\":%zu,\"pre_fft_near_cells\":%zu,\"active_cells\":%zu,"
                            "\"peak_batch_particles\":%llu,\"ghost_bytes\":%llu}\n",
                            mesh, fixture, denominator, potential ? "true" : "false", repeat,
                            cold ? "true" : "false", seconds,
                            static_cast<unsigned long long>(diagnostics.totalParticles),
                            static_cast<unsigned long long>(diagnostics.evaluatedParticles), cells,
                            depth, early, activeCells,
                            static_cast<unsigned long long>(diagnostics.peakBatchParticles),
                            static_cast<unsigned long long>(diagnostics.ghostBytes));
                        std::fflush(stdout);
                    }
                }
            }
        }
    }
    pmmmDestroy();
    setOptions(saved);
}
