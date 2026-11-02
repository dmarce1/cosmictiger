/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <shared_mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <hpx/serialization/array.hpp>
#include <hpx/serialization/output_archive.hpp>
#include <hpx/threading_base/annotated_function.hpp>

#include <cosmictiger/core/containers.hpp>
#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/core/profiler.hpp>
#include <cosmictiger/core/safe_io.hpp>
#include <cosmictiger/gravity/nitadori/mesh_size.hpp>
#include <cosmictiger/math/math.hpp>
#include <cosmictiger/math/vector3.hpp>
#include <cosmictiger/particles/domain.hpp>
#include <cosmictiger/particles/particle_mesh_sort.hpp>
#include <cosmictiger/particles/particles.hpp>
#if COSMICTIGER_WITH_SPH
#include <cosmictiger/hydro/sph/sph_particles.hpp>
#include <cosmictiger/particles/sph_migration.hpp>
#endif

constexpr bool kVerbose = true;

void domainsTransmitParticles(Vector<Particle>);

void domainsInitRebounds();
void domainsTransmitBoxes(std::unordered_map<size_t, DomainT> boxes);
void domainsReboundSort(Vector<double> bounds, int depth);
Vector<size_t> domainsCountBelow(Vector<double> bounds, int depth);
std::vector<std::vector<size_t>> domainsCountCoarseSlabs(int depth, int coarseMeshSize);
Vector<size_t> domainsGetLoads();

HPX_PLAIN_ACTION(domainsCountBelow, DomainsCountBelowAction);
HPX_PLAIN_ACTION(domainsCountCoarseSlabs, DomainsCountCoarseSlabsAction);
HPX_PLAIN_ACTION(domainsGetLoads, DomainsGetLoadsAction);
HPX_PLAIN_ACTION(domainsReboundSort, DomainsReboundSortAction);
HPX_PLAIN_ACTION(domainsBegin, DomainsBeginAction);
HPX_PLAIN_ACTION(domainsEnd, DomainsEndAction);
HPX_PLAIN_ACTION(domainsInitRebounds, DomainsInitReboundsAction);
HPX_PLAIN_ACTION(domainsTransmitParticles, DomainsTransmitParticlesAction);
HPX_PLAIN_ACTION(domainsTransmitBoxes, DomainsTransmitBoxesAction);

static Vector<ParticleInteger> freeIndices;
static Vector<Vector<Particle>> transParticles;
static std::unordered_map<size_t, DomainT> boxesByKey;

static int findParticleDomain(Vector3<double> const& x, size_t key = 1, int depth = 0);
static void domainsCheck();
static Vector<DomainLocal> localDomains;

namespace detail {

template <typename Record>
size_t migrationPayloadBytes(Record const& record, size_t count) {
    std::vector<char> buffer;
    Vector<Record> records(count, record);
    hpx::serialization::output_archive archive(buffer);
    archive << records;
    return archive.bytes_written();
}

Vector<size_t> countDomainSplits(Vector<double> const& bounds, int depth) {
    profile::ScopedPhase profilePhase("domains.rebound.countSplits");
    Vector<hpx::future<Vector<size_t>>> pending;
    for (auto const& locality : hpxLocalities()) {
        pending.push_back(hpx::async<DomainsCountBelowAction>(locality, bounds, depth));
    }
    Vector<size_t> counts(bounds.size(), 0);
    for (auto& future : pending) {
        auto const localCounts = future.get();
        for (size_t i = 0; i < counts.size(); ++i) {
            counts[i] += localCounts[i];
        }
    }
    return counts;
}

// Sum each locality's coarse-slab histograms for the domains at this depth.
std::vector<std::vector<size_t>> countDomainCoarseSlabs(int depth, int coarseMeshSize) {
    profile::ScopedPhase profilePhase("domains.rebound.countSlabs");
    Vector<hpx::future<std::vector<std::vector<size_t>>>> pending;
    for (auto const& locality : hpxLocalities()) {
        pending.push_back(
            hpx::async<DomainsCountCoarseSlabsAction>(locality, depth, coarseMeshSize));
    }
    std::vector<std::vector<size_t>> counts;
    for (auto& future : pending) {
        auto const localCounts = future.get();
        if (counts.empty()) {
            counts.resize(localCounts.size());
        }
        ALWAYS_ASSERT(localCounts.size() == counts.size());
        for (size_t i = 0; i < localCounts.size(); ++i) {
            if (localCounts[i].empty()) {
                continue;
            }
            if (counts[i].empty()) {
                counts[i].resize(localCounts[i].size(), 0);
            }
            ALWAYS_ASSERT(localCounts[i].size() == counts[i].size());
            for (size_t slab = 0; slab < localCounts[i].size(); ++slab) {
                counts[i][slab] += localCounts[i][slab];
            }
        }
    }
    return counts;
}

// PMMM particle domains cut only on coarse-mesh planes, so no cell of the coarse
// mesh or of any finer power-of-two mesh ever spans two localities. The cut at
// this depth bisects the locality range; each child must keep enough coarse
// cells along this axis for the cuts it still owes on it. Among the admissible
// planes, the one whose particle count comes closest to the balanced target
// wins, not the plane nearest the exact-balance coordinate; ties prefer the
// plane nearest the middle of the parent, then the lower plane.
Vector<double> alignedDomainCuts(
    Vector<DomainGlobal> const& domains, int depth, int coarseMeshSize) {
    auto const histograms = countDomainCoarseSlabs(depth, coarseMeshSize);
    ALWAYS_ASSERT(histograms.size() == domains.size());
    int const axis = depth % kDimensions;
    auto const coarseIndex = [coarseMeshSize](double coordinate) {
        auto const index = std::llround(coordinate * coarseMeshSize);
        ALWAYS_ASSERT(
            index >= 0 && index <= coarseMeshSize && double(index) / coarseMeshSize == coordinate);
        return int(index);
    };
    Vector<double> bounds(domains.size());
    for (size_t i = 0; i < domains.size(); ++i) {
        auto const& domain = domains[i];
        int const processes = domain.processRange.second - domain.processRange.first;
        bounds[i] = domain.box.begin[axis];
        if (processes <= 1) {
            continue;
        }
        if (!pmmm::detail::isPowerOfTwo(uint64_t(processes))) {
            throw std::runtime_error("PMMM particle domains require a power-of-two locality count");
        }
        int levels = 0;
        while ((1 << levels) < processes) {
            ++levels;
        }
        int remainingCuts = 0;
        for (int level = 1; level < levels; ++level) {
            if ((depth + level) % kDimensions == axis) {
                ++remainingCuts;
            }
        }
        int const need = 1 << remainingCuts;
        int const lo = coarseIndex(domain.box.begin[axis]);
        int const hi = coarseIndex(domain.box.end[axis]);
        if (hi - lo < 2 * need) {
            throw std::runtime_error(
                "PMMM coarse mesh has too few cells per axis for the locality count");
        }
        auto const& histogram = histograms[i];
        ALWAYS_ASSERT(histogram.size() == size_t(coarseMeshSize));
        size_t const targetCount = size_t(processes / 2) * domain.totalCount / processes;
        size_t below = 0;
        for (int slab = lo; slab < lo + need; ++slab) {
            below += histogram[slab];
        }
        int best = -1;
        size_t bestError = std::numeric_limits<size_t>::max();
        int bestCenterDistance = std::numeric_limits<int>::max();
        for (int plane = lo + need; plane <= hi - need; ++plane) {
            size_t const error = below > targetCount ? below - targetCount : targetCount - below;
            int const centerDistance = std::abs(2 * plane - (lo + hi));
            if (error < bestError || (error == bestError && centerDistance < bestCenterDistance)) {
                best = plane;
                bestError = error;
                bestCenterDistance = centerDistance;
            }
            below += histogram[plane];
        }
        ALWAYS_ASSERT(best > lo && best < hi);
        bounds[i] = double(best) / coarseMeshSize;
    }
    return bounds;
}

}    // namespace detail

double domainsGetLoadImbalance() {
    auto const loads = domainsGetLoads();
    size_t max = 0;
    double avg = 0;
    for (auto const load : loads) {
        max = std::max(load, max);
        avg += load;
    }
    avg /= loads.size();
    double const imbalance = max / avg - 1.0;
    return imbalance;
}

Vector<size_t> domainsGetLoads() {
    Vector<hpx::future<Vector<size_t>>> futs;
    for (auto const& c : hpxChildren()) {
        futs.push_back(hpx::async<DomainsGetLoadsAction>(c));
    }
    Vector<size_t> loads;
    loads.push_back(particlesSize());
    for (auto& f : futs) {
        auto theseLoads = f.get();
        loads.insert(loads.end(), theseLoads.begin(), theseLoads.end());
    }
    return loads;
}

void domainsSave(FILE* fp) {
    size_t size = boxesByKey.size();
    fwrite((char const*) &size, sizeof(size_t), 1, fp);
    for (auto i = boxesByKey.begin(); i != boxesByKey.end(); i++) {
        fwrite(&(*i), sizeof(std::pair<size_t, DomainT>), 1, fp);
    }
    fwrite(localDomains.data(), sizeof(DomainLocal), hpxSize(), fp);
}

void domainsLoad(FILE* fp) {
    boxesByKey = decltype(boxesByKey)();
    size_t size;
    FREAD(&size, sizeof(size_t), 1, fp);
    for (int i = 0; i < size; i++) {
        std::pair<size_t, DomainT> entry;
        FREAD(&entry, sizeof(entry), 1, fp);
        boxesByKey[entry.first] = entry.second;
    }
    localDomains.resize(hpxSize());
    FREAD(localDomains.data(), sizeof(DomainLocal), hpxSize(), fp);
}

Range<double> domainsRange(size_t key) {
    return boxesByKey[key].box;
}

void domainsTransmitBoxes(std::unordered_map<size_t, DomainT> boxes) {
    Vector<hpx::future<void>> futs;
    for (auto const& c : hpxChildren()) {
        futs.push_back(hpx::async<DomainsTransmitBoxesAction>(c, boxes));
    }
    boxesByKey = std::move(boxes);
    hpx::wait_all(futs.begin(), futs.end());
}

void domainsInitRebounds() {
    Vector<hpx::future<void>> futs;
    for (auto const& c : hpxChildren()) {
        futs.push_back(hpx::async<DomainsInitReboundsAction>(c));
    }
    localDomains.resize(1);
    localDomains[0].box = unitBox<double>();
    localDomains[0].processRange.first = 0;
    localDomains[0].processRange.second = hpxSize();
    localDomains[0].particleRange.first = 0;
    localDomains[0].particleRange.second = particlesSize();
    localDomains[0].depth = 0;
    //	PRINT( "Setting init rebound on %i %i %i\n", hpx_rank(), 0, particles_size());
    hpx::wait_all(futs.begin(), futs.end());
}

Vector<size_t> domainsCountBelow(Vector<double> bounds, int depth) {
    assert(bounds.size() == localDomains.size());
    Vector<size_t> counts(bounds.size());
    Vector<hpx::future<void>> futs;
    for (int i = 0; i < localDomains.size(); i++) {
        if (localDomains[i].depth == depth) {
            futs.push_back(hpx::async(hpx::annotated_function(
                [i, &bounds, depth, &counts]() {
                    //							PRINT("???? %i %i %i %i \n", hpx_rank(), i,
                    // local_domains[i].part_range.first, local_domains[i].part_range.second);
                    auto const rng = localDomains[i].particleRange;
                    // Root-created initial conditions leave other localities empty
                    // until this rebalance migrates their particles. Empty ranges
                    // contribute zero without dividing by the local population.
                    if (rng.first == rng.second) {
                        counts[i] = 0;
                        return;
                    }
                    int const numberOfThreads = std::max(2 * (size_t) (rng.second - rng.first) *
                            hpxHardwareConcurrency() / (size_t) particlesSize(),
                        (size_t) 1);
                    Vector<hpx::future<void>> futs;
                    int const xdim = depth % kDimensions;
                    Position const xmid(bounds[i]);
                    std::atomic<size_t> count(0);
                    auto const func = [numberOfThreads, rng, xdim, xmid, &count](int proc) {
                        size_t const begin =
                            rng.first + (size_t) proc * (rng.second - rng.first) / numberOfThreads;
                        size_t const end = rng.first +
                            (size_t) (proc + 1) * (rng.second - rng.first) / numberOfThreads;
                        size_t thisCount = 0;
                        for (size_t i = begin; i < end; i++) {
                            if (particlesPos(xdim, i) < xmid) {
                                thisCount++;
                            }
                        }
                        count += thisCount;
                    };

                    for (int proc = 1; proc < numberOfThreads; proc++) {
                        futs.push_back(hpx::async(
                            hpx::annotated_function(func, "particles.domainsCountBelowChunk"),
                            proc));
                    }
                    func(0);
                    hpx::wait_all(futs.begin(), futs.end());
                    counts[i] = count;
                },
                "particles.domainsCountBelow")));
        }
    }
    hpx::wait_all(futs.begin(), futs.end());
    return counts;
}

// Per-domain histogram of this locality's particles over the coarse-mesh slabs
// along the cut axis. Domains at other depths return empty histograms.
std::vector<std::vector<size_t>> domainsCountCoarseSlabs(int depth, int coarseMeshSize) {
    std::vector<std::vector<size_t>> counts(localDomains.size());
    Vector<hpx::future<void>> futs;
    int const xdim = depth % kDimensions;
    for (int i = 0; i < localDomains.size(); i++) {
        if (localDomains[i].depth != depth) {
            continue;
        }
        counts[i].assign(size_t(coarseMeshSize), 0);
        futs.push_back(hpx::async(hpx::annotated_function(
            [i, xdim, coarseMeshSize, &counts]() {
                auto const rng = localDomains[i].particleRange;
                if (rng.first == rng.second) {
                    return;
                }
                int const numberOfThreads = std::max(2 * (size_t) (rng.second - rng.first) *
                        hpxHardwareConcurrency() / (size_t) particlesSize(),
                    (size_t) 1);
                std::vector<std::vector<size_t>> partial(
                    numberOfThreads, std::vector<size_t>(size_t(coarseMeshSize), 0));
                auto const func = [numberOfThreads, rng, xdim, coarseMeshSize, &partial](int proc) {
                    size_t const begin =
                        rng.first + (size_t) proc * (rng.second - rng.first) / numberOfThreads;
                    size_t const end = rng.first +
                        (size_t) (proc + 1) * (rng.second - rng.first) / numberOfThreads;
                    auto& slabs = partial[proc];
                    for (size_t j = begin; j < end; j++) {
                        slabs[detail::particleMeshCoordinate(
                            particlesPos(xdim, j).raw(), uint32_t(coarseMeshSize))]++;
                    }
                };
                Vector<hpx::future<void>> chunks;
                for (int proc = 1; proc < numberOfThreads; proc++) {
                    chunks.push_back(hpx::async(
                        hpx::annotated_function(func, "particles.domainsCountCoarseSlabsChunk"),
                        proc));
                }
                func(0);
                hpx::wait_all(chunks.begin(), chunks.end());
                for (auto const& slabs : partial) {
                    for (int slab = 0; slab < coarseMeshSize; ++slab) {
                        counts[i][slab] += slabs[slab];
                    }
                }
            },
            "particles.domainsCountCoarseSlabs")));
    }
    hpx::wait_all(futs.begin(), futs.end());
    return counts;
}

void domainsReboundSort(Vector<double> bounds, int depth) {
    assert(bounds.size() == localDomains.size());
    Vector<size_t> mids(bounds.size());
    Vector<hpx::future<void>> futs;
    for (int i = 0; i < localDomains.size(); i++) {
        if (localDomains[i].depth == depth) {
            futs.push_back(hpx::async(hpx::annotated_function(
                [i, &bounds, depth, &mids]() {
                    mids[i] = particlesSort(
                        localDomains[i].particleRange, bounds[i], depth % kDimensions);
                    ///	PRINT( "!!!!!!!! %i %i %e %i %i\n", hpx_rank(), mids[i], bounds[i],
                    /// local_domains[i].part_range.first, local_domains[i].part_range.second);
                },
                "particles.domainsReboundSort")));
        }
    }
    hpx::wait_all(futs.begin(), futs.end());
    Vector<DomainLocal> newDomains;
    for (int i = 0; i < localDomains.size(); i++) {
        if (localDomains[i].processRange.second - localDomains[i].processRange.first > 1) {
            DomainLocal left;
            DomainLocal right;
            left = localDomains[i];
            right = localDomains[i];
            left.box.end[depth % kDimensions] = right.box.begin[depth % kDimensions] = bounds[i];
            left.particleRange.second = right.particleRange.first = mids[i];
            left.processRange.second = right.processRange.first =
                (localDomains[i].processRange.first + localDomains[i].processRange.second) / 2;
            left.depth = right.depth = localDomains[i].depth + 1;
            newDomains.push_back(left);
            newDomains.push_back(right);
        } else {
            newDomains.push_back(localDomains[i]);
        }
    }
    localDomains = std::move(newDomains);
}

void domainsRebound() {
    {
        profile::ScopedPhase profilePhase("domains.rebound.initialize");
        domainsInitRebounds();
    }
    //	PRINT( "Starting rebound\n");
    Vector<DomainGlobal> domains(1);
    boxesByKey = decltype(boxesByKey)();
    domains[0].box = unitBox<double>();
    domains[0].processRange.first = 0;
    domains[0].processRange.second = hpxSize();
    // Both gravity solvers balance the current particle population. Global FMM
    // cuts anywhere; PMMM cuts on coarse-mesh planes so no mesh cell spans two
    // localities, choosing the admissible plane with the best particle balance.
    auto const& gravityOptions = getOptions().gravity;
    int const coarseMeshSize = usesMeshGravity(gravityOptions) ?
        pmmm::detail::resolvedCoarseMeshSize(gravityOptions.meshSize,
            gravityOptions.pmmmCoarseMeshSize, gravityOptions.pmmmNearWidth) :
        0;
    domains[0].totalCount = 0;
    auto const loads = []() {
        profile::ScopedPhase profilePhase("domains.rebound.loads");
        return domainsGetLoads();
    }();
    for (auto const count : loads) {
        domains[0].totalCount += count;
    }
    domains[0].midhi = 1.0;
    domains[0].midlo = 0.0;
    domains[0].key = 1;
    boxesByKey[domains[0].key].box = domains[0].box;
    boxesByKey[domains[0].key].rank =
        domains[0].processRange.second - domains[0].processRange.first > 1 ?
        -1 :
        domains[0].processRange.first;
    Vector<DomainGlobal> newDomains;
    std::swap(domains, newDomains);
    int depth = 0;
    auto const allLeaves = [](Vector<DomainGlobal> const& domains) {
        bool all = true;
        for (int i = 0; i < domains.size(); i++) {
            if (domains[i].processRange.second - domains[i].processRange.first > 1) {
                all = false;
                break;
            }
        }
        return all;
    };
    //	PRINT( "Cycling rebound\n");
    while (!allLeaves(newDomains)) {
        //		PRINT( "Cycle\n");
        domains = std::move(newDomains);
        Vector<size_t> counts(domains.size());
        Vector<double> bounds;
        if (coarseMeshSize > 0) {
            bounds = detail::alignedDomainCuts(domains, depth, coarseMeshSize);
        } else {
            for (int iter = 0; iter < kDomainReboundIterations; ++iter) {
                bounds.clear();
                for (auto const& domain : domains) {
                    bounds.push_back((domain.midhi + domain.midlo) * 0.5);
                }
                counts = detail::countDomainSplits(bounds, depth);
                for (size_t i = 0; i < domains.size(); ++i) {
                    int const firstProcess = domains[i].processRange.first;
                    int const lastProcess = domains[i].processRange.second;
                    int const middleProcess = (firstProcess + lastProcess) / 2;
                    ALWAYS_ASSERT(lastProcess - firstProcess >= 1);
                    size_t const targetCount = size_t(middleProcess - firstProcess) *
                        domains[i].totalCount / (lastProcess - firstProcess);
                    if (counts[i] < targetCount) {
                        domains[i].midlo = bounds[i];
                    } else {
                        domains[i].midhi = bounds[i];
                    }
                }
            }
            bounds.clear();
            for (auto const& domain : domains) {
                bounds.push_back(
                    std::min((domain.midhi + domain.midlo) * 0.5, std::nextafter(1.0, 0.0)));
            }
        }
        for (size_t i = 0; i < domains.size(); ++i) {
            // Counting and sorting use Position comparisons. Store the same
            // representable plane so migration agrees after repeated cuts on
            // an axis, including when the double midpoint rounds up to one.
            // Coarse-mesh planes are exact in both representations.
            double const midx = positionDomainCoordinate(Position(bounds[i]));
            boxesByKey[domains[i].key].midx = midx;
            bounds[i] = midx;
        }
        // Child targets must use the count at the actual final split plane.
        counts = detail::countDomainSplits(bounds, depth);
        {
            profile::ScopedPhase profilePhase("domains.rebound.sort");
            Vector<hpx::future<void>> sortFuts;
            for (int i = 0; i < hpxSize(); i++) {
                sortFuts.push_back(
                    hpx::async<DomainsReboundSortAction>(hpxLocalities()[i], bounds, depth));
            }
            hpx::wait_all(sortFuts.begin(), sortFuts.end());
        }
        newDomains.resize(0);
        for (int i = 0; i < domains.size(); i++) {
            if (domains[i].processRange.second - domains[i].processRange.first > 1) {
                DomainGlobal left;
                DomainGlobal right;
                left = domains[i];
                right = domains[i];
                left.box.end[depth % kDimensions] = right.box.begin[depth % kDimensions] =
                    bounds[i];
                left.processRange.second = right.processRange.first =
                    (domains[i].processRange.first + domains[i].processRange.second) / 2;
                int const leftdim = (depth + 1) % kDimensions;
                int const rightdim = (depth + 1) % kDimensions;
                left.totalCount = counts[i];
                right.totalCount = domains[i].totalCount - counts[i];
                left.midhi = left.box.end[leftdim];
                left.midlo = left.box.begin[leftdim];
                right.midhi = right.box.end[rightdim];
                right.midlo = right.box.begin[rightdim];
                left.key = domains[i].key << 1;
                right.key = (domains[i].key << 1) + 1;
                boxesByKey[left.key].box = left.box;
                boxesByKey[right.key].box = right.box;
                boxesByKey[left.key].rank = left.processRange.second - left.processRange.first > 1 ?
                    -1 :
                    left.processRange.first;
                boxesByKey[right.key].rank =
                    right.processRange.second - right.processRange.first > 1 ?
                    -1 :
                    right.processRange.first;
                newDomains.push_back(left);
                newDomains.push_back(right);
            } else {
                newDomains.push_back(domains[i]);
            }
        }
        depth++;
    }
    {
        profile::ScopedPhase profilePhase("domains.rebound.publish");
        domainsTransmitBoxes(boxesByKey);
    }
}

void domainsTransmitParticles(Vector<Particle> parts) {
    static MutexType mutex;
    std::lock_guard<MutexType> lock(mutex);
    transParticles.push_back(parts);
}

#if COSMICTIGER_WITH_SPH
// Ordinary SPH timesteps migrate the complete gas state together with its
// gravity particle. Category indices are local and are rebuilt at the receiver.
struct GasMigrationState {
    Vector3<float> acceleration, predictedAcceleration, gravityAcceleration;
    float supportRadius, internalEnergy, predictedInternalEnergyRate, internalEnergyRate, density;
    float mass;
    HopkinsParticleState hopkins;
    char converged, semiactive;
    Pair<float> viscosity;    // Alpha and its previous conformal-time rate.
    float divergence;         // Previous physical divergence for Cullen-Dehnen and Price et al.
    gadget4::State chemistry;
    Pair<float> chemistryEos;
    sph_hierarchical::SourceState hierarchicalState;
    float gradHCorrection, balsaraFactor;
    template <class Archive>
    void serialize(Archive& arc, unsigned) {
        arc & acceleration;
        arc & predictedAcceleration;
        arc & gravityAcceleration;
        arc & supportRadius;
        arc & internalEnergy;
        arc & predictedInternalEnergyRate;
        arc & internalEnergyRate;
        arc & density;
        arc & converged;
        arc & semiactive;
        if (getOptions().hydro.method == "mfv") {
            arc & mass;
        }
        if (getOptions().hydro.method != "sph") {
            arc & hopkins;
        }
        if (getOptions().hydro.viscosityEvolution != "none") {
            arc & viscosity;
        }
        if (getOptions().hydro.viscosityGradientSwitch()) {
            arc & divergence;
        }
        if (getOptions().chemistry.enabled) {
            arc & chemistry;
        }
        if (getOptions().chemistry.enabled || getOptions().starFormation.enabled) {
            arc & chemistryEos;
        }
        if (getOptions().run.hierarchicalGravity && getOptions().hydro.method == "sph") {
            arc & hierarchicalState;
            arc & gradHCorrection;
            arc & balsaraFactor;
        }
    }
};
struct SphMigrationParticle {
    Particle part;
    GasMigrationState gas{};
    template <class Archive>
    void serialize(Archive& arc, unsigned) {
        arc & part;
        if (part.type == kSphType) {
            arc & gas;
        }
    }
};
static Vector<SphMigrationParticle> retainedSphParticles;
static detail::SphMigrationSuffix<ParticleInteger> hierarchicalSphMigration;
struct SphMigrationParcel {
    int sourceRank;
    size_t sequence;
    Vector<SphMigrationParticle> particles;
};
static Vector<SphMigrationParcel> receivedSphParticles;
void receiveSphParticles(Vector<SphMigrationParticle> parts, int sourceRank, size_t sequence) {
    static MutexType mutex;
    std::lock_guard<MutexType> lock(mutex);
    receivedSphParticles.push_back({sourceRank, sequence, std::move(parts)});
}
HPX_PLAIN_ACTION(receiveSphParticles, ReceiveSphParticlesAction);
static SphMigrationParticle packSphParticle(ParticleInteger index) {
    SphMigrationParticle result;
    result.part = particlesGetParticle(index);
    ALWAYS_ASSERT(result.part.type == kDarkMatterType || result.part.type == kSphType ||
        result.part.type == kStarType);
    if (result.part.type == kSphType) {
        auto const i = particlesCatIndex(index);
        auto& gas = result.gas;
        gas.supportRadius = sphParticlesThermodynamicStates[i].supportRadius;
        gas.acceleration = sphParticlesAccelerationStates[i].acceleration;
        gas.predictedInternalEnergyRate = sphParticlesPredictedInternalEnergyRates[i];
        gas.internalEnergyRate = sphParticlesInternalEnergyRates[i];
        gas.density = sphParticlesDensities[i];
        gas.converged = sphParticlesConvergenceFlags[i];
        gas.semiactive = sphParticlesSemiactiveFlags[i];
        gas.internalEnergy = sphParticlesThermodynamicStates[i].internalEnergy;
        if (getOptions().hydro.method == "mfv") {
            gas.mass = sphParticlesMass(i);
        }
        if (getOptions().hydro.method != "sph") {
            gas.hopkins = sphParticlesHopkinsState(i);
        }
        if (getOptions().hydro.viscosityEvolution != "none") {
            auto const& state = sphParticlesViscosityStates[i];
            gas.viscosity = {state.alpha, state.rate};
        }
        if (getOptions().hydro.viscosityGradientSwitch()) {
            gas.divergence = sphParticlesCullenDehnenStates[i].divergence;
        }
        if (getOptions().chemistry.enabled) {
            gas.chemistry = sphParticlesChemistryStates[i];
        }
        if (getOptions().chemistry.enabled || getOptions().starFormation.enabled) {
            gas.chemistryEos = sphParticlesChemistryEos[i];
        }
        if (getOptions().run.hierarchicalGravity && getOptions().hydro.method == "sph") {
            gas.hierarchicalState = sphParticlesHierarchicalStates[i];
            gas.gradHCorrection = sphParticlesGradHCorrections[i];
            gas.balsaraFactor =
                getOptions().hydro.balsaraSwitch ? sphParticlesBalsaraFactors[i] : 1.0f;
        }
        for (int dim = 0; dim < kDimensions; dim++) {
            gas.predictedAcceleration[dim] = sphParticlesPredictedAccelerations[dim][i];
            gas.gravityAcceleration[dim] = sphParticlesGravityAccelerations[dim][i];
        }
    }
    return result;
}
static void unpackSphParticle(
    SphMigrationParticle record, ParticleInteger index, ParticleInteger gasIndex) {
    if (record.part.type == kSphType) {
        record.part.catIndex = gasIndex;
    }
    particlesSetParticle(record.part, index);
    if (record.part.type == kSphType) {
        auto const i = gasIndex;
        auto const& gas = record.gas;
        sphParticlesParticleIndex(i) = index;
        sphParticlesThermodynamicStates[i].supportRadius = gas.supportRadius;
        sphParticlesAccelerationStates[i].acceleration = gas.acceleration;
        sphParticlesPredictedInternalEnergyRates[i] = gas.predictedInternalEnergyRate;
        sphParticlesInternalEnergyRates[i] = gas.internalEnergyRate;
        sphParticlesDensities[i] = gas.density;
        sphParticlesConvergenceFlags[i] = gas.converged;
        sphParticlesSemiactiveFlags[i] = gas.semiactive;
        sphParticlesThermodynamicStates[i].internalEnergy = gas.internalEnergy;
        if (getOptions().hydro.method == "mfv") {
            sphParticlesMfvMass(i) = gas.mass;
        }
        if (getOptions().hydro.method != "sph") {
            sphParticlesHopkinsState(i) = gas.hopkins;
        }
        if (getOptions().hydro.viscosityEvolution != "none") {
            sphParticlesViscosityStates[i] = {
                gas.viscosity.first, gas.viscosity.second, gas.viscosity.first};
        }
        if (getOptions().hydro.viscosityGradientSwitch()) {
            sphParticlesCullenDehnenStates[i] = {gas.divergence, 0.0f, 0.0f, 0.0f};
        }
        if (getOptions().chemistry.enabled) {
            sphParticlesChemistryStates[i] = gas.chemistry;
        }
        if (getOptions().chemistry.enabled || getOptions().starFormation.enabled) {
            sphParticlesChemistryEos[i] = gas.chemistryEos;
        }
        if (getOptions().run.hierarchicalGravity && getOptions().hydro.method == "sph") {
            sphParticlesHierarchicalStates[i] = gas.hierarchicalState;
            sphParticlesGradHCorrections[i] = gas.gradHCorrection;
            if (getOptions().hydro.balsaraSwitch) {
                sphParticlesBalsaraFactors[i] = gas.balsaraFactor;
            }
        }
        for (int dim = 0; dim < kDimensions; dim++) {
            sphParticlesPredictedAccelerations[dim][i] = gas.predictedAcceleration[dim];
            sphParticlesGravityAccelerations[dim][i] = gas.gravityAcceleration[dim];
        }
    }
}
static void beginHierarchicalSphMigration(int rung) {
    if (hpxSize() == 1) {
        return;
    }
    auto const range = particlesCurrentRange();
    hierarchicalSphMigration = detail::planSphMigrationSuffix(
        range.first, range.second, particlesSize(), sphParticlesSize(), [](ParticleInteger index) {
            if (particlesType(index) != kSphType) {
                return kNoParticleIndex;
            }
            ParticleInteger const gasIndex = particlesCatIndex(index);
            if (gasIndex < 0) {
                throw std::logic_error("Missing gas slot in hierarchical SPH migration");
            }
            return gasIndex;
        });
    retainedSphParticles.clear();
    Vector<hpx::future<void>> pending;
    try {
        for (auto const& child : hpxChildren()) {
            pending.push_back(hpx::async<DomainsBeginAction>(child, rung));
        }
        auto const box = domainsFindMyBox();
        std::unordered_map<int, Vector<SphMigrationParticle>> sends;
        size_t parcelSequence = 0;
        size_t const parcelCapacity = domainsMigrationParcelCapacity(true);
        // The inactive prefix and its rung-stack offsets remain untouched.
        // Faster particles in the suffix may already be at a later physical
        // time, so only this completed rung is eligible to leave its domain.
        for (ParticleInteger i = range.first; i < range.second; ++i) {
            if (particlesType(i) == kSphType) {
                ParticleInteger const gasIndex = particlesCatIndex(i);
                sphParticlesParticleIndex(gasIndex) = i;
            }
            auto record = packSphParticle(i);
            if (record.part.r != rung) {
                retainedSphParticles.push_back(std::move(record));
                continue;
            }
            Vector3<double> x;
            for (int dim = 0; dim < kDimensions; ++dim) {
                x[dim] = positionDomainCoordinate(record.part.x[dim]);
            }
            if (box.contains(x)) {
                retainedSphParticles.push_back(std::move(record));
            } else {
                auto const rank = findParticleDomain(x);
                auto& send = sends[rank];
                send.push_back(std::move(record));
                if (send.size() >= parcelCapacity) {
                    pending.push_back(hpx::async<ReceiveSphParticlesAction>(
                        hpxLocalities()[rank], std::move(send), hpxRank(), parcelSequence++));
                    send.clear();
                }
            }
        }
        for (auto& entry : sends) {
            if (!entry.second.empty()) {
                pending.push_back(
                    hpx::async<ReceiveSphParticlesAction>(hpxLocalities()[entry.first],
                        std::move(entry.second), hpxRank(), parcelSequence++));
            }
        }
    } catch (...) {
        hpx::wait_all(pending.begin(), pending.end());
        throw;
    }
    hpx::wait_all(pending.begin(), pending.end());
    for (auto& future : pending) {
        future.get();
    }
}

static void endHierarchicalSphMigration() {
    if (hpxSize() == 1) {
        return;
    }
    Vector<hpx::future<void>> pending;
    try {
        for (auto const& child : hpxChildren()) {
            pending.push_back(hpx::async<DomainsEndAction>(child));
        }
        if (particlesCurrentRange().first != hierarchicalSphMigration.begin ||
            particlesSize() != hierarchicalSphMigration.end) {
            throw std::logic_error("Hierarchical SPH migration range changed before completion");
        }
        std::sort(receivedSphParticles.begin(), receivedSphParticles.end(),
            [](auto const& first, auto const& second) {
                return first.sourceRank < second.sourceRank ||
                    (first.sourceRank == second.sourceRank && first.sequence < second.sequence);
            });
        for (auto& parcel : receivedSphParticles) {
            for (auto& record : parcel.particles) {
                retainedSphParticles.push_back(std::move(record));
            }
        }
        receivedSphParticles.clear();
        ParticleInteger gasCount = 0;
        for (auto const& record : retainedSphParticles) {
            gasCount += record.part.type == kSphType;
        }
        // Removing high slots first keeps every remaining old slot valid.
        // A displaced inactive gas slot repairs its main-particle category
        // index in sphParticlesRemove; the main inactive prefix never moves.
        for (auto const gasIndex : hierarchicalSphMigration.gasIndices) {
            sphParticlesRemove(gasIndex);
        }
        ParticleInteger gasIndex = sphParticlesSize();
        particlesResize(hierarchicalSphMigration.begin + retainedSphParticles.size());
        sphParticlesResize(gasIndex + gasCount, false);
        for (size_t i = 0; i < retainedSphParticles.size(); ++i) {
            unpackSphParticle(
                retainedSphParticles[i], hierarchicalSphMigration.begin + i, gasIndex);
            gasIndex += retainedSphParticles[i].part.type == kSphType;
        }
        hierarchicalSphMigration.gasIndices.clear();
        retainedSphParticles.clear();
    } catch (...) {
        hpx::wait_all(pending.begin(), pending.end());
        throw;
    }
    hpx::wait_all(pending.begin(), pending.end());
    for (auto& future : pending) {
        future.get();
    }
}

static void beginSphMigration(int rung) {
    if (usesHierarchicalIntegration(getOptions().run)) {
        beginHierarchicalSphMigration(rung);
        return;
    }
    Vector<hpx::future<void>> pending;
    for (auto const& child : hpxChildren()) {
        pending.push_back(hpx::async<DomainsBeginAction>(child, rung));
    }
    retainedSphParticles.clear();
    auto const box = domainsFindMyBox();
    std::unordered_map<int, Vector<SphMigrationParticle>> sends;
    size_t parcelSequence = 0;
    size_t const parcelCapacity = domainsMigrationParcelCapacity(true);
    // The ordinary integrator retains every rung in the same particle range.
    // Moving every out-of-domain particle also handles domain rebalancing.
    for (ParticleInteger i = 0; i < particlesSize(); i++) {
        Vector3<double> x;
        for (int dim = 0; dim < kDimensions; dim++) {
            x[dim] = positionDomainCoordinate(particlesPos(dim, i));
        }
        auto record = packSphParticle(i);
        if (box.contains(x)) {
            retainedSphParticles.push_back(std::move(record));
        } else {
            auto const rank = findParticleDomain(x);
            auto& send = sends[rank];
            send.push_back(std::move(record));
            if (send.size() >= parcelCapacity) {
                pending.push_back(hpx::async<ReceiveSphParticlesAction>(
                    hpxLocalities()[rank], std::move(send), hpxRank(), parcelSequence++));
                send.clear();
            }
        }
    }
    for (auto& entry : sends) {
        if (!entry.second.empty()) {
            pending.push_back(hpx::async<ReceiveSphParticlesAction>(hpxLocalities()[entry.first],
                std::move(entry.second), hpxRank(), parcelSequence++));
        }
    }
    for (auto& future : pending) {
        future.get();
    }
}
static void endSphMigration() {
    if (usesHierarchicalIntegration(getOptions().run)) {
        endHierarchicalSphMigration();
        return;
    }
    Vector<hpx::future<void>> pending;
    for (auto const& child : hpxChildren()) {
        pending.push_back(hpx::async<DomainsEndAction>(child));
    }
    // Arrival order depends on action scheduling. Reassemble each sender's
    // original particle order, with senders ordered by rank, before tree builds.
    std::sort(receivedSphParticles.begin(), receivedSphParticles.end(),
        [](auto const& first, auto const& second) {
            return first.sourceRank < second.sourceRank ||
                (first.sourceRank == second.sourceRank && first.sequence < second.sequence);
        });
    for (auto& parcel : receivedSphParticles) {
        for (auto& record : parcel.particles) {
            retainedSphParticles.push_back(std::move(record));
        }
    }
    receivedSphParticles.clear();
    ParticleInteger gasCount = 0;
    for (auto const& record : retainedSphParticles) {
        gasCount += record.part.type == kSphType;
    }
    particlesResize(retainedSphParticles.size());
    sphParticlesResize(gasCount, false);
    ParticleInteger gasIndex = 0;
    for (ParticleInteger i = 0; i < retainedSphParticles.size(); i++) {
        unpackSphParticle(retainedSphParticles[i], i, gasIndex);
        gasIndex += retainedSphParticles[i].part.type == kSphType;
    }
    retainedSphParticles.clear();
    for (auto& future : pending) {
        future.get();
    }
}

#endif

size_t domainsMigrationPayloadBytes(bool gas, size_t count) {
#if COSMICTIGER_WITH_SPH
    if (gas) {
        SphMigrationParticle record{};
        record.part.type = kSphType;
        return detail::migrationPayloadBytes(record, count);
    }
#else
    if (gas) {
        throw std::logic_error("SPH migration is unavailable in this build");
    }
#endif
    return detail::migrationPayloadBytes(Particle{}, count);
}

size_t domainsMigrationRecordBytes(bool gas) {
    // Measure the actual serializer, including enabled optional fields. Difference
    // of two vector sizes removes fixed archive/vector headers. Verify affine
    // record growth against a third sample rather than assuming sizeof(Record).
    size_t const one = domainsMigrationPayloadBytes(gas, 1);
    size_t const two = domainsMigrationPayloadBytes(gas, 2);
    size_t const stride = two - one;
    if (two <= one || domainsMigrationPayloadBytes(gas, 3) != two + stride ||
        one > stride + kMigrationParcelOverheadBytes ||
        stride > kMaxMigrationParcelBytes - kMigrationParcelOverheadBytes) {
        throw std::runtime_error("Unsupported variable-length migration record serialization");
    }
    return stride;
}

size_t domainsMigrationParcelCapacity(bool gas) {
    return (kMaxMigrationParcelBytes - kMigrationParcelOverheadBytes) /
        domainsMigrationRecordBytes(gas);
}

void domainsBegin(int rung) {
#if COSMICTIGER_WITH_SPH
    if (getOptions().hydro.enabled) {
        beginSphMigration(rung);
        return;
    }
#endif
    Vector<hpx::future<void>> futs;
    auto children = hpxChildren();
    for (auto& c : children) {
        futs.push_back(hpx::async<DomainsBeginAction>(c, rung));
    }
    auto const myDomain = domainsFindMyBox();
    size_t const parcelCapacity = domainsMigrationParcelCapacity(false);
    freeIndices.resize(0);
    MutexType mutex;
    int const numberOfThreads = hpxHardwareConcurrency();
    auto const currentRange = particlesCurrentRange();
    std::atomic<ParticleInteger> nextBegin(0);
    constexpr int kChunkSize = 1024;
    for (int proc = 0; proc < numberOfThreads; proc++) {
        futs.push_back(hpx::async(hpx::annotated_function(
            [proc, numberOfThreads, &mutex, &nextBegin, myDomain, currentRange, rung,
                parcelCapacity]() {
                std::unordered_map<int, Vector<Particle>> sends;
                Vector<hpx::future<void>> futs;
                Vector<ParticleInteger> myFreeIndices;
                ParticleInteger begin = nextBegin++ * kChunkSize + currentRange.first;
                while (begin < currentRange.second) {
                    ParticleInteger const end = std::min(begin + kChunkSize, currentRange.second);
                    for (ParticleInteger i = begin; i < end; i++) {
                        // Ordinary timesteps drift every rung together. Rebalance
                        // all particles before building their common gravity tree.
                        if (!getOptions().run.hierarchicalGravity ||
                            !getOptions().gravity.enabled || particlesRung(i) == rung) {
                            Vector3<double> x;
                            for (ParticleInteger dim = 0; dim < kDimensions; dim++) {
                                x[dim] = positionDomainCoordinate(particlesPos(dim, i));
                            }
                            if (!myDomain.contains(x)) {
                                ParticleInteger const rank = findParticleDomain(x);
                                auto& send = sends[rank];
                                send.push_back(particlesGetParticle(i));
                                if (send.size() >= parcelCapacity) {
                                    //				PRINT( "Sending %i from %i to %i\n",
                                    // send.size(),
                                    // hpx_rank(), rank);
                                    futs.push_back(hpx::async<DomainsTransmitParticlesAction>(
                                        hpxLocalities()[rank], std::move(send)));
                                }
                                myFreeIndices.push_back(i);
                            }
                        }
                    }
                    begin = nextBegin++ * kChunkSize + currentRange.first;
                }
                for (auto i = sends.begin(); i != sends.end(); i++) {
                    if (i->second.size()) {
                        //			PRINT( "Sending %i from %i to %i\n", i->second.size(),
                        // hpx_rank(),i->first);
                        futs.push_back(hpx::async<DomainsTransmitParticlesAction>(
                            hpxLocalities()[i->first], std::move(i->second)));
                    }
                }
                {
                    std::lock_guard<MutexType> lock(mutex);
                    freeIndices.insert(
                        freeIndices.end(), myFreeIndices.begin(), myFreeIndices.end());
                }
                hpx::wait_all(futs.begin(), futs.end());
            },
            "particles.collectDomainTransfers")));
    }

    hpx::wait_all(futs.begin(), futs.end());
}

void domainsEnd() {
#if COSMICTIGER_WITH_SPH
    if (getOptions().hydro.enabled) {
        endSphMigration();
        return;
    }
#endif
    Vector<hpx::future<void>> futs;
    auto children = hpxChildren();
    for (auto& c : children) {
        futs.push_back(hpx::async<DomainsEndAction>(c));
    }
    auto const particleCompare = [](Particle a, Particle b) {
        for (ParticleInteger dim = 0; dim < kDimensions; dim++) {
            if (a.x[dim] < b.x[dim]) {
                return true;
            } else if (a.x[dim] > b.x[dim]) {
                return false;
            }
        }
        return false;
    };
    hpx::future<void> fut1;
    hpx::future<void> fut2;
    if (freeIndices.size()) {
        fut1 = hpx::sort(annotatedParallelPolicy("particles.sortFreeIndices"), freeIndices.begin(),
            freeIndices.end());
    } else {
        fut1 = hpx::make_ready_future();
    }
    Vector<Particle> allTransParticles;
    Vector<ParticleInteger> offsets(transParticles.size());
    ParticleInteger sz = 0;
    for (int i = 0; i < transParticles.size(); i++) {
        offsets[i] = sz;
        sz += transParticles[i].size();
    }
    allTransParticles.resize(sz);
    Vector<hpx::future<void>> futs1;
    int const numberOfThreads = hpxHardwareConcurrency();
    std::atomic<int> index(0);
    for (int proc = 0; proc < numberOfThreads; proc++) {
        futs1.push_back(hpx::async(hpx::annotated_function(
            [&index, &allTransParticles, &offsets]() {
                int i;
                while ((i = index++) < transParticles.size()) {
                    for (ParticleInteger j = 0; j < transParticles[i].size(); j++) {
                        allTransParticles[offsets[i] + j] = transParticles[i][j];
                    }
                }
            },
            "particles.gatherDomainTransfers")));
    }
    hpx::wait_all(futs1.begin(), futs1.end());
    transParticles = decltype(transParticles)();
    if (allTransParticles.size()) {
        fut2 = hpx::sort(annotatedParallelPolicy("particles.sortDomainTransfers"),
            allTransParticles.begin(), allTransParticles.end(), particleCompare);
    } else {
        fut2 = hpx::make_ready_future();
    }
    fut1.get();
    fut2.get();
    if (freeIndices.size() < allTransParticles.size()) {
        ParticleInteger const diff = allTransParticles.size() - freeIndices.size();
        for (ParticleInteger i = 0; i < diff; i++) {
            freeIndices.push_back(particlesSize() + i);
        }
        particlesResize(particlesSize() + diff);
    } else {
        while (freeIndices.size() > allTransParticles.size()) {
            auto const p = particlesGetParticle(particlesSize() - 1);
            particlesSetParticle(p, freeIndices.back());
            freeIndices.pop_back();
            particlesResize(particlesSize() - 1);
        }
    }
    // PRINT("unloading particles on %i\n", hpx_rank());
    for (int proc = 0; proc < numberOfThreads; proc++) {
        futs.push_back(hpx::async(hpx::annotated_function(
            [numberOfThreads, proc, &allTransParticles]() {
                ParticleInteger const begin = (size_t) proc * freeIndices.size() / numberOfThreads;
                ParticleInteger const end =
                    (size_t) (proc + 1) * freeIndices.size() / numberOfThreads;
                for (ParticleInteger i = begin; i < end; i++) {
                    particlesSetParticle(allTransParticles[i], freeIndices[i]);
                }
            },
            "particles.installDomainTransfers")));
    }
    hpx::wait_all(futs.begin(), futs.end());
    // PRINT("Done on %i\n", hpx_rank());
    freeIndices = decltype(freeIndices)();
#ifdef DOMAINS_CHECK
    domainsCheck();
#endif
}

Range<double> domainsFindMyBox() {
    return localDomains[hpxRank()].box;
}

int domainsFindRank(Vector3<double> const& x) {
    if (hpxSize() == 1) {
        return 0;
    }
    if (boxesByKey.find(1) == boxesByKey.end()) {
        throw std::runtime_error("Particle domains have not been decomposed");
    }
    return findParticleDomain(x);
}

static void domainsCheck() {
    std::atomic<int> fail(0);
    auto const myDomain = domainsFindMyBox();
    int const numberOfThreads = hpxHardwareConcurrency();
    Vector<hpx::future<void>> futs;
    for (int proc = 0; proc < numberOfThreads; proc++) {
        futs.push_back(hpx::async(hpx::annotated_function(
            [proc, numberOfThreads, &fail, myDomain]() {
                auto const rng = particlesCurrentRange();
                ParticleInteger const begin =
                    size_t(proc) * (rng.second - rng.first) / numberOfThreads + rng.first;
                ParticleInteger const end =
                    size_t(proc + 1) * (rng.second - rng.first) / numberOfThreads + rng.first;
                for (ParticleInteger i = begin; i < end; i++) {
                    Vector3<double> x;
                    for (int dim = 0; dim < kDimensions; dim++) {
                        x[dim] = positionDomainCoordinate(particlesPos(dim, i));
                    }
                    if (!myDomain.contains(x)) {
                        PRINT("particle out of range %li of %li %e %e %e rung %i\n", (long long) i,
                            (long long) particlesSize(), x[0], x[1], x[2], particlesRung(i));
                        PRINT("%s\n", domainsFindMyBox().toString().c_str());
                        ALWAYS_ASSERT(false);
                        fail++;
                    }
                }
            },
            "particles.domainsCheck")));
    }
    hpx::wait_all(futs.begin(), futs.end());
    if (int(fail)) {
        THROW_ERROR("particle out of range!\n");
    }
}

/*
 pair<int, double> domains_find_ewald_level(double theta, size_t key, range<double> box, int depth,
 double rb) { pair<int, double> rc; double rmax = 0.0; bool root = key == 1; double xc = 0.5 *
 (box.begin[XDIM] + box.end[XDIM]); double yc = 0.5 * (box.begin[YDIM] + box.end[YDIM]); double zc =
 0.5 * (box.begin[ZDIM] + box.end[ZDIM]); const double r = sqrtf(sqr(0.5 * (box.end[XDIM] -
 box.begin[XDIM]), 0.5 * (box.end[YDIM] - box.begin[YDIM]), 0.5 * (box.end[ZDIM] -
 box.begin[ZDIM]))); const double h = get_options().hsoft; bool ewald_satisfied = false; if (rb <=
 0.0) { ewald_satisfied = (r < 0.25 * (theta / (theta + 1)) && r < 0.125 - 0.5 * h); } else {
 ewald_satisfied = (r < 0.5 * (theta / (1.0 + theta)) - rb && r < 0.5 - rb - h);
 }
 if (ewald_satisfied) {
 rc.first = depth;
 rc.second = r;
 } else {
 const auto& entry = boxes_by_key[key];
 key <<= 1;
 pair<int, double> rcl, rcr;
 if (entry.rank == -1) {
 const int xdim = depth % NDIM;
 range<double> cboxr = box;
 range<double> cboxl = box;
 double midx = (box.begin[xdim] + box.end[xdim]) * 0.5;
 cboxl.end[xdim] = cboxr.begin[xdim] = midx;
 rcl = domains_find_ewald_level(theta, key, cboxl, depth + 1, rb);
 rcr = domains_find_ewald_level(theta, key | 1, cboxl, depth + 1, rb);
 } else {
 double max_span = 0.0;
 int xdim;
 for (int dim = 0; dim < NDIM; dim++) {
 double span = box.end[dim] - box.begin[dim];
 if (span > max_span) {
 max_span = span;
 xdim = dim;
 }
 }
 range<double> cboxr = box;
 range<double> cboxl = box;
 double midx = (box.begin[xdim] + box.end[xdim]) * 0.5;
 cboxl.end[xdim] = cboxr.begin[xdim] = midx;
 rcl = domains_find_ewald_level(theta, key, cboxl, depth + 1, rb);
 rcr = domains_find_ewald_level(theta, key | 1, cboxl, depth + 1, rb);
 }
 if (rb <= 0.0) {
 if (rcr.second > rcl.first) {
 rc = rcr;
 } else {
 rc = rcl;
 }
 } else {
 if (rcl.first > rcr.first) {
 rc = rcl;
 } else {
 rc = rcr;
 }
 }
 }
 if (root && rb <= 0.0) {
 auto rcb = domains_find_ewald_level(theta, key, box, depth, rc.second);
 if (rcb.first > rc.first) {
 rc = rcb;
 }
 }
 return rc;
 }

 */

static int findParticleDomain(Vector3<double> const& x, size_t key, int depth) {
    auto const& entry = boxesByKey[key];
    if (entry.rank == -1) {
        int const xdim = depth % kDimensions;
        key <<= 1;
        if (x[xdim] >= entry.midx) {
            key += 1;
        }
        return findParticleDomain(x, key, depth + 1);
    } else {
        return entry.rank;
    }
}
