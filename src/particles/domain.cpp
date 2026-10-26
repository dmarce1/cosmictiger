/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <algorithm>
#include <shared_mutex>
#include <unordered_map>

#include <cosmictiger/core/containers.hpp>
#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/core/safe_io.hpp>
#include <cosmictiger/math/math.hpp>
#include <cosmictiger/particles/domain.hpp>
#include <cosmictiger/particles/mesh_domain.hpp>
#include <cosmictiger/particles/particles.hpp>
#include <cosmictiger/sph/sph_particles.hpp>

constexpr bool kVerbose = true;

void domainsTransmitParticles(Vector<Particle>);

void domainsInitRebounds();
void domainsTransmitBoxes(std::unordered_map<size_t, DomainT> boxes);
void domainsReboundSort(Vector<double> bounds, int depth);
Vector<size_t> domainsCountBelow(Vector<double> bounds, int depth);
Vector<size_t> domainsGetLoads();

HPX_PLAIN_ACTION(domainsCountBelow, DomainsCountBelowAction);
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

static int findParticleDomain(Array<double, kDimensions> const& x, size_t key = 1, int depth = 0);
static void domainsCheck();
static Vector<DomainLocal> localDomains;
static int alignedMeshSize = 0;

namespace detail {

bool usesMeshDomains() {
    auto const& gravity = getOptions().gravity;
    return gravity.enabled && usesMeshGravity(gravity);
}

bool meshLayoutAligned() {
    uint32_t const meshSize = getOptions().gravity.meshSize;
    auto const aligned = [meshSize](double value) {
        return mesh_domain::boundaryCoordinate(
                   mesh_domain::boundaryIndex(value, meshSize), meshSize) == value;
    };
    auto const alignedBox = [&](Range<double> const& box) {
        for (int dim = 0; dim < kDimensions; ++dim) {
            if (!aligned(box.begin[dim]) || !aligned(box.end[dim])) {
                return false;
            }
        }
        return true;
    };
    for (auto const& [key, domain] : boxesByKey) {
        if (!alignedBox(domain.box) || (domain.rank == -1 && !aligned(domain.midx))) {
            return false;
        }
    }
    for (auto const& domain : localDomains) {
        if (!alignedBox(domain.box)) {
            return false;
        }
    }
    return !boxesByKey.empty() && localDomains.size() == size_t(hpxSize());
}

bool meshDomainContains(Range<double> const& box, ParticleInteger particle) {
    uint32_t const meshSize = getOptions().gravity.meshSize;
    for (int dim = 0; dim < kDimensions; ++dim) {
        if (!mesh_domain::contains(particlesPos(dim, particle).raw(),
                mesh_domain::boundaryIndex(box.begin[dim], meshSize),
                mesh_domain::boundaryIndex(box.end[dim], meshSize), meshSize)) {
            return false;
        }
    }
    return true;
}

int findMeshParticleDomain(ParticleInteger particle) {
    uint32_t const meshSize = getOptions().gravity.meshSize;
    size_t key = 1;
    int depth = 0;
    while (boxesByKey.at(key).rank == -1) {
        uint32_t const plane = mesh_domain::boundaryIndex(boxesByKey.at(key).midx, meshSize);
        bool const left = mesh_domain::belowBoundary(
            particlesPos(depth % kDimensions, particle).raw(), plane, meshSize);
        key = 2 * key + !left;
        ++depth;
    }
    return boxesByKey.at(key).rank;
}

ParticleInteger partitionMeshDomain(Pair<ParticleInteger> range, uint32_t plane, int dimension) {
    uint32_t const meshSize = getOptions().gravity.meshSize;
    auto const below = [&](ParticleInteger index) {
        return mesh_domain::belowBoundary(particlesPos(dimension, index).raw(), plane, meshSize);
    };
    auto begin = range.first;
    auto end = range.second;
    while (begin < end) {
        if (below(begin)) {
            ++begin;
        } else {
            --end;
            if (begin < end) {
                particlesSwap(begin, end);
            }
        }
    }
    return begin;
}

Vector<size_t> countDomainSplits(Vector<double> const& bounds, int depth) {
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

void chooseMeshDomainSplits(Vector<DomainGlobal> const& domains, int depth, Vector<double>& bounds,
    Vector<size_t>& counts) {
    struct SplitSearch {
        uint32_t begin;
        uint32_t end;
        uint32_t lower;
        uint32_t upper;
        uint32_t best;
        uint32_t geometric;
        uint32_t ranks;
        uint32_t leftRanks;
    };
    uint32_t const meshSize = getOptions().gravity.meshSize;
    Vector<SplitSearch> searches(domains.size());
    bounds.resize(domains.size());
    for (size_t i = 0; i < domains.size(); ++i) {
        auto const& domain = domains[i];
        auto& search = searches[i];
        search.begin = mesh_domain::boundaryIndex(domain.box.begin[depth % kDimensions], meshSize);
        search.end = mesh_domain::boundaryIndex(domain.box.end[depth % kDimensions], meshSize);
        search.lower = search.begin;
        search.upper = search.end;
        search.ranks = domain.processRange.second - domain.processRange.first;
        search.leftRanks = search.ranks / 2;
        // Prefer the rank-proportional geometric plane when several planes
        // have equal particle balance, including completely empty domains.
        search.best = search.begin +
            (uint64_t(search.end - search.begin) * search.leftRanks + (search.ranks - 1) / 2) /
                search.ranks;
        search.geometric = search.best;
        bounds[i] = mesh_domain::boundaryCoordinate(search.best, meshSize);
    }
    counts = countDomainSplits(bounds, depth);
    auto const consider = [&](size_t i, uint32_t plane, size_t below) {
        auto& search = searches[i];
        if (search.ranks > 1 &&
            mesh_domain::betterSplit(plane, below, search.best, counts[i], domains[i].totalCount,
                search.leftRanks, search.ranks, search.begin, search.end)) {
            search.best = plane;
            counts[i] = below;
        }
    };
    // Counts are monotone in the integer plane. Locate the first count at or
    // above the exact rank-weighted target, then compare its predecessor.
    while (true) {
        bool searching = false;
        Vector<uint32_t> planes(domains.size());
        for (size_t i = 0; i < domains.size(); ++i) {
            auto const& search = searches[i];
            planes[i] = search.lower + (search.upper - search.lower) / 2;
            bounds[i] = mesh_domain::boundaryCoordinate(planes[i], meshSize);
            searching |= search.ranks > 1 && search.lower < search.upper;
        }
        if (!searching) {
            break;
        }
        auto const trialCounts = countDomainSplits(bounds, depth);
        for (size_t i = 0; i < domains.size(); ++i) {
            auto& search = searches[i];
            if (search.ranks <= 1 || search.lower == search.upper) {
                continue;
            }
            consider(i, planes[i], trialCounts[i]);
            if (mesh_domain::compareTarget(
                    trialCounts[i], domains[i].totalCount, search.leftRanks, search.ranks) < 0) {
                search.lower = planes[i] + 1;
            } else {
                search.upper = planes[i];
            }
        }
    }
    for (int predecessor = 0; predecessor <= 1; ++predecessor) {
        for (size_t i = 0; i < domains.size(); ++i) {
            auto const& search = searches[i];
            uint32_t const plane = search.lower - (predecessor && search.lower > search.begin);
            bounds[i] = mesh_domain::boundaryCoordinate(plane, meshSize);
        }
        auto const trialCounts = countDomainSplits(bounds, depth);
        for (size_t i = 0; i < domains.size(); ++i) {
            consider(i, mesh_domain::boundaryIndex(bounds[i], meshSize), trialCounts[i]);
        }
    }
    // A run of empty cells gives several planes the same optimal count.
    // Find the member of that plateau nearest the geometric target rather
    // than letting the binary-search path determine the tie.
    for (auto& search : searches) {
        search.lower = std::min(search.best, search.geometric);
        search.upper = std::max(search.best, search.geometric);
    }
    while (true) {
        bool searching = false;
        Vector<uint32_t> planes(domains.size());
        for (size_t i = 0; i < domains.size(); ++i) {
            auto const& search = searches[i];
            bool const towardRight = search.best < search.geometric;
            planes[i] = search.lower + (search.upper - search.lower + towardRight) / 2;
            bounds[i] = mesh_domain::boundaryCoordinate(planes[i], meshSize);
            searching |= search.ranks > 1 && search.lower < search.upper;
        }
        if (!searching) {
            break;
        }
        auto const trialCounts = countDomainSplits(bounds, depth);
        for (size_t i = 0; i < domains.size(); ++i) {
            auto& search = searches[i];
            if (search.ranks <= 1 || search.lower == search.upper) {
                continue;
            }
            if (search.best < search.geometric) {
                if (trialCounts[i] <= counts[i]) {
                    search.lower = planes[i];
                } else {
                    search.upper = planes[i] - 1;
                }
            } else if (trialCounts[i] >= counts[i]) {
                search.upper = planes[i];
            } else {
                search.lower = planes[i] + 1;
            }
        }
    }
    for (size_t i = 0; i < domains.size(); ++i) {
        bounds[i] = mesh_domain::boundaryCoordinate(searches[i].lower, meshSize);
    }
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
    alignedMeshSize = detail::usesMeshDomains() && detail::meshLayoutAligned() ?
        getOptions().gravity.meshSize :
        0;
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
    alignedMeshSize = detail::usesMeshDomains() && detail::meshLayoutAligned() ?
        getOptions().gravity.meshSize :
        0;
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
            futs.push_back(hpx::async([i, &bounds, depth, &counts]() {
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
                bool const mesh = detail::usesMeshDomains();
                uint32_t const meshSize = getOptions().gravity.meshSize;
                uint32_t const plane = mesh ? mesh_domain::boundaryIndex(bounds[i], meshSize) : 0;
                Position const xmid = mesh ? Position{} : Position(bounds[i]);
                std::atomic<size_t> count(0);
                auto const func = [numberOfThreads, rng, xdim, xmid, mesh, plane, meshSize, &count](
                                      int proc) {
                    size_t const begin =
                        rng.first + (size_t) proc * (rng.second - rng.first) / numberOfThreads;
                    size_t const end = rng.first +
                        (size_t) (proc + 1) * (rng.second - rng.first) / numberOfThreads;
                    size_t thisCount = 0;
                    for (size_t i = begin; i < end; i++) {
                        if (mesh ? mesh_domain::belowBoundary(
                                       particlesPos(xdim, i).raw(), plane, meshSize) :
                                   particlesPos(xdim, i) < xmid) {
                            thisCount++;
                        }
                    }
                    count += thisCount;
                };

                for (int proc = 1; proc < numberOfThreads; proc++) {
                    futs.push_back(hpx::async(func, proc));
                }
                func(0);
                hpx::wait_all(futs.begin(), futs.end());
                counts[i] = count;
            }));
        }
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
            futs.push_back(hpx::async([i, &bounds, depth, &mids]() {
                mids[i] = detail::usesMeshDomains() ?
                    detail::partitionMeshDomain(localDomains[i].particleRange,
                        mesh_domain::boundaryIndex(bounds[i], getOptions().gravity.meshSize),
                        depth % kDimensions) :
                    particlesSort(localDomains[i].particleRange, bounds[i], depth % kDimensions);
                ///	PRINT( "!!!!!!!! %i %i %e %i %i\n", hpx_rank(), mids[i], bounds[i],
                /// local_domains[i].part_range.first, local_domains[i].part_range.second);
            }));
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
    domainsInitRebounds();
    //	PRINT( "Starting rebound\n");
    Vector<DomainGlobal> domains(1);
    boxesByKey = decltype(boxesByKey)();
    domains[0].box = unitBox<double>();
    domains[0].processRange.first = 0;
    domains[0].processRange.second = hpxSize();
    domains[0].totalCount = getOptions().initial.numberOfParticles;
    bool const mesh = detail::usesMeshDomains();
    if (mesh) {
        domains[0].totalCount = 0;
        for (auto const count : domainsGetLoads()) {
            domains[0].totalCount += count;
        }
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
        if (mesh) {
            detail::chooseMeshDomainSplits(domains, depth, bounds, counts);
            for (size_t i = 0; i < domains.size(); ++i) {
                boxesByKey[domains[i].key].midx = bounds[i];
            }
        } else {
            for (int iter = 0; iter < kDomainReboundIterations; iter++) {
                for (auto& c : counts) {
                    c = 0;
                }
                Vector<double> bounds;
                for (int i = 0; i < domains.size(); i++) {
                    double const midx = (domains[i].midhi + domains[i].midlo) * 0.5;
                    bounds.push_back(midx);
                }
                //			PRINT( "Counting\n");
                Vector<hpx::future<Vector<size_t>>> countFuts;
                for (int i = 0; i < hpxSize(); i++) {
                    countFuts.push_back(
                        hpx::async<DomainsCountBelowAction>(hpxLocalities()[i], bounds, depth));
                }
                for (auto& f : countFuts) {
                    auto const count = f.get();
                    for (int i = 0; i < count.size(); i++) {
                        counts[i] += count[i];
                    }
                }
                //			PRINT( "Done counting\n");
                for (int i = 0; i < domains.size(); i++) {
                    //				PRINT( "%i\n", i);
                    int const firstProcess = domains[i].processRange.first;
                    int const lastProcess = domains[i].processRange.second;
                    int const middleProcess = (firstProcess + lastProcess) / 2;
                    ALWAYS_ASSERT(lastProcess - firstProcess >= 1);
                    size_t const targetCount = (size_t) (middleProcess - firstProcess) *
                        domains[i].totalCount / (lastProcess - firstProcess);
                    double const midx = (domains[i].midhi + domains[i].midlo) * 0.5;
                    if (counts[i] < targetCount) {
                        domains[i].midlo = midx;
                    } else {
                        domains[i].midhi = midx;
                    }
                    //				PRINT("----- %i %li %e %e\n", i, counts[i], domains[i].midlo,
                    // domains[i].midhi);
                }
            }
            for (int i = 0; i < domains.size(); i++) {
                if (domains[i].processRange.second - domains[i].processRange.first) {
                    double const midx = (domains[i].midhi + domains[i].midlo) * 0.5;
                    boxesByKey[domains[i].key].midx = midx;
                    bounds.push_back(midx);
                } else {
                    bounds.push_back(0.0);
                }
            }
        }
        Vector<hpx::future<void>> sortFuts;
        for (int i = 0; i < hpxSize(); i++) {
            sortFuts.push_back(
                hpx::async<DomainsReboundSortAction>(hpxLocalities()[i], bounds, depth));
        }
        hpx::wait_all(sortFuts.begin(), sortFuts.end());
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
    domainsTransmitBoxes(boxesByKey);
}

void domainsTransmitParticles(Vector<Particle> parts) {
    static MutexType mutex;
    std::lock_guard<MutexType> lock(mutex);
    transParticles.push_back(parts);
}

// Ordinary SPH timesteps migrate the complete gas state together with its
// gravity particle. Category indices are local and are rebuilt at the receiver.
struct GasMigrationState {
    Array<float, kDimensions> acceleration, predictedAcceleration, gravityAcceleration;
    float supportRadius, internalEnergy, predictedInternalEnergyRate, internalEnergyRate, density;
    char converged, semiactive;
    Pair<float> viscosity;    // Alpha and its previous conformal-time rate.
    float divergence;         // Previous physical divergence for Cullen-Dehnen.
    Array<float, primordial::kAbundanceCount + 1> chemistry;
    Pair<float> chemistryEos;
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
        if (getOptions().hydro.viscosityEvolution != "none") {
            arc & viscosity;
        }
        if (getOptions().hydro.viscosityEvolution == "cullenAndDehnen") {
            arc & divergence;
        }
        if (getOptions().chemistry.enabled) {
            arc & chemistry;
            arc & chemistryEos;
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
static Vector<Vector<SphMigrationParticle>> receivedSphParticles;
void receiveSphParticles(Vector<SphMigrationParticle> parts) {
    static MutexType mutex;
    std::lock_guard<MutexType> lock(mutex);
    receivedSphParticles.push_back(std::move(parts));
}
HPX_PLAIN_ACTION(receiveSphParticles, ReceiveSphParticlesAction);
static SphMigrationParticle packSphParticle(ParticleInteger index) {
    SphMigrationParticle result;
    result.part = particlesGetParticle(index);
    ALWAYS_ASSERT(result.part.type == kDarkMatterType || result.part.type == kSphType);
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
        if (getOptions().hydro.viscosityEvolution != "none") {
            auto const& state = sphParticlesViscosityStates[i];
            gas.viscosity = {state.alpha, state.rate};
        }
        if (getOptions().hydro.viscosityEvolution == "cullenAndDehnen") {
            gas.divergence = sphParticlesCullenDehnenStates[i].divergence;
        }
        if (getOptions().chemistry.enabled) {
            auto const& state = sphParticlesChemistryStates[i];
            for (int species = 0; species < primordial::kAbundanceCount; ++species) {
                gas.chemistry[species] = state.abundances[species].raw();
            }
            gas.chemistry[primordial::kAbundanceCount] = state.thermalEnergy;
            gas.chemistryEos = sphParticlesChemistryEos[i];
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
        if (getOptions().hydro.viscosityEvolution != "none") {
            sphParticlesViscosityStates[i] = {
                gas.viscosity.first, gas.viscosity.second, gas.viscosity.first};
        }
        if (getOptions().hydro.viscosityEvolution == "cullenAndDehnen") {
            sphParticlesCullenDehnenStates[i] = {gas.divergence, 0.0f, 0.0f, 0.0f};
        }
        if (getOptions().chemistry.enabled) {
            auto& state = sphParticlesChemistryStates[i];
            for (int species = 0; species < primordial::kAbundanceCount; ++species) {
                state.abundances[species].setRaw(gas.chemistry[species]);
            }
            state.thermalEnergy = gas.chemistry[primordial::kAbundanceCount];
            sphParticlesChemistryEos[i] = gas.chemistryEos;
        }
        for (int dim = 0; dim < kDimensions; dim++) {
            sphParticlesPredictedAccelerations[dim][i] = gas.predictedAcceleration[dim];
            sphParticlesGravityAccelerations[dim][i] = gas.gravityAcceleration[dim];
        }
    }
}
static void beginSphMigration(int rung) {
    Vector<hpx::future<void>> pending;
    for (auto const& child : hpxChildren()) {
        pending.push_back(hpx::async<DomainsBeginAction>(child, rung));
    }
    retainedSphParticles.clear();
    auto const box = domainsFindMyBox();
    std::unordered_map<int, Vector<SphMigrationParticle>> sends;
    // The ordinary integrator retains every rung in the same particle range.
    // Moving every out-of-domain particle also handles domain rebalancing.
    for (ParticleInteger i = 0; i < particlesSize(); i++) {
        Array<double, kDimensions> x;
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
            if (send.size() >= kMaxParticlesPerParcel) {
                pending.push_back(
                    hpx::async<ReceiveSphParticlesAction>(hpxLocalities()[rank], std::move(send)));
                send.clear();
            }
        }
    }
    for (auto& entry : sends) {
        if (!entry.second.empty()) {
            pending.push_back(hpx::async<ReceiveSphParticlesAction>(
                hpxLocalities()[entry.first], std::move(entry.second)));
        }
    }
    for (auto& future : pending) {
        future.get();
    }
}
static void endSphMigration() {
    Vector<hpx::future<void>> pending;
    for (auto const& child : hpxChildren()) {
        pending.push_back(hpx::async<DomainsEndAction>(child));
    }
    for (auto& parcel : receivedSphParticles) {
        for (auto& record : parcel) {
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

void domainsBegin(int rung) {
    bool const mesh = detail::usesMeshDomains();
    if (hpxRank() == 0 && mesh && alignedMeshSize != getOptions().gravity.meshSize) {
        // Older mesh checkpoints used continuous domain cuts. Rebuild them
        // collectively before the first migration, preserving the file layout.
        domainsRebound();
    }
    if (getOptions().hydro.enabled) {
        beginSphMigration(rung);
        return;
    }
    Vector<hpx::future<void>> futs;
    auto children = hpxChildren();
    for (auto& c : children) {
        futs.push_back(hpx::async<DomainsBeginAction>(c, rung));
    }
    auto const myDomain = domainsFindMyBox();
    freeIndices.resize(0);
    MutexType mutex;
    int const numberOfThreads = hpxHardwareConcurrency();
    auto const currentRange = particlesCurrentRange();
    std::atomic<ParticleInteger> nextBegin(0);
    constexpr int kChunkSize = 1024;
    for (int proc = 0; proc < numberOfThreads; proc++) {
        futs.push_back(hpx::async([proc, numberOfThreads, &mutex, &nextBegin, myDomain,
                                      currentRange, rung, mesh]() {
            std::unordered_map<int, Vector<Particle>> sends;
            Vector<hpx::future<void>> futs;
            Vector<ParticleInteger> myFreeIndices;
            ParticleInteger begin = nextBegin++ * kChunkSize + currentRange.first;
            while (begin < currentRange.second) {
                ParticleInteger const end = std::min(begin + kChunkSize, currentRange.second);
                for (ParticleInteger i = begin; i < end; i++) {
                    // Ordinary timesteps drift every rung together. Rebalance
                    // all particles before building their common gravity tree.
                    if (!getOptions().run.hierarchicalGravity || particlesRung(i) == rung) {
                        Array<double, kDimensions> x;
                        if (!mesh) {
                            for (ParticleInteger dim = 0; dim < kDimensions; dim++) {
                                x[dim] = positionDomainCoordinate(particlesPos(dim, i));
                            }
                        }
                        bool const contained =
                            mesh ? detail::meshDomainContains(myDomain, i) : myDomain.contains(x);
                        if (!contained) {
                            ParticleInteger const rank =
                                mesh ? detail::findMeshParticleDomain(i) : findParticleDomain(x);
                            auto& send = sends[rank];
                            send.push_back(particlesGetParticle(i));
                            if (send.size() >= kMaxParticlesPerParcel) {
                                //				PRINT( "Sending %i from %i to %i\n", send.size(),
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
                freeIndices.insert(freeIndices.end(), myFreeIndices.begin(), myFreeIndices.end());
            }
            hpx::wait_all(futs.begin(), futs.end());
        }));
    }

    hpx::wait_all(futs.begin(), futs.end());
}

void domainsEnd() {
    if (getOptions().hydro.enabled) {
        endSphMigration();
        return;
    }
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
        fut1 = hpx::sort(PAR_EXECUTION_POLICY, freeIndices.begin(), freeIndices.end());
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
        futs1.push_back(hpx::async([&index, &allTransParticles, &offsets]() {
            int i;
            while ((i = index++) < transParticles.size()) {
                for (ParticleInteger j = 0; j < transParticles[i].size(); j++) {
                    allTransParticles[offsets[i] + j] = transParticles[i][j];
                }
            }
        }));
    }
    hpx::wait_all(futs1.begin(), futs1.end());
    transParticles = decltype(transParticles)();
    if (allTransParticles.size()) {
        fut2 = hpx::sort(PAR_EXECUTION_POLICY, allTransParticles.begin(), allTransParticles.end(),
            particleCompare);
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
        futs.push_back(hpx::async([numberOfThreads, proc, &allTransParticles]() {
            ParticleInteger const begin = (size_t) proc * freeIndices.size() / numberOfThreads;
            ParticleInteger const end = (size_t) (proc + 1) * freeIndices.size() / numberOfThreads;
            for (ParticleInteger i = begin; i < end; i++) {
                particlesSetParticle(allTransParticles[i], freeIndices[i]);
            }
        }));
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

static void domainsCheck() {
    std::atomic<int> fail(0);
    auto const myDomain = domainsFindMyBox();
    int const numberOfThreads = hpxHardwareConcurrency();
    Vector<hpx::future<void>> futs;
    for (int proc = 0; proc < numberOfThreads; proc++) {
        futs.push_back(hpx::async([proc, numberOfThreads, &fail, myDomain]() {
            auto const rng = particlesCurrentRange();
            ParticleInteger const begin =
                size_t(proc) * (rng.second - rng.first) / numberOfThreads + rng.first;
            ParticleInteger const end =
                size_t(proc + 1) * (rng.second - rng.first) / numberOfThreads + rng.first;
            for (ParticleInteger i = begin; i < end; i++) {
                Array<double, kDimensions> x;
                for (int dim = 0; dim < kDimensions; dim++) {
                    x[dim] = positionDomainCoordinate(particlesPos(dim, i));
                }
                bool const contained = detail::usesMeshDomains() ?
                    detail::meshDomainContains(myDomain, i) :
                    myDomain.contains(x);
                if (!contained) {
                    PRINT("particle out of range %li of %li %e %e %e rung %i\n", (long long) i,
                        (long long) particlesSize(), x[0], x[1], x[2], particlesRung(i));
                    PRINT("%s\n", domainsFindMyBox().toString().c_str());
                    ALWAYS_ASSERT(false);
                    fail++;
                }
            }
        }));
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

static int findParticleDomain(Array<double, kDimensions> const& x, size_t key, int depth) {
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
