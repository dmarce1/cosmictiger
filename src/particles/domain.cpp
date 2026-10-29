/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <algorithm>
#include <cmath>
#include <shared_mutex>
#include <unordered_map>

#include <hpx/threading_base/annotated_function.hpp>

#include <cosmictiger/core/containers.hpp>
#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/core/safe_io.hpp>
#include <cosmictiger/math/math.hpp>
#include <cosmictiger/particles/domain.hpp>
#include <cosmictiger/particles/particles.hpp>
#if COSMICTIGER_WITH_SPH
#include <cosmictiger/sph/sph_particles.hpp>
#endif

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

namespace detail {

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
    domainsInitRebounds();
    //	PRINT( "Starting rebound\n");
    Vector<DomainGlobal> domains(1);
    boxesByKey = decltype(boxesByKey)();
    domains[0].box = unitBox<double>();
    domains[0].processRange.first = 0;
    domains[0].processRange.second = hpxSize();
    // Both gravity solvers balance the current particle population independently
    // of the FFT slabs. PMMM cell fragments are reduced by the mesh exchange.
    domains[0].totalCount = 0;
    for (auto const count : domainsGetLoads()) {
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
            // Counting and sorting use Position comparisons. Store the same
            // representable plane so migration agrees after repeated cuts on
            // an axis, including when the double midpoint rounds up to one.
            double const midpoint =
                std::min((domain.midhi + domain.midlo) * 0.5, std::nextafter(1.0, 0.0));
            double const midx = positionDomainCoordinate(Position(midpoint));
            boxesByKey[domain.key].midx = midx;
            bounds.push_back(midx);
        }
        // Child targets must use the count at the actual final split plane.
        counts = detail::countDomainSplits(bounds, depth);
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

#if COSMICTIGER_WITH_SPH
// Ordinary SPH timesteps migrate the complete gas state together with its
// gravity particle. Category indices are local and are rebuilt at the receiver.
struct GasMigrationState {
    Array<float, kDimensions> acceleration, predictedAcceleration, gravityAcceleration;
    float supportRadius, internalEnergy, predictedInternalEnergyRate, internalEnergyRate, density;
    char converged, semiactive;
    Pair<float> viscosity;    // Alpha and its previous conformal-time rate.
    float divergence;         // Previous physical divergence for Cullen-Dehnen and Price et al.
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
        if (getOptions().hydro.viscosityGradientSwitch()) {
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
        if (getOptions().hydro.viscosityGradientSwitch()) {
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
        if (getOptions().hydro.viscosityGradientSwitch()) {
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

#endif
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
    freeIndices.resize(0);
    MutexType mutex;
    int const numberOfThreads = hpxHardwareConcurrency();
    auto const currentRange = particlesCurrentRange();
    std::atomic<ParticleInteger> nextBegin(0);
    constexpr int kChunkSize = 1024;
    for (int proc = 0; proc < numberOfThreads; proc++) {
        futs.push_back(hpx::async(hpx::annotated_function(
            [proc, numberOfThreads, &mutex, &nextBegin, myDomain, currentRange, rung]() {
                std::unordered_map<int, Vector<Particle>> sends;
                Vector<hpx::future<void>> futs;
                Vector<ParticleInteger> myFreeIndices;
                ParticleInteger begin = nextBegin++ * kChunkSize + currentRange.first;
                while (begin < currentRange.second) {
                    ParticleInteger const end = std::min(begin + kChunkSize, currentRange.second);
                    for (ParticleInteger i = begin; i < end; i++) {
                        // Ordinary timesteps drift every rung together. Rebalance
                        // all particles before building their common gravity tree.
                        if (!integration::kBuildUsesHierarchical || !getOptions().gravity.enabled ||
                            particlesRung(i) == rung) {
                            Array<double, kDimensions> x;
                            for (ParticleInteger dim = 0; dim < kDimensions; dim++) {
                                x[dim] = positionDomainCoordinate(particlesPos(dim, i));
                            }
                            if (!myDomain.contains(x)) {
                                ParticleInteger const rank = findParticleDomain(x);
                                auto& send = sends[rank];
                                send.push_back(particlesGetParticle(i));
                                if (send.size() >= kMaxParticlesPerParcel) {
                                    //				PRINT( "Sending %i from %i to %i\n",
                                    //send.size(),
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
                    Array<double, kDimensions> x;
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
