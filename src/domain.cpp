/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 This program is free software; you can redistribute it and/or
 modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation; either version 2
 of the License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */

#include <shared_mutex>
#include <unordered_map>

#include <cosmictiger/containers.hpp>
#include <cosmictiger/domain.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/particles.hpp>
#include <cosmictiger/safe_io.hpp>
#include <cosmictiger/sph_particles.hpp>

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

static int findParticleDomain(Array<double, NDIM> const& x, size_t key = 1, int depth = 0);
static void domainsCheck();
static Vector<DomainLocal> localDomains;

double domainsGetLoadImbalance() {
    profilerEnter(__FUNCTION__);
    auto const loads = domainsGetLoads();
    size_t max = 0;
    double avg = 0;
    for (auto const load : loads) {
        max = std::max(load, max);
        avg += load;
    }
    avg /= loads.size();
    double const imbalance = max / avg - 1.0;
    profilerExit();
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
    localDomains[0].procRange.first = 0;
    localDomains[0].procRange.second = hpxSize();
    localDomains[0].partRange.first = 0;
    localDomains[0].partRange.second = particlesSize();
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
                auto const rng = localDomains[i].partRange;
                int const numberThreads = std::max(2 * (size_t) (rng.second - rng.first) *
                        hpxHardwareConcurrency() / (size_t) particlesSize(),
                    (size_t) 1);
                Vector<hpx::future<void>> futs;
                int const xdim = depth % NDIM;
                Fixed32 const xmid = bounds[i];
                std::atomic<size_t> count(0);
                auto const func = [numberThreads, rng, xdim, xmid, &count, &counts](int proc) {
                    size_t const begin =
                        rng.first + (size_t) proc * (rng.second - rng.first) / numberThreads;
                    size_t const end =
                        rng.first + (size_t) (proc + 1) * (rng.second - rng.first) / numberThreads;
                    size_t thisCount = 0;
                    for (size_t i = begin; i < end; i++) {
                        if (particlesPos(xdim, i) < xmid) {
                            thisCount++;
                        }
                    }
                    count += thisCount;
                };

                for (int proc = 1; proc < numberThreads; proc++) {
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
                mids[i] = particlesSort(localDomains[i].partRange, bounds[i], depth % NDIM);
                ///	PRINT( "!!!!!!!! %i %i %e %i %i\n", hpx_rank(), mids[i], bounds[i],
                /// local_domains[i].part_range.first, local_domains[i].part_range.second);
            }));
        }
    }
    hpx::wait_all(futs.begin(), futs.end());
    Vector<DomainLocal> newDomains;
    for (int i = 0; i < localDomains.size(); i++) {
        if (localDomains[i].procRange.second - localDomains[i].procRange.first > 1) {
            DomainLocal left;
            DomainLocal right;
            left = localDomains[i];
            right = localDomains[i];
            left.box.end[depth % NDIM] = right.box.begin[depth % NDIM] = bounds[i];
            left.partRange.second = right.partRange.first = mids[i];
            left.procRange.second = right.procRange.first =
                (localDomains[i].procRange.first + localDomains[i].procRange.second) / 2;
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
    profilerEnter(__FUNCTION__);
    domainsInitRebounds();
    //	PRINT( "Starting rebound\n");
    Vector<DomainGlobal> domains(1);
    boxesByKey = decltype(boxesByKey)();
    domains[0].box = unitBox<double>();
    domains[0].procRange.first = 0;
    domains[0].procRange.second = hpxSize();
    domains[0].totalCount = getOptions().numberParticles;
    domains[0].midhi = 1.0;
    domains[0].midlo = 0.0;
    domains[0].key = 1;
    boxesByKey[domains[0].key].box = domains[0].box;
    boxesByKey[domains[0].key].rank = domains[0].procRange.second - domains[0].procRange.first > 1 ?
        -1 :
        domains[0].procRange.first;
    Vector<DomainGlobal> newDomains;
    std::swap(domains, newDomains);
    int depth = 0;
    auto const allLeaves = [](Vector<DomainGlobal> const& domains) {
        bool all = true;
        for (int i = 0; i < domains.size(); i++) {
            if (domains[i].procRange.second - domains[i].procRange.first > 1) {
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
        for (int iter = 0; iter < DOMAIN_REBOUND_ITERS; iter++) {
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
                int const firstProc = domains[i].procRange.first;
                int const lastProc = domains[i].procRange.second;
                int const midProc = (firstProc + lastProc) / 2;
                ALWAYS_ASSERT(lastProc - firstProc >= 1);
                size_t const targetCount =
                    (size_t) (midProc - firstProc) * domains[i].totalCount / (lastProc - firstProc);
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
        Vector<double> bounds;
        for (int i = 0; i < domains.size(); i++) {
            if (domains[i].procRange.second - domains[i].procRange.first) {
                double const midx = (domains[i].midhi + domains[i].midlo) * 0.5;
                boxesByKey[domains[i].key].midx = midx;
                bounds.push_back(midx);
            } else {
                bounds.push_back(0.0);
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
            if (domains[i].procRange.second - domains[i].procRange.first > 1) {
                DomainGlobal left;
                DomainGlobal right;
                left = domains[i];
                right = domains[i];
                left.box.end[depth % NDIM] = right.box.begin[depth % NDIM] = bounds[i];
                left.procRange.second = right.procRange.first =
                    (domains[i].procRange.first + domains[i].procRange.second) / 2;
                int const leftdim = (depth + 1) % NDIM;
                int const rightdim = (depth + 1) % NDIM;
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
                boxesByKey[left.key].rank =
                    left.procRange.second - left.procRange.first > 1 ? -1 : left.procRange.first;
                boxesByKey[right.key].rank =
                    right.procRange.second - right.procRange.first > 1 ? -1 : right.procRange.first;
                newDomains.push_back(left);
                newDomains.push_back(right);
            } else {
                newDomains.push_back(domains[i]);
            }
        }
        depth++;
    }
    domainsTransmitBoxes(boxesByKey);
    profilerExit();
}

void domainsTransmitParticles(Vector<Particle> parts) {
    static MutexType mutex;
    std::lock_guard<MutexType> lock(mutex);
    transParticles.push_back(parts);
}

// Ordinary SPH timesteps migrate the complete gas state together with its
// gravity particle. Category indices are local and are rebuilt at the receiver.
struct GasMigrationState {
    Array<float, NCHEMFRACS> frac, dfrac;
    Array<float, NDIM> dvel, predictedVelocity, gravity;
    float alpha, h, energy, dfcold;
    float dvValue, kapValue, e0Value, de3Value, de2Value, fp2Value, preValue, fp1Value, s2Value,
        fcValue, rhValue, tcValue;
    char orValue, saValue, stValue;
    template <class Archive>
    void serialize(Archive& arc, unsigned) {
        arc & frac;
        arc & dfrac;
        arc & dvel;
        arc & predictedVelocity;
        arc & gravity;
        arc & alpha;
        arc & h;
        arc & energy;
        arc & dfcold;
        arc & dvValue;
        arc & kapValue;
        arc & e0Value;
        arc & de3Value;
        arc & de2Value;
        arc & fp2Value;
        arc & preValue;
        arc & fp1Value;
        arc & s2Value;
        arc & fcValue;
        arc & rhValue;
        arc & tcValue;
        arc & orValue;
        arc & saValue;
        arc & stValue;
    }
};
struct SphMigrationParticle {
    Particle part;
    GasMigrationState gas{};
    Array<float, 4> auxiliary{};
    Array<char, 2> flags{};
    template <class Archive>
    void serialize(Archive& arc, unsigned) {
        arc & part;
        arc & auxiliary;
        arc & flags;
        if (part.type == SPH_TYPE) {
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
    ALWAYS_ASSERT(result.part.type != STAR_TYPE);
    result.auxiliary[0] = particlesSoftlen(index);
    result.auxiliary[1] = particlesZeta(index);
    result.auxiliary[2] = particlesRho(index);
    result.auxiliary[3] = particlesDivv(index);
    result.flags[0] = particlesConverged(index);
    result.flags[1] = particlesSemiactive(index);
    if (result.part.type == SPH_TYPE) {
        auto const i = particlesCatIndex(index);
        auto& gas = result.gas;
        gas.frac = sphParticlesR1[i].frac;
        gas.alpha = sphParticlesR1[i].alpha;
        gas.h = sphParticlesR2[i].h;
        gas.dfrac = sphParticlesR5[i].dfrac;
        gas.dfcold = sphParticlesR5[i].dfcold;
        gas.dvel = sphParticlesR6[i].dvel;
        gas.dvValue = sphParticlesDv[i];
        gas.kapValue = sphParticlesKap[i];
        gas.e0Value = sphParticlesE0[i];
        gas.de3Value = sphParticlesDe3[i];
        gas.de2Value = sphParticlesDe2[i];
        gas.fp2Value = sphParticlesFp2[i];
        gas.preValue = sphParticlesPre[i];
        gas.fp1Value = sphParticlesFp1[i];
        gas.s2Value = sphParticlesS2[i];
        gas.fcValue = sphParticlesFc[i];
        gas.rhValue = sphParticlesRh[i];
        gas.tcValue = sphParticlesTc[i];
        gas.orValue = sphParticlesOr[i];
        gas.saValue = sphParticlesSa[i];
        gas.stValue = sphParticlesSt[i];
#ifdef ENTROPY
        gas.energy = sphParticlesR2[i].entropy;
#else
        gas.energy = sphParticlesR2[i].eint;
#endif
        for (int dim = 0; dim < NDIM; dim++) {
            gas.predictedVelocity[dim] = sphParticlesDv2[dim][i];
            gas.gravity[dim] = sphParticlesG[dim][i];
        }
    }
    return result;
}
static void unpackSphParticle(
    SphMigrationParticle record, ParticleInteger index, ParticleInteger gasIndex) {
    if (record.part.type == SPH_TYPE) {
        record.part.catIndex = gasIndex;
    }
    particlesSetParticle(record.part, index);
    particlesSoftlen(index) = record.auxiliary[0];
    particlesZeta(index) = record.auxiliary[1];
    particlesRho(index) = record.auxiliary[2];
    particlesDivv(index) = record.auxiliary[3];
    particlesConverged(index) = record.flags[0];
    particlesSemiactive(index) = record.flags[1];
    if (record.part.type == SPH_TYPE) {
        auto const i = gasIndex;
        auto const& gas = record.gas;
        sphParticlesDmIndex(i) = index;
        sphParticlesR1[i].frac = gas.frac;
        sphParticlesR1[i].alpha = gas.alpha;
        sphParticlesR2[i].h = gas.h;
        sphParticlesR5[i].dfrac = gas.dfrac;
        sphParticlesR5[i].dfcold = gas.dfcold;
        sphParticlesR6[i].dvel = gas.dvel;
        sphParticlesDv[i] = gas.dvValue;
        sphParticlesKap[i] = gas.kapValue;
        sphParticlesE0[i] = gas.e0Value;
        sphParticlesDe3[i] = gas.de3Value;
        sphParticlesDe2[i] = gas.de2Value;
        sphParticlesFp2[i] = gas.fp2Value;
        sphParticlesPre[i] = gas.preValue;
        sphParticlesFp1[i] = gas.fp1Value;
        sphParticlesS2[i] = gas.s2Value;
        sphParticlesFc[i] = gas.fcValue;
        sphParticlesRh[i] = gas.rhValue;
        sphParticlesTc[i] = gas.tcValue;
        sphParticlesOr[i] = gas.orValue;
        sphParticlesSa[i] = gas.saValue;
        sphParticlesSt[i] = gas.stValue;
#ifdef ENTROPY
        sphParticlesR2[i].entropy = gas.energy;
#else
        sphParticlesR2[i].eint = gas.energy;
#endif
        for (int dim = 0; dim < NDIM; dim++) {
            sphParticlesDv2[dim][i] = gas.predictedVelocity[dim];
            sphParticlesG[dim][i] = gas.gravity[dim];
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
        Array<double, NDIM> x;
        for (int dim = 0; dim < NDIM; dim++) {
            x[dim] = particlesPos(dim, i).toDouble();
        }
        auto record = packSphParticle(i);
        if (box.contains(x)) {
            retainedSphParticles.push_back(std::move(record));
        } else {
            auto const rank = findParticleDomain(x);
            auto& send = sends[rank];
            send.push_back(std::move(record));
            if (send.size() >= MAX_PARTICLES_PER_PARCEL) {
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
        gasCount += record.part.type == SPH_TYPE;
    }
    particlesResize(retainedSphParticles.size());
    sphParticlesResize(gasCount, false);
    ParticleInteger gasIndex = 0;
    for (ParticleInteger i = 0; i < retainedSphParticles.size(); i++) {
        unpackSphParticle(retainedSphParticles[i], i, gasIndex);
        gasIndex += retainedSphParticles[i].part.type == SPH_TYPE;
    }
    retainedSphParticles.clear();
    for (auto& future : pending) {
        future.get();
    }
}

void domainsBegin(int rung) {
    if (getOptions().sph) {
        beginSphMigration(rung);
        return;
    }
    profilerEnter(__FUNCTION__);
    Vector<hpx::future<void>> futs;
    auto children = hpxChildren();
    for (auto& c : children) {
        futs.push_back(hpx::async<DomainsBeginAction>(c, rung));
    }
    auto const myDomain = domainsFindMyBox();
    freeIndices.resize(0);
    MutexType mutex;
    int const numberThreads = hpxHardwareConcurrency();
    auto const currentRange = particlesCurrentRange();
    std::atomic<ParticleInteger> nextBegin(0);
    constexpr int kChunkSize = 1024;
    for (int proc = 0; proc < numberThreads; proc++) {
        futs.push_back(hpx::async([proc, numberThreads, &mutex, &nextBegin, myDomain, currentRange,
                                      rung]() {
            std::unordered_map<int, Vector<Particle>> sends;
            Vector<hpx::future<void>> futs;
            Vector<ParticleInteger> myFreeIndices;
            ParticleInteger begin = nextBegin++ * kChunkSize + currentRange.first;
            while (begin < currentRange.second) {
                ParticleInteger const end = std::min(begin + kChunkSize, currentRange.second);
                for (ParticleInteger i = begin; i < end; i++) {
                    if (particlesRung(i) == rung) {
                        Array<double, NDIM> x;
                        for (ParticleInteger dim = 0; dim < NDIM; dim++) {
                            x[dim] = particlesPos(dim, i).toDouble();
                        }
                        if (!myDomain.contains(x)) {
                            ParticleInteger const rank = findParticleDomain(x);
                            auto& send = sends[rank];
                            send.push_back(particlesGetParticle(i));
                            if (send.size() >= MAX_PARTICLES_PER_PARCEL) {
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
    profilerExit();
}

void domainsEnd() {
    if (getOptions().sph) {
        endSphMigration();
        return;
    }
    profilerEnter(__FUNCTION__);
    Vector<hpx::future<void>> futs;
    auto children = hpxChildren();
    for (auto& c : children) {
        futs.push_back(hpx::async<DomainsEndAction>(c));
    }
    auto const particleCompare = [](Particle a, Particle b) {
        for (ParticleInteger dim = 0; dim < NDIM; dim++) {
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
    int const numberThreads = hpxHardwareConcurrency();
    std::atomic<int> index(0);
    for (int proc = 0; proc < numberThreads; proc++) {
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
    for (int proc = 0; proc < numberThreads; proc++) {
        futs.push_back(hpx::async([numberThreads, proc, &allTransParticles]() {
            ParticleInteger const begin = (size_t) proc * freeIndices.size() / numberThreads;
            ParticleInteger const end = (size_t) (proc + 1) * freeIndices.size() / numberThreads;
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
    profilerExit();
}

Range<double> domainsFindMyBox() {
    return localDomains[hpxRank()].box;
}

static void domainsCheck() {
    std::atomic<int> fail(0);
    auto const myDomain = domainsFindMyBox();
    int const numberThreads = hpxHardwareConcurrency();
    Vector<hpx::future<void>> futs;
    for (int proc = 0; proc < numberThreads; proc++) {
        futs.push_back(hpx::async([proc, numberThreads, &fail, myDomain]() {
            auto const rng = particlesCurrentRange();
            ParticleInteger const begin =
                size_t(proc) * (rng.second - rng.first) / numberThreads + rng.first;
            ParticleInteger const end =
                size_t(proc + 1) * (rng.second - rng.first) / numberThreads + rng.first;
            for (ParticleInteger i = begin; i < end; i++) {
                Array<double, NDIM> x;
                for (int dim = 0; dim < NDIM; dim++) {
                    x[dim] = particlesPos(dim, i).toDouble();
                }
                if (!myDomain.contains(x)) {
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

static int findParticleDomain(Array<double, NDIM> const& x, size_t key, int depth) {
    auto const& entry = boxesByKey[key];
    if (entry.rank == -1) {
        int const xdim = depth % NDIM;
        key <<= 1;
        if (x[xdim] >= entry.midx) {
            key += 1;
        }
        return findParticleDomain(x, key, depth + 1);
    } else {
        return entry.rank;
    }
}
