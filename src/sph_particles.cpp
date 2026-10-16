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

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <unordered_map>

#define SPH_PARTICLES_CPP
#include <cosmictiger/constants.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/sph.hpp>
#include <cosmictiger/sph_particles.hpp>
#include <cosmictiger/stars.hpp>

static ParticleInteger capacity = 0;
static ParticleInteger size = 0;

struct LineIdType {
    int proc;
    ParticleInteger index;
    inline bool operator==(LineIdType other) const {
        return proc == other.proc && index == other.index;
    }
};

struct LineIdHash {
    inline size_t operator()(LineIdType id) const {
        ParticleInteger const lineSize = getOptions().particleCacheLineSize;
        ParticleInteger const i = id.index / lineSize;
        return i * (hpxSize() - 1) + ((id.proc < hpxRank()) ? id.proc : id.proc - 1);
    }
};

struct LineIdHashLo {
    inline size_t operator()(LineIdType id) const {
        LineIdHash hash;
        return hash(id) % PART_CACHE_SIZE;
    }
};

struct LineIdHashHi {
    inline size_t operator()(LineIdType id) const {
        LineIdHash hash;
        return hash(id) / PART_CACHE_SIZE;
    }
};

static Array<Fixed32, NDIM> const* sphParticlesCacheReadLine(LineIdType lineId);
static Pair<float> const* sphParticlesEnergyAndSmoothlenCacheReadLine(LineIdType lineId);
static Pair<float, char> const* sphParticlesFcoldCacheReadLine(LineIdType lineId);
static char const* sphParticlesRungCacheReadLine(LineIdType lineId);
static Array<float, NDIM> const* sphParticlesVelsCacheReadLine(LineIdType lineId);
static float const* sphParticlesKappasCacheReadLine(LineIdType lineId);
static float const* sphParticlesRhoCacheReadLine(LineIdType lineId);
static AuxiliaryQuantities const* sphParticlesAuxCacheReadLine(LineIdType lineId);

static Vector<Array<Fixed32, NDIM>> sphParticlesFetchCacheLine(ParticleInteger index);
static Vector<char> sphParticlesFetchRungCacheLine(ParticleInteger index);
static Vector<Array<float, NDIM>> sphParticlesFetchVelsCacheLine(ParticleInteger index);
static Vector<float> sphParticlesFetchRhoCacheLine(ParticleInteger index);
static Vector<float> sphParticlesFetchKappasCacheLine(ParticleInteger index);
static Vector<Pair<float>> sphParticlesFetchEnergyAndSmoothlenCacheLine(ParticleInteger index);
static Vector<Pair<float, char>> sphParticlesFetchFcoldCacheLine(ParticleInteger index);
static Vector<AuxiliaryQuantities> sphParticlesFetchAuxCacheLine(ParticleInteger index);

HPX_PLAIN_ACTION(sphParticlesFetchCacheLine, SphParticlesFetchCacheLineAction);
HPX_PLAIN_ACTION(sphParticlesFetchEnergyAndSmoothlenCacheLine,
    SphParticlesFetchEnergyAndSmoothlenCacheLineAction);
HPX_PLAIN_ACTION(sphParticlesFetchFcoldCacheLine, SphParticlesFetchFcoldCacheLineAction);
HPX_PLAIN_ACTION(sphParticlesFetchRungCacheLine, SphParticlesFetchRungCacheLineAction);
HPX_PLAIN_ACTION(sphParticlesFetchVelsCacheLine, SphParticlesFetchVelsCacheLineAction);
HPX_PLAIN_ACTION(sphParticlesFetchRhoCacheLine, SphParticlesFetchRhoCacheLineAction);
HPX_PLAIN_ACTION(sphParticlesFetchKappasCacheLine, SphParticlesFetchKappasCacheLineAction);
#ifdef ENTROPY
HPX_PLAIN_ACTION(sphParticlesCacheFreeEntr, SphParticlesCacheFreeEntrAction);
#else
HPX_PLAIN_ACTION(sphParticlesCacheFreeEint, SphParticlesCacheFreeEintAction);
#endif
HPX_PLAIN_ACTION(sphParticlesCacheFree1, SphParticlesCacheFree1Action);
HPX_PLAIN_ACTION(sphParticlesCacheFree2, SphParticlesCacheFree2Action);
HPX_PLAIN_ACTION(sphParticlesFetchAuxCacheLine, SphParticlesFetchAuxCacheLineAction);

static Array<std::unordered_map<LineIdType,
                 std::shared_ptr<hpx::shared_future<Vector<Array<Fixed32, NDIM>>>>, LineIdHashHi>,
    PART_CACHE_SIZE>
    particleCache;
static Array<std::unordered_map<LineIdType,
                 std::shared_ptr<hpx::shared_future<Vector<Pair<float>>>>, LineIdHashHi>,
    PART_CACHE_SIZE>
    energyAndSmoothingLengthParticleCache;
static Array<std::unordered_map<LineIdType,
                 std::shared_ptr<hpx::shared_future<Vector<Pair<float, char>>>>, LineIdHashHi>,
    PART_CACHE_SIZE>
    coldFractionParticleCache;
static Array<
    std::unordered_map<LineIdType, std::shared_ptr<hpx::shared_future<Vector<char>>>, LineIdHashHi>,
    PART_CACHE_SIZE>
    rungParticleCache;
static Array<std::unordered_map<LineIdType,
                 std::shared_ptr<hpx::shared_future<Vector<Array<float, NDIM>>>>, LineIdHashHi>,
    PART_CACHE_SIZE>
    particleVelocityCache;
static Array<std::unordered_map<LineIdType, std::shared_ptr<hpx::shared_future<Vector<float>>>,
                 LineIdHashHi>,
    PART_CACHE_SIZE>
    conductivityParticleCache;
static Array<std::unordered_map<LineIdType, std::shared_ptr<hpx::shared_future<Vector<float>>>,
                 LineIdHashHi>,
    PART_CACHE_SIZE>
    densityParticleCache;
static Array<std::unordered_map<LineIdType,
                 std::shared_ptr<hpx::shared_future<Vector<AuxiliaryQuantities>>>, LineIdHashHi>,
    PART_CACHE_SIZE>
    auxCache;
static Array<SpinlockType, PART_CACHE_SIZE> mutexes;
static Array<SpinlockType, PART_CACHE_SIZE> entrMutexes;
static Array<SpinlockType, PART_CACHE_SIZE> fcoldMutexes;
static Array<SpinlockType, PART_CACHE_SIZE> rungMutexes;
static Array<SpinlockType, PART_CACHE_SIZE> velsMutexes;
static Array<SpinlockType, PART_CACHE_SIZE> kappasMutexes;
static Array<SpinlockType, PART_CACHE_SIZE> rhoMutexes;
static Array<SpinlockType, PART_CACHE_SIZE> auxMutexes;
static int groupCacheEpoch = 0;

template <typename Action, typename T>
static T const* readSphCacheLine(LineIdType lineId,
    Array<std::unordered_map<LineIdType, std::shared_ptr<hpx::shared_future<Vector<T>>>,
              LineIdHashHi>,
        PART_CACHE_SIZE>& cache,
    Array<SpinlockType, PART_CACHE_SIZE>& locks) {
    size_t const bin = LineIdHashLo()(lineId);
    std::shared_ptr<hpx::shared_future<Vector<T>>> future;
    {
        std::lock_guard<SpinlockType> lock(locks[bin]);
        auto const entry = cache[bin].find(lineId);
        if (entry != cache[bin].end()) {
            future = entry->second;
        }
    }
    if (!future) {
        auto promise = std::make_shared<hpx::promise<Vector<T>>>();
        auto candidate =
            std::make_shared<hpx::shared_future<Vector<T>>>(promise->get_future().share());
        bool inserted;
        {
            std::lock_guard<SpinlockType> lock(locks[bin]);
            auto const entry = cache[bin].emplace(lineId, candidate);
            inserted = entry.second;
            future = entry.first->second;
        }
        if (inserted) {
            try {
                hpx::apply([promise, lineId]() {
                    try {
                        auto remote =
                            hpx::async<Action>(hpxLocalities()[lineId.proc], lineId.index);
                        promise->set_value(remote.get());
                    } catch (...) {
                        promise->set_exception(std::current_exception());
                    }
                });
            } catch (...) {
                promise->set_exception(std::current_exception());
            }
        }
    }
    return future->get().data();
}

HPX_PLAIN_ACTION(sphParticlesMaximumSmoothingLength, SphParticlesMaximumSmoothingLengthAction);

float sphParticlesMaximumSmoothingLength() {
    Vector<hpx::future<float>> futs;
    for (auto& c : hpxChildren()) {
        futs.push_back(hpx::async<SphParticlesMaximumSmoothingLengthAction>(c));
    }
    int const numberOfThreads = hpxHardwareConcurrency();
    for (int proc = 0; proc < numberOfThreads; proc++) {
        futs.push_back(hpx::async([numberOfThreads, proc]() {
            float maxh = 0.0;
            ParticleInteger const b = (size_t) proc * sphParticlesSize() / numberOfThreads;
            ParticleInteger const e = (size_t) (proc + 1) * sphParticlesSize() / numberOfThreads;
            for (int i = b; i < e; i++) {
                maxh = std::max(maxh, sphParticlesSmoothingLength(i));
            }
            return maxh;
        }));
    }
    float maxh = 0.f;
    for (auto& f : futs) {
        maxh = std::max(maxh, f.get());
    }
    PRINT("done\n");
    return maxh;
}

HPX_PLAIN_ACTION(sphParticlesApplyUpdates, SphParticlesApplyUpdatesAction);
std::pair<double, double> sphParticlesApplyUpdates(
    int minrung, int phase, float t0, float tau, float w) {
    if (!(phase == 2 || tau > 0.0)) {
        return std::pair<double, double>();
    }
    profilerEnter(__FUNCTION__);
    double err = 0.0;
    double norm = 0.0;
    Vector<hpx::future<std::pair<double, double>>> futs;
    for (auto& c : hpxChildren()) {
        futs.push_back(hpx::async<SphParticlesApplyUpdatesAction>(c, minrung, phase, t0, tau, w));
    }
    int const numberOfThreads = hpxHardwareConcurrency();
    for (int proc = 0; proc < numberOfThreads; proc++) {
        futs.push_back(hpx::async([t0, numberOfThreads, proc, phase, minrung, tau, w]() {
            auto const chem = getOptions().chem;
            auto const stars = getOptions().stars;
            double error = 0.0;
            double norm = 0.0;
            ParticleInteger const b = (size_t) proc * sphParticlesSize() / numberOfThreads;
            ParticleInteger const e = (size_t) (proc + 1) * sphParticlesSize() / numberOfThreads;
            for (int i = b; i < e; i++) {
                ParticleInteger const k = sphParticlesDmIndex(i);
                auto const rung2 = sphParticlesRung(i);
                float const dt2 = 0.5f * t0 / (1 << rung2);
                if (phase == 0) {
                    if (rung2 >= minrung && !sphParticlesIsstar(i)) {
                        for (int dim = 0; dim < NDIM; dim++) {
                            sphParticlesDvel0(dim, i) = sphParticlesDvel(dim, i);
                        }
                        for (int dim = 0; dim < NDIM; dim++) {
                            particlesVel(dim, k) += sphParticlesDvel(dim, i) * dt2;
                        }
#ifdef ENTROPY
                        ALWAYS_ASSERT(sphParticlesEntr(i) > 0.0);
                        sphParticlesEntr0(i) = sphParticlesDentr2(i);
                        sphParticlesEntr(i) += sphParticlesDentr2(i) * dt2;
                        if (sphParticlesDentr2(i) < -1.0e-7 / sphParticlesEntr(i)) {
                            ALWAYS_ASSERT(sphParticlesDentr2(i) >= 0.f);
                        }
                        ALWAYS_ASSERT(sphParticlesEntr(i) > 0.0);
#else
                        ALWAYS_ASSERT(sphParticlesEint(i) > 0.0);
                        sphParticlesEint0(i) = sphParticlesDeint2(i);
                        sphParticlesEint(i) += sphParticlesDeint2(i) * dt2;
                        ALWAYS_ASSERT(sphParticlesEint(i) > 0.0);
#endif
                    }
                } else if (phase == 1) {
                    if (rung2 >= minrung && !sphParticlesIsstar(i)) {
                        for (int dim = 0; dim < NDIM; dim++) {
                            particlesVel(dim, k) +=
                                (sphParticlesDvel(dim, i) - sphParticlesDvel0(dim, i)) * dt2;
                        }
#ifdef ENTROPY
                        sphParticlesEntr(i) += (-sphParticlesEntr0(i)) * dt2;
                        if (sphParticlesEntr(i) <= 0.0) {
                            PRINT("%e %e\n", sphParticlesEntr(i), (-sphParticlesEntr0(i)) * dt2);
                        }
                        ALWAYS_ASSERT(sphParticlesEntr(i) > 0.0);
                        sphParticlesEntr(i) += (sphParticlesDentr2(i)) * dt2;
                        if (sphParticlesEntr(i) <= 0.0) {
                            PRINT("%e %e\n", sphParticlesEntr(i), (sphParticlesDentr2(i)) * dt2);
                        }
                        ALWAYS_ASSERT(sphParticlesEntr(i) > 0.0);
#else
                        sphParticlesEint(i) += (-sphParticlesEint0(i)) * dt2;
                        if (sphParticlesEint(i) <= 0.0) {
                            PRINT("%e %e\n", sphParticlesEint(i), (-sphParticlesEint0(i)) * dt2);
                        }
                        ALWAYS_ASSERT(sphParticlesEint(i) > 0.0);
                        sphParticlesEint(i) += (sphParticlesDeint2(i)) * dt2;
                        if (sphParticlesEint(i) <= 0.0) {
                            PRINT("%e %e\n", sphParticlesEint(i), (sphParticlesDeint2(i)) * dt2);
                        }
                        ALWAYS_ASSERT(sphParticlesEint(i) > 0.0);
#endif
                    }
                } else if (phase == 2) {
                    if (rung2 >= minrung && !sphParticlesIsstar(i)) {
                        for (int dim = 0; dim < NDIM; dim++) {
                            particlesVel(dim, k) += sphParticlesDvel(dim, i) * dt2;
                        }
#ifdef ENTROPY
                        sphParticlesEntr(i) += sphParticlesDentr1(i) * 2.0 * dt2;
                        ALWAYS_ASSERT(sphParticlesEntr(i) > 0.0);
                        sphParticlesEntr(i) += sphParticlesDentr2(i) * dt2;
                        ALWAYS_ASSERT(sphParticlesEntr(i) > 0.0);
#else
                        sphParticlesEint(i) += sphParticlesDeint1(i) * 2.0 * dt2;
                        ALWAYS_ASSERT(sphParticlesEint(i) > 0.0);
                        sphParticlesEint(i) += sphParticlesDeint2(i) * dt2;
                        ALWAYS_ASSERT(sphParticlesEint(i) > 0.0);
#endif
                        if (chem) {
                            for (int fi = 0; fi < NCHEMFRACS; fi++) {
                                auto& frac = sphParticlesFrac(fi, i);
                                auto dfrac = sphParticlesDchem(i)[fi];
                                if (frac + dfrac * dt2 * 2.0 < 0.0f) {
                                    PRINT("-----> %e %e\n", frac, frac + dfrac * dt2 * 2.0);
                                }
                                frac += dfrac * dt2 * 2.0;
                            }
                        }
                    }
                }
#ifndef ENTROPY
                if (rung2 >= minrung && !sphParticlesIsstar(i)) {
                    // Diagnostics and interpolation must use the corrected internal energy.
                    sphParticlesPressure(i) =
                        (getOptions().gamma - 1.0) * sphParticlesRho(i) * sphParticlesEint(i);
                }
#endif
            }
            return std::make_pair(error, norm);
        }));
    }
    for (auto& f : futs) {
        auto tmp = f.get();
        err = std::max(err, tmp.first);
        norm += tmp.second;
    }
    profilerExit();
    return std::make_pair(err, 1.0);
}

HPX_PLAIN_ACTION(sphParticlesSoftlens2smoothlens, SphParticlesSoftlens2smoothlensAction);

void sphParticlesSoftlens2smoothlens(int minrung) {
    profilerEnter(__FUNCTION__);
    double err = 0.0;
    double norm = 0.0;
    Vector<hpx::future<void>> futs;
    for (auto& c : hpxChildren()) {
        futs.push_back(hpx::async<SphParticlesSoftlens2smoothlensAction>(c, minrung));
    }
    int const numberOfThreads = hpxHardwareConcurrency();
    for (int proc = 0; proc < numberOfThreads; proc++) {
        futs.push_back(hpx::async([numberOfThreads, proc, minrung]() {
            ParticleInteger const b = (size_t) proc * sphParticlesSize() / numberOfThreads;
            ParticleInteger const e = (size_t) (proc + 1) * sphParticlesSize() / numberOfThreads;
            for (int i = b; i < e; i++) {
                ParticleInteger const k = sphParticlesDmIndex(i);
                //		if( particles_rung(k) >= minrung || particles_semiactive(k)) {
                sphParticlesSmoothingLength(i) = particlesSoftlen(k);
                //	}
            }
        }));
    }
    for (auto& f : futs) {
        f.get();
    }
    profilerExit();
}

float sphParticlesTemperature(ParticleInteger i, float a) {
    if (sphParticlesIsstar(i)) {
        return 0.0;
    } else {
        double const codeToEnergyDensity =
            getOptions().codeToG / (getOptions().codeToCm * sqr(getOptions().codeToS));    // 7
        double const codeToEnergy =
            sqr(getOptions().codeToCm) / (sqr(getOptions().codeToS));                          // 7
        double const codeToDensity = pow(getOptions().codeToCm, -3) * getOptions().codeToG;    // 10
        double const codeToEntropy = codeToEnergy / pow(codeToDensity, getOptions().gamma - 1.0);
        double const h = sphParticlesSmoothingLength(i);
        double const hp = sphParticlesHp(i);
        double const hn = sphParticlesHn(i);
        double const h2 = sphParticlesH2(i);
        double const yValue = sphParticlesY(i);
        double const hep = sphParticlesHep(i);
        double const hepp = sphParticlesHepp(i);
        double const hValue = sphParticlesH(i);
        double const he = yValue - hep - hepp;
#ifdef HOPKINS
        double rho = sphParticlesRhoRho(i);
#else
        double rho = sphParticlesRho(i);
#endif
        double const hfrac = 1.0 - sphParticlesColdMass(i);
        double n = (hValue + 2.f * hp + .5f * h2 + .25f * he + .5f * hep + .75f * hepp) * 1.0 /
            (1.0 - sphParticlesZ(i));
        rho *= codeToDensity * pow(a, -3);
        n *= constants::kAvo * rho;    // 8
        double gamma = sphParticlesGamma(i);
        double cv = 1.0 / (gamma - 1.0);    // 4
        cv *= double(constants::kB);        // 1
#ifdef ENTROPY
        double entr = sphParticlesEntr(i);
        entr *= codeToEntropy;
        double const eint = entr * pow(rho * hfrac, getOptions().gamma - 1.0) / (gamma - 1.0);
#else
        double const eint = sphParticlesEint(i) * codeToEnergy;
#endif
        double t = rho * eint / (n * cv);
        if (hValue < 0.0) {
            if (hValue < -5.0e-3) {
                PRINT("NEGATIVE H\n");
                PRINT("%e %e %e %e %e %e %e\n", hValue, hp, hn, h2, he, hep, hepp);
                abort();
            }
        }
        if (t > TMAX) {
            PRINT("T == %e %e %e %e %e\n", t, eint, eint, rho, h);
            abort();
        }
        if (t < 0.0) {
            PRINT("T == %e %e %e %e %e\n", t, eint, eint, rho, h);
        }
        return t;
    }
}

float sphParticlesMmw(ParticleInteger i) {
    double const hp = sphParticlesHp(i);
    double const hn = sphParticlesHn(i);
    double const h2 = sphParticlesH2(i);
    double const yValue = sphParticlesY(i);
    double const hep = sphParticlesHep(i);
    double const hepp = sphParticlesHepp(i);
    double const hValue = sphParticlesH(i);
    double const he = yValue - hep - hepp;
    double n = hValue + 2.f * hp + .5f * h2 + .25f * he + .5f * hep + .75f * hepp;
    //	PRINT( "%e\n", 1.0 / n);
    return 1.0 / n;
}

namespace detail {

template <typename Visitor>
void visitSphCheckpointArrays(Visitor&& visit) {
    size_t const count = sphParticlesSize();
    // Keep stage fields as well as the physical fields: fine rungs may be staggered at checkpoints.
    visit(sphParticlesR1, count);
    visit(sphParticlesR2, count);
    visit(sphParticlesR5, count);
    visit(sphParticlesR6, count);
    visit(sphParticlesDm, count);
    visit(sphParticlesDv, count);
    visit(sphParticlesKap, count);
    visit(sphParticlesE0, count);
    visit(sphParticlesDe3, count);
    visit(sphParticlesDe2, count);
    visit(sphParticlesFp2, count);
    visit(sphParticlesPre, count);
    visit(sphParticlesFp1, count);
    visit(sphParticlesS2, count);
    visit(sphParticlesFc, count);
    visit(sphParticlesRh, count);
    visit(sphParticlesTc, count);
    visit(sphParticlesOr, count);
    visit(sphParticlesSa, count);
    visit(sphParticlesSt, count);
    for (int dim = 0; dim < NDIM; dim++) {
        visit(sphParticlesDv2[dim], count);
        visit(sphParticlesG[dim], count);
    }
}

uint32_t sphCheckpointMode() {
#ifdef ENTROPY
    return 1;
#else
    return 0;
#endif
}

}    // namespace detail

void sphParticlesSwap(ParticleInteger i, ParticleInteger j) {
    detail::visitSphCheckpointArrays(
        [i, j](auto* values, size_t) { std::swap(values[i], values[j]); });
}

void sphParticlesSwap2(ParticleInteger i, ParticleInteger j) {
    sphParticlesSwap(i, j);
}

ParticleInteger sphParticlesSort(Pair<ParticleInteger> rng, Fixed32 xmid, int xdim) {
    ParticleInteger begin = rng.first;
    ParticleInteger end = rng.second;
    ParticleInteger lo = begin;
    ParticleInteger hi = end;
    ParticleInteger j = end;
    while (lo < hi) {
        if (sphParticlesPos(xdim, lo) >= xmid) {
            while (lo != hi) {
                hi--;
                if (sphParticlesPos(xdim, hi) < xmid) {
                    sphParticlesSwap(lo, hi);
                    break;
                }
            }
        }
        lo++;
    }
    return hi;
}

ParticleInteger sphParticlesSize() {
    return size;
}

template <typename T>
void sphParticlesArrayResize(T*& ptr, ParticleInteger newCapacity, bool reg) {
    T* newPtr;
#ifdef USE_CUDA
    if (reg) {
        cudaMallocManaged(&newPtr, sizeof(T) * newCapacity);
    } else {
        newPtr = (T*) malloc(sizeof(T) * newCapacity);
    }
#else
    newPtr = (T*) malloc(sizeof(T) * newCapacity);
    memset(newPtr, T(0), newCapacity);
#endif
    if (capacity > 0) {
        hpx_copy(PAR_EXECUTION_POLICY, ptr, ptr + size, newPtr).get();
#ifdef USE_CUDA
        if (reg) {
            cudaFree(ptr);
        } else {
            free(ptr);
        }
#else
        free(ptr);
#endif
    }
    ptr = newPtr;
}

void sphParticlesResize(ParticleInteger sz, bool parts2) {
    bool const chem = getOptions().chem;
    bool const stars = getOptions().stars;
    if (sz > capacity) {
        ParticleInteger newCapacity = std::max(capacity, (ParticleInteger) 100);
        while (newCapacity < sz) {
            newCapacity = size_t(101) * newCapacity / size_t(100);
        }
        //	PRINT("Resizing sph_particles to %li from %li\n", new_capacity, capacity);
        sphParticlesArrayResize(sphParticlesRh, newCapacity, true);
        sphParticlesArrayResize(sphParticlesSt, newCapacity, true);
        sphParticlesArrayResize(sphParticlesFc, newCapacity, true);
        sphParticlesArrayResize(sphParticlesDv, newCapacity, true);
        sphParticlesArrayResize(sphParticlesR1, newCapacity, true);
        sphParticlesArrayResize(sphParticlesR2, newCapacity, true);
        sphParticlesArrayResize(sphParticlesR5, newCapacity, true);
        sphParticlesArrayResize(sphParticlesR6, newCapacity, true);
        sphParticlesArrayResize(sphParticlesKap, newCapacity, true);
        sphParticlesArrayResize(sphParticlesOr, newCapacity, true);
        sphParticlesArrayResize(sphParticlesSa, newCapacity, true);
        sphParticlesArrayResize(sphParticlesDm, newCapacity, true);
        sphParticlesArrayResize(sphParticlesE0, newCapacity, true);
        sphParticlesArrayResize(sphParticlesDe2, newCapacity, true);
        sphParticlesArrayResize(sphParticlesDe3, newCapacity, true);
        sphParticlesArrayResize(sphParticlesFp1, newCapacity, true);
        sphParticlesArrayResize(sphParticlesFp2, newCapacity, true);
        sphParticlesArrayResize(sphParticlesPre, newCapacity, true);
        sphParticlesArrayResize(sphParticlesS2, newCapacity, true);
        sphParticlesArrayResize(sphParticlesTc, newCapacity, false);
        for (int dim = 0; dim < NDIM; dim++) {
            sphParticlesArrayResize(sphParticlesDv2[dim], newCapacity, true);
            sphParticlesArrayResize(sphParticlesG[dim], newCapacity, true);
        }
        capacity = newCapacity;
    }
    ParticleInteger numberOfNewParticles = sz - size;
    ParticleInteger offset = particlesSize();
    if (parts2) {
        particlesResize(particlesSize() + numberOfNewParticles);
    }
    int oldsz = size;
    size = sz;
    for (int i = 0; i < numberOfNewParticles; i++) {
        ParticleInteger const slot = oldsz + i;
        sphParticlesR1[slot] = {};
        sphParticlesR2[slot] = {};
        sphParticlesR5[slot] = {};
        sphParticlesR6[slot] = {};
        sphParticlesDm[slot] = NO_INDEX;
        sphParticlesDv[slot] = 0.0f;
        sphParticlesKap[slot] = 0.0f;
        sphParticlesE0[slot] = 0.0f;
        sphParticlesDe3[slot] = 0.0f;
        sphParticlesDe2[slot] = 0.0f;
        sphParticlesFp2[slot] = 0.0f;
        sphParticlesPre[slot] = 0.0f;
        sphParticlesFp1[slot] = 0.0f;
        sphParticlesS2[slot] = 0.0f;
        sphParticlesFc[slot] = 0.0f;
        sphParticlesRh[slot] = 0.0f;
        sphParticlesTc[slot] = 0.0f;
        sphParticlesOr[slot] = 0;
        sphParticlesSa[slot] = 0;
        sphParticlesSt[slot] = 0;
        for (int dim = 0; dim < NDIM; dim++) {
            sphParticlesDv2[dim][slot] = 0.0f;
            sphParticlesG[dim][slot] = 0.0f;
        }
        if (parts2) {
            particlesCatIndex(offset + i) = oldsz + i;
            particlesType(offset + i) = SPH_TYPE;
            sphParticlesDmIndex(oldsz + i) = offset + i;
        }
        sphParticlesAlpha(oldsz + i) = getOptions().alpha0;
        sphParticlesDivv(oldsz + i) = 0.f;
        sphParticlesSemiactive(oldsz + i) = 0.0f;
        if (stars) {
            sphParticlesColdMass((oldsz + i)) = 0.f;
        }
        sphParticlesIsstar((oldsz + i)) = false;
#ifdef ENTROPY
        sphParticlesDentr1(oldsz + i) = 0.f;
        sphParticlesDentr2(oldsz + i) = 0.f;
#else
        sphParticlesDeint1(oldsz + i) = 0.f;
        sphParticlesDeint2(oldsz + i) = 0.f;
#endif
        for (int dim = 0; dim < NDIM; dim++) {
            sphParticlesDvel(dim, oldsz + i) = 0.0f;
            sphParticlesDvel0(dim, oldsz + i) = 0.0f;
            sphParticlesGforce(dim, oldsz + i) = 0.0f;
        }
    }
}

void sphParticlesFree() {
    bool const stars = getOptions().stars;
    free(sphParticlesDm);
    bool cuda = false;
#ifdef USE_CUDA
#ifdef SPH_TOTAL_ENERGY
    cuda = true;
#endif
#endif
    CUDA_CHECK(cudaFree(sphParticlesR2));
    CUDA_CHECK(cudaFree(sphParticlesR5));
    CUDA_CHECK(cudaFree(sphParticlesR6));
    CUDA_CHECK(cudaFree(sphParticlesR1));
    if (stars) {
        //		CUDA_CHECK(cudaFree(sph_particles_fZ));
        //			CUDA_CHECK(cudaFree(sph_particles_fY));
        //			CUDA_CHECK(cudaFree(sph_particles_sn));
        //			CUDA_CHECK(cudaFree(sph_particles_dz));
    }
}

void sphParticlesResolveWithParticles() {
    int const numberOfThreads = hpxHardwareConcurrency();
    std::vector<hpx::future<void>> futs;
    for (int proc = 0; proc < numberOfThreads; proc++) {
        futs.push_back(hpx::async([proc, numberOfThreads] {
            ParticleInteger const b = (size_t) proc * sphParticlesSize() / numberOfThreads;
            ParticleInteger const e = (size_t) (proc + 1) * sphParticlesSize() / numberOfThreads;
            for (ParticleInteger i = b; i < e; i++) {
                int const j = sphParticlesDmIndex(i);
                particlesCatIndex(j) = i;
            }
        }));
    }
    hpx::wait_all(futs.begin(), futs.end());
}

void sphParticlesGlobalReadPos(
    ParticleGlobalRange range, Fixed32* x, Fixed32* y, Fixed32* z, ParticleInteger offset) {
    ParticleInteger const lineSize = getOptions().particleCacheLineSize;
    if (range.range.first != range.range.second) {
        if (range.proc == hpxRank()) {
            ParticleInteger const dif = offset - range.range.first;
            ParticleInteger const sz = range.range.second - range.range.first;
            for (ParticleInteger i = range.range.first; i < range.range.second; i++) {
                int const j = offset + i - range.range.first;
                x[j] = sphParticlesPos(XDIM, i);
                y[j] = sphParticlesPos(YDIM, i);
                z[j] = sphParticlesPos(ZDIM, i);
            }
        } else {
            LineIdType lineId;
            lineId.proc = range.proc;
            ParticleInteger const startLine = (range.range.first / lineSize) * lineSize;
            ParticleInteger const stopLine = ((range.range.second - 1) / lineSize) * lineSize;
            ParticleInteger destIndex = offset;
            for (ParticleInteger line = startLine; line <= stopLine; line += lineSize) {
                lineId.index = line;
                auto const* ptr = sphParticlesCacheReadLine(lineId);
                auto const begin = std::max(lineId.index, range.range.first);
                auto const end = std::min(lineId.index + lineSize, range.range.second);
                for (ParticleInteger i = begin; i < end; i++) {
                    ParticleInteger const srcIndex = i - lineId.index;
                    x[destIndex] = ptr[srcIndex][XDIM];
                    y[destIndex] = ptr[srcIndex][YDIM];
                    z[destIndex] = ptr[srcIndex][ZDIM];
                    destIndex++;
                }
            }
        }
    }
}

static Array<Fixed32, NDIM> const* sphParticlesCacheReadLine(LineIdType lineId) {
    return readSphCacheLine<SphParticlesFetchCacheLineAction>(lineId, particleCache, mutexes);
}

static Vector<Array<Fixed32, NDIM>> sphParticlesFetchCacheLine(ParticleInteger index) {
    ParticleInteger const lineSize = getOptions().particleCacheLineSize;
    Vector<Array<Fixed32, NDIM>> line(lineSize);
    ParticleInteger const begin = (index / lineSize) * lineSize;
    ParticleInteger const end = std::min(sphParticlesSize(), begin + lineSize);
    for (ParticleInteger i = begin; i < end; i++) {
        for (int dim = 0; dim < NDIM; dim++) {
            line[i - begin][dim] = sphParticlesPos(dim, i);
        }
    }
    return line;
}

void sphParticlesGlobalReadFcold(
    ParticleGlobalRange range, float* cfrac, char* isstar, ParticleInteger offset) {
    ParticleInteger const lineSize = getOptions().particleCacheLineSize;
    int const sz = offset + range.range.second - range.range.first;
    if (range.range.first != range.range.second) {
        if (range.proc == hpxRank()) {
            ParticleInteger const dif = offset - range.range.first;
            ParticleInteger const sz = range.range.second - range.range.first;
            for (ParticleInteger i = range.range.first; i < range.range.second; i++) {
                int const j = offset + i - range.range.first;
                if (cfrac) {
                    cfrac[j] = sphParticlesColdMass(i);
                }
                if (isstar) {
                    isstar[j] = sphParticlesIsstar(i);
                }
            }
        } else {
            LineIdType lineId;
            lineId.proc = range.proc;
            ParticleInteger const startLine = (range.range.first / lineSize) * lineSize;
            ParticleInteger const stopLine = ((range.range.second - 1) / lineSize) * lineSize;
            ParticleInteger destIndex = offset;
            for (ParticleInteger line = startLine; line <= stopLine; line += lineSize) {
                lineId.index = line;
                auto const* ptr = sphParticlesFcoldCacheReadLine(lineId);
                auto const begin = std::max(lineId.index, range.range.first);
                auto const end = std::min(lineId.index + lineSize, range.range.second);
                for (ParticleInteger i = begin; i < end; i++) {
                    ParticleInteger const srcIndex = i - lineId.index;
                    Pair<float, char> const& part = ptr[srcIndex];
                    if (cfrac) {
                        cfrac[destIndex] = part.first;
                    }
                    if (isstar) {
                        isstar[destIndex] = part.second;
                    }
                    destIndex++;
                }
            }
        }
    }
}

static Pair<float, char> const* sphParticlesFcoldCacheReadLine(LineIdType lineId) {
    return readSphCacheLine<SphParticlesFetchFcoldCacheLineAction>(
        lineId, coldFractionParticleCache, fcoldMutexes);
}

static Vector<Pair<float, char>> sphParticlesFetchFcoldCacheLine(ParticleInteger index) {
    ParticleInteger const lineSize = getOptions().particleCacheLineSize;
    Vector<Pair<float, char>> line(lineSize);
    ParticleInteger const begin = (index / lineSize) * lineSize;
    ParticleInteger const end = std::min(sphParticlesSize(), begin + lineSize);
    for (ParticleInteger i = begin; i < end; i++) {
        line[i - begin].first = sphParticlesColdMass(i);
        line[i - begin].second = sphParticlesIsstar(i);
    }
    return line;
}

void sphParticlesGlobalReadEnergyAndSmoothlen(
    ParticleGlobalRange range, float* ent, float* h, ParticleInteger offset) {
    ParticleInteger const lineSize = getOptions().particleCacheLineSize;
    int const sz = offset + range.range.second - range.range.first;
    if (range.range.first != range.range.second) {
        if (range.proc == hpxRank()) {
            ParticleInteger const dif = offset - range.range.first;
            ParticleInteger const sz = range.range.second - range.range.first;
            for (ParticleInteger i = range.range.first; i < range.range.second; i++) {
                int const j = offset + i - range.range.first;
                if (ent) {
#ifdef ENTROPY
                    ent[j] = sphParticlesEntr(i);
#else
                    ent[j] = sphParticlesEint(i);
#endif
                }
                if (h) {
                    h[j] = sphParticlesSmoothingLength(i);
                }
            }
        } else {
            LineIdType lineId;
            lineId.proc = range.proc;
            ParticleInteger const startLine = (range.range.first / lineSize) * lineSize;
            ParticleInteger const stopLine = ((range.range.second - 1) / lineSize) * lineSize;
            ParticleInteger destIndex = offset;
            for (ParticleInteger line = startLine; line <= stopLine; line += lineSize) {
                lineId.index = line;
                auto const* ptr = sphParticlesEnergyAndSmoothlenCacheReadLine(lineId);
                auto const begin = std::max(lineId.index, range.range.first);
                auto const end = std::min(lineId.index + lineSize, range.range.second);
                for (ParticleInteger i = begin; i < end; i++) {
                    ParticleInteger const srcIndex = i - lineId.index;
                    if (ent) {
                        ent[destIndex] = ptr[srcIndex].first;
                    }
                    if (h) {
                        h[destIndex] = ptr[srcIndex].second;
                    }
                    destIndex++;
                }
            }
        }
    }
}

static Pair<float> const* sphParticlesEnergyAndSmoothlenCacheReadLine(LineIdType lineId) {
    return readSphCacheLine<SphParticlesFetchEnergyAndSmoothlenCacheLineAction>(
        lineId, energyAndSmoothingLengthParticleCache, entrMutexes);
}

static Vector<Pair<float>> sphParticlesFetchEnergyAndSmoothlenCacheLine(ParticleInteger index) {
    ParticleInteger const lineSize = getOptions().particleCacheLineSize;
    Vector<Pair<float>> line(lineSize);
    ParticleInteger const begin = (index / lineSize) * lineSize;
    ParticleInteger const end = std::min(sphParticlesSize(), begin + lineSize);
    for (ParticleInteger i = begin; i < end; i++) {
#ifdef ENTROPY
        line[i - begin].first = sphParticlesEntr(i);
#else
        line[i - begin].first = sphParticlesEint(i);
#endif
        line[i - begin].second = sphParticlesSmoothingLength(i);
    }
    return line;
}

void sphParticlesGlobalReadRungs(ParticleGlobalRange range, char* rungs, ParticleInteger offset) {
    ParticleInteger const lineSize = getOptions().particleCacheLineSize;
    if (range.range.first != range.range.second) {
        if (range.proc == hpxRank()) {
            ParticleInteger const dif = offset - range.range.first;
            ParticleInteger const sz = range.range.second - range.range.first;
            for (ParticleInteger i = range.range.first; i < range.range.second; i++) {
                int const j = offset + i - range.range.first;
                rungs[j] = sphParticlesRung(i);
            }
        } else {
            LineIdType lineId;
            lineId.proc = range.proc;
            ParticleInteger const startLine = (range.range.first / lineSize) * lineSize;
            ParticleInteger const stopLine = ((range.range.second - 1) / lineSize) * lineSize;
            ParticleInteger destIndex = offset;
            for (ParticleInteger line = startLine; line <= stopLine; line += lineSize) {
                lineId.index = line;
                auto const* ptr = sphParticlesRungCacheReadLine(lineId);
                auto const begin = std::max(lineId.index, range.range.first);
                auto const end = std::min(lineId.index + lineSize, range.range.second);
                for (ParticleInteger i = begin; i < end; i++) {
                    ParticleInteger const srcIndex = i - lineId.index;
                    rungs[destIndex] = ptr[srcIndex];
                    destIndex++;
                }
            }
        }
    }
}

static char const* sphParticlesRungCacheReadLine(LineIdType lineId) {
    return readSphCacheLine<SphParticlesFetchRungCacheLineAction>(
        lineId, rungParticleCache, rungMutexes);
}

static Vector<char> sphParticlesFetchRungCacheLine(ParticleInteger index) {
    ParticleInteger const lineSize = getOptions().particleCacheLineSize;
    Vector<char> line(lineSize);
    ParticleInteger const begin = (index / lineSize) * lineSize;
    ParticleInteger const end = std::min(sphParticlesSize(), begin + lineSize);
    for (ParticleInteger i = begin; i < end; i++) {
        line[i - begin] = sphParticlesRung(i);
    }
    return line;
}

void sphParticlesGlobalReadVels(
    ParticleGlobalRange range, float* vx, float* vy, float* vz, ParticleInteger offset) {
    ParticleInteger const lineSize = getOptions().particleCacheLineSize;
    if (range.range.first != range.range.second) {
        if (range.proc == hpxRank()) {
            ParticleInteger const dif = offset - range.range.first;
            ParticleInteger const sz = range.range.second - range.range.first;
            for (ParticleInteger i = range.range.first; i < range.range.second; i++) {
                int const j = offset + i - range.range.first;
                int const k = sphParticlesDmIndex(i);
                vx[j] = particlesVel(XDIM, k);
                vy[j] = particlesVel(YDIM, k);
                vz[j] = particlesVel(ZDIM, k);
            }
        } else {
            LineIdType lineId;
            lineId.proc = range.proc;
            ParticleInteger const startLine = (range.range.first / lineSize) * lineSize;
            ParticleInteger const stopLine = ((range.range.second - 1) / lineSize) * lineSize;
            ParticleInteger destIndex = offset;
            for (ParticleInteger line = startLine; line <= stopLine; line += lineSize) {
                lineId.index = line;
                auto const* ptr = sphParticlesVelsCacheReadLine(lineId);
                auto const begin = std::max(lineId.index, range.range.first);
                auto const end = std::min(lineId.index + lineSize, range.range.second);
                for (ParticleInteger i = begin; i < end; i++) {
                    ParticleInteger const srcIndex = i - lineId.index;
                    vx[destIndex] = ptr[srcIndex][XDIM];
                    vy[destIndex] = ptr[srcIndex][YDIM];
                    vz[destIndex] = ptr[srcIndex][ZDIM];
                    destIndex++;
                }
            }
        }
    }
}

static Array<float, NDIM> const* sphParticlesVelsCacheReadLine(LineIdType lineId) {
    return readSphCacheLine<SphParticlesFetchVelsCacheLineAction>(
        lineId, particleVelocityCache, velsMutexes);
}

static Vector<Array<float, NDIM>> sphParticlesFetchVelsCacheLine(ParticleInteger index) {
    ParticleInteger const lineSize = getOptions().particleCacheLineSize;
    Vector<Array<float, NDIM>> line(lineSize);
    ParticleInteger const begin = (index / lineSize) * lineSize;
    ParticleInteger const end = std::min(sphParticlesSize(), begin + lineSize);
    for (ParticleInteger i = begin; i < end; i++) {
        int const k = sphParticlesDmIndex(i);
        line[i - begin][XDIM] = particlesVel(k, XDIM);
        line[i - begin][YDIM] = particlesVel(k, YDIM);
        line[i - begin][ZDIM] = particlesVel(k, ZDIM);
    }
    return line;
}

void sphParticlesGlobalReadKappas(
    ParticleGlobalRange range, float* kappas, ParticleInteger offset) {
    ParticleInteger const lineSize = getOptions().particleCacheLineSize;
    if (range.range.first != range.range.second) {
        if (range.proc == hpxRank()) {
            ParticleInteger const dif = offset - range.range.first;
            ParticleInteger const sz = range.range.second - range.range.first;
            for (ParticleInteger i = range.range.first; i < range.range.second; i++) {
                int const j = offset + i - range.range.first;
                kappas[j] = sphParticlesKappa(i);
            }
        } else {
            LineIdType lineId;
            lineId.proc = range.proc;
            ParticleInteger const startLine = (range.range.first / lineSize) * lineSize;
            ParticleInteger const stopLine = ((range.range.second - 1) / lineSize) * lineSize;
            ParticleInteger destIndex = offset;
            for (ParticleInteger line = startLine; line <= stopLine; line += lineSize) {
                lineId.index = line;
                auto const* ptr = sphParticlesKappasCacheReadLine(lineId);
                auto const begin = std::max(lineId.index, range.range.first);
                auto const end = std::min(lineId.index + lineSize, range.range.second);
                for (ParticleInteger i = begin; i < end; i++) {
                    ParticleInteger const srcIndex = i - lineId.index;
                    kappas[destIndex] = ptr[srcIndex];
                    destIndex++;
                }
            }
        }
    }
}

static float const* sphParticlesKappasCacheReadLine(LineIdType lineId) {
    return readSphCacheLine<SphParticlesFetchKappasCacheLineAction>(
        lineId, conductivityParticleCache, kappasMutexes);
}

static Vector<float> sphParticlesFetchKappasCacheLine(ParticleInteger index) {
    ParticleInteger const lineSize = getOptions().particleCacheLineSize;
    Vector<float> line(lineSize);
    ParticleInteger const begin = (index / lineSize) * lineSize;
    ParticleInteger const end = std::min(sphParticlesSize(), begin + lineSize);
    for (ParticleInteger i = begin; i < end; i++) {
        line[i - begin] = sphParticlesKappa(i);
    }
    return line;
}

void sphParticlesGlobalReadRho(ParticleGlobalRange range, float* rho, ParticleInteger offset) {
    ParticleInteger const lineSize = getOptions().particleCacheLineSize;
    if (range.range.first != range.range.second) {
        if (range.proc == hpxRank()) {
            ParticleInteger const dif = offset - range.range.first;
            ParticleInteger const sz = range.range.second - range.range.first;
            for (ParticleInteger i = range.range.first; i < range.range.second; i++) {
                int const j = offset + i - range.range.first;
#ifdef HOPKINS
                rho[j] = sphParticlesRhoRho(i);
#else
                rho[j] = sphParticlesRho(i);
#endif
            }
        } else {
            LineIdType lineId;
            lineId.proc = range.proc;
            ParticleInteger const startLine = (range.range.first / lineSize) * lineSize;
            ParticleInteger const stopLine = ((range.range.second - 1) / lineSize) * lineSize;
            ParticleInteger destIndex = offset;
            for (ParticleInteger line = startLine; line <= stopLine; line += lineSize) {
                lineId.index = line;
                auto const* ptr = sphParticlesRhoCacheReadLine(lineId);
                auto const begin = std::max(lineId.index, range.range.first);
                auto const end = std::min(lineId.index + lineSize, range.range.second);
                for (ParticleInteger i = begin; i < end; i++) {
                    ParticleInteger const srcIndex = i - lineId.index;
                    rho[destIndex] = ptr[srcIndex];
                    destIndex++;
                }
            }
        }
    }
}

static float const* sphParticlesRhoCacheReadLine(LineIdType lineId) {
    return readSphCacheLine<SphParticlesFetchRhoCacheLineAction>(
        lineId, densityParticleCache, rhoMutexes);
}

static Vector<float> sphParticlesFetchRhoCacheLine(ParticleInteger index) {
    ParticleInteger const lineSize = getOptions().particleCacheLineSize;
    Vector<float> line(lineSize);
    ParticleInteger const begin = (index / lineSize) * lineSize;
    ParticleInteger const end = std::min(sphParticlesSize(), begin + lineSize);
    for (ParticleInteger i = begin; i < end; i++) {
#ifdef HOPKINS
        line[i - begin] = sphParticlesRhoRho(i);
#else
        line[i - begin] = sphParticlesRho(i);
#endif
    }
    return line;
}

void sphParticlesGlobalReadAux(ParticleGlobalRange range, float* alpha, float* omega, float* omegaP,
    float* pre, float* shearv, Array<float, NCHEMFRACS>* fracs, ParticleInteger offset) {
    ParticleInteger const lineSize = getOptions().particleCacheLineSize;
    if (range.range.first == range.range.second) {
        return;
    }
    if (range.proc == hpxRank()) {
        ParticleInteger const dif = offset - range.range.first;
        ParticleInteger const sz = range.range.second - range.range.first;
        for (ParticleInteger i = range.range.first; i < range.range.second; i++) {
            int const j = offset + i - range.range.first;
            if (alpha) {
                alpha[j] = sphParticlesAlpha(i);
            }
            if (omega) {
                omega[j] = sphParticlesOmega(i);
            }
            if (omegaP) {
                omegaP[j] = sphParticlesOmegaP(i);
            }
            if (pre) {
                pre[j] = sphParticlesPressure(i);
            }
            if (shearv) {
                shearv[j] = sphParticlesShear(i);
            }
            if (fracs) {
                fracs[j] = sphParticlesChem(i);
            }
        }
    } else {
        LineIdType lineId;
        lineId.proc = range.proc;
        ParticleInteger const startLine = (range.range.first / lineSize) * lineSize;
        ParticleInteger const stopLine = ((range.range.second - 1) / lineSize) * lineSize;
        ParticleInteger destIndex = offset;
        for (ParticleInteger line = startLine; line <= stopLine; line += lineSize) {
            lineId.index = line;
            auto const* ptr = sphParticlesAuxCacheReadLine(lineId);
            auto const begin = std::max(lineId.index, range.range.first);
            auto const end = std::min(lineId.index + lineSize, range.range.second);
            for (ParticleInteger i = begin; i < end; i++) {
                ParticleInteger const srcIndex = i - lineId.index;
                int const j = destIndex;
                if (alpha) {
                    alpha[j] = ptr[srcIndex].alpha;
                }
                if (omega) {
                    omega[j] = ptr[srcIndex].omega;
                }
                if (omegaP) {
                    omegaP[j] = ptr[srcIndex].omegaP;
                }
                if (pre) {
                    pre[j] = ptr[srcIndex].pre;
                }
                if (shearv) {
                    shearv[j] = ptr[srcIndex].shearv;
                }
                if (fracs) {
                    fracs[j] = ptr[srcIndex].fracs;
                }
                destIndex++;
            }
        }
    }
}

static AuxiliaryQuantities const* sphParticlesAuxCacheReadLine(LineIdType lineId) {
    return readSphCacheLine<SphParticlesFetchAuxCacheLineAction>(lineId, auxCache, auxMutexes);
}

static Vector<AuxiliaryQuantities> sphParticlesFetchAuxCacheLine(ParticleInteger index) {
    ParticleInteger const lineSize = getOptions().particleCacheLineSize;
    Vector<AuxiliaryQuantities> line(lineSize);
    ParticleInteger const begin = (index / lineSize) * lineSize;
    ParticleInteger const end = std::min(sphParticlesSize(), begin + lineSize);
    for (ParticleInteger i = begin; i < end; i++) {
        line[i - begin] = sphParticlesAuxQuantities(i);
    }
    return line;
}

void sphParticlesCacheFree1() {
    profilerEnter(__FUNCTION__);
    static auto const stars = getOptions().stars;
    Vector<hpx::future<void>> futs;
    for (auto const& c : hpxChildren()) {
        futs.push_back(hpx::async<SphParticlesCacheFree1Action>(c));
    }
    int const numberOfThreads = hpxHardwareConcurrency();
    energyAndSmoothingLengthParticleCache = decltype(energyAndSmoothingLengthParticleCache)();
    particleVelocityCache = decltype(particleVelocityCache)();
    auxCache = decltype(auxCache)();
    hpx::wait_all(futs.begin(), futs.end());
    profilerExit();
}

#ifdef ENTROPY
void sphParticlesCacheFreeEntr() {
    profilerEnter(__FUNCTION__);
    static auto const stars = getOptions().stars;
    Vector<hpx::future<void>> futs;
    for (auto const& c : hpxChildren()) {
        futs.push_back(hpx::async<SphParticlesCacheFreeEntrAction>(c));
    }
    int const numberOfThreads = hpxHardwareConcurrency();
    energyAndSmoothingLengthParticleCache = decltype(energyAndSmoothingLengthParticleCache)();
    hpx::wait_all(futs.begin(), futs.end());
    profilerExit();
}
#else
void sphParticlesCacheFreeEint() {
    profilerEnter(__FUNCTION__);
    static auto const stars = getOptions().stars;
    Vector<hpx::future<void>> futs;
    for (auto const& c : hpxChildren()) {
        futs.push_back(hpx::async<SphParticlesCacheFreeEintAction>(c));
    }
    int const numberOfThreads = hpxHardwareConcurrency();
    energyAndSmoothingLengthParticleCache = decltype(energyAndSmoothingLengthParticleCache)();
    hpx::wait_all(futs.begin(), futs.end());
    profilerExit();
}
#endif

void sphParticlesCacheFree2() {
    profilerEnter(__FUNCTION__);
    static auto const stars = getOptions().stars;
    Vector<hpx::future<void>> futs;
    for (auto const& c : hpxChildren()) {
        futs.push_back(hpx::async<SphParticlesCacheFree2Action>(c));
    }
    int const numberOfThreads = hpxHardwareConcurrency();
    particleCache = decltype(particleCache)();
    energyAndSmoothingLengthParticleCache = decltype(energyAndSmoothingLengthParticleCache)();
    rungParticleCache = decltype(rungParticleCache)();
    coldFractionParticleCache = decltype(coldFractionParticleCache)();
    particleVelocityCache = decltype(particleVelocityCache)();
    conductivityParticleCache = decltype(conductivityParticleCache)();
    densityParticleCache = decltype(densityParticleCache)();
    auxCache = decltype(auxCache)();
    hpx::wait_all(futs.begin(), futs.end());
    profilerExit();
}

void sphParticlesLoad(FILE* fp) {
    uint32_t header[7];
    ParticleInteger savedCount;
    FREAD(header, sizeof(uint32_t), 7, fp);
    FREAD(&savedCount, sizeof(ParticleInteger), 1, fp);
    if (header[0] != 0x43545350U || header[1] != 1 || header[2] != sizeof(ParticleInteger) ||
        header[3] != sizeof(SphRecord1) || header[4] != sizeof(SphRecord2) || header[5] != NDIM ||
        header[6] != detail::sphCheckpointMode() || savedCount != sphParticlesSize()) {
        THROW_ERROR("Incompatible SPH checkpoint payload\n");
    }
    detail::visitSphCheckpointArrays([fp](auto* values, size_t count) {
        if (count != 0) {
            FREAD(values, sizeof(*values), count, fp);
        }
    });
}

void sphParticlesSave(FILE* fp) {
    uint32_t header[7] = {0x43545350U, 1, sizeof(ParticleInteger), sizeof(SphRecord1),
        sizeof(SphRecord2), NDIM, detail::sphCheckpointMode()};
    ParticleInteger count = sphParticlesSize();
    FWRITE(header, sizeof(uint32_t), 7, fp);
    FWRITE(&count, sizeof(ParticleInteger), 1, fp);
    detail::visitSphCheckpointArrays([fp](auto* values, size_t count) {
        if (count != 0) {
            FWRITE(values, sizeof(*values), count, fp);
        }
    });
}

void sphCheckpointTest() {
    Vector<unsigned char> expected;
    FILE* file = std::tmpfile();
    if (file == nullptr) {
        THROW_ERROR("Unable to create SPH checkpoint test file\n");
    }
    sphParticlesSave(file);
    detail::visitSphCheckpointArrays([&expected](auto* values, size_t count) {
        size_t const bytes = sizeof(*values) * count;
        if (bytes != 0) {
            auto* data = reinterpret_cast<unsigned char*>(values);
            expected.insert(expected.end(), data, data + bytes);
            std::memset(data, 0x5a, bytes);
        }
    });
    std::rewind(file);
    sphParticlesLoad(file);
    size_t offset = 0;
    detail::visitSphCheckpointArrays([&expected, &offset](auto* values, size_t count) {
        size_t const bytes = sizeof(*values) * count;
        if (bytes != 0) {
            if (std::memcmp(values, expected.data() + offset, bytes) != 0) {
                THROW_ERROR("SPH checkpoint did not restore every gas field\n");
            }
            offset += bytes;
        }
    });
    if (sphParticlesSize() >= 2) {
        detail::visitSphCheckpointArrays([](auto* values, size_t) {
            std::memset(values, 0x11, sizeof(*values));
            std::memset(values + 1, 0x22, sizeof(*values));
        });
        auto checkOrder = [](unsigned char first, unsigned char second) {
            detail::visitSphCheckpointArrays([first, second](auto* values, size_t) {
                auto const* bytes = reinterpret_cast<unsigned char const*>(values);
                for (size_t i = 0; i < sizeof(*values); i++) {
                    if (bytes[i] != first || bytes[sizeof(*values) + i] != second) {
                        THROW_ERROR("SPH reorder did not preserve every gas field\n");
                    }
                }
            });
        };
        sphParticlesSwap(0, 1);
        checkOrder(0x22, 0x11);
        sphParticlesSwap2(0, 1);
        checkOrder(0x11, 0x22);
        std::rewind(file);
        sphParticlesLoad(file);
        PRINT("SPH reorder passed (all fields, gravity and predictor stages)\n");
    }
    ParticleInteger const savedSize = size;
    size = 0;
    std::rewind(file);
    sphParticlesSave(file);
    std::rewind(file);
    sphParticlesLoad(file);
    size = savedSize;
    std::fclose(file);
    PRINT("SPH checkpoint roundtrip passed (all fields and empty gas payload)\n");
}

HPX_PLAIN_ACTION(sphParticlesResetConverged, SphParticlesResetConvergedAction);

void sphParticlesResetConverged() {
    int const numberOfThreads = hpxHardwareConcurrency();
    std::vector<hpx::future<void>> futs;
    for (auto& c : hpxChildren()) {
        futs.push_back(hpx::async<SphParticlesResetConvergedAction>(c));
    }
    for (int proc = 0; proc < numberOfThreads; proc++) {
        futs.push_back(hpx::async([proc, numberOfThreads] {
            ParticleInteger const b = (size_t) proc * sphParticlesSize() / numberOfThreads;
            ParticleInteger const e = (size_t) (proc + 1) * sphParticlesSize() / numberOfThreads;
            for (ParticleInteger i = b; i < e; i++) {
                sphParticlesConverged(i) = false;
            }
        }));
    }
    hpx::wait_all(futs.begin(), futs.end());
}

HPX_PLAIN_ACTION(sphParticlesResetSemiactive, SphParticlesResetSemiactiveAction);

void sphParticlesResetSemiactive() {
    int const numberOfThreads = hpxHardwareConcurrency();
    std::vector<hpx::future<void>> futs;
    for (auto& c : hpxChildren()) {
        futs.push_back(hpx::async<SphParticlesResetSemiactiveAction>(c));
    }
    for (int proc = 0; proc < numberOfThreads; proc++) {
        futs.push_back(hpx::async([proc, numberOfThreads] {
            ParticleInteger const b = (size_t) proc * sphParticlesSize() / numberOfThreads;
            ParticleInteger const e = (size_t) (proc + 1) * sphParticlesSize() / numberOfThreads;
            for (ParticleInteger i = b; i < e; i++) {
                sphParticlesSemiactive(i) = false;
            }
        }));
    }
    hpx::wait_all(futs.begin(), futs.end());
}
