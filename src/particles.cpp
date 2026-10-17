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

constexpr bool kVerbose = true;
#define PARTICLES_CPP

#include <cstring>
#include <exception>
#include <memory>
#include <type_traits>
#include <unordered_map>

#include <gsl/gsl_rng.h>

#include <cosmictiger/flops.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/kernel.hpp>
#include <cosmictiger/particle_selection.hpp>
#include <cosmictiger/particles.hpp>
#include <cosmictiger/safe_io.hpp>
#include <cosmictiger/sph_particles.hpp>
#include <cosmictiger/stars.hpp>

struct LineIdType;

static Vector<GroupInteger> particlesGroupRefreshCacheLine(ParticleInteger index);

static Vector<GroupParticle> particlesGroupFetchCacheLine(ParticleInteger index);
static GroupParticle const* particlesGroupCacheReadLine(LineIdType lineId);
void particlesGroupCacheFree();

struct ParticlesCacheEntry {
    Array<Fixed32, NDIM> x;
    char type;
    float zeta1;
    float zeta2;
    template <class A>
    void serialize(A&& arc, unsigned) {
        arc & type;
        arc & x;
        arc & zeta1;
        arc & zeta2;
    }
};

static Vector<ParticlesCacheEntry> particlesFetchCacheLine(ParticleInteger index);
static ParticlesCacheEntry const* particlesCacheReadLine(LineIdType lineId);
void particlesCacheFree();

static Vector<Array<float, NDIM>> particlesFetchCacheLineVels(ParticleInteger index);
static Array<float, NDIM> const* particlesCacheReadLineVels(LineIdType lineId);

static Vector<float> particlesFetchCacheLineSoftlens(ParticleInteger index);
static Vector<char> particlesFetchCacheLineRungs(ParticleInteger index);
static float const* particlesCacheReadLineSoftlens(LineIdType lineId);
static char const* particlesCacheReadLineRungs(LineIdType lineId);
void particlesCacheFreeSoftlens();

static void particlesSetGlobalOffset(Vector<size_t>);

static SharedMutexType sharedMutex;

static ParticleInteger size = 0;
static ParticleInteger capacity = 0;
static Vector<size_t> globalOffsets;
static ParticleInteger rungBegin;
static std::vector<ParticleInteger> rungBegins;

HPX_PLAIN_ACTION(particlesCacheFree, ParticlesCacheFreeAction);
HPX_PLAIN_ACTION(particlesIncGroupCacheEpoch, ParticlesIncGroupCacheEpochAction);
HPX_PLAIN_ACTION(particlesDestroy, ParticlesDestroyAction);
HPX_PLAIN_ACTION(particlesFetchCacheLineSoftlens, ParticlesFetchCacheLineSoftlensAction);
HPX_PLAIN_ACTION(particlesFetchCacheLineVels, ParticlesFetchCacheLineVelsAction);
HPX_PLAIN_ACTION(particlesFetchCacheLineRungs, ParticlesFetchCacheLineRungsAction);
HPX_PLAIN_ACTION(particlesFetchCacheLine, ParticlesFetchCacheLineAction);
HPX_PLAIN_ACTION(particlesGroupRefreshCacheLine, ParticlesGroupRefreshCacheLineAction);
HPX_PLAIN_ACTION(particlesGroupFetchCacheLine, ParticlesGroupFetchCacheLineAction);
HPX_PLAIN_ACTION(particlesRandomInit, ParticlesRandomInitAction);
HPX_PLAIN_ACTION(particlesSample, ParticlesSampleAction);
HPX_PLAIN_ACTION(particlesGroupsInit, ParticlesGroupsInitAction);
HPX_PLAIN_ACTION(particlesGroupsDestroy, ParticlesGroupsDestroyAction);
HPX_PLAIN_ACTION(particlesSetGlobalOffset, ParticlesSetGlobalOffsetAction);
HPX_PLAIN_ACTION(particlesSetTracers, ParticlesSetTracersAction);
HPX_PLAIN_ACTION(particlesGetTracers, ParticlesGetTracersAction);
HPX_PLAIN_ACTION(particlesGetSample, ParticlesGetSampleAction);
/*
 struct particle_ref {
 int index;
 bool operator<(const particle_ref& other) const {
 return particles_sph_index(index) < particles_sph_index(other.index);
 }
 bool operator<(const particle& other) const {
 return particles_sph_index(index) < other.sph_index;
 }
 operator particle() const {
 return particles_get_particle(index);
 }
 particle_ref operator=(const particle& other) {
 particles_set_particle(other, index);
 return *this;
 }
 };

 bool operator<(const particle& a, const particle& b) {
 return a.sph_index < b.sph_index;
 }

 void swap(particle_ref a, particle_ref b) {
 particle c = (particle) a;
 a = (particle) b;
 b = c;
 }

 struct particle_iterator {
 using iterator_category = std::random_access_iterator_tag;
 using difference_type = int;
 using value_type = particle;
 using pointer = int;  // or also value_type*
 using reference = particle_ref&;  // or also value_type&
 int index;
 particle_ref operator*() const {
 particle_ref ref;
 ref.index = index;
 return ref;
 }
 int operator-(const particle_iterator& other) const {
 return index - other.index;
 }
 particle_iterator operator+(int i) const {
 particle_iterator j;
 j.index = index + i;
 return j;
 }
 particle_iterator operator-(int i) const {
 particle_iterator j;
 j.index = index - i;
 return j;
 }
 particle_iterator& operator--() {
 index--;
 return *this;
 }
 particle_iterator& operator--(int) {
 index--;
 return *this;
 }
 particle_iterator& operator++() {
 index++;
 return *this;
 }
 particle_iterator& operator++(int) {
 index++;
 return *this;
 }
 bool operator!=(const particle_iterator& other) const {
 return index != other.index;
 }
 bool operator==(const particle_iterator& other) const {
 return index == other.index;
 }
 bool operator<(const particle_iterator& other) const {
 return index < other.index;
 }
 };
 */

HPX_PLAIN_ACTION(particlesPopRungs, ParticlesPopRungsAction);
HPX_PLAIN_ACTION(particlesPushRungs, ParticlesPushRungsAction);
HPX_PLAIN_ACTION(particlesActivePct, ParticlesActivePctAction);

double particlesActivePct() {
    Vector<hpx::future<double>> futs;
    for (auto& c : hpxChildren()) {
        futs.push_back(hpx::async<ParticlesActivePctAction>(c));
    }
    double numActive = particlesSize() - rungBegin;
    for (auto& f : futs) {
        numActive += f.get();
    }
    if (hpxRank() == 0) {
        numActive /= getOptions().numberOfParticles;
    }
    return numActive;
}

HPX_PLAIN_ACTION(particlesDisplace, ParticlesDisplaceAction);

void particlesDisplace(double dx, double dy, double dz) {
    Vector<hpx::future<void>> futs;
    for (auto& c : hpxChildren()) {
        futs.push_back(hpx::async<ParticlesDisplaceAction>(c, dx, dy, dz));
    }
    int const numberOfThreads = hpxHardwareConcurrency();
    for (int proc = 0; proc < numberOfThreads; proc++) {
        futs.push_back(hpx::async([numberOfThreads, proc, dx, dy, dz]() {
            ParticleInteger const begin = (size_t) proc * particlesSize() / numberOfThreads;
            ParticleInteger const end = (size_t) (proc + 1) * particlesSize() / numberOfThreads;
            for (ParticleInteger i = begin; i < end; i++) {
                double x = particlesPos(XDIM, i).toDouble();
                double y = particlesPos(YDIM, i).toDouble();
                double z = particlesPos(ZDIM, i).toDouble();
                x = fmod(x + dx, 1.0);
                y = fmod(y + dy, 1.0);
                z = fmod(z + dz, 1.0);
                particlesPos(XDIM, i) = x;
                particlesPos(YDIM, i) = y;
                particlesPos(ZDIM, i) = z;
            }
        }));
    }
    hpx::wait_all(futs.begin(), futs.end());
    for (auto& f : futs) {
        f.get();
    }
}

void particlesPopRungs() {
    Vector<hpx::future<void>> futs;
    for (auto& c : hpxChildren()) {
        futs.push_back(hpx::async<ParticlesPopRungsAction>(c));
    }
    rungBegin = rungBegins.back();
    rungBegins.pop_back();
    hpx::wait_all(futs.begin(), futs.end());
}

void particlesPushRungs() {
    Vector<hpx::future<void>> futs;
    for (auto& c : hpxChildren()) {
        futs.push_back(hpx::async<ParticlesPushRungsAction>(c));
    }
    rungBegins.push_back(rungBegin);
    hpx::wait_all(futs.begin(), futs.end());
}

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

static Array<std::unordered_map<LineIdType,
                 std::shared_ptr<hpx::shared_future<Vector<ParticlesCacheEntry>>>, LineIdHashHi>,
    PART_CACHE_SIZE>
    particleCache;
static Array<SpinlockType, PART_CACHE_SIZE> mutexes;
static int groupCacheEpoch = 0;

static Array<
    std::unordered_map<LineIdType, std::shared_ptr<hpx::shared_future<Vector<char>>>, LineIdHashHi>,
    PART_CACHE_SIZE>
    particleRungCache;
static Array<SpinlockType, PART_CACHE_SIZE> mutexesRungs;

static Array<std::unordered_map<LineIdType, std::shared_ptr<hpx::shared_future<Vector<float>>>,
                 LineIdHashHi>,
    PART_CACHE_SIZE>
    particleSofteningLengthCache;
static Array<SpinlockType, PART_CACHE_SIZE> mutexesSoftlens;

static Array<std::unordered_map<LineIdType,
                 std::shared_ptr<hpx::shared_future<Vector<Array<float, NDIM>>>>, LineIdHashHi>,
    PART_CACHE_SIZE>
    particleVelocityCache;
static Array<SpinlockType, PART_CACHE_SIZE> velsMutexes;

struct GroupCacheEntry {
    std::shared_ptr<hpx::shared_future<Vector<GroupParticle>>> data;
    int epoch;
};

static Array<std::unordered_map<LineIdType, GroupCacheEntry, LineIdHashHi>, PART_CACHE_SIZE>
    groupParticleCache;
static Array<SpinlockType, PART_CACHE_SIZE> groupMutexes;

template <typename Action, typename T>
static T const* readParticleCacheLine(LineIdType lineId,
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
                hpx::post([promise, lineId]() {
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

/*void particles_sort_by_sph(pair<part_int> rng) {
 particle_iterator b;
 particle_iterator e;
 b.index = rng.first;
 e.index = rng.second;
 std::sort(b, e);
 }*/

Vector<OutputParticle> particlesGetSample(Range<double> const& box) {
    Vector<hpx::future<Vector<OutputParticle>>> futs;
    Vector<OutputParticle> output;
    for (auto const& c : hpxChildren()) {
        futs.push_back(hpx::async<ParticlesGetSampleAction>(c, box));
    }
    for (ParticleInteger i = 0; i < particlesSize(); i++) {
        Array<double, NDIM> x;
        for (int dim = 0; dim < NDIM; dim++) {
            x[dim] = particlesPos(dim, i).toDouble();
        }
        if (box.contains(x)) {
            OutputParticle data;
            for (int dim = 0; dim < NDIM; dim++) {
                data.x[dim] = particlesPos(dim, i);
                data.v[dim] = particlesVel(dim, i);
            }
            data.r = particlesRung(i);
            output.push_back(data);
        }
    }
    for (auto& f : futs) {
        auto vec = f.get();
        output.insert(output.end(), vec.begin(), vec.end());
    }
    return std::move(output);
}

HPX_PLAIN_ACTION(particlesApplyUpdates, ParticlesApplyUpdatesAction);

int particlesApplyUpdates(int minrung, float t0, float a) {
    profilerEnter(__FUNCTION__);
    Vector<hpx::future<int>> futs;
    for (auto& c : hpxChildren()) {
        futs.push_back(hpx::async<ParticlesApplyUpdatesAction>(c, minrung, t0, a));
    }
    int const numberOfThreads = hpxHardwareConcurrency();
    for (int proc = 0; proc < numberOfThreads; proc++) {
        futs.push_back(hpx::async([t0, numberOfThreads, proc, minrung, a]() {
            int maxRung = 0;
            float const cfl = getOptions().cfl;
            ParticleInteger const b = (size_t) proc * particlesSize() / numberOfThreads;
            ParticleInteger const e = (size_t) (proc + 1) * particlesSize() / numberOfThreads;
            for (int i = b; i < e; i++) {
                auto& rung = particlesRung(i);
                if (particlesRung(i) >= minrung && particlesType(i) == DARK_MATTER_TYPE) {
                    auto& vx = particlesVel(XDIM, i);
                    auto& vy = particlesVel(YDIM, i);
                    auto& vz = particlesVel(ZDIM, i);
                    auto& gx = particlesGravityAcceleration(XDIM, i);
                    auto& gy = particlesGravityAcceleration(YDIM, i);
                    auto& gz = particlesGravityAcceleration(ZDIM, i);
                    auto const& divv = particlesDivv(i);
                    auto const h = particlesSoftlen(i);
                    float const dtDivv = cfl *
                        (3.f +
                            dlogsmoothXDlogh(h, getOptions().hydroMinSupportRadius,
                                getOptions().hydroMaxSupportRadius)) *
                        a / (fabs(divv) + 1e-37f);
                    float dt = std::min(dtDivv, (float) (rungDt[rung] * t0));
                    auto newRung = (int) ceilf(log2f(t0 / dt));
                    rung = newRung;
                    maxRung = std::max(maxRung, (int) rung);
                    dt = 0.5f * rungDt[rung] * t0;
                    vx += dt * gx;
                    vy += dt * gy;
                    vz += dt * gz;
                }
            }
            return maxRung;
        }));
    }
    int maxRung = 0;
    for (auto& f : futs) {
        maxRung = std::max(maxRung, f.get());
    }
    profilerExit();
    return maxRung;
}

Vector<OutputParticle> particlesGetTracers() {
    Vector<hpx::future<Vector<OutputParticle>>> futs;
    Vector<OutputParticle> output;
    for (auto const& c : hpxChildren()) {
        futs.push_back(hpx::async<ParticlesGetTracersAction>(c));
    }
    for (ParticleInteger i = 0; i < particlesSize(); i++) {
        if (particlesTracer(i)) {
            OutputParticle data;
            for (int dim = 0; dim < NDIM; dim++) {
                data.x[dim] = particlesPos(dim, i);
                data.v[dim] = particlesVel(dim, i);
            }
            data.r = particlesRung(i);
            output.push_back(data);
        }
    }
    for (auto& f : futs) {
        auto vec = f.get();
        output.insert(output.end(), vec.begin(), vec.end());
    }
    return std::move(output);
}

void particlesSetTracers(size_t count) {
    hpx::future<void> fut;
    if (hpxRank() < hpxSize() - 1) {
        fut = hpx::async<ParticlesSetTracersAction>(
            hpxLocalities()[hpxRank() + 1], count + particlesSize());
    }

    if (particlesSize() > 0) {
        size_t const totalCount = getOptions().numberOfParticles * (getOptions().sph ? 2 : 1);
        detail::setTracerFlags(
            &particlesTracer(0), particlesSize(), count, totalCount, getOptions().tracerCount);
    }

    if (hpxRank() < hpxSize() - 1) {
        fut.get();
    }
}

std::unordered_map<int, ParticleInteger> particlesGroupsInit() {
    Vector<hpx::future<std::unordered_map<int, ParticleInteger>>> futs;
    for (auto const& c : hpxChildren()) {
        futs.push_back(hpx::async<ParticlesGroupsInitAction>(c));
    }

    ALWAYS_ASSERT(!particlesGrp);
    particlesGrp = new std::atomic<GroupInteger>[size];
    hpx_fill(PAR_EXECUTION_POLICY, particlesGrp, particlesGrp + particlesSize(), NO_GROUP).get();
    groupCacheEpoch = 0;

    std::unordered_map<int, ParticleInteger> map;
    map[hpxRank()] = particlesSize();
    for (auto& f : futs) {
        auto thisMap = f.get();
        for (auto i = thisMap.begin(); i != thisMap.end(); i++) {
            map[i->first] = i->second;
        }
    }

    if (hpxRank() == 0) {
        Vector<size_t> offsets(hpxSize());
        offsets[0] = 0;
        for (int i = 0; i < hpxSize() - 1; i++) {
            offsets[i + 1] = map[i] + offsets[i];
        }
        particlesSetGlobalOffset(std::move(offsets));
    }

    return map;
}

int particlesGroupHome(GroupInteger grp) {
    int begin = 0;
    int end = hpxSize();
    while (end - begin > 1) {
        int mid = (begin + end) / 2;
        if (grp >= globalOffsets[mid]) {
            begin = mid;
        } else {
            end = mid;
        }
    }
    return begin;
}

static void particlesSetGlobalOffset(Vector<size_t> map) {
    particlesGlobalOffset = map[hpxRank()];
    Vector<hpx::future<void>> futs;
    for (auto const& c : hpxChildren()) {
        futs.push_back(hpx::async<ParticlesSetGlobalOffsetAction>(c, map));
    }
    particlesGlobalOffset = map[hpxRank()];
    globalOffsets = std::move(map);
    hpx::wait_all(futs.begin(), futs.end());
}

void particlesGroupsDestroy() {
    Vector<hpx::future<void>> futs;
    for (auto const& c : hpxChildren()) {
        futs.push_back(hpx::async<ParticlesGroupsDestroyAction>(c));
    }
    int const numberOfThreads = hpxHardwareConcurrency();
    Vector<hpx::future<void>> futs2;
    for (int proc = 0; proc < numberOfThreads; proc++) {
        futs2.push_back(hpx::async([numberOfThreads, proc]() {
            ParticleInteger const begin = (size_t) proc * particlesSize() / numberOfThreads;
            ParticleInteger const end = (size_t) (proc + 1) * particlesSize() / numberOfThreads;
            for (ParticleInteger i = begin; i < end; i++) {
                particlesLastgroup(i) = particlesGroup(i);
            }
        }));
    }
    hpx::wait_all(futs2.begin(), futs2.end());
    delete[] (particlesGrp);
    particlesGrp = nullptr;
    groupParticleCache = decltype(groupParticleCache)();
    hpx::wait_all(futs.begin(), futs.end());
}

void particlesCacheFree() {
    for (size_t i = 0; i < PART_CACHE_SIZE; i++) {
        particleSofteningLengthCache[i].clear();
        particleRungCache[i].clear();
        particleVelocityCache[i].clear();
    }
    profilerEnter(__FUNCTION__);
    Vector<hpx::future<void>> futs;
    for (auto const& c : hpxChildren()) {
        futs.push_back(hpx::async<ParticlesCacheFreeAction>(c));
    }
    particleCache = decltype(particleCache)();
    hpx::wait_all(futs.begin(), futs.end());
    profilerExit();
}

static std::function<void()> undoLastMemadvise = nullptr;

void particlesMemadviseGpu() {
#ifdef USE_CUDA
    cudaSetDevice();
    if (undoLastMemadvise) {
        undoLastMemadvise();
    }
    int deviceid = cudaGetDevice();
    auto const rng = particlesCurrentRange();
    auto const begin = rng.first;
    auto const count = rng.second - begin;
    CUDA_CHECK(cudaMemAdvise(particlesVelData() + begin, count * sizeof(Array<float, NDIM>),
        cudaMemAdviseSetAccessedBy, deviceid));
    CUDA_CHECK(cudaMemAdvise(
        &particlesRung(0) + begin, count * sizeof(char), cudaMemAdviseSetAccessedBy, deviceid));
    undoLastMemadvise = [begin, count, deviceid]() {
        CUDA_CHECK(cudaMemAdvise(particlesVelData() + begin, count * sizeof(Array<float, NDIM>),
            cudaMemAdviseUnsetAccessedBy, deviceid));
        CUDA_CHECK(cudaMemAdvise(&particlesRung(0) + begin, count * sizeof(char),
            cudaMemAdviseUnsetAccessedBy, deviceid));
    };
#endif
}

void particlesMemadviseCpu() {
#ifdef USE_CUDA
    cudaSetDevice();
    int deviceid = cudaGetDevice();
    if (undoLastMemadvise) {
        undoLastMemadvise();
    }
    auto const rng = particlesCurrentRange();
    auto const begin = rng.first;
    auto const count = rng.second - begin;
    CUDA_CHECK(cudaMemAdvise(particlesVelData() + begin, count * sizeof(Array<float, NDIM>),
        cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
    CUDA_CHECK(cudaMemAdvise(&particlesRung(0) + begin, count * sizeof(char),
        cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
    undoLastMemadvise = [begin, count]() {
        CUDA_CHECK(cudaMemAdvise(particlesVelData() + begin, count * sizeof(Array<float, NDIM>),
            cudaMemAdviseUnsetPreferredLocation, cudaCpuDeviceId));
        CUDA_CHECK(cudaMemAdvise(&particlesRung(0) + begin, count * sizeof(char),
            cudaMemAdviseUnsetPreferredLocation, cudaCpuDeviceId));
    };
    for (int dim = 0; dim < NDIM; dim++) {
        CUDA_CHECK(cudaMemPrefetchAsync(
            &particlesPos(dim, begin), count * sizeof(Fixed32), cudaCpuDeviceId, 0));
    }
    CUDA_CHECK(cudaMemPrefetchAsync(
        &particlesVel(XDIM, begin), count * sizeof(Array<float, NDIM>), cudaCpuDeviceId, 0));
    CUDA_CHECK(
        cudaMemPrefetchAsync(&particlesRung(begin), count * sizeof(char), cudaCpuDeviceId, 0));
#endif
}

void particlesGlobalReadPosAndGroup(ParticleGlobalRange range, Fixed32* x, Fixed32* y, Fixed32* z,
    GroupInteger* g, ParticleInteger offset) {
    ParticleInteger const lineSize = getOptions().particleCacheLineSize;
    if (range.range.first != range.range.second) {
        if (range.proc == hpxRank()) {
            ParticleInteger const dif = offset - range.range.first;
            ParticleInteger const sz = range.range.second - range.range.first;
            std::memcpy(x + offset, &particlesPos(XDIM, range.range.first), sizeof(float) * sz);
            std::memcpy(y + offset, &particlesPos(YDIM, range.range.first), sizeof(float) * sz);
            std::memcpy(z + offset, &particlesPos(ZDIM, range.range.first), sizeof(float) * sz);
            for (int i = 0; i < sz; i++) {
                g[i + offset] = particlesGroup(range.range.first + i);
            }
        } else {
            LineIdType lineId;
            lineId.proc = range.proc;
            ParticleInteger const startLine = (range.range.first / lineSize) * lineSize;
            ParticleInteger const stopLine = ((range.range.second - 1) / lineSize) * lineSize;
            ParticleInteger destIndex = offset;
            for (ParticleInteger line = startLine; line <= stopLine; line += lineSize) {
                lineId.index = line;
                auto const* ptr = particlesGroupCacheReadLine(lineId);
                auto const begin = std::max(lineId.index, range.range.first);
                auto const end = std::min(lineId.index + lineSize, range.range.second);
                for (ParticleInteger i = begin; i < end; i++) {
                    ParticleInteger const srcIndex = i - lineId.index;
                    x[destIndex] = ptr[srcIndex].x[XDIM];
                    y[destIndex] = ptr[srcIndex].x[YDIM];
                    z[destIndex] = ptr[srcIndex].x[ZDIM];
                    g[destIndex] = ptr[srcIndex].g;
                    destIndex++;
                }
            }
        }
    }
}

static GroupParticle const* particlesGroupCacheReadLine(LineIdType lineId) {
    size_t const bin = LineIdHashLo()(lineId);
    int const epoch = groupCacheEpoch;
    std::shared_ptr<hpx::shared_future<Vector<GroupParticle>>> future;
    {
        std::lock_guard<SpinlockType> lock(groupMutexes[bin]);
        auto const entry = groupParticleCache[bin].find(lineId);
        if (entry != groupParticleCache[bin].end() && entry->second.epoch >= epoch) {
            future = entry->second.data;
        }
    }
    if (!future) {
        auto promise = std::make_shared<hpx::promise<Vector<GroupParticle>>>();
        auto candidate = std::make_shared<hpx::shared_future<Vector<GroupParticle>>>(
            promise->get_future().share());
        std::shared_ptr<hpx::shared_future<Vector<GroupParticle>>> old;
        bool inserted = false;
        {
            std::lock_guard<SpinlockType> lock(groupMutexes[bin]);
            auto& entry = groupParticleCache[bin][lineId];
            if (!entry.data || entry.epoch < epoch) {
                old = entry.data;
                entry.data = candidate;
                entry.epoch = epoch;
                inserted = true;
            }
            future = entry.data;
        }
        if (inserted) {
            try {
                hpx::post([promise, lineId, old]() {
                    try {
                        if (old) {
                            auto data = old->get();
                            auto groups = hpx::async<ParticlesGroupRefreshCacheLineAction>(
                                hpxLocalities()[lineId.proc], lineId.index)
                                              .get();
                            for (int i = 0; i < groups.size(); i++) {
                                data[i].g = groups[i];
                            }
                            promise->set_value(std::move(data));
                        } else {
                            promise->set_value(hpx::async<ParticlesGroupFetchCacheLineAction>(
                                hpxLocalities()[lineId.proc], lineId.index)
                                    .get());
                        }
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

HPX_PLAIN_ACTION(particlesRungCounts, ParticlesRungCountsAction);

Vector<size_t> particlesRungCounts() {
    profilerEnter(__FUNCTION__);
    Vector<hpx::future<Vector<size_t>>> futs;
    for (auto& c : hpxChildren()) {
        futs.push_back(hpx::async<ParticlesRungCountsAction>(c));
    }
    int const numberOfThreads = hpxHardwareConcurrency();
    for (int proc = 0; proc < numberOfThreads; proc++) {
        futs.push_back(hpx::async([proc, numberOfThreads]() {
            Vector<size_t> counts;
            ParticleInteger const b = (size_t) proc * particlesSize() / numberOfThreads;
            ParticleInteger const e = (size_t) (proc + 1) * particlesSize() / numberOfThreads;
            for (ParticleInteger i = b; i < e; i++) {
                auto const rung = particlesRung(i);
                if (counts.size() <= rung) {
                    counts.resize(rung + 1, 0);
                }
                counts[rung]++;
            }
            return counts;
        }));
    }
    Vector<size_t> counts;
    for (auto& f : futs) {
        auto theseCounts = f.get();
        if (counts.size() < theseCounts.size()) {
            counts.resize(theseCounts.size(), 0);
        }
        for (int i = 0; i < theseCounts.size(); i++) {
            counts[i] += theseCounts[i];
        }
    }
    if (hpxRank() == 0) {
        size_t tot = 0;
        for (int i = 0; i < counts.size(); i++) {
            tot += counts[i];
        }
        //	ALWAYS_ASSERT(tot == numberOfParticles);
    }
    profilerExit();
    return counts;
}

HPX_PLAIN_ACTION(particlesSetMinrung, ParticlesSetMinrungAction);

void particlesSetMinrung(int minrung) {
    profilerEnter(__FUNCTION__);
    Vector<hpx::future<void>> futs;
    for (auto& c : hpxChildren()) {
        futs.push_back(hpx::async<ParticlesSetMinrungAction>(c, minrung));
    }
    int const numberOfThreads = hpxHardwareConcurrency();
    for (int proc = 0; proc < numberOfThreads; proc++) {
        futs.push_back(hpx::async([proc, numberOfThreads, minrung]() {
            ParticleInteger const b = (size_t) proc * particlesSize() / numberOfThreads;
            ParticleInteger const e = (size_t) (proc + 1) * particlesSize() / numberOfThreads;
            for (ParticleInteger i = b; i < e; i++) {
                auto& rung = particlesRung(i);
                rung = std::max((int) rung, minrung);
            }
        }));
    }
    for (auto& f : futs) {
        f.get();
    }
    profilerExit();
}

void particlesGlobalReadSoftlens(ParticleGlobalRange range, float* h, ParticleInteger offset) {
    static bool const sph = getOptions().sph;
    ParticleInteger const lineSize = getOptions().particleCacheLineSize;
    if (range.range.first != range.range.second) {
        if (range.proc == hpxRank()) {
            ParticleInteger const dif = offset - range.range.first;
            ParticleInteger const sz = range.range.second - range.range.first;
            std::memcpy(h + offset, &particlesSoftlen(range.range.first), sizeof(float) * sz);
        } else {
            LineIdType lineId;
            lineId.proc = range.proc;
            ParticleInteger const startLine = (range.range.first / lineSize) * lineSize;
            ParticleInteger const stopLine = ((range.range.second - 1) / lineSize) * lineSize;
            ParticleInteger destIndex = offset;
            for (ParticleInteger line = startLine; line <= stopLine; line += lineSize) {
                lineId.index = line;
                auto const* ptr = particlesCacheReadLineSoftlens(lineId);
                auto const begin = std::max(lineId.index, range.range.first);
                auto const end = std::min(lineId.index + lineSize, range.range.second);
                for (ParticleInteger i = begin; i < end; i++) {
                    ParticleInteger const srcIndex = i - lineId.index;
                    h[destIndex] = ptr[srcIndex];
                    destIndex++;
                }
            }
        }
    }
}

static float const* particlesCacheReadLineSoftlens(LineIdType lineId) {
    return readParticleCacheLine<ParticlesFetchCacheLineSoftlensAction>(
        lineId, particleSofteningLengthCache, mutexesSoftlens);
}

void particlesGlobalReadPos(ParticleGlobalRange range, Fixed32* x, Fixed32* y, Fixed32* z,
    char* types, float* zeta, ParticleInteger offset) {
    static bool const sph = getOptions().sph;
    ParticleInteger const lineSize = getOptions().particleCacheLineSize;
    if (range.range.first != range.range.second) {
        if (range.proc == hpxRank()) {
            ParticleInteger const dif = offset - range.range.first;
            ParticleInteger const sz = range.range.second - range.range.first;
            std::memcpy(x + offset, &particlesPos(XDIM, range.range.first), sizeof(Fixed32) * sz);
            std::memcpy(y + offset, &particlesPos(YDIM, range.range.first), sizeof(Fixed32) * sz);
            std::memcpy(z + offset, &particlesPos(ZDIM, range.range.first), sizeof(Fixed32) * sz);
            if (zeta) {
                std::memcpy(zeta + offset, &particlesZeta(range.range.first), sizeof(float) * sz);
            }
            if (types) {
                for (int i = range.range.first; i < range.range.second; i++) {
                    int const j = offset + i - range.range.first;
                    if (getOptions().sph) {
                        ParticleInteger const k = particlesCatIndex(i);
                        int type = particlesType(i);
                        types[j] = type;
                    } else {
                        types[j] = DARK_MATTER_TYPE;
                    }
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
                auto const* ptr = particlesCacheReadLine(lineId);
                auto const begin = std::max(lineId.index, range.range.first);
                auto const end = std::min(lineId.index + lineSize, range.range.second);
                for (ParticleInteger i = begin; i < end; i++) {
                    ParticleInteger const srcIndex = i - lineId.index;
                    x[destIndex] = ptr[srcIndex].x[XDIM];
                    y[destIndex] = ptr[srcIndex].x[YDIM];
                    z[destIndex] = ptr[srcIndex].x[ZDIM];
                    if (types) {
                        if (getOptions().sph) {
                            types[destIndex] = ptr[srcIndex].type;
                        } else {
                            types[destIndex] = DARK_MATTER_TYPE;
                        }
                    }
                    if (zeta) {
                        zeta[destIndex] = ptr[srcIndex].zeta1;
                    }
                    destIndex++;
                }
            }
        }
    }
}

static ParticlesCacheEntry const* particlesCacheReadLine(LineIdType lineId) {
    return readParticleCacheLine<ParticlesFetchCacheLineAction>(lineId, particleCache, mutexes);
}

static Vector<ParticlesCacheEntry> particlesFetchCacheLine(ParticleInteger index) {
    bool const sph = getOptions().sph;
    bool const vsoft = getOptions().vsoft;
    ParticleInteger const lineSize = getOptions().particleCacheLineSize;
    Vector<ParticlesCacheEntry> line(lineSize);
    ParticleInteger const begin = (index / lineSize) * lineSize;
    ParticleInteger const end = std::min(particlesSize(), begin + lineSize);
    for (ParticleInteger i = begin; i < end; i++) {
        auto& ln = line[i - begin];
        ln.zeta1 = ln.zeta2 = 0.0f;
        for (int dim = 0; dim < NDIM; dim++) {
            ln.x[dim] = particlesPos(dim, i);
        }
        if (getOptions().sph) {
            int type = particlesType(i);
            auto const kk = particlesCatIndex(i);
            ln.type = type;
        } else {
            ln.type = DARK_MATTER_TYPE;
        }
        if (getOptions().vsoft) {
            ln.zeta1 = particlesZeta(i);
        }
    }
    return line;
}

static Vector<float> particlesFetchCacheLineSoftlens(ParticleInteger index) {
    static bool const sph = getOptions().sph;
    ParticleInteger const lineSize = getOptions().particleCacheLineSize;
    Vector<float> line(lineSize);
    ParticleInteger const begin = (index / lineSize) * lineSize;
    ParticleInteger const end = std::min(particlesSize(), begin + lineSize);
    for (ParticleInteger i = begin; i < end; i++) {
        auto& ln = line[i - begin];
        ln = particlesSoftlen(i);
    }
    return line;
}

static Vector<GroupParticle> particlesGroupFetchCacheLine(ParticleInteger index) {
    ParticleInteger const lineSize = getOptions().particleCacheLineSize;
    Vector<GroupParticle> line(lineSize);
    ParticleInteger const begin = (index / lineSize) * lineSize;
    ParticleInteger const end = std::min(particlesSize(), begin + lineSize);
    for (ParticleInteger i = begin; i < end; i++) {
        for (int dim = 0; dim < NDIM; dim++) {
            line[i - begin].x[dim] = particlesPos(dim, i);
        }
        line[i - begin].g = particlesGroup(i);
    }
    return line;
}

void particlesIncGroupCacheEpoch() {
    ParticleInteger const lineSize = getOptions().particleCacheLineSize;
    Vector<hpx::future<void>> futs;
    auto const children = hpxChildren();
    for (auto const& c : children) {
        futs.push_back(hpx::async<ParticlesIncGroupCacheEpochAction>(c));
    }
    groupCacheEpoch++;
    hpx::wait_all(futs.begin(), futs.end());
}

static Vector<GroupInteger> particlesGroupRefreshCacheLine(ParticleInteger index) {
    ParticleInteger const lineSize = getOptions().particleCacheLineSize;
    Vector<GroupInteger> line(lineSize);
    ParticleInteger const begin = (index / lineSize) * lineSize;
    ParticleInteger const end = std::min(particlesSize(), begin + lineSize);
    for (ParticleInteger i = begin; i < end; i++) {
        line[i - begin] = (GroupInteger) particlesGroup(i);
    }
    return line;
}

void particlesDestroy() {
    Vector<hpx::future<void>> futs;
    auto const children = hpxChildren();
    for (auto const& c : children) {
        futs.push_back(hpx::async<ParticlesDestroyAction>(c));
    }
    particlesX = decltype(particlesX)();
    particlesGravityAccelerations = decltype(particlesGravityAccelerations)();
    particlesV = decltype(particlesV)();
    particlesP = decltype(particlesP)();
    particlesR = decltype(particlesR)();
    hpx::wait_all(futs.begin(), futs.end());
}

ParticleInteger particlesSize() {
    return size;
}

template <class T>
void particlesArrayResize(T*& ptr, ParticleInteger newCapacity, bool reg) {
    T* newPtr;
#ifdef USE_CUDA
    if (reg) {
        cudaMallocManaged(&newPtr, sizeof(T) * newCapacity);
    } else {
        newPtr = (T*) malloc(sizeof(T) * newCapacity);
    }
#else
    newPtr = (T*) malloc(sizeof(T) * newCapacity);
#endif
    if (capacity > 0) {
        static_assert(std::is_trivially_copyable_v<T>);
        std::memcpy(newPtr, ptr, sizeof(T) * size);
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

HPX_PLAIN_ACTION(particlesResetConverged, ParticlesResetConvergedAction);

void particlesResetConverged() {
    int const numberOfThreads = hpxHardwareConcurrency();
    std::vector<hpx::future<void>> futs;
    for (auto& c : hpxChildren()) {
        futs.push_back(hpx::async<ParticlesResetConvergedAction>(c));
    }
    for (int proc = 0; proc < numberOfThreads; proc++) {
        futs.push_back(hpx::async([proc, numberOfThreads] {
            ParticleInteger const b = (size_t) proc * particlesSize() / numberOfThreads;
            ParticleInteger const e = (size_t) (proc + 1) * particlesSize() / numberOfThreads;
            for (ParticleInteger i = b; i < e; i++) {
                particlesConverged(i) = false;
            }
        }));
    }
    hpx::wait_all(futs.begin(), futs.end());
}

void particlesResize(ParticleInteger sz, bool lock) {
    //	PRINT( "Resizing particles to %i\n", sz);
    if (sz > capacity) {
        if (lock) {
            sharedMutex.lock();
        }
        ParticleInteger newCapacity = std::max(capacity, (ParticleInteger) 100);
        while (newCapacity < sz) {
            newCapacity = size_t(105) * newCapacity / size_t(100);
        }
        //		PRINT("%i: Resizing particles to %li from %li\n", hpx_rank(), new_capacity,
        // capacity);
        for (int dim = 0; dim < NDIM; dim++) {
            particlesArrayResize(particlesX[dim], newCapacity, true);
        }
        particlesArrayResize(particlesV, newCapacity, true);
        particlesArrayResize(particlesR, newCapacity, true);
        if (getOptions().vsoft || getOptions().sph) {
            particlesArrayResize(particlesSa, newCapacity, true);
            particlesArrayResize(particlesC, newCapacity, true);
            particlesArrayResize(particlesZ, newCapacity, true);
            particlesArrayResize(particlesS, newCapacity, true);
            particlesArrayResize(particlesDv, newCapacity, true);
        }
        if (getOptions().doGroups) {
            particlesArrayResize(particlesLgrp, newCapacity, false);
            for (ParticleInteger i = capacity; i < newCapacity; i++) {
                particlesLgrp[i] = NO_GROUP;
            }
        }
        if (getOptions().saveForce || getOptions().vsoft) {
            for (int dim = 0; dim < NDIM; dim++) {
                particlesArrayResize(particlesGravityAccelerations[dim], newCapacity, true);
            }
            particlesArrayResize(particlesP, newCapacity, true);
        }
        if (getOptions().doTracers) {
            particlesArrayResize(particlesTr, newCapacity, false);
        }
        if (getOptions().sph) {
            particlesArrayResize(particlesSph, newCapacity, true);
            particlesArrayResize(particlesTy, newCapacity, true);
            particlesArrayResize(particlesRh, newCapacity, true);
        }
        capacity = newCapacity;
        if (lock) {
            sharedMutex.unlock();
        }
    }
    int oldsz = size;
    size = sz;
    if (getOptions().sph) {
        for (int i = oldsz; i < sz; i++) {
            particlesCatIndex(i) = NO_INDEX;
            particlesType(i) = DARK_MATTER_TYPE;
            particlesSoftlen(i) = getOptions().gravitySupportRadius;
            particlesZeta(i) = 0.0f;
            particlesDivv(i) = 0.0f;
            particlesRho(i) = 0.0f;
            particlesConverged(i) = false;
            particlesSemiactive(i) = false;
        }
    }
}

void particlesFree() {
    for (int dim = 0; dim < NDIM; dim++) {
#ifdef USE_CUDA
        CUDA_CHECK(cudaFree(particlesX[dim]));
#else
        free(particlesX[dim]);
#endif
    }
#ifdef USE_CUDA
    CUDA_CHECK(cudaFree(particlesV));
#else
    free(particlesV);
#endif
#ifdef USE_CUDA
    CUDA_CHECK(cudaFree(particlesR));
#else
    free(particlesR);
#endif
    if (getOptions().doGroups) {
        free(particlesLgrp);
    }
    if (getOptions().saveForce || getOptions().vsoft) {
        for (int dim = 0; dim < NDIM; dim++) {
#ifdef USE_CUDA
            CUDA_CHECK(cudaFree(particlesGravityAccelerations[dim]));
#else
            free(particlesGravityAccelerations[dim]);
#endif
        }
#ifdef USE_CUDA
        CUDA_CHECK(cudaFree(particlesP));
#else
        free(particlesP);
#endif
    }
    if (getOptions().doTracers) {
        free(particlesTr);
    }
    if (getOptions().sph) {
#ifdef USE_CUDA
        CUDA_CHECK(cudaFree(particlesTy));
        CUDA_CHECK(cudaFree(particlesSph));
        CUDA_CHECK(cudaFree(particlesRh));
#else
        free(particlesTy);
        free(particlesSph);
        free(particlesRh);
#endif
    }
    if (getOptions().sph || getOptions().vsoft) {
#ifdef USE_CUDA
        CUDA_CHECK(cudaFree(particlesS));
        CUDA_CHECK(cudaFree(particlesZ));
        CUDA_CHECK(cudaFree(particlesDv));
        CUDA_CHECK(cudaFree(particlesSa));
        CUDA_CHECK(cudaFree(particlesC));
#else
        free(particlesS);
        free(particlesZ);
        free(particlesDv);
        free(particlesSa);
        free(particlesC);
#endif
    }
}

void particlesRandomInit() {
    Vector<hpx::future<void>> futs;
    auto const children = hpxChildren();
    for (auto const& c : children) {
        futs.push_back(hpx::async<ParticlesRandomInitAction>(c));
    }
    size_t const totalNumberOfParticles = getOptions().numberOfParticles;
    size_t const begin = (size_t) (hpxRank()) * totalNumberOfParticles / hpxSize();
    size_t const end = (size_t) (hpxRank() + 1) * totalNumberOfParticles / hpxSize();
    size_t const localParticleCount = end - begin;
    particlesResize(localParticleCount);
    int const numberOfThreads = hpxHardwareConcurrency();
    for (int proc = 0; proc < numberOfThreads; proc++) {
        futs.push_back(hpx::async([proc, numberOfThreads]() {
            ParticleInteger const begin = (size_t) proc * particlesSize() / numberOfThreads;
            ParticleInteger const end = (size_t) (proc + 1) * particlesSize() / numberOfThreads;
            int const seed = 4321 * (hpxRank() * numberOfThreads + proc) + 42;
            gsl_rng* rndgen = gsl_rng_alloc(gsl_rng_taus);
            gsl_rng_set(rndgen, seed);
            for (ParticleInteger i = begin; i < end; i++) {
                for (int dim = 0; dim < NDIM; dim++) {
                    particlesPos(dim, i) = gsl_rng_uniform(rndgen);
                    particlesVel(dim, i) = 0.0f;
                }
                particlesRung(i) = 0;
            }
            gsl_rng_free(rndgen);
        }));
    }
    hpx::wait_all(futs.begin(), futs.end());
}

Pair<Array<double, NDIM>, double> particlesEnclosingSphere(Pair<ParticleInteger> rng) {
    auto totRng = particlesCurrentRange();
    auto totalNumberOfParticles = totRng.second - totRng.first;
    int const numberOfThreads =
        (rng.second - rng.first) * hpxHardwareConcurrency() / totalNumberOfParticles;

    auto const func1 = [numberOfThreads, rng](int proc) {
        Range<double> box;
        for (int dim = 0; dim < NDIM; dim++) {
            box.begin[dim] = 1.0;
            box.end[dim] = 0.0;
        }
        auto const begin = (size_t) (proc) * (rng.second - rng.first) * numberOfThreads + rng.first;
        auto const end =
            (size_t) (proc + 1) * (rng.second - rng.first) * numberOfThreads + rng.first;
        for (ParticleInteger i = begin; i < end; i++) {
            for (int dim = 0; dim < NDIM; dim++) {
                auto const x = particlesPos(dim, i).toDouble();
                box.begin[dim] = std::min(box.begin[dim], x);
                box.end[dim] = std::max(box.end[dim], x);
            }
        }
        return box;
    };
    Vector<hpx::future<Range<double>>> futs1;
    for (int proc = 1; proc < numberOfThreads; proc++) {
        futs1.push_back(hpx::async(func1, proc));
    }
    auto box = func1(0);
    for (auto& fut : futs1) {
        auto thisBox = fut.get();
        for (int dim = 0; dim < NDIM; dim++) {
            box.begin[dim] = std::min(box.begin[dim], thisBox.begin[dim]);
            box.end[dim] = std::max(box.end[dim], thisBox.end[dim]);
        }
    }
    Array<double, NDIM> xc;
    for (int dim = 0; dim < NDIM; dim++) {
        xc[dim] = 0.5 * (box.begin[dim] + box.end[dim]);
    }
    auto const func2 = [numberOfThreads, rng, xc](int proc) {
        double rmax = 0.0;
        auto const begin = (size_t) (proc) * (rng.second - rng.first) * numberOfThreads + rng.first;
        auto const end =
            (size_t) (proc + 1) * (rng.second - rng.first) * numberOfThreads + rng.first;
        for (ParticleInteger i = begin; i < end; i++) {
            double r2 = 0.0;
            for (int dim = 0; dim < NDIM; dim++) {
                double const x = particlesPos(dim, i).toDouble();
                r2 += sqr(x - xc[dim]);
            }
            rmax = std::max(rmax, sqrt(r2));
        }
        return rmax;
    };
    Vector<hpx::future<double>> futs2;
    for (int proc = 1; proc < numberOfThreads; proc++) {
        futs2.push_back(hpx::async(func2, proc));
    }
    auto rmax = func2(0);
    for (auto& fut : futs2) {
        rmax = std::max(rmax, fut.get());
    }
    Pair<Array<double, NDIM>, double> rc;
    rc.first = xc;
    rc.second = rmax;
    return rc;
}

ParticleInteger particlesSort(Pair<ParticleInteger> rng, double xm, int xdim) {
    ParticleInteger begin = rng.first;
    ParticleInteger end = rng.second;
    ParticleInteger lo = begin;
    ParticleInteger hi = end;
    Fixed32 xmid(xm);
    bool const doGroups = getOptions().doGroups;
    bool const doTracers = getOptions().doTracers;
    auto& xptrDim = particlesX[xdim];
    auto& x = particlesX[XDIM];
    auto& y = particlesX[YDIM];
    auto& z = particlesX[ZDIM];
    int flops = 0;
    while (lo < hi) {
        flops++;
        if (xptrDim[lo] >= xmid) {
            while (lo != hi) {
                hi--;
                flops++;
                if (xptrDim[hi] < xmid) {
                    std::swap(x[hi], x[lo]);
                    std::swap(y[hi], y[lo]);
                    std::swap(z[hi], z[lo]);
                    std::swap(particlesV[hi], particlesV[lo]);
                    std::swap(particlesR[hi], particlesR[lo]);
                    if (doGroups) {
                        std::swap(particlesLgrp[hi], particlesLgrp[lo]);
                    }
                    if (doTracers) {
                        std::swap(particlesTr[hi], particlesTr[lo]);
                    }
                    if (getOptions().sph) {
                        std::swap(particlesSph[hi], particlesSph[lo]);
                        std::swap(particlesTy[hi], particlesTy[lo]);
                    }
                    if (getOptions().vsoft) {
                        std::swap(particlesDv[hi], particlesDv[lo]);
                        std::swap(particlesS[hi], particlesS[lo]);
                    }
                    break;
                }
            }
        }
        lo++;
    }
    return hi;
    addCpuFlops(flops);
}

Pair<ParticleInteger, ParticleInteger> particlesCurrentRange() {
    Pair<ParticleInteger, ParticleInteger> rc;
    rc.first = rungBegin;
    rc.second = particlesSize();
    return rc;
}

HPX_PLAIN_ACTION(particlesSortByRung, ParticlesSortByRungAction);

void particlesSortByRung(int minrung) {
    profilerEnter(__FUNCTION__);
    Vector<hpx::future<void>> futs;
    for (auto& c : hpxChildren()) {
        futs.push_back(hpx::async<ParticlesSortByRungAction>(c, minrung));
    }
    if (minrung == 0) {
        rungBegin = 0;
    } else {
        ParticleInteger begin;
        ParticleInteger end;
        begin = rungBegin;
        end = particlesSize();
        ParticleInteger lo = begin;
        ParticleInteger hi = end;
        bool const doGroups = getOptions().doGroups;
        bool const doTracers = getOptions().doTracers;
        auto& x = particlesX[XDIM];
        auto& y = particlesX[YDIM];
        auto& z = particlesX[ZDIM];
        while (lo < hi) {
            if (particlesRung(lo) >= minrung) {
                while (lo != hi) {
                    hi--;
                    if (particlesRung(hi) < minrung) {
                        std::swap(x[hi], x[lo]);
                        std::swap(y[hi], y[lo]);
                        std::swap(z[hi], z[lo]);
                        std::swap(particlesV[hi], particlesV[lo]);
                        std::swap(particlesR[hi], particlesR[lo]);
                        if (doGroups) {
                            std::swap(particlesLgrp[hi], particlesLgrp[lo]);
                        }
                        if (doTracers) {
                            std::swap(particlesTr[hi], particlesTr[lo]);
                        }
                        if (getOptions().sph) {
                            std::swap(particlesSph[hi], particlesSph[lo]);
                            std::swap(particlesTy[hi], particlesTy[lo]);
                        }
                        if (getOptions().vsoft) {
                            std::swap(particlesDv[hi], particlesDv[lo]);
                            std::swap(particlesS[hi], particlesS[lo]);
                        }
                        break;
                    }
                }
            }
            lo++;
        }
        rungBegin = hi;
    }

    hpx::wait_all(futs.begin(), futs.end());
    profilerExit();
}

void particlesGlobalReadRungs(ParticleGlobalRange range, char* r, ParticleInteger offset) {
    ParticleInteger const lineSize = getOptions().particleCacheLineSize;
    if (range.range.first != range.range.second) {
        if (range.proc == hpxRank()) {
            ParticleInteger const dif = offset - range.range.first;
            ParticleInteger const sz = range.range.second - range.range.first;
            std::memcpy(r + offset, &particlesRung(range.range.first), sizeof(char) * sz);
        } else {
            LineIdType lineId;
            lineId.proc = range.proc;
            ParticleInteger const startLine = (range.range.first / lineSize) * lineSize;
            ParticleInteger const stopLine = ((range.range.second - 1) / lineSize) * lineSize;
            ParticleInteger destIndex = offset;
            for (ParticleInteger line = startLine; line <= stopLine; line += lineSize) {
                lineId.index = line;
                auto const* ptr = particlesCacheReadLineRungs(lineId);
                auto const begin = std::max(lineId.index, range.range.first);
                auto const end = std::min(lineId.index + lineSize, range.range.second);
                for (ParticleInteger i = begin; i < end; i++) {
                    ParticleInteger const srcIndex = i - lineId.index;
                    r[destIndex] = ptr[srcIndex];
                    destIndex++;
                }
            }
        }
    }
}

static char const* particlesCacheReadLineRungs(LineIdType lineId) {
    return readParticleCacheLine<ParticlesFetchCacheLineRungsAction>(
        lineId, particleRungCache, mutexesRungs);
}

static Vector<char> particlesFetchCacheLineRungs(ParticleInteger index) {
    ParticleInteger const lineSize = getOptions().particleCacheLineSize;
    Vector<char> line(lineSize);
    ParticleInteger const begin = (index / lineSize) * lineSize;
    ParticleInteger const end = std::min(particlesSize(), begin + lineSize);
    for (ParticleInteger i = begin; i < end; i++) {
        auto& ln = line[i - begin];
        ln = particlesRung(i);
    }
    return line;
}

Vector<ParticleSample> particlesSample(int cnt) {
    bool const saveForce = getOptions().saveForce;
    Vector<ParticleSample> parts;
    Vector<hpx::future<Vector<ParticleSample>>> futs;
    if (hpxRank() == 0) {
        auto const& localities = hpxLocalities();
        for (int i = 1; i < localities.size(); i++) {
            ParticleInteger const b = (size_t) i * cnt / localities.size();
            ParticleInteger const e = (size_t) (i + 1) * cnt / localities.size();
            ParticleInteger const thisCnt = e - b;
            futs.push_back(hpx::async<ParticlesSampleAction>(localities[i], thisCnt));
        }
        ParticleInteger const b = 0;
        ParticleInteger const e = (size_t) (1) * cnt / localities.size();
        cnt = e - b;
    }
    int const seed = 4321 * hpxRank() + 42;
    gsl_rng* rndgen = gsl_rng_alloc(gsl_rng_taus);
    gsl_rng_set(rndgen, seed);
    PRINT("Selecting %i particles\n", cnt);
    for (ParticleInteger i = 0; i < cnt; i++) {
        ParticleSample sample;
        ParticleInteger const index =
            ((size_t) gsl_rng_get(rndgen) * (size_t) gsl_rng_get(rndgen)) % particlesSize();
        for (int dim = 0; dim < NDIM; dim++) {
            sample.x[dim] = particlesPos(dim, index);
        }
        if (saveForce) {
            for (int dim = 0; dim < NDIM; dim++) {
                sample.g[dim] = particlesGravityAcceleration(dim, index);
            }
            sample.p = particlesPot(index);
        }
        parts.push_back(sample);
    }
    gsl_rng_free(rndgen);
    PRINT("Done\n");
    for (auto& f : futs) {
        auto const theseParticles = f.get();
        parts.insert(parts.end(), theseParticles.begin(), theseParticles.end());
    }
    return std::move(parts);
}

void particlesLoad(FILE* fp) {
    ParticleInteger sphSize = 0;
    ParticleInteger starsSize0 = 0;
    bool const vsoft = getOptions().vsoft;
    ParticleInteger size;
    FREAD(&size, sizeof(ParticleInteger), 1, fp);
    if (getOptions().sph) {
        FREAD(&sphSize, sizeof(ParticleInteger), 1, fp);
        FREAD(&starsSize0, sizeof(ParticleInteger), 1, fp);
        PRINT("Reading %i total particles %i stars %i sph\n", size, starsSize0, sphSize);
        if (starsSize0 != 0 || sphSize < 0 || sphSize > size) {
            THROW_ERROR("Unsupported star payload or invalid gas count in checkpoint\n");
        }
        particlesResize(size);
        sphParticlesResize(sphSize, false);
    } else {
        PRINT("Reading %i total particles.\n", size);
        particlesResize(size);
    }
    PRINT("particles_size  = %i\n", particlesSize());
    FREAD(&particlesPos(XDIM, 0), sizeof(Fixed32), particlesSize(), fp);
    FREAD(&particlesPos(YDIM, 0), sizeof(Fixed32), particlesSize(), fp);
    FREAD(&particlesPos(ZDIM, 0), sizeof(Fixed32), particlesSize(), fp);
    FREAD(particlesVelData(), sizeof(Array<float, NDIM>), particlesSize(), fp);
    FREAD(&particlesRung(0), sizeof(char), particlesSize(), fp);
    if (getOptions().doGroups) {
        FREAD(&particlesLastgroup(0), sizeof(GroupInteger), particlesSize(), fp);
    }
    if (getOptions().doTracers) {
        FREAD(&particlesTracer(0), sizeof(char), particlesSize(), fp);
    }
    if (getOptions().vsoft) {
        FREAD(&particlesSoftlen(0), sizeof(float), particlesSize(), fp);
        FREAD(&particlesDivv(0), sizeof(float), particlesSize(), fp);
    }
    if (getOptions().sph) {
        FREAD(&particlesType(0), sizeof(char), particlesSize(), fp);
        FREAD(&particlesCatIndex(0), sizeof(ParticleInteger), particlesSize(), fp);
        sphParticlesLoad(fp);
    }
    FREAD(&size, sizeof(int), 1, fp);
    rungBegins.resize(size);
    FREAD(&rungBegin, sizeof(int), 1, fp);
    FREAD(rungBegins.data(), sizeof(int), rungBegins.size(), fp);
}

void particlesSave(FILE* fp) {
    ParticleInteger sphSize = getOptions().sph ? sphParticlesSize() : 0;
    ParticleInteger starsSize0 = 0;
    bool const vsoft = getOptions().vsoft;
    ParticleInteger size = particlesSize();
    fwrite(&size, sizeof(ParticleInteger), 1, fp);
    if (getOptions().sph) {
        fwrite(&sphSize, sizeof(ParticleInteger), 1, fp);
        fwrite(&starsSize0, sizeof(ParticleInteger), 1, fp);
    }
    fwrite(&particlesPos(XDIM, 0), sizeof(Fixed32), particlesSize(), fp);
    fwrite(&particlesPos(YDIM, 0), sizeof(Fixed32), particlesSize(), fp);
    fwrite(&particlesPos(ZDIM, 0), sizeof(Fixed32), particlesSize(), fp);
    fwrite(particlesVelData(), sizeof(Array<float, NDIM>), particlesSize(), fp);
    fwrite(&particlesRung(0), sizeof(char), particlesSize(), fp);
    if (getOptions().doGroups) {
        fwrite(&particlesLastgroup(0), sizeof(GroupInteger), particlesSize(), fp);
    }
    if (getOptions().doTracers) {
        fwrite(&particlesTracer(0), sizeof(char), particlesSize(), fp);
    }
    if (getOptions().vsoft) {
        fwrite(&particlesSoftlen(0), sizeof(float), particlesSize(), fp);
        fwrite(&particlesDivv(0), sizeof(float), particlesSize(), fp);
    }
    if (getOptions().sph) {
        fwrite(&particlesType(0), sizeof(char), particlesSize(), fp);
        fwrite(&particlesCatIndex(0), sizeof(ParticleInteger), particlesSize(), fp);
        sphParticlesSave(fp);
    }
    size = rungBegins.size();
    fwrite(&size, sizeof(int), 1, fp);
    fwrite(&rungBegin, sizeof(int), 1, fp);
    fwrite(rungBegins.data(), sizeof(int), rungBegins.size(), fp);
}

void particlesSaveGlass(char const* filename) {
    FILE* fp = fopen(filename, "wb");
    ParticleInteger size = getOptions().particlesPerDimension;
    fwrite(&size, sizeof(ParticleInteger), 1, fp);
    for (ParticleInteger i = 0; i < particlesSize(); i++) {
        if (particlesType(i) == DARK_MATTER_TYPE) {
            fwrite(&particlesPos(XDIM, i), sizeof(Fixed32), 1, fp);
        }
    }
    for (ParticleInteger i = 0; i < particlesSize(); i++) {
        if (particlesType(i) == DARK_MATTER_TYPE) {
            fwrite(&particlesPos(YDIM, i), sizeof(Fixed32), 1, fp);
        }
    }
    for (ParticleInteger i = 0; i < particlesSize(); i++) {
        if (particlesType(i) == DARK_MATTER_TYPE) {
            fwrite(&particlesPos(ZDIM, i), sizeof(Fixed32), 1, fp);
        }
    }
    for (ParticleInteger i = 0; i < particlesSize(); i++) {
        if (particlesType(i) == SPH_TYPE) {
            fwrite(&particlesPos(XDIM, i), sizeof(Fixed32), 1, fp);
        }
    }
    for (ParticleInteger i = 0; i < particlesSize(); i++) {
        if (particlesType(i) == SPH_TYPE) {
            fwrite(&particlesPos(YDIM, i), sizeof(Fixed32), 1, fp);
        }
    }
    for (ParticleInteger i = 0; i < particlesSize(); i++) {
        if (particlesType(i) == SPH_TYPE) {
            fwrite(&particlesPos(ZDIM, i), sizeof(Fixed32), 1, fp);
        }
    }
    fclose(fp);
}

void particlesResolveWithSphParticles() {
    int const numberOfThreads = hpxHardwareConcurrency();
    std::vector<hpx::future<void>> futs;
    for (int proc = 0; proc < numberOfThreads; proc++) {
        futs.push_back(hpx::async([proc, numberOfThreads] {
            ParticleInteger const b = (size_t) proc * particlesSize() / numberOfThreads;
            ParticleInteger const e = (size_t) (proc + 1) * particlesSize() / numberOfThreads;
            for (ParticleInteger i = b; i < e; i++) {
                int const type = particlesType(i);
                if (type == SPH_TYPE) {
                    ParticleInteger const index = particlesCatIndex(i);
                    sphParticlesParticleIndex(index) = i;
                } else if (type == STAR_TYPE) {
                    ParticleInteger const index = particlesCatIndex(i);
                    starsGet(index).particleIndex = i;
                }
            }
        }));
    }
    hpx::wait_all(futs.begin(), futs.end());
}

void particlesGlobalReadVels(
    ParticleGlobalRange range, float* vx, float* vy, float* vz, ParticleInteger offset) {
    static bool const sph = getOptions().sph;
    ParticleInteger const lineSize = getOptions().particleCacheLineSize;
    if (range.range.first != range.range.second) {
        if (range.proc == hpxRank()) {
            ParticleInteger const dif = offset - range.range.first;
            ParticleInteger const sz = range.range.second - range.range.first;
            for (ParticleInteger i = 0; i < sz; i++) {
                vx[offset + i] = particlesVel(XDIM, range.range.first + i);
                vy[offset + i] = particlesVel(YDIM, range.range.first + i);
                vz[offset + i] = particlesVel(ZDIM, range.range.first + i);
            }
        } else {
            LineIdType lineId;
            lineId.proc = range.proc;
            ParticleInteger const startLine = (range.range.first / lineSize) * lineSize;
            ParticleInteger const stopLine = ((range.range.second - 1) / lineSize) * lineSize;
            ParticleInteger destIndex = offset;
            for (ParticleInteger line = startLine; line <= stopLine; line += lineSize) {
                lineId.index = line;
                auto const* ptr = particlesCacheReadLineVels(lineId);
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

static Array<float, NDIM> const* particlesCacheReadLineVels(LineIdType lineId) {
    return readParticleCacheLine<ParticlesFetchCacheLineVelsAction>(
        lineId, particleVelocityCache, velsMutexes);
}

static Vector<Array<float, NDIM>> particlesFetchCacheLineVels(ParticleInteger index) {
    static bool const sph = getOptions().sph;
    static bool const vsoft = getOptions().vsoft;
    ParticleInteger const lineSize = getOptions().particleCacheLineSize;
    Vector<Array<float, NDIM>> line(lineSize);
    ParticleInteger const begin = (index / lineSize) * lineSize;
    ParticleInteger const end = std::min(particlesSize(), begin + lineSize);
    for (ParticleInteger i = begin; i < end; i++) {
        auto& ln = line[i - begin];
        for (int dim = 0; dim < NDIM; dim++) {
            ln[dim] = particlesVel(dim, i);
        }
    }
    return line;
}

HPX_PLAIN_ACTION(particlesSumEnergies, ParticlesSumEnergiesAction);

EnergiesT particlesSumEnergies() {
    profilerEnter(__FUNCTION__);
    std::vector<hpx::future<EnergiesT>> futs;
    for (auto& c : hpxChildren()) {
        futs.push_back(hpx::async<ParticlesSumEnergiesAction>(c));
    }
    EnergiesT energies;
    int const numberOfThreads = hpxHardwareConcurrency();
    for (int proc = 0; proc < numberOfThreads; proc++) {
        futs.push_back(hpx::async([numberOfThreads, proc]() {
            EnergiesT energies;
            ParticleInteger const b = (size_t) proc * particlesSize() / numberOfThreads;
            ParticleInteger const e = (size_t) (proc + 1) * particlesSize() / numberOfThreads;
            for (ParticleInteger i = b; i != e; i++) {
                float const vx = particlesVel(XDIM, i);
                float const vy = particlesVel(YDIM, i);
                float const vz = particlesVel(ZDIM, i);
                float const m = particlesMass(i);
                if (getOptions().saveForce || getOptions().vsoft) {
                    energies.pot += 0.5 * m * particlesPot(i);
                }
                energies.kin += 0.5 * m * (sqr(vx) + sqr(vy) + sqr(vz));
                energies.xmom += m * vx;
                energies.ymom += m * vy;
                energies.zmom += m * vz;
                energies.nmom += m * std::sqrt(sqr(vx) + sqr(vy) + sqr(vz));
                if (particlesType(i) == SPH_TYPE) {
                    int const kk = particlesCatIndex(i);
#ifdef HOPKINS
                    energies.therm += m * sphParticlesEintRho(kk);
#else
                    energies.therm += m * sphParticlesEint(kk);
#endif
                }
            }
            return energies;
        }));
    }

    for (auto& f : futs) {
        energies += f.get();
    }
    profilerExit();
    return energies;
}

SharedMutexType& particlesSharedMutex() {
    return sharedMutex;
}

Vector<int> particlesGetLocal(Vector<Pair<ParticleInteger>> const& ranges) {
    Vector<int> rc;
    size_t sz = 0;
    for (auto const& r : ranges) {
        sz += r.second - r.first;
    }

    rc.resize(NDIM * sz);

    auto const func1 = [&ranges, &rc, sz]() {
        std::shared_lock<SharedMutexType> lock(sharedMutex);
        ParticleInteger i = 0;
        for (int j = 0; j < ranges.size() / 2; j++) {
            auto const& r = ranges[j];
            auto const count = r.second - r.first;
            for (int dim = 0; dim < NDIM; dim++) {
                memcpy(&rc[i + dim * sz], &particlesPos(dim, r.first), count * sizeof(Fixed32));
            }
            i += count;
        }
    };
    auto const func2 = [&ranges, &rc, sz]() {
        std::shared_lock<SharedMutexType> lock(sharedMutex);
        ParticleInteger i = sz;
        for (int j = ranges.size() - 1; j >= (int) ranges.size() / 2; j--) {
            auto const& r = ranges[j];
            auto const count = r.second - r.first;
            i -= count;
            for (int dim = 0; dim < NDIM; dim++) {
                memcpy(&rc[i + dim * sz], &particlesPos(dim, r.first), count * sizeof(Fixed32));
            }
        }
    };
    auto fut1 = hpx::async(func1);
    func2();
    fut1.get();
    return rc;
}

HPX_PLAIN_ACTION(particlesGetLocal, ParticlesGetLocalAction);

hpx::future<Vector<int>> particlesGet(int rank, Vector<Pair<ParticleInteger>> const& ranges) {
    return hpx::async<ParticlesGetLocalAction>(hpxLocalities()[rank], ranges);
}

HPX_PLAIN_ACTION(particlesReadGadget4, ParticlesReadGadget4Action);

GadgetIoHeader particlesReadGadget4(std::string fnameBase) {
    PRINT("Reading gadget4 file\n");
    Vector<hpx::future<void>> futs;
    for (auto& c : hpxChildren()) {
        futs.push_back(hpx::async<ParticlesReadGadget4Action>(c, fnameBase));
    }
    std::string fname = fnameBase;
    if (hpxSize() > 1) {
        fname += std::string(".") + std::to_string(hpxRank());
    }
    FILE* fp = fopen(fname.c_str(), "rb");
    if (!fp) {
        THROW_ERROR("Unable to open %s for writing.\n", fname.c_str());
    }
    int size;
    GadgetIoHeader header;

    FREAD(&size, sizeof(int), 1, fp);
    PRINT("%i\n", size);
    FREAD(&header, sizeof(GadgetIoHeader), 1, fp);
    FREAD(&size, sizeof(int), 1, fp);
    PRINT("%i\n", size);

    PRINT("Print nparts = %lli\n", header.npart[1]);
    auto const numberOfParticles = header.npart[1];
    particlesResize(numberOfParticles);

    FREAD(&size, sizeof(int), 1, fp);
    PRINT("%i\n", size);
    for (ParticleInteger i = 0; i < numberOfParticles; i++) {
        for (int dim = 0; dim < NDIM; dim++) {
            float pos;
            FREAD(&pos, sizeof(float), 1, fp);
            pos /= header.boxSize;
            particlesPos(dim, i) = pos;
        }
    }
    FREAD(&size, sizeof(int), 1, fp);
    PRINT("%i\n", size);

    FREAD(&size, sizeof(int), 1, fp);
    PRINT("%i\n", size);
    for (ParticleInteger i = 0; i < numberOfParticles; i++) {
        for (int dim = 0; dim < NDIM; dim++) {
            float vel;
            FREAD(&vel, sizeof(float), 1, fp);
            particlesVel(dim, i) = vel;
        }
    }
    auto const a = 1.0 / (header.redshift + 1.0);
    for (int i = 0; i < numberOfParticles; i++) {
        //	PRINT( "%e %e %e\n", particles_pos(XDIM,i).to_double(),
        // particles_pos(YDIM,i).to_double(), particles_pos(ZDIM,i).to_double());
        for (int dim = 0; dim < NDIM; dim++) {
            particlesVel(dim, i) *=
                sqrt(a) * a * 1e5 * getOptions().codeToS / getOptions().codeToCm;
        }
    }
    FREAD(&size, sizeof(int), 1, fp);
    PRINT("%i\n", size);

    fclose(fp);
    hpx::wait_all(futs.begin(), futs.end());
    return header;
}
