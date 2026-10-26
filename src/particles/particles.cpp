/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

constexpr bool kVerbose = true;
#define PARTICLES_CPP

#include <algorithm>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <gsl/gsl_rng.h>

#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/safe_io.hpp>
#include <cosmictiger/cuda/cuda_memory_advice.hpp>
#include <cosmictiger/io/output.hpp>
#include <cosmictiger/particles/particle_partition.hpp>
#include <cosmictiger/particles/particle_selection.hpp>
#include <cosmictiger/particles/particles.hpp>
#include <cosmictiger/sph/sph_particles.hpp>

struct LineIdType;

static Vector<GroupInteger> particlesGroupRefreshCacheLine(ParticleInteger index);

static Vector<GroupParticle> particlesGroupFetchCacheLine(ParticleInteger index);
static GroupParticle const* particlesGroupCacheReadLine(LineIdType lineId);
void particlesGroupCacheFree();

struct ParticlesCacheEntry {
    Array<Position, kDimensions> x;
    char type;
    template <class A>
    void serialize(A&& arc, unsigned) {
        arc & type;
        arc & x;
    }
};

static Vector<ParticlesCacheEntry> particlesFetchCacheLine(ParticleInteger index);
static ParticlesCacheEntry const* particlesCacheReadLine(LineIdType lineId);
void particlesCacheFree();

static Vector<Array<float, kDimensions>> particlesFetchCacheLineVels(ParticleInteger index);
static Array<float, kDimensions> const* particlesCacheReadLineVels(LineIdType lineId);

static Vector<char> particlesFetchCacheLineRungs(ParticleInteger index);
static char const* particlesCacheReadLineRungs(LineIdType lineId);

static void particlesSetGlobalOffset(Vector<size_t>);

static SharedMutexType sharedMutex;

static ParticleInteger size = 0;
static ParticleInteger capacity = 0;
static std::vector<void (*)()> particleReleaseHooks;
static std::atomic<uint64_t> remoteParticleFetches{0};
static Vector<size_t> globalOffsets;
static ParticleInteger rungBegin;
static std::vector<ParticleInteger> rungBegins;

HPX_PLAIN_ACTION(particlesCacheFree, ParticlesCacheFreeAction);
HPX_PLAIN_ACTION(particlesIncGroupCacheEpoch, ParticlesIncGroupCacheEpochAction);
HPX_PLAIN_ACTION(particlesDestroy, ParticlesDestroyAction);
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
        numActive /= getOptions().initial.numberOfParticles;
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
                double x = particlesPos(kXDimension, i).toDouble();
                double y = particlesPos(kYDimension, i).toDouble();
                double z = particlesPos(kZDimension, i).toDouble();
                x = fmod(x + dx, 1.0);
                y = fmod(y + dy, 1.0);
                z = fmod(z + dz, 1.0);
                particlesPos(kXDimension, i) = x;
                particlesPos(kYDimension, i) = y;
                particlesPos(kZDimension, i) = z;
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
        ParticleInteger const lineSize = getOptions().compute.particleCacheLineSize;
        ParticleInteger const i = id.index / lineSize;
        return i * (hpxSize() - 1) + ((id.proc < hpxRank()) ? id.proc : id.proc - 1);
    }
};

struct LineIdHashLo {
    inline size_t operator()(LineIdType id) const {
        LineIdHash hash;
        return hash(id) % kParticleCacheSize;
    }
};

struct LineIdHashHi {
    inline size_t operator()(LineIdType id) const {
        LineIdHash hash;
        return hash(id) / kParticleCacheSize;
    }
};

static Array<std::unordered_map<LineIdType,
                 std::shared_ptr<hpx::shared_future<Vector<ParticlesCacheEntry>>>, LineIdHashHi>,
    kParticleCacheSize>
    particleCache;
static Array<SpinlockType, kParticleCacheSize> mutexes;
static int groupCacheEpoch = 0;

static Array<
    std::unordered_map<LineIdType, std::shared_ptr<hpx::shared_future<Vector<char>>>, LineIdHashHi>,
    kParticleCacheSize>
    particleRungCache;
static Array<SpinlockType, kParticleCacheSize> mutexesRungs;

static Array<
    std::unordered_map<LineIdType,
        std::shared_ptr<hpx::shared_future<Vector<Array<float, kDimensions>>>>, LineIdHashHi>,
    kParticleCacheSize>
    particleVelocityCache;
static Array<SpinlockType, kParticleCacheSize> velsMutexes;

struct GroupCacheEntry {
    std::shared_ptr<hpx::shared_future<Vector<GroupParticle>>> data;
    int epoch;
};

static Array<std::unordered_map<LineIdType, GroupCacheEntry, LineIdHashHi>, kParticleCacheSize>
    groupParticleCache;
static Array<SpinlockType, kParticleCacheSize> groupMutexes;

template <typename Action, typename T>
static T const* readParticleCacheLine(LineIdType lineId,
    Array<std::unordered_map<LineIdType, std::shared_ptr<hpx::shared_future<Vector<T>>>,
              LineIdHashHi>,
        kParticleCacheSize>& cache,
    Array<SpinlockType, kParticleCacheSize>& locks) {
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
                        if constexpr (std::is_same_v<Action, ParticlesFetchCacheLineAction>) {
                            remoteParticleFetches.fetch_add(1, std::memory_order_relaxed);
                        }
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
        Array<double, kDimensions> x;
        for (int dim = 0; dim < kDimensions; dim++) {
            x[dim] = positionDomainCoordinate(particlesPos(dim, i));
        }
        if (box.contains(x)) {
            OutputParticle data;
            for (int dim = 0; dim < kDimensions; dim++) {
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

Vector<OutputParticle> particlesGetTracers() {
    Vector<hpx::future<Vector<OutputParticle>>> futs;
    Vector<OutputParticle> output;
    for (auto const& c : hpxChildren()) {
        futs.push_back(hpx::async<ParticlesGetTracersAction>(c));
    }
    for (ParticleInteger i = 0; i < particlesSize(); i++) {
        if (particlesTracer(i)) {
            OutputParticle data;
            for (int dim = 0; dim < kDimensions; dim++) {
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
        size_t const totalCount =
            getOptions().initial.numberOfParticles * (getOptions().hydro.enabled ? 2 : 1);
        detail::setTracerFlags(&particlesTracer(0), particlesSize(), count, totalCount,
            getOptions().analysis.tracerCount);
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
    hpx_fill(PAR_EXECUTION_POLICY, particlesGrp, particlesGrp + particlesSize(), kNoParticleGroup)
        .get();
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

void particlesCacheFreeLocal() {
    for (size_t i = 0; i < kParticleCacheSize; i++) {
        particleRungCache[i].clear();
        particleVelocityCache[i].clear();
    }
    particleCache = decltype(particleCache)();
    groupParticleCache = decltype(groupParticleCache)();
}

void particlesCacheFree() {
    particlesCacheFreeLocal();
    Vector<hpx::future<void>> futs;
    for (auto const& c : hpxChildren()) {
        futs.push_back(hpx::async<ParticlesCacheFreeAction>(c));
    }
    hpx::wait_all(futs.begin(), futs.end());
}

static std::function<void()> undoLastMemadvise = nullptr;

void particlesMemadviseGpu() {
#ifdef USE_CUDA
    if (!cudaEnabled()) {
        return;
    }
    cudaSetDevice();
    if (undoLastMemadvise) {
        undoLastMemadvise();
    }
    int deviceid = cudaGetDevice();
    auto const rng = particlesCurrentRange();
    auto const begin = rng.first;
    auto const count = rng.second - begin;
    CUDA_CHECK(cudaAdviseMemory(particlesVelData() + begin,
        count * sizeof(Array<float, kDimensions>), cudaMemAdviseSetAccessedBy, deviceid));
    CUDA_CHECK(cudaAdviseMemory(
        &particlesRung(0) + begin, count * sizeof(char), cudaMemAdviseSetAccessedBy, deviceid));
    undoLastMemadvise = [begin, count, deviceid]() {
        CUDA_CHECK(cudaAdviseMemory(particlesVelData() + begin,
            count * sizeof(Array<float, kDimensions>), cudaMemAdviseUnsetAccessedBy, deviceid));
        CUDA_CHECK(cudaAdviseMemory(&particlesRung(0) + begin, count * sizeof(char),
            cudaMemAdviseUnsetAccessedBy, deviceid));
    };
#endif
}

void particlesMemadviseCpu() {
#ifdef USE_CUDA
    if (!cudaEnabled()) {
        return;
    }
    cudaSetDevice();
    int deviceid = cudaGetDevice();
    if (undoLastMemadvise) {
        undoLastMemadvise();
    }
    auto const rng = particlesCurrentRange();
    auto const begin = rng.first;
    auto const count = rng.second - begin;
    CUDA_CHECK(
        cudaAdviseMemory(particlesVelData() + begin, count * sizeof(Array<float, kDimensions>),
            cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
    CUDA_CHECK(cudaAdviseMemory(&particlesRung(0) + begin, count * sizeof(char),
        cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
    undoLastMemadvise = [begin, count]() {
        CUDA_CHECK(
            cudaAdviseMemory(particlesVelData() + begin, count * sizeof(Array<float, kDimensions>),
                cudaMemAdviseUnsetPreferredLocation, cudaCpuDeviceId));
        CUDA_CHECK(cudaAdviseMemory(&particlesRung(0) + begin, count * sizeof(char),
            cudaMemAdviseUnsetPreferredLocation, cudaCpuDeviceId));
    };
    CudaStream stream(deviceid);
    for (int dim = 0; dim < kDimensions; dim++) {
        stream.prefetchAsync(&particlesPos(dim, begin), count * sizeof(Position), cudaCpuDeviceId);
    }
    stream.prefetchAsync(&particlesVel(kXDimension, begin),
        count * sizeof(Array<float, kDimensions>), cudaCpuDeviceId);
    stream.prefetchAsync(&particlesRung(begin), count * sizeof(char), cudaCpuDeviceId);
    stream.wait();
#endif
}

void particlesGlobalReadPosAndGroup(ParticleGlobalRange range, Position* x, Position* y,
    Position* z, GroupInteger* g, ParticleInteger offset) {
    ParticleInteger const lineSize = getOptions().compute.particleCacheLineSize;
    if (range.range.first != range.range.second) {
        if (range.proc == hpxRank()) {
            ParticleInteger const dif = offset - range.range.first;
            ParticleInteger const sz = range.range.second - range.range.first;
            std::memcpy(
                x + offset, &particlesPos(kXDimension, range.range.first), sizeof(Position) * sz);
            std::memcpy(
                y + offset, &particlesPos(kYDimension, range.range.first), sizeof(Position) * sz);
            std::memcpy(
                z + offset, &particlesPos(kZDimension, range.range.first), sizeof(Position) * sz);
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
                    x[destIndex] = ptr[srcIndex].x[kXDimension];
                    y[destIndex] = ptr[srcIndex].x[kYDimension];
                    z[destIndex] = ptr[srcIndex].x[kZDimension];
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
    return counts;
}

HPX_PLAIN_ACTION(particlesSetMinrung, ParticlesSetMinrungAction);

void particlesSetMinrung(int minrung) {
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
}

void particlesGlobalReadPos(ParticleGlobalRange range, Position* x, Position* y, Position* z,
    char* types, ParticleInteger offset) {
    static bool const sph = getOptions().hydro.enabled;
    ParticleInteger const lineSize = getOptions().compute.particleCacheLineSize;
    if (range.range.first != range.range.second) {
        if (range.proc == hpxRank()) {
            ParticleInteger const dif = offset - range.range.first;
            ParticleInteger const sz = range.range.second - range.range.first;
            std::memcpy(
                x + offset, &particlesPos(kXDimension, range.range.first), sizeof(Position) * sz);
            std::memcpy(
                y + offset, &particlesPos(kYDimension, range.range.first), sizeof(Position) * sz);
            std::memcpy(
                z + offset, &particlesPos(kZDimension, range.range.first), sizeof(Position) * sz);
            if (types) {
                for (int i = range.range.first; i < range.range.second; i++) {
                    int const j = offset + i - range.range.first;
                    if (getOptions().hydro.enabled) {
                        ParticleInteger const k = particlesCatIndex(i);
                        int type = particlesType(i);
                        types[j] = type;
                    } else {
                        types[j] = kDarkMatterType;
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
                    x[destIndex] = ptr[srcIndex].x[kXDimension];
                    y[destIndex] = ptr[srcIndex].x[kYDimension];
                    z[destIndex] = ptr[srcIndex].x[kZDimension];
                    if (types) {
                        if (getOptions().hydro.enabled) {
                            types[destIndex] = ptr[srcIndex].type;
                        } else {
                            types[destIndex] = kDarkMatterType;
                        }
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
    bool const sph = getOptions().hydro.enabled;
    ParticleInteger const lineSize = getOptions().compute.particleCacheLineSize;
    Vector<ParticlesCacheEntry> line(lineSize);
    ParticleInteger const begin = (index / lineSize) * lineSize;
    ParticleInteger const end = std::min(particlesSize(), begin + lineSize);
    for (ParticleInteger i = begin; i < end; i++) {
        auto& ln = line[i - begin];
        for (int dim = 0; dim < kDimensions; dim++) {
            ln.x[dim] = particlesPos(dim, i);
        }
        if (getOptions().hydro.enabled) {
            int type = particlesType(i);
            auto const kk = particlesCatIndex(i);
            ln.type = type;
        } else {
            ln.type = kDarkMatterType;
        }
    }
    return line;
}

static Vector<GroupParticle> particlesGroupFetchCacheLine(ParticleInteger index) {
    ParticleInteger const lineSize = getOptions().compute.particleCacheLineSize;
    Vector<GroupParticle> line(lineSize);
    ParticleInteger const begin = (index / lineSize) * lineSize;
    ParticleInteger const end = std::min(particlesSize(), begin + lineSize);
    for (ParticleInteger i = begin; i < end; i++) {
        for (int dim = 0; dim < kDimensions; dim++) {
            line[i - begin].x[dim] = particlesPos(dim, i);
        }
        line[i - begin].g = particlesGroup(i);
    }
    return line;
}

void particlesIncGroupCacheEpoch() {
    ParticleInteger const lineSize = getOptions().compute.particleCacheLineSize;
    Vector<hpx::future<void>> futs;
    auto const children = hpxChildren();
    for (auto const& c : children) {
        futs.push_back(hpx::async<ParticlesIncGroupCacheEpochAction>(c));
    }
    groupCacheEpoch++;
    hpx::wait_all(futs.begin(), futs.end());
}

static Vector<GroupInteger> particlesGroupRefreshCacheLine(ParticleInteger index) {
    ParticleInteger const lineSize = getOptions().compute.particleCacheLineSize;
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

ParticleInteger particlesOwnedSize() {
    return size;
}

uint64_t particlesRemoteFetchCount() {
    return remoteParticleFetches.load(std::memory_order_relaxed);
}

template <class T>
void particlesArrayResize(T*& ptr, ParticleInteger newCapacity, bool reg) {
    T* newPtr = static_cast<T*>(reg ? allocateManagedMemory(sizeof(T) * newCapacity) :
                                      std::malloc(sizeof(T) * newCapacity));
    if (newPtr == nullptr) {
        throw std::bad_alloc();
    }
    if (ptr != nullptr) {
        static_assert(std::is_trivially_copyable_v<T>);
        std::memcpy(newPtr, ptr, sizeof(T) * size);
        if (reg) {
            freeManagedMemory(ptr);
        } else {
            std::free(ptr);
        }
    }
    ptr = newPtr;
}

static void particlesResizeStorage(ParticleInteger sz, bool lock) {
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
        for (int dim = 0; dim < kDimensions; dim++) {
            particlesArrayResize(particlesX[dim], newCapacity, true);
        }
        particlesArrayResize(particlesV, newCapacity, true);
        particlesArrayResize(particlesR, newCapacity, true);
        if (getOptions().analysis.groups) {
            particlesArrayResize(particlesLgrp, newCapacity, false);
            for (ParticleInteger i = capacity; i < newCapacity; i++) {
                particlesLgrp[i] = kNoParticleGroup;
            }
        }
        if (saveParticleAcceleration(getOptions().analysis)) {
            for (int dim = 0; dim < kDimensions; dim++) {
                particlesArrayResize(particlesGravityAccelerations[dim], newCapacity, true);
            }
        }
        if (saveParticlePotential(getOptions().analysis)) {
            particlesArrayResize(particlesP, newCapacity, true);
        }
        if (getOptions().analysis.tracers) {
            particlesArrayResize(particlesTr, newCapacity, false);
        }
        if (getOptions().hydro.enabled) {
            particlesArrayResize(particlesSph, newCapacity, true);
            particlesArrayResize(particlesTy, newCapacity, true);
        }
        capacity = newCapacity;
        if (lock) {
            sharedMutex.unlock();
        }
    }
    int oldsz = size;
    size = sz;
    if (getOptions().hydro.enabled) {
        for (int i = oldsz; i < sz; i++) {
            particlesCatIndex(i) = kNoParticleIndex;
            particlesType(i) = kDarkMatterType;
        }
    }
}

void particlesResize(ParticleInteger sz, bool lock) {
    if (sz != size) {
        particlesResizeStorage(sz, lock);
    }
}

void particlesRegisterReleaseHook(void (*hook)()) {
    particleReleaseHooks.push_back(hook);
}

void particlesFree() {
    for (auto hook : particleReleaseHooks) {
        hook();
    }
    for (int dim = 0; dim < kDimensions; dim++) {
        freeManagedMemory(particlesX[dim]);
    }
    freeManagedMemory(particlesV);
    freeManagedMemory(particlesR);
    if (getOptions().analysis.groups) {
        free(particlesLgrp);
    }
    if (saveParticleAcceleration(getOptions().analysis)) {
        for (int dim = 0; dim < kDimensions; dim++) {
            freeManagedMemory(particlesGravityAccelerations[dim]);
        }
    }
    if (saveParticlePotential(getOptions().analysis)) {
        freeManagedMemory(particlesP);
    }
    if (getOptions().analysis.tracers) {
        free(particlesTr);
    }
    if (getOptions().hydro.enabled) {
        freeManagedMemory(particlesTy);
        freeManagedMemory(particlesSph);
    }
}

void particlesRandomInit() {
    Vector<hpx::future<void>> futs;
    auto const children = hpxChildren();
    for (auto const& c : children) {
        futs.push_back(hpx::async<ParticlesRandomInitAction>(c));
    }
    size_t const totalNumberOfParticles = getOptions().initial.numberOfParticles;
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
                for (int dim = 0; dim < kDimensions; dim++) {
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

Pair<Array<double, kDimensions>, double> particlesEnclosingSphere(Pair<ParticleInteger> rng) {
    auto totRng = particlesCurrentRange();
    auto totalNumberOfParticles = totRng.second - totRng.first;
    int const numberOfThreads =
        (rng.second - rng.first) * hpxHardwareConcurrency() / totalNumberOfParticles;

    auto const func1 = [numberOfThreads, rng](int proc) {
        Range<double> box;
        for (int dim = 0; dim < kDimensions; dim++) {
            box.begin[dim] = 1.0;
            box.end[dim] = 0.0;
        }
        auto const begin = (size_t) (proc) * (rng.second - rng.first) * numberOfThreads + rng.first;
        auto const end =
            (size_t) (proc + 1) * (rng.second - rng.first) * numberOfThreads + rng.first;
        for (ParticleInteger i = begin; i < end; i++) {
            for (int dim = 0; dim < kDimensions; dim++) {
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
        for (int dim = 0; dim < kDimensions; dim++) {
            box.begin[dim] = std::min(box.begin[dim], thisBox.begin[dim]);
            box.end[dim] = std::max(box.end[dim], thisBox.end[dim]);
        }
    }
    Array<double, kDimensions> xc;
    for (int dim = 0; dim < kDimensions; dim++) {
        xc[dim] = 0.5 * (box.begin[dim] + box.end[dim]);
    }
    auto const func2 = [numberOfThreads, rng, xc](int proc) {
        double rmax = 0.0;
        auto const begin = (size_t) (proc) * (rng.second - rng.first) * numberOfThreads + rng.first;
        auto const end =
            (size_t) (proc + 1) * (rng.second - rng.first) * numberOfThreads + rng.first;
        for (ParticleInteger i = begin; i < end; i++) {
            double r2 = 0.0;
            for (int dim = 0; dim < kDimensions; dim++) {
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
    Pair<Array<double, kDimensions>, double> rc;
    rc.first = xc;
    rc.second = rmax;
    return rc;
}

ParticleInteger particlesSort(Pair<ParticleInteger> rng, double xm, int xdim) {
    addCpuFlops(1);
    return particlesPartition(rng, Position(xm), xdim);
}

ParticleInteger particlesPartition(Pair<ParticleInteger> rng, Position xmid, int xdim) {
    ParticleInteger begin = rng.first;
    ParticleInteger end = rng.second;
    ParticleInteger lo = begin;
    ParticleInteger hi = end;
    auto& xptrDim = particlesX[xdim];
    while (lo < hi) {
        if (xptrDim[lo] >= xmid) {
            while (lo != hi) {
                hi--;
                if (xptrDim[hi] < xmid) {
                    particlesSwap(hi, lo);
                    break;
                }
            }
        }
        lo++;
    }
    return hi;
}

Pair<ParticleInteger, ParticleInteger> particlesCurrentRange() {
    Pair<ParticleInteger, ParticleInteger> rc;
    rc.first = rungBegin;
    rc.second = particlesSize();
    return rc;
}

HPX_PLAIN_ACTION(particlesSortByRung, ParticlesSortByRungAction);

void particlesSortByRung(int minrung) {
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
        end = particlesOwnedSize();
        ParticleInteger lo = begin;
        ParticleInteger hi = end;
        while (lo < hi) {
            if (particlesRung(lo) >= minrung) {
                while (lo != hi) {
                    hi--;
                    if (particlesRung(hi) < minrung) {
                        particlesSwap(hi, lo);
                        break;
                    }
                }
            }
            lo++;
        }
        rungBegin = hi;
    }

    hpx::wait_all(futs.begin(), futs.end());
}

void particlesGlobalReadRungs(ParticleGlobalRange range, char* r, ParticleInteger offset) {
    ParticleInteger const lineSize = getOptions().compute.particleCacheLineSize;
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
    ParticleInteger const lineSize = getOptions().compute.particleCacheLineSize;
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
    auto const& analysis = getOptions().analysis;
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
        ParticleSample sample{};
        ParticleInteger const index =
            ((size_t) gsl_rng_get(rndgen) * (size_t) gsl_rng_get(rndgen)) % particlesSize();
        for (int dim = 0; dim < kDimensions; dim++) {
            sample.x[dim] = particlesPos(dim, index);
        }
        if (saveParticleAcceleration(analysis)) {
            for (int dim = 0; dim < kDimensions; dim++) {
                sample.g[dim] = particlesGravityAcceleration(dim, index);
            }
        }
        if (saveParticlePotential(analysis)) {
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
    ParticleInteger size;
    FREAD(&size, sizeof(ParticleInteger), 1, fp);
    if (getOptions().hydro.enabled) {
        FREAD(&sphSize, sizeof(ParticleInteger), 1, fp);
        PRINT("Reading %i total particles %i sph\n", size, sphSize);
        if (sphSize < 0 || sphSize > size) {
            THROW_ERROR("Invalid gas count in checkpoint\n");
        }
        particlesResize(size);
        sphParticlesResize(sphSize, false);
    } else {
        PRINT("Reading %i total particles.\n", size);
        particlesResize(size);
    }
    PRINT("particles_size  = %i\n", particlesSize());
    FREAD(&particlesPos(kXDimension, 0), sizeof(Position), particlesSize(), fp);
    FREAD(&particlesPos(kYDimension, 0), sizeof(Position), particlesSize(), fp);
    FREAD(&particlesPos(kZDimension, 0), sizeof(Position), particlesSize(), fp);
    FREAD(particlesVelData(), sizeof(Array<float, kDimensions>), particlesSize(), fp);
    FREAD(&particlesRung(0), sizeof(char), particlesSize(), fp);
    if (getOptions().analysis.groups) {
        FREAD(&particlesLastgroup(0), sizeof(GroupInteger), particlesSize(), fp);
    }
    if (getOptions().analysis.tracers) {
        FREAD(&particlesTracer(0), sizeof(char), particlesSize(), fp);
    }
    if (getOptions().hydro.enabled) {
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
    ParticleInteger sphSize = getOptions().hydro.enabled ? sphParticlesSize() : 0;
    ParticleInteger size = particlesSize();
    fwrite(&size, sizeof(ParticleInteger), 1, fp);
    if (getOptions().hydro.enabled) {
        fwrite(&sphSize, sizeof(ParticleInteger), 1, fp);
    }
    fwrite(&particlesPos(kXDimension, 0), sizeof(Position), particlesSize(), fp);
    fwrite(&particlesPos(kYDimension, 0), sizeof(Position), particlesSize(), fp);
    fwrite(&particlesPos(kZDimension, 0), sizeof(Position), particlesSize(), fp);
    fwrite(particlesVelData(), sizeof(Array<float, kDimensions>), particlesSize(), fp);
    fwrite(&particlesRung(0), sizeof(char), particlesSize(), fp);
    if (getOptions().analysis.groups) {
        fwrite(&particlesLastgroup(0), sizeof(GroupInteger), particlesSize(), fp);
    }
    if (getOptions().analysis.tracers) {
        fwrite(&particlesTracer(0), sizeof(char), particlesSize(), fp);
    }
    if (getOptions().hydro.enabled) {
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
    FILE* fp = fopen(outputPath(filename).c_str(), "wb");
    ParticleInteger size = getOptions().initial.particlesPerDimension;
    fwrite(&size, sizeof(ParticleInteger), 1, fp);
    for (ParticleInteger i = 0; i < particlesSize(); i++) {
        if (particlesType(i) == kDarkMatterType) {
            Fixed32 position(particlesPos(kXDimension, i));
            fwrite(&position, sizeof(Fixed32), 1, fp);
        }
    }
    for (ParticleInteger i = 0; i < particlesSize(); i++) {
        if (particlesType(i) == kDarkMatterType) {
            Fixed32 position(particlesPos(kYDimension, i));
            fwrite(&position, sizeof(Fixed32), 1, fp);
        }
    }
    for (ParticleInteger i = 0; i < particlesSize(); i++) {
        if (particlesType(i) == kDarkMatterType) {
            Fixed32 position(particlesPos(kZDimension, i));
            fwrite(&position, sizeof(Fixed32), 1, fp);
        }
    }
    for (ParticleInteger i = 0; i < particlesSize(); i++) {
        if (particlesType(i) == kSphType) {
            Fixed32 position(particlesPos(kXDimension, i));
            fwrite(&position, sizeof(Fixed32), 1, fp);
        }
    }
    for (ParticleInteger i = 0; i < particlesSize(); i++) {
        if (particlesType(i) == kSphType) {
            Fixed32 position(particlesPos(kYDimension, i));
            fwrite(&position, sizeof(Fixed32), 1, fp);
        }
    }
    for (ParticleInteger i = 0; i < particlesSize(); i++) {
        if (particlesType(i) == kSphType) {
            Fixed32 position(particlesPos(kZDimension, i));
            fwrite(&position, sizeof(Fixed32), 1, fp);
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
                if (type == kSphType) {
                    ParticleInteger const index = particlesCatIndex(i);
                    sphParticlesParticleIndex(index) = i;
                }
            }
        }));
    }
    hpx::wait_all(futs.begin(), futs.end());
}

void particlesGlobalReadVels(
    ParticleGlobalRange range, float* vx, float* vy, float* vz, ParticleInteger offset) {
    static bool const sph = getOptions().hydro.enabled;
    ParticleInteger const lineSize = getOptions().compute.particleCacheLineSize;
    if (range.range.first != range.range.second) {
        if (range.proc == hpxRank()) {
            ParticleInteger const dif = offset - range.range.first;
            ParticleInteger const sz = range.range.second - range.range.first;
            for (ParticleInteger i = 0; i < sz; i++) {
                vx[offset + i] = particlesVel(kXDimension, range.range.first + i);
                vy[offset + i] = particlesVel(kYDimension, range.range.first + i);
                vz[offset + i] = particlesVel(kZDimension, range.range.first + i);
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
                    vx[destIndex] = ptr[srcIndex][kXDimension];
                    vy[destIndex] = ptr[srcIndex][kYDimension];
                    vz[destIndex] = ptr[srcIndex][kZDimension];
                    destIndex++;
                }
            }
        }
    }
}

static Array<float, kDimensions> const* particlesCacheReadLineVels(LineIdType lineId) {
    return readParticleCacheLine<ParticlesFetchCacheLineVelsAction>(
        lineId, particleVelocityCache, velsMutexes);
}

static Vector<Array<float, kDimensions>> particlesFetchCacheLineVels(ParticleInteger index) {
    static bool const sph = getOptions().hydro.enabled;
    ParticleInteger const lineSize = getOptions().compute.particleCacheLineSize;
    Vector<Array<float, kDimensions>> line(lineSize);
    ParticleInteger const begin = (index / lineSize) * lineSize;
    ParticleInteger const end = std::min(particlesSize(), begin + lineSize);
    for (ParticleInteger i = begin; i < end; i++) {
        auto& ln = line[i - begin];
        for (int dim = 0; dim < kDimensions; dim++) {
            ln[dim] = particlesVel(dim, i);
        }
    }
    return line;
}

HPX_PLAIN_ACTION(particlesSumEnergies, ParticlesSumEnergiesAction);

EnergiesT particlesSumEnergies() {
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
                float const vx = particlesVel(kXDimension, i);
                float const vy = particlesVel(kYDimension, i);
                float const vz = particlesVel(kZDimension, i);
                float const m = particlesMass(i);
                if (saveParticlePotential(getOptions().analysis)) {
                    energies.pot += 0.5 * m * particlesPot(i);
                }
                energies.kin += 0.5 * m * (sqr(vx) + sqr(vy) + sqr(vz));
                energies.xmom += m * vx;
                energies.ymom += m * vy;
                energies.zmom += m * vz;
                energies.nmom += m * std::sqrt(sqr(vx) + sqr(vy) + sqr(vz));
                if (particlesType(i) == kSphType) {
                    int const kk = particlesCatIndex(i);
                    energies.therm += m * sphParticlesInternalEnergy(kk);
                }
            }
            return energies;
        }));
    }

    for (auto& f : futs) {
        energies += f.get();
    }
    return energies;
}

SharedMutexType& particlesSharedMutex() {
    return sharedMutex;
}

Vector<Position> particlesGetLocal(Vector<Pair<ParticleInteger>> const& ranges) {
    Vector<Position> rc;
    size_t sz = 0;
    for (auto const& r : ranges) {
        sz += r.second - r.first;
    }

    rc.resize(kDimensions * sz);

    auto const func1 = [&ranges, &rc, sz]() {
        std::shared_lock<SharedMutexType> lock(sharedMutex);
        ParticleInteger i = 0;
        for (int j = 0; j < ranges.size() / 2; j++) {
            auto const& r = ranges[j];
            auto const count = r.second - r.first;
            for (int dim = 0; dim < kDimensions; dim++) {
                memcpy(&rc[i + dim * sz], &particlesPos(dim, r.first), count * sizeof(Position));
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
            for (int dim = 0; dim < kDimensions; dim++) {
                memcpy(&rc[i + dim * sz], &particlesPos(dim, r.first), count * sizeof(Position));
            }
        }
    };
    auto fut1 = hpx::async(func1);
    func2();
    fut1.get();
    return rc;
}

HPX_PLAIN_ACTION(particlesGetLocal, ParticlesGetLocalAction);

hpx::future<Vector<Position>> particlesGet(int rank, Vector<Pair<ParticleInteger>> const& ranges) {
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
        for (int dim = 0; dim < kDimensions; dim++) {
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
        for (int dim = 0; dim < kDimensions; dim++) {
            float vel;
            FREAD(&vel, sizeof(float), 1, fp);
            particlesVel(dim, i) = vel;
        }
    }
    auto const a = 1.0 / (header.redshift + 1.0);
    for (int i = 0; i < numberOfParticles; i++) {
        //	PRINT( "%e %e %e\n", particles_pos(XDIM,i).to_double(),
        // particles_pos(YDIM,i).to_double(), particles_pos(ZDIM,i).to_double());
        for (int dim = 0; dim < kDimensions; dim++) {
            particlesVel(dim, i) *=
                sqrt(a) * a * 1e5 * getOptions().units.timeSeconds / getOptions().units.lengthCm;
        }
    }
    FREAD(&size, sizeof(int), 1, fp);
    PRINT("%i\n", size);

    fclose(fp);
    hpx::wait_all(futs.begin(), futs.end());
    return header;
}
