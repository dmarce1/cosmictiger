/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

constexpr bool kVerbose = true;
#define PARTICLES_CPP

#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if COSMICTIGER_WITH_GRAVITY
#include <gsl/gsl_rng.h>
#endif
#include <hpx/threading_base/annotated_function.hpp>

#include <cosmictiger/core/file_handle.hpp>
#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/safe_io.hpp>
#include <cosmictiger/cuda/cuda_memory_advice.hpp>
#include <cosmictiger/io/output.hpp>
#include <cosmictiger/math/vector3.hpp>
#include <cosmictiger/particles/particle_partition.hpp>
#include <cosmictiger/particles/particle_selection.hpp>
#include <cosmictiger/particles/particles.hpp>
#if COSMICTIGER_WITH_SPH
#include <cosmictiger/hydro/sph/sph_particles.hpp>
#endif

struct LineIdType;

static Vector<GroupInteger> particlesGroupRefreshCacheLine(ParticleInteger index);

static Vector<GroupParticle> particlesGroupFetchCacheLine(ParticleInteger index);
static GroupParticle const* particlesGroupCacheReadLine(LineIdType lineId);
void particlesGroupCacheFree();

struct ParticlesCacheEntry {
    Vector3<Position> x;
    char type;
    float mass;
    template <class A>
    void serialize(A&& arc, unsigned) {
        arc & type;
        arc & x;
        if (getOptions().hydro.method == "mfv") {
            arc & mass;
        }
    }
};

static Vector<ParticlesCacheEntry> particlesFetchCacheLine(ParticleInteger index);
static ParticlesCacheEntry const* particlesCacheReadLine(LineIdType lineId);
void particlesCacheFree();

static Vector<Vector3<float>> particlesFetchCacheLineVels(ParticleInteger index);
static Vector3<float> const* particlesCacheReadLineVels(LineIdType lineId);

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
#if COSMICTIGER_WITH_GRAVITY
HPX_PLAIN_ACTION(particlesSample, ParticlesSampleAction);
#endif
HPX_PLAIN_ACTION(particlesGroupsInit, ParticlesGroupsInitAction);
HPX_PLAIN_ACTION(particlesGroupsDestroy, ParticlesGroupsDestroyAction);
HPX_PLAIN_ACTION(particlesSetGlobalOffset, ParticlesSetGlobalOffsetAction);
HPX_PLAIN_ACTION(particlesSetTracers, ParticlesSetTracersAction);
HPX_PLAIN_ACTION(particlesSize, ParticlesTracerCountAction);
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
        futs.push_back(hpx::async(hpx::annotated_function(
            [numberOfThreads, proc, dx, dy, dz]() {
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
            },
            "particles.displace")));
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

static Array<std::unordered_map<LineIdType,
                 std::shared_ptr<hpx::shared_future<Vector<Vector3<float>>>>, LineIdHashHi>,
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
                hpx::post(hpx::annotated_function(
                    [promise, lineId]() {
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
                    },
                    "particles.readCacheLine"));
            } catch (...) {
                promise->set_exception(std::current_exception());
            }
        }
    }
    return future->get().data();
}

ParticleInteger particlesPartitionDarkMatter(Pair<ParticleInteger> rng) {
    if (!getOptions().hydro.enabled) {
        return rng.second;
    }
    ParticleInteger lo = rng.first;
    ParticleInteger hi = rng.second;
    while (lo < hi) {
        if (particlesType(lo) != kDarkMatterType) {
            while (lo != hi) {
                hi--;
                if (particlesType(hi) == kDarkMatterType) {
                    particlesSwap(hi, lo);
                    break;
                }
            }
        }
        lo++;
    }
    return hi;
}

Vector<OutputParticle> particlesGetSample(Range<double> const& box) {
    Vector<hpx::future<Vector<OutputParticle>>> futs;
    Vector<OutputParticle> output;
    for (auto const& c : hpxChildren()) {
        futs.push_back(hpx::async<ParticlesGetSampleAction>(c, box));
    }
    for (ParticleInteger i = 0; i < particlesSize(); i++) {
        Vector3<double> x;
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

void particlesSetTracers(size_t count, size_t totalCount) {
    if (hpxRank() == 0) {
        Vector<hpx::future<ParticleInteger>> counts;
        for (size_t rank = 1; rank < hpxLocalities().size(); ++rank) {
            counts.push_back(hpx::async<ParticlesTracerCountAction>(hpxLocalities()[rank]));
        }
        totalCount = particlesSize();
        hpx::wait_all(counts.begin(), counts.end());
        for (auto& result : counts) {
            size_t const localCount = result.get();
            if (localCount > std::numeric_limits<size_t>::max() - totalCount) {
                throw std::overflow_error("Tracer particle count overflow");
            }
            totalCount += localCount;
        }
    }
    hpx::future<void> fut;
    if (hpxRank() < hpxSize() - 1) {
        fut = hpx::async<ParticlesSetTracersAction>(
            hpxLocalities()[hpxRank() + 1], count + particlesSize(), totalCount);
    }

    if (particlesSize() > 0) {
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
    hpx_fill(annotatedParallelPolicy("particles.groupsInit"), particlesGrp,
        particlesGrp + particlesSize(), kNoParticleGroup)
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
        futs2.push_back(hpx::async(hpx::annotated_function(
            [numberOfThreads, proc]() {
                ParticleInteger const begin = (size_t) proc * particlesSize() / numberOfThreads;
                ParticleInteger const end = (size_t) (proc + 1) * particlesSize() / numberOfThreads;
                for (ParticleInteger i = begin; i < end; i++) {
                    particlesLastgroup(i) = particlesGroup(i);
                }
            },
            "particles.groupsDestroy")));
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
    CUDA_WARN(cudaAdviseMemory(particlesVelData() + begin, count * sizeof(Vector3<float>),
        cudaMemAdviseSetAccessedBy, deviceid));
    CUDA_WARN(cudaAdviseMemory(
        &particlesRung(0) + begin, count * sizeof(char), cudaMemAdviseSetAccessedBy, deviceid));
    // The CUDA drift reads species for lightcone crossings when hydro is on.
    bool const types = getOptions().hydro.enabled;
    if (types) {
        CUDA_WARN(cudaAdviseMemory(
            particlesTy + begin, count * sizeof(char), cudaMemAdviseSetAccessedBy, deviceid));
    }
    undoLastMemadvise = [begin, count, deviceid, types]() {
        CUDA_WARN(cudaAdviseMemory(particlesVelData() + begin, count * sizeof(Vector3<float>),
            cudaMemAdviseUnsetAccessedBy, deviceid));
        CUDA_WARN(cudaAdviseMemory(&particlesRung(0) + begin, count * sizeof(char),
            cudaMemAdviseUnsetAccessedBy, deviceid));
        if (types) {
            CUDA_WARN(cudaAdviseMemory(
                particlesTy + begin, count * sizeof(char), cudaMemAdviseUnsetAccessedBy, deviceid));
        }
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
    CUDA_CHECK(cudaAdviseMemory(particlesVelData() + begin, count * sizeof(Vector3<float>),
        cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
    CUDA_WARN(cudaAdviseMemory(&particlesRung(0) + begin, count * sizeof(char),
        cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
    bool const types = getOptions().hydro.enabled;
    if (types) {
        CUDA_WARN(cudaAdviseMemory(particlesTy + begin, count * sizeof(char),
            cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
    }
    undoLastMemadvise = [begin, count, types]() {
        CUDA_CHECK(cudaAdviseMemory(particlesVelData() + begin, count * sizeof(Vector3<float>),
            cudaMemAdviseUnsetPreferredLocation, cudaCpuDeviceId));
        CUDA_WARN(cudaAdviseMemory(&particlesRung(0) + begin, count * sizeof(char),
            cudaMemAdviseUnsetPreferredLocation, cudaCpuDeviceId));
        if (types) {
            CUDA_WARN(cudaAdviseMemory(particlesTy + begin, count * sizeof(char),
                cudaMemAdviseUnsetPreferredLocation, cudaCpuDeviceId));
        }
    };
    CudaStream stream(deviceid);
    for (int dim = 0; dim < kDimensions; dim++) {
        stream.prefetchAsync(&particlesPos(dim, begin), count * sizeof(Position), cudaCpuDeviceId);
    }
    stream.prefetchAsync(
        &particlesVel(kXDimension, begin), count * sizeof(Vector3<float>), cudaCpuDeviceId);
    stream.prefetchAsync(&particlesRung(begin), count * sizeof(char), cudaCpuDeviceId);
    if (types) {
        stream.prefetchAsync(particlesTy + begin, count * sizeof(char), cudaCpuDeviceId);
    }
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
                hpx::post(hpx::annotated_function(
                    [promise, lineId, old]() {
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
                    },
                    "particles.readGroupCacheLine"));
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
        futs.push_back(hpx::async(hpx::annotated_function(
            [proc, numberOfThreads]() {
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
            },
            "particles.rungCounts")));
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
        futs.push_back(hpx::async(hpx::annotated_function(
            [proc, numberOfThreads, minrung]() {
                ParticleInteger const b = (size_t) proc * particlesSize() / numberOfThreads;
                ParticleInteger const e = (size_t) (proc + 1) * particlesSize() / numberOfThreads;
                for (ParticleInteger i = b; i < e; i++) {
                    auto& rung = particlesRung(i);
                    rung = std::max((int) rung, minrung);
                }
            },
            "particles.setMinrung")));
    }
    for (auto& f : futs) {
        f.get();
    }
}

void particlesGlobalReadMasses(ParticleGlobalRange range, float* masses, ParticleInteger offset) {
    if (range.range.first == range.range.second) {
        return;
    }
    if (range.proc == hpxRank()) {
        for (ParticleInteger i = range.range.first; i < range.range.second; ++i) {
            masses[offset + i - range.range.first] = particlesMass(i);
        }
        return;
    }
    ParticleInteger const lineSize = getOptions().compute.particleCacheLineSize;
    ParticleInteger const startLine = (range.range.first / lineSize) * lineSize;
    ParticleInteger const stopLine = ((range.range.second - 1) / lineSize) * lineSize;
    ParticleInteger destination = offset;
    for (ParticleInteger line = startLine; line <= stopLine; line += lineSize) {
        auto const* values = particlesCacheReadLine({range.proc, line});
        auto const begin = std::max(line, range.range.first);
        auto const end = std::min(line + lineSize, range.range.second);
        for (ParticleInteger i = begin; i < end; ++i) {
            auto const& value = values[i - line];
            masses[destination++] = getOptions().hydro.method == "mfv" ?
                value.mass :
                (getOptions().hydro.enabled ?
                        float(value.type == kDarkMatterType ? getOptions().initial.darkMatterMass :
                                                              getOptions().initial.gasMass) :
                        1.0f);
        }
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
    std::shared_lock<SharedMutexType> lock(sharedMutex);
    bool const sph = getOptions().hydro.enabled;
    ParticleInteger const lineSize = getOptions().compute.particleCacheLineSize;
    Vector<ParticlesCacheEntry> line(lineSize);
    ParticleInteger const begin = (index / lineSize) * lineSize;
    ParticleInteger const end = std::min(particlesSize(), begin + lineSize);
    for (ParticleInteger i = begin; i < end; i++) {
        auto& ln = line[i - begin];
        // A final owned cache line may include Dehnen's temporary imported tail.
        // Imported gas has no local hydro category; its mass is fetched separately
        // from the source owner, never through the local gas mirror.
        bool const importedGas = getOptions().hydro.enabled && particlesType(i) == kSphType &&
            particlesCatIndex(i) == kNoParticleIndex;
        ln.mass = importedGas ? 0.0f : particlesMass(i);
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
    std::shared_lock<SharedMutexType> lock(sharedMutex);
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
    std::shared_lock<SharedMutexType> lock(sharedMutex);
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
    particlesLongAccelerations = decltype(particlesLongAccelerations)();
    particlesV = decltype(particlesV)();
    particlesP = decltype(particlesP)();
    particlesR = decltype(particlesR)();
    freeManagedMemory(particlesGravitySampleIndices);
    freeManagedMemory(particlesGravitySampleAccelerations);
    particlesGravitySampleIndices = nullptr;
    particlesGravitySampleAccelerations = nullptr;
    freeManagedMemory(particlesStarFormationStates);
    particlesStarFormationStates = nullptr;
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

static void particlesEnsureGravitySampleStorage() {
    if (getOptions().gravity.errorInterval > 0 && particlesGravitySampleIndices == nullptr &&
        capacity > 0) {
        particlesArrayResize(particlesGravitySampleIndices, capacity, true);
        std::fill_n(particlesGravitySampleIndices, capacity, ParticleInteger{-1});
    }
}

static void particlesEnsureStarFormationStorage() {
    if (getOptions().starFormation.enabled && particlesStarFormationStates == nullptr &&
        capacity > 0) {
        particlesArrayResize(particlesStarFormationStates, capacity, true);
        std::fill_n(particlesStarFormationStates, capacity, StarFormationState{});
    }
}

#if !COSMICTIGER_WITH_GRAVITY
static void particlesClearGravityDiagnostics(ParticleInteger begin, ParticleInteger end) {
    if (begin >= end) {
        return;
    }
    // These output arrays have no force pass to populate them in an SPH-only build.
    if (saveParticleAcceleration(getOptions().analysis)) {
        for (int dim = 0; dim < kDimensions; ++dim) {
            std::fill_n(particlesGravityAccelerations[dim] + begin, end - begin, 0.0f);
        }
    }
    if (saveParticlePotential(getOptions().analysis)) {
        std::fill_n(particlesP + begin, end - begin, 0.0f);
    }
}
#endif

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
            for (ParticleInteger i = size; i < newCapacity; i++) {
                particlesLgrp[i] = kNoParticleGroup;
            }
        }
        if (saveParticleAcceleration(getOptions().analysis)) {
            for (int dim = 0; dim < kDimensions; dim++) {
                particlesArrayResize(particlesGravityAccelerations[dim], newCapacity, true);
            }
        }
        if (particlesHasLongAcceleration(getOptions())) {
            for (int dim = 0; dim < kDimensions; dim++) {
                particlesArrayResize(particlesLongAccelerations[dim], newCapacity, true);
            }
        }
        if (saveParticlePotential(getOptions().analysis)) {
            particlesArrayResize(particlesP, newCapacity, true);
        }
        if (getOptions().analysis.tracers) {
            particlesArrayResize(particlesTr, newCapacity, false);
        }
        if (getOptions().gravity.errorInterval > 0 || particlesGravitySampleIndices != nullptr) {
            bool const firstAllocation = particlesGravitySampleIndices == nullptr;
            particlesArrayResize(particlesGravitySampleIndices, newCapacity, true);
            ParticleInteger const begin = firstAllocation ? 0 : size;
            std::fill(particlesGravitySampleIndices + begin,
                particlesGravitySampleIndices + newCapacity, ParticleInteger{-1});
        }
        if (getOptions().hydro.enabled) {
            particlesArrayResize(particlesSph, newCapacity, true);
            particlesArrayResize(particlesTy, newCapacity, true);
        }
        if (getOptions().starFormation.enabled || particlesStarFormationStates != nullptr) {
            bool const firstAllocation = particlesStarFormationStates == nullptr;
            particlesArrayResize(particlesStarFormationStates, newCapacity, true);
            ParticleInteger const begin = firstAllocation ? 0 : size;
            std::fill(particlesStarFormationStates + begin,
                particlesStarFormationStates + newCapacity, StarFormationState{});
        }
        capacity = newCapacity;
        if (lock) {
            sharedMutex.unlock();
        }
    }
    particlesEnsureGravitySampleStorage();
    particlesEnsureStarFormationStorage();
    ParticleInteger const oldsz = size;
    size = sz;
    if (sz > oldsz) {
        if (getOptions().analysis.groups) {
            std::fill(particlesLgrp + oldsz, particlesLgrp + sz, kNoParticleGroup);
        }
        if (getOptions().analysis.tracers) {
            std::fill(particlesTr + oldsz, particlesTr + sz, char{0});
        }
    }
#if !COSMICTIGER_WITH_GRAVITY
    particlesClearGravityDiagnostics(oldsz, sz);
#endif
    if (particlesGravitySampleIndices != nullptr) {
        for (ParticleInteger i = oldsz; i < sz; ++i) {
            particlesGravitySample(i) = -1;
        }
    }
    if (particlesStarFormationStates != nullptr) {
        std::fill(particlesStarFormationStates + oldsz,
            particlesStarFormationStates + std::max(ParticleInteger(oldsz), sz),
            StarFormationState{});
    }
    if (getOptions().hydro.enabled) {
        for (ParticleInteger i = oldsz; i < sz; i++) {
            particlesCatIndex(i) = kNoParticleIndex;
            particlesType(i) = kDarkMatterType;
        }
    }
}

void particlesResize(ParticleInteger sz, bool lock) {
    if (sz != size) {
        particlesResizeStorage(sz, lock);
    } else {
        particlesEnsureGravitySampleStorage();
        particlesEnsureStarFormationStorage();
    }
}

static Pair<uint64_t> particlesStarFormationCount() {
    particlesEnsureStarFormationStorage();
    std::unordered_set<uint64_t> ids;
    uint64_t missing = 0;
    for (ParticleInteger index = 0; index < particlesOwnedSize(); ++index) {
        uint64_t const id = particlesStarFormationStates[index].id;
        if (id == 0) {
            ++missing;
        } else if (!ids.insert(id).second) {
            throw std::logic_error("Duplicate initial star-formation identity");
        }
    }
    return {uint64_t(particlesOwnedSize()), missing};
}

static void particlesAssignStarFormationIds() {
    // Fixed scenarios initialize their full population on rank zero. Assign
    // source-order IDs there before the first migration, independently of MPI size.
    for (ParticleInteger index = 0; index < particlesOwnedSize(); ++index) {
        particlesStarFormationStates[index].id = uint64_t(index) + 1;
    }
}
HPX_PLAIN_ACTION(particlesStarFormationCount, ParticlesStarFormationCountAction);

void particlesInitializeStarFormationIds() {
    if (!getOptions().starFormation.enabled) {
        return;
    }
    if (hpxRank() != 0) {
        throw std::logic_error("Initialize star-formation IDs from rank zero only");
    }
    std::vector<hpx::future<Pair<uint64_t>>> counts;
    std::exception_ptr failure;
    try {
        counts.reserve(hpxLocalities().size());
        for (auto const& locality : hpxLocalities()) {
            counts.push_back(hpx::async<ParticlesStarFormationCountAction>(locality));
        }
    } catch (...) {
        failure = std::current_exception();
    }
    uint64_t total = 0;
    uint64_t missing = 0;
    hpx::wait_all(counts);
    for (auto& future : counts) {
        try {
            auto const count = future.get();
            if (count.first > std::numeric_limits<uint64_t>::max() - total) {
                throw std::overflow_error("Star-formation particle identity overflow");
            }
            total += count.first;
            missing += count.second;
        } catch (...) {
            if (!failure) {
                failure = std::current_exception();
            }
        }
    }
    if (failure) {
        std::rethrow_exception(failure);
    }
    if (missing == 0) {
        return;
    }
    if (missing != total || total != uint64_t(particlesOwnedSize()) ||
        (getOptions().run.scenario == "cosmology" && getOptions().run.test.empty())) {
        throw std::logic_error(
            "Distributed initial conditions must supply globally stable star-formation IDs");
    }
    particlesAssignStarFormationIds();
}

void particlesConvertToStar(ParticleInteger index, double formationTime) {
#if COSMICTIGER_WITH_SPH
    uint64_t timeBits;
    std::memcpy(&timeBits, &formationTime, sizeof(timeBits));
    if (!getOptions().starFormation.enabled || index < 0 || index >= particlesOwnedSize() ||
        particlesType(index) != kSphType ||
        (timeBits & 0x7ff0000000000000ULL) == 0x7ff0000000000000ULL || formationTime < 0.0 ||
        particlesStarFormationStates[index].id == 0) {
        throw std::invalid_argument("Invalid synchronized gas-to-star conversion");
    }
    sphParticlesRemove(particlesCatIndex(index));
    particlesType(index) = kStarType;
    particlesCatIndex(index) = kNoParticleIndex;
    auto& state = particlesStarFormationStates[index];
    state.formationTime = formationTime;
    state.specificStarFormationRate = 0.0;
#else
    throw std::logic_error("Star formation requires SPH");
#endif
}

void particlesRegisterReleaseHook(void (*hook)()) {
    particleReleaseHooks.push_back(hook);
}

void particlesGravitySamplesBegin(size_t sampleCount) {
    if (sampleCount > 0 && getOptions().gravity.errorInterval <= 0) {
        throw std::logic_error("Gravity sample capture requires gravity.errorInterval > 0");
    }
    if (sampleCount > size_t(std::numeric_limits<ParticleInteger>::max()) ||
        sampleCount > std::numeric_limits<size_t>::max() / sizeof(Vector3<float>)) {
        throw std::length_error("Gravity sample count exceeds the particle index range");
    }
    particlesEnsureGravitySampleStorage();
    Vector3<float>* accelerations = nullptr;
    if (sampleCount > 0) {
        accelerations = static_cast<Vector3<float>*>(
            allocateManagedMemory(sampleCount * sizeof(Vector3<float>)));
        if (accelerations == nullptr) {
            throw std::bad_alloc();
        }
        float const unset = std::numeric_limits<float>::quiet_NaN();
        for (size_t i = 0; i < sampleCount; ++i) {
            for (int dim = 0; dim < kDimensions; ++dim) {
                accelerations[i][dim] = unset;
            }
        }
    }
    freeManagedMemory(particlesGravitySampleAccelerations);
    particlesGravitySampleAccelerations = accelerations;
    if (particlesGravitySampleIndices != nullptr) {
        std::fill_n(particlesGravitySampleIndices, particlesSize(), ParticleInteger{-1});
    }
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
    freeManagedMemory(particlesGravitySampleIndices);
    freeManagedMemory(particlesGravitySampleAccelerations);
    particlesGravitySampleIndices = nullptr;
    particlesGravitySampleAccelerations = nullptr;
    freeManagedMemory(particlesStarFormationStates);
    particlesStarFormationStates = nullptr;
    if (getOptions().analysis.groups) {
        free(particlesLgrp);
    }
    if (saveParticleAcceleration(getOptions().analysis)) {
        for (int dim = 0; dim < kDimensions; dim++) {
            freeManagedMemory(particlesGravityAccelerations[dim]);
        }
    }
    if (particlesHasLongAcceleration(getOptions())) {
        for (int dim = 0; dim < kDimensions; dim++) {
            freeManagedMemory(particlesLongAccelerations[dim]);
            particlesLongAccelerations[dim] = nullptr;
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
        futs.push_back(hpx::async(hpx::annotated_function(
            [proc, numberOfThreads]() {
                ParticleInteger const begin = (size_t) proc * particlesSize() / numberOfThreads;
                ParticleInteger const end = (size_t) (proc + 1) * particlesSize() / numberOfThreads;
                int const seed = 4321 * (hpxRank() * numberOfThreads + proc) + 42;
#if COSMICTIGER_WITH_GRAVITY
                gsl_rng* rndgen = gsl_rng_alloc(gsl_rng_taus);
                gsl_rng_set(rndgen, seed);
#else
                std::mt19937_64 generator(seed);
#endif
                for (ParticleInteger i = begin; i < end; i++) {
                    for (int dim = 0; dim < kDimensions; dim++) {
#if COSMICTIGER_WITH_GRAVITY
                        particlesPos(dim, i) = gsl_rng_uniform(rndgen);
#else
                        particlesPos(dim, i) = std::generate_canonical<double, 53>(generator);
#endif
                        particlesVel(dim, i) = 0.0f;
                    }
                    particlesRung(i) = 0;
                }
#if COSMICTIGER_WITH_GRAVITY
                gsl_rng_free(rndgen);
#endif
            },
            "particles.randomInit")));
    }
    hpx::wait_all(futs.begin(), futs.end());
}

Pair<Vector3<double>, double> particlesEnclosingSphere(Pair<ParticleInteger> rng) {
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
        futs1.push_back(
            hpx::async(hpx::annotated_function(func1, "particles.enclosingSphereBounds"), proc));
    }
    auto box = func1(0);
    for (auto& fut : futs1) {
        auto thisBox = fut.get();
        for (int dim = 0; dim < kDimensions; dim++) {
            box.begin[dim] = std::min(box.begin[dim], thisBox.begin[dim]);
            box.end[dim] = std::max(box.end[dim], thisBox.end[dim]);
        }
    }
    Vector3<double> xc;
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
        futs2.push_back(
            hpx::async(hpx::annotated_function(func2, "particles.enclosingSphereRadius"), proc));
    }
    auto rmax = func2(0);
    for (auto& fut : futs2) {
        rmax = std::max(rmax, fut.get());
    }
    Pair<Vector3<double>, double> rc;
    rc.first = xc;
    rc.second = rmax;
    return rc;
}

ParticleInteger particlesSort(Pair<ParticleInteger> rng, double xm, int xdim) {
    addCpuFlops(1, FlopPrecision::Double);
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

#if COSMICTIGER_WITH_GRAVITY
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
#endif

namespace detail {

constexpr uint64_t kStarFormationParticleMagic = 0x4354534650415254ULL;
constexpr uint64_t kStarFormationParticleVersion = 2;

bool finiteStarFormationScalar(double value) {
    uint64_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & 0x7ff0000000000000ULL) != 0x7ff0000000000000ULL;
}

void loadStarFormationParticles(FILE* fp) {
    uint64_t magic, version;
    FREAD(&magic, sizeof(magic), 1, fp);
    FREAD(&version, sizeof(version), 1, fp);
    if (magic != kStarFormationParticleMagic || version != kStarFormationParticleVersion) {
        THROW_ERROR("Unsupported star-formation particle checkpoint layout\n");
    }
    std::unordered_set<uint64_t> ids;
    for (ParticleInteger index = 0; index < particlesSize(); ++index) {
        auto& state = particlesStarFormationStates[index];
        FREAD(&state.id, sizeof(state.id), 1, fp);
        FREAD(&state.eventCounter, sizeof(state.eventCounter), 1, fp);
        FREAD(&state.formationTime, sizeof(state.formationTime), 1, fp);
        FREAD(&state.specificStarFormationRate, sizeof(state.specificStarFormationRate), 1, fp);
        FREAD(&state.elapsedPhysicalTime, sizeof(state.elapsedPhysicalTime), 1, fp);
        int const type = particlesType(index);
        if (state.id == 0 || !ids.insert(state.id).second ||
            !finiteStarFormationScalar(state.formationTime) ||
            !finiteStarFormationScalar(state.specificStarFormationRate) ||
            !finiteStarFormationScalar(state.elapsedPhysicalTime) ||
            state.specificStarFormationRate < 0.0 || state.elapsedPhysicalTime < 0.0 ||
            (type != kDarkMatterType && type != kSphType && type != kStarType) ||
            ((type == kStarType) != (state.formationTime >= 0.0)) ||
            (type != kSphType && state.specificStarFormationRate != 0.0)) {
            THROW_ERROR("Invalid star-formation particle checkpoint metadata\n");
        }
    }
}

void saveStarFormationParticles(FILE* fp) {
    if (fwrite(&kStarFormationParticleMagic, sizeof(uint64_t), 1, fp) != 1 ||
        fwrite(&kStarFormationParticleVersion, sizeof(uint64_t), 1, fp) != 1) {
        THROW_ERROR("Unable to write star-formation particle checkpoint header\n");
    }
    for (ParticleInteger index = 0; index < particlesSize(); ++index) {
        auto const& state = particlesStarFormationStates[index];
        if (fwrite(&state.id, sizeof(state.id), 1, fp) != 1 ||
            fwrite(&state.eventCounter, sizeof(state.eventCounter), 1, fp) != 1 ||
            fwrite(&state.formationTime, sizeof(state.formationTime), 1, fp) != 1 ||
            fwrite(&state.specificStarFormationRate, sizeof(state.specificStarFormationRate), 1,
                fp) != 1 ||
            fwrite(&state.elapsedPhysicalTime, sizeof(state.elapsedPhysicalTime), 1, fp) != 1) {
            THROW_ERROR("Unable to write star-formation particle checkpoint metadata\n");
        }
    }
}

}    // namespace detail

void particlesLoad(FILE* fp) {
    ParticleInteger size;
    FREAD(&size, sizeof(ParticleInteger), 1, fp);
#if COSMICTIGER_WITH_SPH
    if (getOptions().hydro.enabled) {
        ParticleInteger sphSize = 0;
        FREAD(&sphSize, sizeof(ParticleInteger), 1, fp);
        PRINT("Reading %i total particles %i sph\n", size, sphSize);
        if (sphSize < 0 || sphSize > size) {
            THROW_ERROR("Invalid gas count in checkpoint\n");
        }
        particlesResize(size);
        sphParticlesResize(sphSize, false);
    } else
#endif
    {
        PRINT("Reading %i total particles.\n", size);
        particlesResize(size);
    }
    // Sample identities and captured forces describe only an in-progress force pass.
    // They are intentionally absent from checkpoints, including same-process reloads.
    particlesGravitySamplesBegin(0);
    PRINT("particles_size  = %i\n", particlesSize());
    FREAD(&particlesPos(kXDimension, 0), sizeof(Position), particlesSize(), fp);
    FREAD(&particlesPos(kYDimension, 0), sizeof(Position), particlesSize(), fp);
    FREAD(&particlesPos(kZDimension, 0), sizeof(Position), particlesSize(), fp);
    FREAD(particlesVelData(), sizeof(Vector3<float>), particlesSize(), fp);
    FREAD(&particlesRung(0), sizeof(char), particlesSize(), fp);
    if (getOptions().analysis.groups) {
        FREAD(&particlesLastgroup(0), sizeof(GroupInteger), particlesSize(), fp);
    }
    if (getOptions().analysis.tracers) {
        FREAD(&particlesTracer(0), sizeof(char), particlesSize(), fp);
    }
#if COSMICTIGER_WITH_SPH
    if (getOptions().hydro.enabled) {
        FREAD(&particlesType(0), sizeof(char), particlesSize(), fp);
        FREAD(&particlesCatIndex(0), sizeof(ParticleInteger), particlesSize(), fp);
        sphParticlesLoad(fp);
    }
#endif
    if (getOptions().starFormation.enabled) {
        detail::loadStarFormationParticles(fp);
    }
    int rungCount = 0;
    FREAD(&rungCount, sizeof(rungCount), 1, fp);
    ALWAYS_ASSERT(rungCount >= 0 && rungCount <= kMaxRung);
    rungBegins.resize(rungCount);
    FREAD(&rungBegin, sizeof(ParticleInteger), 1, fp);
    FREAD(rungBegins.data(), sizeof(ParticleInteger), rungBegins.size(), fp);
#if !COSMICTIGER_WITH_GRAVITY
    // Force diagnostics are absent from the checkpoint, including same-process reloads.
    particlesClearGravityDiagnostics(0, particlesSize());
#endif
}

void particlesSave(FILE* fp) {
    ParticleInteger size = particlesSize();
    fwrite(&size, sizeof(ParticleInteger), 1, fp);
#if COSMICTIGER_WITH_SPH
    if (getOptions().hydro.enabled) {
        ParticleInteger const sphSize = sphParticlesSize();
        fwrite(&sphSize, sizeof(ParticleInteger), 1, fp);
    }
#endif
    fwrite(&particlesPos(kXDimension, 0), sizeof(Position), particlesSize(), fp);
    fwrite(&particlesPos(kYDimension, 0), sizeof(Position), particlesSize(), fp);
    fwrite(&particlesPos(kZDimension, 0), sizeof(Position), particlesSize(), fp);
    fwrite(particlesVelData(), sizeof(Vector3<float>), particlesSize(), fp);
    fwrite(&particlesRung(0), sizeof(char), particlesSize(), fp);
    if (getOptions().analysis.groups) {
        fwrite(&particlesLastgroup(0), sizeof(GroupInteger), particlesSize(), fp);
    }
    if (getOptions().analysis.tracers) {
        fwrite(&particlesTracer(0), sizeof(char), particlesSize(), fp);
    }
#if COSMICTIGER_WITH_SPH
    if (getOptions().hydro.enabled) {
        fwrite(&particlesType(0), sizeof(char), particlesSize(), fp);
        fwrite(&particlesCatIndex(0), sizeof(ParticleInteger), particlesSize(), fp);
        sphParticlesSave(fp);
    }
#endif
    if (getOptions().starFormation.enabled) {
        detail::saveStarFormationParticles(fp);
    }
    int const rungCount = rungBegins.size();
    fwrite(&rungCount, sizeof(rungCount), 1, fp);
    fwrite(&rungBegin, sizeof(ParticleInteger), 1, fp);
    fwrite(rungBegins.data(), sizeof(ParticleInteger), rungBegins.size(), fp);
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

#if COSMICTIGER_WITH_SPH
void particlesResolveWithSphParticles() {
    int const numberOfThreads = hpxHardwareConcurrency();
    std::vector<hpx::future<void>> futs;
    for (int proc = 0; proc < numberOfThreads; proc++) {
        futs.push_back(hpx::async(hpx::annotated_function(
            [proc, numberOfThreads] {
                ParticleInteger const b = (size_t) proc * particlesSize() / numberOfThreads;
                ParticleInteger const e = (size_t) (proc + 1) * particlesSize() / numberOfThreads;
                for (ParticleInteger i = b; i < e; i++) {
                    int const type = particlesType(i);
                    if (type == kSphType) {
                        ParticleInteger const index = particlesCatIndex(i);
                        sphParticlesParticleIndex(index) = i;
                    }
                }
            },
            "particles.resolveSphIndices")));
    }
    hpx::wait_all(futs.begin(), futs.end());
}

#endif
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

static Vector3<float> const* particlesCacheReadLineVels(LineIdType lineId) {
    return readParticleCacheLine<ParticlesFetchCacheLineVelsAction>(
        lineId, particleVelocityCache, velsMutexes);
}

static Vector<Vector3<float>> particlesFetchCacheLineVels(ParticleInteger index) {
    static bool const sph = getOptions().hydro.enabled;
    ParticleInteger const lineSize = getOptions().compute.particleCacheLineSize;
    Vector<Vector3<float>> line(lineSize);
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
        futs.push_back(hpx::async(hpx::annotated_function(
            [numberOfThreads, proc]() {
                EnergiesT energies;
                ParticleInteger const b = (size_t) proc * particlesSize() / numberOfThreads;
                ParticleInteger const e = (size_t) (proc + 1) * particlesSize() / numberOfThreads;
                for (ParticleInteger i = b; i != e; i++) {
#if COSMICTIGER_WITH_SPH
                    if (getOptions().hydro.method != "sph" && particlesType(i) == kSphType) {
                        auto const& state = sphParticlesHopkinsState(particlesCatIndex(i));
                        if (state.initialized) {
                            double momentumSquared = 0.0;
                            for (int dim = 0; dim < kDimensions; dim++) {
                                double const component = state.accountedMomentum(dim);
                                momentumSquared += component * component;
                            }
                            double const mass = state.accountedMass();
                            double const kinetic = 0.5 * momentumSquared / mass;
                            if (saveParticlePotential(getOptions().analysis)) {
                                energies.pot += 0.5 * mass * particlesPot(i);
                            }
                            energies.kin += kinetic;
                            energies.therm += state.accountedEnergy() - kinetic;
                            energies.xmom += state.accountedMomentum(0);
                            energies.ymom += state.accountedMomentum(1);
                            energies.zmom += state.accountedMomentum(2);
                            energies.nmom += std::sqrt(momentumSquared);
                            continue;
                        }
                    }
#endif
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
#if COSMICTIGER_WITH_SPH
                    if (particlesType(i) == kSphType) {
                        int const kk = particlesCatIndex(i);
                        energies.therm += m * sphParticlesInternalEnergy(kk);
                    }
#endif
                }
                return energies;
            },
            "particles.sumEnergies")));
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
    auto fut1 = hpx::async(hpx::annotated_function(func1, "particles.getLocalPositions"));
    func2();
    fut1.get();
    return rc;
}

HPX_PLAIN_ACTION(particlesGetLocal, ParticlesGetLocalAction);

hpx::future<Vector<Position>> particlesGet(int rank, Vector<Pair<ParticleInteger>> const& ranges) {
    return hpx::async<ParticlesGetLocalAction>(hpxLocalities()[rank], ranges);
}

namespace detail {
GadgetIoHeader readGadgetHeader(FILE* fp, std::string const& filename) {
    int marker = 0;
    FREAD(&marker, sizeof(marker), 1, fp);
    if (marker != sizeof(GadgetIoHeader)) {
        THROW_ERROR(
            "Unsupported Gadget header layout in %s (expected NTYPES=2)\n", filename.c_str());
    }
    GadgetIoHeader header{};
    FREAD(&header, sizeof(header), 1, fp);
    FREAD(&marker, sizeof(marker), 1, fp);
    if (marker != sizeof(header) || header.npart[0] != 0 || header.npartTotal[0] != 0 ||
        header.npart[1] < 0 || header.npartTotal[1] <= 0 ||
        header.npart[1] > std::numeric_limits<ParticleInteger>::max() ||
        header.npart[1] > header.npartTotal[1] || header.numFiles != hpxSize() ||
        (header.numFiles == 1 && header.npart[1] != header.npartTotal[1]) ||
        !std::isfinite(header.mass[1]) || header.mass[1] <= 0.0 || !std::isfinite(header.boxSize) ||
        header.boxSize <= 0.0 || !std::isfinite(header.redshift) || header.redshift <= -1.0) {
        THROW_ERROR("Unsupported Gadget snapshot in %s: require finite DM-only constant-mass "
                    "data and one file per locality\n",
            filename.c_str());
    }
    return header;
}
}    // namespace detail

GadgetIoHeader particlesReadGadget4Header(std::string const& filenameBase) {
    std::string const filename = filenameBase + (hpxSize() > 1 ? ".0" : "");
    UniqueFile file(fopen(filename.c_str(), "rb"));
    if (!file) {
        THROW_ERROR("Unable to open %s for reading.\n", filename.c_str());
    }
    return detail::readGadgetHeader(file.get(), filename);
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
        THROW_ERROR("Unable to open %s for reading.\n", fname.c_str());
    }
    int size;
    auto const header = detail::readGadgetHeader(fp, fname);
    auto checkBlock = [&] {
        FREAD(&size, sizeof(size), 1, fp);
        uint64_t const bytes = uint64_t(header.npart[1]) * 3 * sizeof(float);
        if (size < 0 || uint64_t(size) != bytes) {
            THROW_ERROR("Invalid Gadget position/velocity block in %s\n", fname.c_str());
        }
    };
    auto const numberOfParticles = header.npart[1];
    particlesResize(numberOfParticles);

    checkBlock();
    for (ParticleInteger i = 0; i < numberOfParticles; i++) {
        for (int dim = 0; dim < kDimensions; dim++) {
            float pos;
            FREAD(&pos, sizeof(float), 1, fp);
            pos /= header.boxSize;
            particlesPos(dim, i) = pos;
        }
    }
    checkBlock();

    checkBlock();
    for (ParticleInteger i = 0; i < numberOfParticles; i++) {
        for (int dim = 0; dim < kDimensions; dim++) {
            float vel;
            FREAD(&vel, sizeof(float), 1, fp);
            particlesVel(dim, i) = vel;
        }
    }
    auto const a = 1.0 / (header.redshift + 1.0);
    for (ParticleInteger i = 0; i < numberOfParticles; i++) {
        //	PRINT( "%e %e %e\n", particles_pos(XDIM,i).to_double(),
        // particles_pos(YDIM,i).to_double(), particles_pos(ZDIM,i).to_double());
        for (int dim = 0; dim < kDimensions; dim++) {
            particlesVel(dim, i) *=
                sqrt(a) * a * 1e5 * getOptions().units.timeSeconds / getOptions().units.lengthCm;
        }
    }
    checkBlock();

    fclose(fp);
    hpx::wait_all(futs.begin(), futs.end());
    for (auto& future : futs) {
        future.get();
    }
    return header;
}
