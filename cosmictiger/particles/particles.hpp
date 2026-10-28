/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#ifndef PARTICLES_HPP_
#define PARTICLES_HPP_

#include <atomic>
#include <cstdint>
#include <fstream>
#include <unordered_map>

#include <cosmictiger/analysis/group_entry.hpp>
#include <cosmictiger/core/containers.hpp>
#include <cosmictiger/core/defs.hpp>
#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/cuda/cuda.hpp>
#include <cosmictiger/math/position.hpp>
#include <cosmictiger/math/range.hpp>

static
#ifdef __CUDACC__
    __constant__
#endif
    float rungDt[kMaxRung] = {1.0 / (1 << 0), 1.0 / (1 << 1), 1.0 / (1 << 2), 1.0 / (1 << 3),
        1.0 / (1 << 4), 1.0 / (1 << 5), 1.0 / (1 << 6), 1.0 / (1 << 7), 1.0 / (1 << 8),
        1.0 / (1 << 9), 1.0 / (1 << 10), 1.0 / (1 << 11), 1.0 / (1 << 12), 1.0 / (1 << 13),
        1.0 / (1 << 14), 1.0 / (1 << 15)};

inline constexpr int kDarkMatterType = 0;
inline constexpr int kSphType = 1;

#ifdef LONG_LONG_PART_INT
using ParticleInteger = long long;
#else
using ParticleInteger = int;
#endif

inline constexpr ParticleInteger kNoParticleIndex = static_cast<ParticleInteger>(0xFFFFFFFFU);

inline constexpr GroupInteger kNoParticleGroup = static_cast<GroupInteger>(0x7FFFFFFFFFFFFFFFLL);

#ifdef PARTICLES_CPP
#define PARTICLES_EXTERN
#else
#define PARTICLES_EXTERN extern
#endif

struct GroupParticle {
    Array<Position, kDimensions> x;
    GroupInteger g;
    template <class A>
    void serialize(A&& arc, unsigned) {
        arc & x;
        arc & g;
    }
};

struct OutputParticle {
    Array<Position, kDimensions> x;
    Array<float, kDimensions> v;
    char r;
    template <class A>
    void serialize(A&& a, unsigned) {
        a & x;
        a & v;
        a & r;
    }
};

inline constexpr int kParticleHeaderTypeCount = 2;

/** Copied from GADGET-4. This declaration retains upstream GPL terms;
 * see THIRD_PARTY_NOTICES.md for source attribution and scope. */
struct GadgetIoHeader {
    long long npart[kParticleHeaderTypeCount]; /**< number of particles of each type in this file */
    long long
        npartTotal[kParticleHeaderTypeCount]; /**< total number of particles of each type in this
snapshot. This can be different from npart if one is dealing with a multi-file snapshot. */
    double
        mass[kParticleHeaderTypeCount]; /**< mass of particles of each type. If 0, then the masses
  are explicitly stored in the mass-block of the snapshot file, otherwise they are omitted */
    double time;                /**< time of snapshot file */
    double redshift;            /**< redshift of snapshot file */
    double boxSize;             /**< box-size of simulation in case periodic boundaries were used */
    int numFiles;               /**< number of files in multi-file snapshot */

    long long ntrees;
    long long totNtrees;

    template <class A>
    void serialize(A&& arc, unsigned) {
        for (int i = 0; i < kParticleHeaderTypeCount; i++) {
            arc& npart[i];
            arc& npartTotal[i];
            arc& mass[i];
        }
        arc & time;
        arc & redshift;
        arc & boxSize;
        arc & numFiles;
    }
};
/*****************************/

struct Particle {
    Array<Position, kDimensions> x;
    Array<float, kDimensions> v;
    GroupInteger lg;
    ParticleInteger catIndex;
    char r;
    char t;
    char type;
    ParticleInteger gravitySample = -1;
    template <class A>
    void serialize(A&& a, unsigned) {
        static bool doGroups = getOptions().analysis.groups;
        static bool doTracers = getOptions().analysis.tracers;
        for (int dim = 0; dim < kDimensions; dim++) {
            a& x[dim];
            a& v[dim];
        }
        a & r;
        if (doGroups) {
            a & lg;
        }
        if (doTracers) {
            a & t;
        }
        if (getOptions().gravity.errorInterval > 0) {
            a & gravitySample;
        }
        if (getOptions().hydro.enabled) {
            a & catIndex;
            a & type;
        }
    }
};

struct ParticleSample {
    Array<Position, kDimensions> x;
    Array<float, kDimensions> g;
    float p;
    template <class A>
    void serialize(A&& a, unsigned) {
        for (int dim = 0; dim < kDimensions; dim++) {
            a& x[dim];
            a& g[dim];
        }
        a & p;
    }
};

#ifdef CHECK_BOUNDS
#define CHECK_PART_BOUNDS(i)                                                                     \
    if (i < 0 || i >= particlesSize()) {                                                         \
        PRINT("particle bound check failure %li should be between %li and %li\n", (long long) i, \
            (long long) 0, (long long) particlesSize());                                         \
        ALWAYS_ASSERT(false);                                                                    \
    }
#else
#define CHECK_PART_BOUNDS(i)
#endif

PARTICLES_EXTERN Array<Position*, kDimensions> particlesX;
PARTICLES_EXTERN Array<float, kDimensions>* particlesV;
PARTICLES_EXTERN char* particlesR;
PARTICLES_EXTERN Array<float*, kDimensions> particlesGravityAccelerations;
PARTICLES_EXTERN float* particlesP;
PARTICLES_EXTERN std::atomic<GroupInteger>* particlesGrp
#ifdef PARTICLES_CPP
    = nullptr
#endif
    ;
PARTICLES_EXTERN ParticleInteger* particlesSph;
PARTICLES_EXTERN GroupInteger* particlesLgrp;
PARTICLES_EXTERN char* particlesTr;
PARTICLES_EXTERN char* particlesTy;
// Transient gravity-check attributes: -1 is unselected, otherwise the global sample slot.
PARTICLES_EXTERN ParticleInteger* particlesGravitySampleIndices;
PARTICLES_EXTERN Array<float, kDimensions>* particlesGravitySampleAccelerations;
PARTICLES_EXTERN size_t particlesGlobalOffset;

struct ParticleGlobalRange {
    int proc;
    Pair<ParticleInteger> range;
};

ParticleInteger particlesSize();
// Source ghosts use a separate arena and never change the owned population.
ParticleInteger particlesOwnedSize();
// Cumulative position/type cache misses requiring a remote particle action.
uint64_t particlesRemoteFetchCount();
std::unordered_map<int, ParticleInteger> particlesGroupsInit();
void particlesGroupsDestroy();
void particlesResize(ParticleInteger, bool lock = true);
void particlesRandomInit();
#if COSMICTIGER_WITH_SPH
void particlesResolveWithSphParticles();
#endif
void particlesPopRungs();
void particlesPushRungs();
void particlesDisplace(double dx, double dy, double dz);
void particlesDestroy();
void particlesSortBySph(Pair<ParticleInteger> rng);
void particlesGlobalReadPos(
    ParticleGlobalRange, Position* x, Position* y, Position* z, char* type, ParticleInteger offset);
inline void particlesGlobalReadPos(
    ParticleGlobalRange range, Position* x, Position* y, Position* z, ParticleInteger offset) {
    particlesGlobalReadPos(range, x, y, z, nullptr, offset);
}
void particlesGlobalReadPosAndGroup(ParticleGlobalRange range, Position* x, Position* y,
    Position* z, GroupInteger* g, ParticleInteger offset);
ParticleInteger particlesSort(Pair<ParticleInteger> rng, double xm, int xdim);
// Swap all attached fields without reading uninitialized saved-force values.
void particlesSwap(ParticleInteger first, ParticleInteger second);
GadgetIoHeader particlesReadGadget4(std::string);
#ifndef __CUDACC__
hpx::future<Vector<Position>> particlesGet(int rank, Vector<Pair<ParticleInteger>> const& ranges);
#endif
void particlesCacheFree();
// Local half of particlesCacheFree; requires a synchronized reorder boundary.
void particlesCacheFreeLocal();
void particlesGroupCacheFree();
Pair<Array<double, kDimensions>, double> particlesEnclosingSphere(Pair<ParticleInteger> rng);
Vector<OutputParticle> particlesGetSample(Range<double> const& box);
Vector<ParticleSample> particlesSample(int cnt);
void particlesLoad(FILE* fp);
void particlesSave(FILE* fp);
void particlesIncGroupCacheEpoch();
void particlesGlobalReadRungs(ParticleGlobalRange range, char* r, ParticleInteger offset);
int particlesGroupHome(GroupInteger);
#ifndef __CUDACC__
SharedMutexType& particlesSharedMutex();
#endif
void particlesSetTracers(size_t count = 0);
Vector<OutputParticle> particlesGetTracers();
// Local operation. A zero count disables capture and releases the compact result buffer.
void particlesGravitySamplesBegin(size_t sampleCount);
void particlesMemadviseCpu();
void particlesGlobalReadVels(
    ParticleGlobalRange range, float* vx, float* vy, float* z, ParticleInteger offset);
void particlesMemadviseGpu();
Vector<size_t> particlesRungCounts();
void particlesSetMinrung(int minrung);
void particlesFree();
// Hooks run before owned storage is released; register each hook once.
void particlesRegisterReleaseHook(void (*hook)());
void particlesSaveGlass(char const* filename);
void particlesSortByRung(int minrung);
double particlesActivePct();
Pair<ParticleInteger, ParticleInteger> particlesCurrentRange();

inline char& particlesType(ParticleInteger index) {
    static bool const sph = getOptions().hydro.enabled;
    if (getOptions().hydro.enabled) {
        CHECK_PART_BOUNDS(index);
        return particlesTy[index];
    } else {
        static char darkMatterType = kDarkMatterType;
        return darkMatterType;
    }
}

inline float particlesMass(ParticleInteger index) {
    if (!getOptions().hydro.enabled) {
        return 1.0f;
    }
    float const sphMass = getOptions().initial.gasMass;
    float const darkMatterMass = getOptions().initial.darkMatterMass;
    if (particlesType(index) != kDarkMatterType) {
        return sphMass;
    } else {
        return darkMatterMass;
    }
}

inline float& particlesPot(ParticleInteger index) {
    CHECK_PART_BOUNDS(index);
    return particlesP[index];
}

inline Position& particlesPos(int dim, ParticleInteger index) {
    CHECK_PART_BOUNDS(index);
    return particlesX[dim][index];
}

inline float& particlesVel(int dim, ParticleInteger index) {
    CHECK_PART_BOUNDS(index);
    return particlesV[index][dim];
}

inline Array<float, kDimensions>* particlesVelData() {
    return particlesV;
}

inline char& particlesRung(ParticleInteger index) {
    CHECK_PART_BOUNDS(index);
    return particlesR[index];
}

inline float& particlesGravityAcceleration(int dim, ParticleInteger index) {
    CHECK_PART_BOUNDS(index);
    return particlesGravityAccelerations[dim][index];
}

inline GroupInteger particlesGroupInit(ParticleInteger index) {
    CHECK_PART_BOUNDS(index);
    return particlesGlobalOffset + index;
}

inline std::atomic<GroupInteger>& particlesGroup(ParticleInteger index) {
    CHECK_PART_BOUNDS(index);
    ASSERT(particlesGrp);
    return particlesGrp[index];
}

inline GroupInteger& particlesLastgroup(ParticleInteger index) {
    CHECK_PART_BOUNDS(index);
    ASSERT(particlesLgrp);
    return particlesLgrp[index];
}

inline char& particlesTracer(ParticleInteger index) {
    CHECK_PART_BOUNDS(index);
    return particlesTr[index];
}

inline ParticleInteger& particlesGravitySample(ParticleInteger index) {
    CHECK_PART_BOUNDS(index);
    return particlesGravitySampleIndices[index];
}

inline void particlesCaptureGravitySample(ParticleInteger index, float gx, float gy, float gz) {
    if (particlesGravitySampleAccelerations != nullptr) {
        ParticleInteger const slot = particlesGravitySample(index);
        if (slot >= 0) {
            particlesGravitySampleAccelerations[slot][kXDimension] = gx;
            particlesGravitySampleAccelerations[slot][kYDimension] = gy;
            particlesGravitySampleAccelerations[slot][kZDimension] = gz;
        }
    }
}

inline ParticleInteger& particlesCatIndex(ParticleInteger index) {
    CHECK_PART_BOUNDS(index);
    return particlesSph[index];
}

inline Particle particlesGetParticle(ParticleInteger index) {
    static bool doGroups = getOptions().analysis.groups;
    static bool doTracers = getOptions().analysis.tracers;
    CHECK_PART_BOUNDS(index);
    Particle p;
    for (int dim = 0; dim < kDimensions; dim++) {
        p.x[dim] = particlesPos(dim, index);
        p.v[dim] = particlesVel(dim, index);
    }
    p.r = particlesRung(index);
    if (doGroups) {
        p.lg = particlesLastgroup(index);
    }
    if (doTracers) {
        p.t = particlesTracer(index);
    }
    if (getOptions().gravity.errorInterval > 0) {
        p.gravitySample = particlesGravitySample(index);
    }
    if (getOptions().hydro.enabled) {
        p.type = particlesType(index);
        p.catIndex = particlesCatIndex(index);
    }
    return p;
}

inline void particlesSetParticle(Particle p, ParticleInteger index) {
    static bool doGroups = getOptions().analysis.groups;
    static bool doTracers = getOptions().analysis.tracers;
    CHECK_PART_BOUNDS(index);
    for (int dim = 0; dim < kDimensions; dim++) {
        particlesPos(dim, index) = p.x[dim];
        particlesVel(dim, index) = p.v[dim];
    }
    particlesRung(index) = p.r;
    if (doGroups) {
        particlesLastgroup(index) = p.lg;
    }
    if (doTracers) {
        particlesTracer(index) = p.t;
    }
    if (getOptions().gravity.errorInterval > 0) {
        particlesGravitySample(index) = p.gravitySample;
    }
    if (getOptions().hydro.enabled) {
        particlesType(index) = p.type;
        particlesCatIndex(index) = p.catIndex;
    }
}

inline bool particlesIsSph(int index) {
    return particlesType(index) == kSphType;
}

struct EnergiesT {
    double therm;
    double heating;
    double pot;
    double kin;
    double tckin;
    double xmom;
    double ymom;
    double cosmic;
    double zmom;
    double nmom;
    EnergiesT() {
        therm = heating = 0.0;
        pot = kin = 0.f;
        xmom = ymom = zmom = nmom = 0.0;
        cosmic = 0.0;
        tckin = 0.0;
    }
    EnergiesT& operator+=(EnergiesT const& other) {
        therm += other.therm;
        heating += other.heating;
        pot += other.pot;
        kin += other.kin;
        cosmic += other.cosmic;
        xmom += other.xmom;
        ymom += other.ymom;
        zmom += other.zmom;
        nmom += other.nmom;
        tckin += other.tckin;
        return *this;
    }
    template <class A>
    void serialize(A&& arc, unsigned) {
        arc & xmom;
        arc & ymom;
        arc & cosmic;
        arc & zmom;
        arc & nmom;
        arc & pot;
        arc & kin;
        arc & tckin;
        if (getOptions().hydro.enabled) {
            arc & therm;
            arc & heating;
        }
    }
};

EnergiesT particlesSumEnergies();

#endif /* PARTICLES_HPP_ */
