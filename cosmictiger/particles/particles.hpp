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
#include <format>
#include <fstream>
#include <iostream>
#include <type_traits>
#include <unordered_map>

#include <cosmictiger/analysis/group_entry.hpp>
#include <cosmictiger/core/containers.hpp>
#include <cosmictiger/core/defs.hpp>
#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/cuda/cuda.hpp>
#include <cosmictiger/math/position.hpp>
#include <cosmictiger/math/range.hpp>
#include <cosmictiger/math/vector3.hpp>
#include <cosmictiger/particles/particle_types.hpp>
#include <cosmictiger/stars/star_formation_state.hpp>

static
#ifdef __CUDACC__
    __constant__
#endif
    float rungDt[kMaxRung] = {1.0 / (1 << 0), 1.0 / (1 << 1), 1.0 / (1 << 2), 1.0 / (1 << 3),
        1.0 / (1 << 4), 1.0 / (1 << 5), 1.0 / (1 << 6), 1.0 / (1 << 7), 1.0 / (1 << 8),
        1.0 / (1 << 9), 1.0 / (1 << 10), 1.0 / (1 << 11), 1.0 / (1 << 12), 1.0 / (1 << 13),
        1.0 / (1 << 14), 1.0 / (1 << 15)};

#ifdef LONG_LONG_PART_INT
using ParticleInteger = long long;
#else
using ParticleInteger = int;
#endif

inline constexpr ParticleInteger kNoParticleIndex = ParticleInteger{-1};

inline constexpr GroupInteger kNoParticleGroup = static_cast<GroupInteger>(0x7FFFFFFFFFFFFFFFLL);

#ifdef PARTICLES_CPP
#define PARTICLES_EXTERN
#else
#define PARTICLES_EXTERN extern
#endif

// Native particle/checkpoint vectors contain exactly the active dimensions.
static_assert(sizeof(ParticleVector<float>) == sizeof(Array<float, kSpatialDimensions>));
static_assert(alignof(ParticleVector<float>) == alignof(Array<float, kSpatialDimensions>));
static_assert(sizeof(ParticleVector<Position>) == sizeof(Array<Position, kSpatialDimensions>));
static_assert(alignof(ParticleVector<Position>) == alignof(Array<Position, kSpatialDimensions>));
static_assert(std::is_trivially_copyable_v<ParticleVector<float>>);
static_assert(std::is_standard_layout_v<ParticleVector<float>>);
static_assert(std::is_trivially_copyable_v<ParticleVector<Position>>);
static_assert(std::is_standard_layout_v<ParticleVector<Position>>);

struct GroupParticle {
    ParticleVector<Position> position;
    GroupInteger groupNumber;
    template <class A>
    void serialize(A&& arc, unsigned) {
        arc & position;
        arc & groupNumber;
    }
};

struct OutputParticle {
    ParticleVector<Position> position;
    ParticleVector<float> velocity;
    char rung;
    template <class A>
    void serialize(A&& a, unsigned) {
        a & position;
        a & velocity;
        a & rung;
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
    double time;                        /**< time of snapshot file */
    double redshift;                    /**< redshift of snapshot file */
    double boxSize; /**< box-size of simulation in case periodic boundaries were used */
    int numFiles;   /**< number of files in multi-file snapshot */

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

// The mesh solvers (pm, p3m) store their gm-scaled long acceleration per particle.
inline bool particlesHasLongAcceleration(Options const& options) {
    return options.gravity.enabled && usesParticleMesh(options.gravity);
}

struct Particle {
    ParticleVector<Position> position;
    ParticleVector<float> velocity;
    ParticleVector<float> longAcceleration;
    GroupInteger groupNumber;
    ParticleInteger catalogIndex;
    char rung;
    char isTracer;
    char type;
    ParticleInteger gravitySample = -1;
    StarFormationState starFormation;
    template <class A>
    void serialize(A&& a, unsigned) {
        auto const& options = getOptions();
        static bool doGroups = options.analysis.groups;
        static bool doTracers = options.analysis.tracers;
        for (int dim = 0; dim < kSpatialDimensions; dim++) {
            a& position[dim];
            a& velocity[dim];
        }
        a & rung;
        if (doGroups) {
            a & groupNumber;
        }
        if (doTracers) {
            a & isTracer;
        }
        if (options.gravity.errorInterval > 0) {
            a & gravitySample;
        }
        if (options.hydro.enabled) {
            a & catalogIndex;
            a & type;
        }
        if (options.starFormation.enabled) {
            a & starFormation;
        }
        if (particlesHasLongAcceleration(options)) {
            for (int dim = 0; dim < kSpatialDimensions; dim++) {
                a& longAcceleration[dim];
            }
        }
    }
};

// A randomly selected particle's position and tree-computed gravity, gathered by
// particlesSample() for comparison against a direct-summation reference.
struct ParticleSample {
    // Particle position.
    ParticleVector<Position> position;
    // Gravitational acceleration from the last kick; zero unless the analysis options
    // save particle accelerations.
    ParticleVector<float> acceleration;
    // Gravitational potential from the last kick; zero unless the analysis options save
    // particle potentials.
    float potential;
    template <class A>
    void serialize(A&& a, unsigned) {
        for (int dim = 0; dim < kSpatialDimensions; dim++) {
            a& position[dim];
            a& acceleration[dim];
        }
        a & potential;
    }
};

#ifdef CHECK_BOUNDS
#define CHECK_PART_BOUNDS(i)                                                                    \
    if (i < 0 || i >= particlesSize()) {                                                        \
        std::cout << std::format(                                                               \
                         "particle bound check failure {:d} should be between {:d} and {:d}\n", \
                         (long long) i, (long long) 0, (long long) particlesSize())             \
                  << std::flush;                                                                \
        ALWAYS_ASSERT(false);                                                                   \
    }
#else
#define CHECK_PART_BOUNDS(i)
#endif

// Derived snapshot fields, gm-scaled like the canonical gravity RHS.
// Allocated only while output.siloTimeDerivatives output is being evaluated/written.
struct GravityDiagnostics {
    ParticleVector<float> acceleration{};
    ParticleVector<float> longAcceleration{};
    ParticleVector<float> nearAcceleration{};
    float potential{};
    float longPotential{};
    float nearPotential{};
};
PARTICLES_EXTERN GravityDiagnostics* particlesGravityDiagnostics;
void particlesAllocateGravityDiagnostics();
void particlesFreeGravityDiagnostics();

PARTICLES_EXTERN Array<Position*, kSpatialDimensions> particlesX;
PARTICLES_EXTERN ParticleVector<float>* particlesV;
PARTICLES_EXTERN char* particlesR;
PARTICLES_EXTERN Array<float*, kSpatialDimensions> particlesGravityAccelerations;
// Long-range (mesh) acceleration of the particle-mesh solvers, scaled by gm. It
// is produced at an outer PM boundary and consumed by the next kick, so it
// follows particle permutations and migration but is never checkpointed.
PARTICLES_EXTERN Array<float*, kSpatialDimensions> particlesLongAccelerations;
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
PARTICLES_EXTERN StarFormationState* particlesStarFormationStates;
// Transient gravity-check attributes: -1 is unselected, otherwise the global sample slot.
PARTICLES_EXTERN ParticleInteger* particlesGravitySampleIndices;
PARTICLES_EXTERN ParticleVector<float>* particlesGravitySampleAccelerations;
PARTICLES_EXTERN size_t particlesGlobalOffset;

struct ParticleGlobalRange {
    int proc;
    Pair<ParticleInteger> range;
    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & proc & range;
    }
};

ParticleInteger particlesSize();
// Source ghosts use a separate arena and never change the owned population.
ParticleInteger particlesOwnedSize();
// Cumulative position/type cache misses requiring a remote particle action.
uint64_t particlesRemoteFetchCount();
std::unordered_map<int, ParticleInteger> particlesGroupsInit();
void particlesGroupsDestroy();
void particlesResize(ParticleInteger, bool lock = true);
// Call once on rank zero after fresh initial conditions; restart preserves saved IDs.
void particlesInitializeStarFormationIds();
// Local conversion at a synchronized boundary with no outstanding force/cache readers.
void particlesConvertToStar(ParticleInteger index, double formationTime);
void particlesRandomInit();
#if COSMICTIGER_WITH_SPH
void particlesResolveWithSphParticles();
#endif
void particlesPopRungs();
void particlesPushRungs();
void particlesDisplace(double dx, double dy, double dz);
void particlesDestroy();
// Moves the dark-matter particles of a range to its front and returns the end
// of that run. Halo finding reads only the dark-matter run (DM-only linking).
ParticleInteger particlesPartitionDarkMatter(Pair<ParticleInteger> rng);
void particlesGlobalReadPos(
    ParticleGlobalRange, Position* x, Position* y, Position* z, char* type, ParticleInteger offset);
void particlesGlobalReadMasses(ParticleGlobalRange range, float* masses, ParticleInteger offset);
inline void particlesGlobalReadPos(
    ParticleGlobalRange range, Position* x, Position* y, Position* z, ParticleInteger offset) {
    particlesGlobalReadPos(range, x, y, z, nullptr, offset);
}
void particlesGlobalReadPosAndGroup(ParticleGlobalRange range, Position* x, Position* y,
    Position* z, GroupInteger* groupNumbers, ParticleInteger offset);
ParticleInteger particlesSort(Pair<ParticleInteger> rng, double xm, int xdim);
// Swap all attached fields without reading uninitialized saved-force values.
void particlesSwap(ParticleInteger first, ParticleInteger second);
// Local-only temporary geometric sorting. No migration, owned-population change,
// or gas sorting may intervene. Ending restores every base-particle field and gas inverse
// index to its original order, leaving persisted hierarchical rung ranges valid.
void particlesBeginTemporarySort();
void particlesEndTemporarySort();
GadgetIoHeader particlesReadGadget4Header(std::string const&);
GadgetIoHeader particlesReadGadget4(std::string);
#ifndef __CUDACC__
hpx::future<Vector<Position>> particlesGet(int rank, Vector<Pair<ParticleInteger>> const& ranges);
#endif
void particlesCacheFree();
// Local half of particlesCacheFree; requires a synchronized reorder boundary.
void particlesCacheFreeLocal();
void particlesGroupCacheFree();
Pair<Vector3<double>, double> particlesEnclosingSphere(Pair<ParticleInteger> rng);
Vector<OutputParticle> particlesGetSample(Range<double> const& box);
#if COSMICTIGER_WITH_GRAVITY
Vector<ParticleSample> particlesSample(int cnt);
#endif
void particlesLoad(FILE* fp);
void particlesSave(FILE* fp);
void particlesIncGroupCacheEpoch();
void particlesGlobalReadRungs(ParticleGlobalRange range, char* r, ParticleInteger offset);
int particlesGroupHome(GroupInteger);
#ifndef __CUDACC__
SharedMutexType& particlesSharedMutex();
#endif
void particlesSetTracers(size_t count = 0, size_t totalCount = 0);
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
    ALWAYS_ASSERT(dim >= 0 && dim < kSpatialDimensions);
    return particlesX[dim][index];
}

inline float& particlesVel(int dim, ParticleInteger index) {
    CHECK_PART_BOUNDS(index);
    ALWAYS_ASSERT(dim >= 0 && dim < kSpatialDimensions);
    return particlesV[index][dim];
}

// Cartesian adapters are for three-axis geometry/visualization scratch only;
// inactive coordinates and velocities have no particle allocation.
inline Position particlesPositionValue(int dim, ParticleInteger index) {
    return dim < kSpatialDimensions ? particlesPos(dim, index) : Position(0.5);
}
inline float particlesVelocityValue(int dim, ParticleInteger index) {
    return dim < kSpatialDimensions ? particlesVel(dim, index) : 0.0f;
}
inline Position* particlesPositionData(int dim, ParticleInteger offset = 0) {
    return dim < kSpatialDimensions ? particlesX[dim] + offset : nullptr;
}

inline ParticleVector<float>* particlesVelData() {
    return particlesV;
}

inline char& particlesRung(ParticleInteger index) {
    CHECK_PART_BOUNDS(index);
    return particlesR[index];
}

inline float& particlesLongAcceleration(int dim, ParticleInteger index) {
    CHECK_PART_BOUNDS(index);
    return particlesLongAccelerations[dim][index];
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
            particlesGravitySampleAccelerations[slot][kXi] = gx;
            if constexpr (kSpatialDimensions > 1) {
                particlesGravitySampleAccelerations[slot][kYi] = gy;
            }
            if constexpr (kSpatialDimensions > 2) {
                particlesGravitySampleAccelerations[slot][kZi] = gz;
            }
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
    for (int dim = 0; dim < kSpatialDimensions; dim++) {
        p.position[dim] = particlesPos(dim, index);
        p.velocity[dim] = particlesVel(dim, index);
    }
    p.rung = particlesRung(index);
    if (doGroups) {
        p.groupNumber = particlesLastgroup(index);
    }
    if (doTracers) {
        p.isTracer = particlesTracer(index);
    }
    if (getOptions().gravity.errorInterval > 0) {
        p.gravitySample = particlesGravitySample(index);
    }
    if (getOptions().hydro.enabled) {
        p.type = particlesType(index);
        p.catalogIndex = particlesCatIndex(index);
    }
    if (getOptions().starFormation.enabled) {
        p.starFormation = particlesStarFormationStates[index];
    }
    if (particlesHasLongAcceleration(getOptions())) {
        for (int dim = 0; dim < kSpatialDimensions; dim++) {
            p.longAcceleration[dim] = particlesLongAcceleration(dim, index);
        }
    }
    return p;
}

inline void particlesSetParticle(Particle p, ParticleInteger index) {
    static bool doGroups = getOptions().analysis.groups;
    static bool doTracers = getOptions().analysis.tracers;
    CHECK_PART_BOUNDS(index);
    for (int dim = 0; dim < kSpatialDimensions; dim++) {
        particlesPos(dim, index) = p.position[dim];
        particlesVel(dim, index) = p.velocity[dim];
    }
    particlesRung(index) = p.rung;
    if (doGroups) {
        particlesLastgroup(index) = p.groupNumber;
    }
    if (doTracers) {
        particlesTracer(index) = p.isTracer;
    }
    if (getOptions().gravity.errorInterval > 0) {
        particlesGravitySample(index) = p.gravitySample;
    }
    if (getOptions().hydro.enabled) {
        particlesType(index) = p.type;
        particlesCatIndex(index) = p.catalogIndex;
    }
    if (getOptions().starFormation.enabled) {
        particlesStarFormationStates[index] = p.starFormation;
    }
    if (particlesHasLongAcceleration(getOptions())) {
        for (int dim = 0; dim < kSpatialDimensions; dim++) {
            particlesLongAcceleration(dim, index) = p.longAcceleration[dim];
        }
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

// Kinetic, potential, momentum and thermal sums; includeGas=false skips SPH gas, which the
// hydro kick sums itself.
EnergiesT particlesSumEnergies(bool includeGas = true);

#endif /* PARTICLES_HPP_ */
