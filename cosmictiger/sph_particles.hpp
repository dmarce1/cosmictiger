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
#pragma once

#include <cosmictiger/chemistry.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/particles.hpp>

#ifdef SPH_PARTICLES_CPP
#define SPH_PARTICLES_EXTERN
#else
#define SPH_PARTICLES_EXTERN extern
#endif
#ifdef CHECK_BOUNDS
#define CHECK_SPH_PART_BOUNDS(i)                                                                 \
    if (i < 0 || i >= sphParticlesSize()) {                                                      \
        PRINT("particle bound check failure %li should be between %li and %li\n", (long long) i, \
            (long long) 0, (long long) sphParticlesSize());                                      \
        ALWAYS_ASSERT(false);                                                                    \
    }
#else
#define CHECK_SPH_PART_BOUNDS(i)
#endif

#define NCHEMFRACS 8
#define CHEM_H 0
#define CHEM_HP 1
#define CHEM_HN 2
#define CHEM_H2 3
#define CHEM_HE 4
#define CHEM_HEP 5
#define CHEM_HEPP 6
#define CHEM_Z 7

class FractionReal {
    unsigned short i_;
    constexpr static float kC0_ = 10000.0f;
    constexpr static float kC0inv_ = 1.0f / kC0_;

public:
    CUDA_EXPORT
    operator float() const {
        return expf(-sqr(i_ * kC0inv_));
    }
    CUDA_EXPORT
    FractionReal& operator=(float y) {
        ALWAYS_ASSERT(y <= 1.0f);
        if (y > 0.0) {
            i_ = kC0_ * sqrtf(-logf(y));
        } else {
            i_ = (1 << 16) - 1;
        }
        return *this;
    }
    template <typename A>
    void serialize(A&& arc, unsigned) {
        arc & i_;
    }
};

struct SphParticle0 {
#ifdef ENTROPY
    float entr0;
#else
    float eint0;
#endif
    Array<float, NCHEMFRACS> chem0;
    template <typename A>
    void serialize(A&& arc, unsigned) {
#ifdef ENTROPY
        arc & entr0;
#else
        arc & eint0;
#endif
        arc & chem0;
    }
};
struct SphParticle {
    Array<float, NCHEMFRACS> frac;
    Array<float, NDIM> dvel;
    float omega;
    float h;
    float shearv;
    float alpha;
#ifdef ENTROPY
    float entropy;
#else
    float eint;
#endif
    float fcold;
    float divv;
    float omegaP;
    float pre;
    template <typename Arc>
    void serialize(Arc&& arc, unsigned) {
        arc & frac;
        arc & dvel;
        arc & omega;
        arc & h;
        arc & shearv;
        arc & alpha;
#ifdef ENTROPY
        arc & entropy;
#else
        arc & eint;
#endif
        arc & fcold;
        arc & divv;
        arc & omegaP;
        arc & pre;
    }
};

struct SphRecord1 {
    Array<float, NCHEMFRACS> frac;
    float alpha;
};

struct SphRecord2 {
#ifdef ENTROPY
    float entropy;
#else
    float eint;
#endif
    float h;
};

struct SphRecord5 {
    Array<float, NCHEMFRACS> dfrac;
    float dfcold;
};

struct SphRecord6 {
    Array<float, NDIM> dvel;
};

/*SPH_PARTICLES_EXTERN array<float*, NDIM> sphParticlesDv1; // dvel_pred
 SPH_PARTICLES_EXTERN float* sphParticlesDe1; // dentr_pred
 SPH_PARTICLES_EXTERN array<float, NCHEMFRACS>* sphParticlesDchem1; // chemistry
 SPH_PARTICLES_EXTERN float* sphParticlesDrc1;
 */

SPH_PARTICLES_EXTERN SphRecord1* sphParticlesR1;
SPH_PARTICLES_EXTERN SphRecord2* sphParticlesR2;
SPH_PARTICLES_EXTERN SphRecord5* sphParticlesR5;
SPH_PARTICLES_EXTERN SphRecord6* sphParticlesR6;

SPH_PARTICLES_EXTERN ParticleInteger* sphParticlesDm;        // dark matter index
SPH_PARTICLES_EXTERN Array<float*, NDIM> sphParticlesDv2;    // dvel_pred
SPH_PARTICLES_EXTERN Array<float*, NDIM> sphParticlesG;      // dvel_pred
SPH_PARTICLES_EXTERN float* sphParticlesDv;
SPH_PARTICLES_EXTERN float* sphParticlesKap;
SPH_PARTICLES_EXTERN float* sphParticlesE0;
SPH_PARTICLES_EXTERN float* sphParticlesDe3;
SPH_PARTICLES_EXTERN float* sphParticlesDe2;
SPH_PARTICLES_EXTERN float* sphParticlesFp2;
SPH_PARTICLES_EXTERN float* sphParticlesPre;
SPH_PARTICLES_EXTERN float* sphParticlesFp1;
SPH_PARTICLES_EXTERN float* sphParticlesS2;
SPH_PARTICLES_EXTERN float* sphParticlesFc;
SPH_PARTICLES_EXTERN float* sphParticlesRh;
SPH_PARTICLES_EXTERN float* sphParticlesTc;
SPH_PARTICLES_EXTERN char* sphParticlesOr;
SPH_PARTICLES_EXTERN char* sphParticlesSa;
SPH_PARTICLES_EXTERN char* sphParticlesSt;

struct AuxiliaryQuantities {
    float omega;
    float shearv;
    float alpha;
    float divv;
    float omegaP;
    float pre;
    Array<float, NCHEMFRACS> fracs;
    template <typename A>
    void serialize(A&& arc, unsigned) {
        arc & omega;
        arc & shearv;
        arc & alpha;
        arc & fracs;
        arc & divv;
        arc & omegaP;
        arc & pre;
    }
};

ParticleInteger sphParticlesSize();
void sphParticlesResize(ParticleInteger sz, bool parts2 = true);
void sphParticlesFree();
void sphParticlesCacheFree1();
void sphParticlesCacheFree2();
#ifdef ENTROPY
void sphParticlesCacheFreeEntr();
#else
void sphParticlesCacheFreeEint();
#endif
void sphParticlesResolveWithParticles();
void sphParticlesSortByParticles(Pair<ParticleInteger> rng);
void sphParticlesSwap(ParticleInteger i, ParticleInteger j);
void sphParticlesSwap2(ParticleInteger i, ParticleInteger j);
ParticleInteger sphParticlesSort(Pair<ParticleInteger> rng, Fixed32 xm, int xdim);
void sphParticlesGlobalReadPos(
    ParticleGlobalRange range, Fixed32* x, Fixed32* y, Fixed32* z, ParticleInteger offset);
void sphParticlesGlobalReadFcold(ParticleGlobalRange range, float*, char*, ParticleInteger offset);
void sphParticlesGlobalReadEnergyAndSmoothlen(
    ParticleGlobalRange range, float*, float*, ParticleInteger offset);
void sphParticlesGlobalReadRungs(ParticleGlobalRange range, char*, ParticleInteger offset);
void sphParticlesGlobalReadVels(
    ParticleGlobalRange range, float*, float*, float*, ParticleInteger offset);
void sphParticlesGlobalReadKappas(ParticleGlobalRange range, float*, ParticleInteger offset);
void sphParticlesGlobalReadRho(ParticleGlobalRange range, float*, ParticleInteger offset);
void sphParticlesGlobalReadAux(ParticleGlobalRange range, float* alpha, float* omega, float* omegaP,
    float* pre, float* shearv, Array<float, NCHEMFRACS>* fracs, ParticleInteger offset);
void sphParticlesResetConverged();
void sphParticlesResetSemiactive();
void sphParticlesLoad(FILE* fp);
void sphParticlesSave(FILE* fp);
void sphCheckpointTest();
float sphParticlesMaximumSmoothingLength();
float sphParticlesTemperature(ParticleInteger, float);
float sphParticlesMmw(ParticleInteger);
void sphParticlesSoftlens2smoothlens(int minrung);

std::pair<double, double> sphParticlesApplyUpdates(int, int, float, float, float = 1.0);

/*
 inline float& sph_particles_SN(part_int index) {
 return sphParticlesSn[index];
 }
 */

#ifdef ENTROPY
inline float& sphParticlesEntr0(ParticleInteger index) {
    return sphParticlesE0[index];
}
#else
inline float& sphParticlesEint0(ParticleInteger index) {
    return sphParticlesE0[index];
}
#endif

inline char& sphParticlesIsstar(ParticleInteger index) {
    return sphParticlesSt[index];
}

inline float& sphParticlesTcool(ParticleInteger index) {
    return sphParticlesTc[index];
}

inline char& sphParticlesConverged(ParticleInteger index) {
    return sphParticlesOr[index];
}

inline char& sphParticlesSemiactive(ParticleInteger index) {
    return sphParticlesSa[index];
}

inline float& sphParticlesKappa(ParticleInteger index) {
    return sphParticlesKap[index];
}

inline float& sphParticlesColdMass(ParticleInteger index) {
    return sphParticlesFc[index];
}

inline float& sphParticlesDcoldMass(ParticleInteger index) {
    return sphParticlesR5[index].dfcold;
}

inline float& sphParticlesAlpha(ParticleInteger index) {
    CHECK_SPH_PART_BOUNDS(index);
    return sphParticlesR1[index].alpha;
}

inline float& sphParticlesFrac(int j, ParticleInteger index) {
    CHECK_SPH_PART_BOUNDS(index);
    return sphParticlesR1[index].frac[j];
}

inline float& sphParticlesShear(ParticleInteger index) {
    CHECK_SPH_PART_BOUNDS(index);
    return sphParticlesS2[index];
}

inline float& sphParticlesZ(ParticleInteger index) {
    CHECK_SPH_PART_BOUNDS(index);
    return sphParticlesR1[index].frac[CHEM_Z];
}

inline void sphParticlesNormalizeFracs(ParticleInteger index) {
    double sum = 0.0;
    for (double fi = 0; fi < NCHEMFRACS; fi++) {
        sum += sphParticlesFrac(fi, index);
    }
    for (double fi = 0; fi < NCHEMFRACS; fi++) {
        sphParticlesFrac(fi, index) /= sum;
    }
}

inline float& sphParticlesH(ParticleInteger index) {
    CHECK_SPH_PART_BOUNDS(index);
    return sphParticlesR1[index].frac[CHEM_H];
}

inline float& sphParticlesHe0(ParticleInteger index) {
    CHECK_SPH_PART_BOUNDS(index);
    return sphParticlesR1[index].frac[CHEM_HE];
}

inline float& sphParticlesHp(ParticleInteger index) {
    CHECK_SPH_PART_BOUNDS(index);
    return sphParticlesR1[index].frac[CHEM_HP];
}

inline float& sphParticlesHn(ParticleInteger index) {
    CHECK_SPH_PART_BOUNDS(index);
    return sphParticlesR1[index].frac[CHEM_HN];
}

inline float& sphParticlesH2(ParticleInteger index) {
    CHECK_SPH_PART_BOUNDS(index);
    return sphParticlesR1[index].frac[CHEM_H2];
}

inline float& sphParticlesHep(ParticleInteger index) {
    CHECK_SPH_PART_BOUNDS(index);
    return sphParticlesR1[index].frac[CHEM_HEP];
}

inline float& sphParticlesHepp(ParticleInteger index) {
    CHECK_SPH_PART_BOUNDS(index);
    return sphParticlesR1[index].frac[CHEM_HEPP];
}

inline float sphParticlesY(ParticleInteger index) {
    CHECK_SPH_PART_BOUNDS(index);
    return sphParticlesHe0(index) + sphParticlesHep(index) + sphParticlesHepp(index);
}

inline float sphParticlesGamma(ParticleInteger index) {
    CHECK_SPH_PART_BOUNDS(index);
    float const hValue = sphParticlesH(index);
    float const hp = sphParticlesHp(index);
    float const hn = sphParticlesHn(index);
    float const h2 = sphParticlesH2(index);
    float const he = sphParticlesHe0(index);
    float const hep = sphParticlesHep(index);
    float const hepp = sphParticlesHepp(index);
    float const zValue = sphParticlesZ(index);
    float const ne = hp - hn + hep + 2.0f * hepp;
    float const n =
        hValue + hp + hn + 0.5f * h2 + 0.25f * he + 0.25f * hep + 0.25f * hepp + 0.1f * zValue + ne;
    float const cv = 1.5f + 0.5f * hn / n;
    return 1.f + 1.f / cv;
}

inline SphRecord1& sphParticlesRec1(ParticleInteger index) {
    return sphParticlesR1[index];
}

inline SphRecord2& sphParticlesRec2(ParticleInteger index) {
    return sphParticlesR2[index];
}

inline SphRecord5& sphParticlesRec5(ParticleInteger index) {
    return sphParticlesR5[index];
}

inline SphRecord6& sphParticlesRec6(ParticleInteger index) {
    return sphParticlesR6[index];
}

inline float& sphParticlesOmega(ParticleInteger index) {
    CHECK_SPH_PART_BOUNDS(index);
    return sphParticlesFp1[index];
}

inline float& sphParticlesOmegaP(ParticleInteger index) {
    CHECK_SPH_PART_BOUNDS(index);
    return sphParticlesFp2[index];
}

inline float& sphParticlesPressure(ParticleInteger index) {
    CHECK_SPH_PART_BOUNDS(index);
    return sphParticlesPre[index];
}

inline ParticleInteger& sphParticlesDmIndex(ParticleInteger index) {
    CHECK_SPH_PART_BOUNDS(index);
    return sphParticlesDm[index];
}

inline Fixed32& sphParticlesPos(int dim, int index) {
    return particlesPos(dim, sphParticlesDmIndex(index));
}

inline float& sphParticlesVel(int dim, int index) {
    return particlesVel(dim, sphParticlesDmIndex(index));
}

inline char& sphParticlesRung(int index) {
    return particlesRung(sphParticlesDmIndex(index));
}

#ifdef ENTROPY
inline float& sphParticlesEntr(ParticleInteger index) {
    CHECK_SPH_PART_BOUNDS(index);
    return sphParticlesR2[index].A;
}

inline float& sphParticlesDentr2(ParticleInteger index) {
    CHECK_SPH_PART_BOUNDS(index);
    return sphParticlesDe3[index];
}

inline float& sphParticlesDentr1(ParticleInteger index) {
    CHECK_SPH_PART_BOUNDS(index);
    return sphParticlesDe2[index];
}

#else
inline float& sphParticlesEint(ParticleInteger index) {
    CHECK_SPH_PART_BOUNDS(index);
    return sphParticlesR2[index].eint;
}

inline float& sphParticlesDeint2(ParticleInteger index) {
    CHECK_SPH_PART_BOUNDS(index);
    return sphParticlesDe3[index];
}

inline float& sphParticlesDeint1(ParticleInteger index) {
    CHECK_SPH_PART_BOUNDS(index);
    return sphParticlesDe2[index];
}
#endif

inline float& sphParticlesDvel0(int dim, ParticleInteger index) {
    CHECK_SPH_PART_BOUNDS(index);
    return sphParticlesDv2[dim][index];
}

inline float& sphParticlesDvel(int dim, ParticleInteger index) {
    CHECK_SPH_PART_BOUNDS(index);
    return sphParticlesR6[index].dvel[dim];
}

inline Array<float, NCHEMFRACS>& sphParticlesChem(ParticleInteger index) {
    CHECK_SPH_PART_BOUNDS(index);
    return sphParticlesR1[index].frac;
}

inline Array<float, NCHEMFRACS>& sphParticlesDchem(ParticleInteger index) {
    CHECK_SPH_PART_BOUNDS(index);
    return sphParticlesR5[index].dfrac;
}

inline float& sphParticlesDivv(ParticleInteger index) {
    CHECK_SPH_PART_BOUNDS(index);
    return sphParticlesDv[index];
}

inline float& sphParticlesGforce(int dim, ParticleInteger index) {
    CHECK_SPH_PART_BOUNDS(index);
    return sphParticlesG[dim][index];
}

inline float sphParticlesEkin(ParticleInteger index) {
    float ekin = 0.f;
    for (int dim = 0; dim < NDIM; dim++) {
        ekin += 0.5f * sqr(sphParticlesVel(dim, index));
    }
    return ekin;
}

inline float& sphParticlesSmoothingLength(ParticleInteger index) {
    CHECK_SPH_PART_BOUNDS(index);
    return sphParticlesR2[index].h;
}

#ifdef HOPKINS

inline float& sphParticlesRhoRho(ParticleInteger index) {
    return sphParticlesRh[index];
}

inline float sphParticlesRhoPre(ParticleInteger index) {
#ifdef ENTROPY
    float const gamma = getOptions().gamma;
    return powf(sphParticlesPressure(index) / sphParticlesEntr(index), 1.0f / gamma);
#else
    float const gamma = getOptions().gamma;
    return sphParticlesPressure(index) / sphParticlesEint(index) / (gamma - 1.f);
#endif
}

inline float sphParticlesEintPre(ParticleInteger index) {
#ifdef ENTROPY
    ALWAYS_ASSERT(sphParticlesEntr(index) > 0.0f);
    float const gamma = getOptions().gamma;
    return sphParticlesPressure(index) / sphParticlesRhoPre(index) / (gamma - 1.f);
#else
    return sphParticlesEint(index);
#endif
}

inline float sphParticlesEintRho(ParticleInteger index) {
#ifdef ENTROPY
    static float const gamma0 = getOptions().gamma;
    float const rho = sphParticlesRhoRho(index);
    float const entropy = sphParticlesEntr(index);
    return entropy * powf(rho, gamma0 - 1.0) / (gamma0 - 1.0);
#else
    return sphParticlesEint(index);
#endif
}

#else

inline float& sphParticlesRho(ParticleInteger index) {
    return sphParticlesRh[index];
}

#ifdef ENTROPY
inline float sphParticlesEint(ParticleInteger index) {
    static float const gamma0 = getOptions().gamma;
    static float const stars = getOptions().stars;
    float hfrac = 1.0f;
    if (stars) {
        hfrac = 1.0f - sphParticlesColdMass(index);
    }
    float const rho = sphParticlesRho(index) * hfrac;
    float const entropy = sphParticlesEntr(index);
    return entropy * powf(rho, gamma0 - 1.0) / (gamma0 - 1.0);
}
#endif
#endif

inline AuxiliaryQuantities sphParticlesAuxQuantities(ParticleInteger index) {
    AuxiliaryQuantities aux{};
    aux.alpha = sphParticlesAlpha(index);
    aux.omega = sphParticlesOmega(index);
    aux.omegaP = sphParticlesOmegaP(index);
    aux.pre = sphParticlesPressure(index);
    aux.divv = sphParticlesDivv(index);
    if (getOptions().diffusion) {
        aux.shearv = sphParticlesShear(index);
    }
    if (getOptions().chem) {
        aux.fracs = sphParticlesChem(index);
    }
    return aux;
}

inline SphParticle getSphParticle(ParticleInteger i) {
    SphParticle part;
    for (int fi = 0; fi < NCHEMFRACS; fi++) {
        part.frac[fi] = sphParticlesFrac(fi, i);
    }
    for (int dim = 0; dim < NDIM; dim++) {
        part.dvel[dim] = sphParticlesDvel(dim, i);
    }
    part.omega = sphParticlesOmega(i);
    part.h = sphParticlesSmoothingLength(i);
    part.shearv = sphParticlesShear(i);
    part.alpha = sphParticlesAlpha(i);
#ifdef ENTROPY
    part.entropy = sphParticlesEntr(i);
#else
    part.eint = sphParticlesEint(i);
#endif
    part.fcold = sphParticlesColdMass(i);
    part.divv = sphParticlesDivv(i);
    return part;
}
