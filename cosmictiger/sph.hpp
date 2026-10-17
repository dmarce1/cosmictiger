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

#ifndef SPH_HPP_
#define SPH_HPP_

#define SPH_KERNEL_ORDER 5

#include <atomic>

#include <cosmictiger/constants.hpp>
#include <cosmictiger/defs.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/range.hpp>
#include <cosmictiger/stack_vector.hpp>
#include <cosmictiger/tree.hpp>

#define SPH_AV_HU 0
#define SPH_AV_MM 1
#define SPH_AV_CONSTANT 2

struct SphValues {
    float vx;
    float vy;
    float vz;
    float rho;
    float p;
    template <typename A>
    void serialize(A&& arc, unsigned) {
        arc & vx;
        arc & vy;
        arc & vz;
        arc & rho;
        arc & p;
    }
};

#ifndef __CUDACC__
struct SphTreeNeighborReturn {
    Fixed32Range innerBox;
    Fixed32Range outerBox;
    SphValues valueAt;
    bool hasValueAt;
    CUDA_EXPORT
    SphTreeNeighborReturn() {
        hasValueAt = false;
        for (int dim = 0; dim < NDIM; dim++) {
            innerBox.begin[dim] = 1.9;
            innerBox.end[dim] = -0.9;
            outerBox.begin[dim] = 1.9;
            outerBox.end[dim] = -0.9;
        }
    }
    CUDA_EXPORT
    SphTreeNeighborReturn& operator+=(SphTreeNeighborReturn const& other) {
        for (int dim = 0; dim < NDIM; dim++) {
            innerBox.begin[dim] = std::min(innerBox.begin[dim], other.innerBox.begin[dim]);
            innerBox.end[dim] = std::max(innerBox.end[dim], other.innerBox.end[dim]);
            outerBox.begin[dim] = std::min(outerBox.begin[dim], other.outerBox.begin[dim]);
            outerBox.end[dim] = std::max(outerBox.end[dim], other.outerBox.end[dim]);
        }
        if (!hasValueAt && other.hasValueAt) {
            valueAt = other.valueAt;
            hasValueAt = true;
        }
        return *this;
    }
    template <typename A>
    void serialize(A&& arc, unsigned) {
        arc & hasValueAt;
        arc & innerBox;
        arc & outerBox;
        arc & valueAt;
    }
};
#endif

#define SPH_TREE_NEIGHBOR_BOXES 0
#define SPH_TREE_NEIGHBOR_NEIGHBORS 1
#define SPH_TREE_NEIGHBOR_VALUE_AT 2
#define SPH_INTERACTIONS_I 1
#define SPH_INTERACTIONS_J 2
#define SPH_INTERACTIONS_IJ 3
#define SPH_SET_ACTIVE 1
#define SPH_SET_ALL 4

struct SphTreeNeighborParameters {
    int runType = 0;
    float hWt = 1.0f;
    int minRung = 0;
    int seti = 0;
    int seto = 0;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    template <typename T>
    void serialize(T&& arc, unsigned) {
        arc & seti;
        arc & seto;
        arc & runType;
        arc & hWt;
        arc & minRung;
        arc & x;
        arc & y;
        arc & z;
    }
};

struct SphRunReturn {
    float hmin;
    float hmax;
    double flops;
    int maxRungHydro;
    int maxRungGrav;
    int maxRung;
    float maxVsig;
    bool rc;
    float ekin;
    float ent;
    float etherm;
    float momx;
    float momy;
    float momz;
    float vol;
    float dtinvCfl;
    float dtinvVisc;
    float dtinvDiff;
    float dtinvCond;
    float dtinvDivv;
    float dtinvAcc;
    float dtinvOmega;
    SphRunReturn() {
        flops = 0.0;
        ent = 0.0;
        hmax = 0.0;
        hmin = std::numeric_limits<float>::max();
        maxRungHydro = 0;
        maxRungGrav = 0;
        maxRung = 0;
        maxVsig = 0.0;
        etherm = vol = ekin = momx = momy = momz = 0.0;
        dtinvCfl = 0.0f;
        dtinvVisc = 0.0f;
        dtinvDiff = 0.0f;
        dtinvCond = 0.0f;
        dtinvDivv = 0.0f;
        dtinvAcc = 0.0f;
        dtinvOmega = 0.0f;
        rc = false;
    }
    template <typename A>
    void serialize(A&& arc, unsigned) {
        arc & rc;
        arc & flops;
        arc & dtinvCfl;
        arc & dtinvVisc;
        arc & dtinvDiff;
        arc & dtinvCond;
        arc & dtinvDivv;
        arc & dtinvAcc;
        arc & hmin;
        arc & hmax;
        arc & maxRungHydro;
        arc & maxRungGrav;
        arc & maxRung;
        arc & maxVsig;
        arc & ekin;
        arc & momx;
        arc & momy;
        arc & momz;
        arc & vol;
        arc & ent;
        arc & etherm;
        arc & dtinvOmega;
    }
    SphRunReturn& operator+=(SphRunReturn const& other) {
        flops += other.flops;
        hmax = std::max(hmax, other.hmax);
        hmin = std::min(hmin, other.hmin);
        maxRungHydro = std::max(maxRungHydro, other.maxRungHydro);
        maxRungGrav = std::max(maxRungGrav, other.maxRungGrav);
        maxRung = std::max(maxRung, other.maxRung);
        maxVsig = std::max(maxVsig, other.maxVsig);
        dtinvCfl = std::max(dtinvCfl, other.dtinvCfl);
        dtinvVisc = std::max(dtinvVisc, other.dtinvVisc);
        dtinvAcc = std::max(dtinvAcc, other.dtinvAcc);
        dtinvDiff = std::max(dtinvDiff, other.dtinvDiff);
        dtinvCond = std::max(dtinvCond, other.dtinvCond);
        dtinvDivv = std::max(dtinvDivv, other.dtinvDivv);
        dtinvOmega = std::max(dtinvOmega, other.dtinvOmega);
        ekin += other.ekin;
        momx += other.momx;
        momy += other.momy;
        momz += other.momz;
        vol += other.vol;
        etherm += other.etherm;
        ent += other.ent;
        rc = rc || other.rc;
        return *this;
    }
};

struct SphRunParameters {
    bool vsoft;
    int avType;
    int runType;
    int set;
    int phase;
    int minRung;
    float t0;
    float a;
    float adot;
    bool tzero;
    float cfl;
    float hmin;
    float hmax;
    int maxRung;
    float yAcceleration;
    float tau;
    int iter;
    float maxDt;
    float alpha0;
    float alpha1;
    float beta;
    float viscosityEpsilon;
    float alphaDecay;
    float omegaM;
    float h0;
    float damping;
    bool diffusion;
    bool conduction;
    double codeToS;
    double codeToG;
    double codeToCm;
    bool stars;
    SphRunParameters() {
        iter = 0;
        auto const opts = getOptions();
        hmin = opts.hydroMinSupportRadius;
        hmax = opts.hydroMaxSupportRadius;
        stars = opts.stars;
        vsoft = opts.vsoft;
        avType = opts.viscType;
        codeToS = opts.codeToS;
        codeToG = opts.codeToG;
        codeToCm = opts.codeToCm;
        damping = opts.damping;
        diffusion = opts.diffusion;
        conduction = opts.conduction;
        yAcceleration = opts.yAcceleration;
        maxDt = 1e30;
        alpha0 = opts.alpha0;
        alpha1 = opts.alpha1;
        beta = opts.beta;
        viscosityEpsilon = opts.viscosityEpsilon;
        alphaDecay = opts.alphaDecay;
        if (opts.test == "") {
            h0 = opts.hubble * constants::kH0 * opts.codeToS;
            omegaM = opts.omegaM;
        } else {
            h0 = omegaM = 0.0;
        }
    }
    template <typename A>
    void serialize(A&& arc, unsigned) {
        arc & alpha0;
        arc & alpha1;
        arc & beta;
        arc & viscosityEpsilon;
        arc & alphaDecay;
        arc & damping;
        arc & diffusion;
        arc & conduction;
        arc & codeToS;
        arc & codeToG;
        arc & codeToCm;
        arc & stars;
        arc & hmin;
        arc & hmax;
        arc & omegaM;
        arc & vsoft;
        arc & h0;
        arc & adot;
        arc & iter;
        arc & maxDt;
        arc & yAcceleration;
        arc & avType;
        arc & tau;
        arc & tzero;
        arc & t0;
        arc & phase;
        arc & runType;
        arc & set;
        arc & minRung;
        arc & a;
        arc & cfl;
        arc & maxRung;
    }
};

#define SPH_RUN_PREHYDRO1 0
#define SPH_RUN_PREHYDRO2 1
#define SPH_RUN_HYDRO 2
#define SPH_RUN_AUX 3
#define SPH_RUN_RUNGS 4
#define SPH_RUN_COND_INIT 5
#define SPH_RUN_CONDUCTION 6

float sphApplyDiffusionUpdate(int minrung, float toler);
void sphInitDiffusion();

struct CondUpdateReturn {
    float errMax;
    float errRms;
    size_t n;
    template <typename A>
    void serialize(A& arc, unsigned) {
        arc & errMax;
        arc & errRms;
        arc & n;
    }
};
CondUpdateReturn sphApplyConductionUpdate(int minrung);
SphRunReturn sphRun(SphRunParameters params, bool cuda = false);
#ifndef __CUDACC__
hpx::future<SphTreeNeighborReturn> sphTreeNeighbor(
    SphTreeNeighborParameters params, TreeId self, Vector<TreeId> checklist, int level = 0);
#endif
Vector<SphValues> sphValuesAt(Vector<double> x, Vector<double> y, Vector<double> z);
/*
 template<typename T>
 CUDA_EXPORT
 inline T sph_divW(T r, Thinv, Th3inv) {
 const T c0 = T(21.f/M_PI/2.f);
 const T C = c0 * h3inv * sqr(hinv);
 const T q = r * hinv;
 T w = 120.0;
 w = fmaf(q, w, T(120));
 w = fmaf(q, w, -T(300));
 w = fmaf(q, w, T(240));
 w = fmaf(q, w, -T(60));
 w *= C;
 return w;
 }
 */

#endif /* SPH_HPP_ */
