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

#ifndef KICK_HPP_
#define KICK_HPP_

#include <atomic>

#include <cosmictiger/sph_particles.hpp>
#include <cosmictiger/tree.hpp>

#define KICK_C_MAX 32

struct CudaKickData {
    TreeNode* treeNodes;
    Fixed32* x;
    Fixed32* y;
    Fixed32* z;
    Array<float, NDIM>* vel;
    float* sphGx;
    float* sphGy;
    float* sphGz;
    ParticleInteger* catIndex;
    char* type;
    bool sph;
    float dmMass;
    float sphMass;
    bool vsoft;
    char* rungs;
    float* gx;
    float* gy;
    float* gz;
    float* pot;
};

struct KickReturn;

#ifdef __CUDACC__

#include <cosmictiger/stack_vector.hpp>

#define KICK_PP_MAX (32 * 29)

struct ExpansionType {
    Array<Fixed32, NDIM> pos;
    Expansion<float> expansion;
};

struct SearchParameters {
    int self;
    int phase;
};

struct ForceType {
    float gx;
    float gy;
    float gz;
    float phi;
};

struct CudaKickSharedMemory {
    union {
        struct {
            Array<Fixed32, KICK_PP_MAX> x;
            Array<Fixed32, KICK_PP_MAX> y;
            Array<Fixed32, KICK_PP_MAX> z;
            Array<float, KICK_PP_MAX> mass;
        } xValue;
        Array<MultiPos, KICK_C_MAX> mpos;
    };
    DeviceVector<ForceType> f;
    StackVector<int> echecks;
    StackVector<int> dchecks;
    DeviceVector<int> leaflist;
    DeviceVector<int> closelist;
    DeviceVector<int> cplist;
    DeviceVector<int> cclist;
    DeviceVector<int> pclist;
    DeviceVector<int> nextlist;
    DeviceVector<ExpansionType> l;
    DeviceVector<SearchParameters> params;
    BarrierType barrier;
};
#endif

struct KickReturn {
    int maxRung;
    double pot;
    double kin;
    double dkin;
    double xmom;
    double ymom;
    double zmom;
    double nmom;
    double dmKineticCorrection;
    double dmXMomentumCorrection;
    double dmYMomentumCorrection;
    double dmZMomentumCorrection;
    double dmMomentumNormCorrection;

    CUDA_EXPORT
    KickReturn() {
        maxRung = 0;
        pot = 0.0;
        kin = 0.0;
        xmom = ymom = zmom = nmom = 0.0;
        dkin = 0.0;
        dmKineticCorrection = 0.0;
        dmXMomentumCorrection = 0.0;
        dmYMomentumCorrection = 0.0;
        dmZMomentumCorrection = 0.0;
        dmMomentumNormCorrection = 0.0;
    }
    CUDA_EXPORT
    KickReturn& operator+=(KickReturn const& other) {
        if (other.maxRung > maxRung) {
            maxRung = other.maxRung;
        }
        pot += other.pot;
        kin += other.kin;
        xmom += other.xmom;
        ymom += other.ymom;
        zmom += other.zmom;
        nmom += other.nmom;
        dkin += other.dkin;
        dmKineticCorrection += other.dmKineticCorrection;
        dmXMomentumCorrection += other.dmXMomentumCorrection;
        dmYMomentumCorrection += other.dmYMomentumCorrection;
        dmZMomentumCorrection += other.dmZMomentumCorrection;
        dmMomentumNormCorrection += other.dmMomentumNormCorrection;

        return *this;
    }
    template <class A>
    void serialize(A&& arc, unsigned) {
        arc & maxRung;
        arc & pot;
        arc & kin;
        arc & xmom;
        arc & ymom;
        arc & zmom;
        arc & nmom;
        arc & dkin;
        arc & dmKineticCorrection;
        arc & dmXMomentumCorrection;
        arc & dmYMomentumCorrection;
        arc & dmZMomentumCorrection;
        arc & dmMomentumNormCorrection;
    }
};

struct KickParameters {
    bool htime;
    float sign;
    int minLevel;
    int minRung;
    float a;
    float t0;
    float theta;
    float h;
    float eta;
    float gm;
    bool velInit;
    bool saveForce;
    bool firstCall;
    bool gpu;
    float nodeLoad;
    float maxDt;
    bool ascending;
    bool descending;
    bool top;
    bool doPhi;
    KickParameters() {
        htime = true;
        maxDt = 1e30;
        doPhi = true;
        velInit = false;
        sign = 1.0;
    }
    template <class A>
    void serialize(A&& arc, unsigned) {
        arc & htime;
        arc & ascending;
        arc & descending;
        arc & top;
        arc & maxDt;
        arc & minRung;
        arc & gpu;
        arc & a;
        arc & t0;
        arc & theta;
        arc & firstCall;
        arc & h;
        arc & eta;
        arc & gm;
        arc & sign;
        arc & saveForce;
        arc & nodeLoad;
        arc & minLevel;
        arc & doPhi;
        arc & velInit;
    }
};

struct KickWorkitem {
    Expansion<float> l;
    Array<Fixed32, NDIM> pos;
    TreeId self;
    Vector<TreeId> dchecklist;
    Vector<TreeId> echecklist;
};

struct KickWorkspace;

#ifndef __CUDACC__
KickReturn kick(KickParameters, Expansion<float> l, Array<Fixed32, NDIM> pos, TreeId self,
    Vector<TreeId> dchecklist, Vector<TreeId> echecklist, std::shared_ptr<KickWorkspace>);
#endif
void kickShowTimings();
void kickSetRc(KickReturn kr);
#ifdef USE_CUDA
KickReturn cudaExecuteKicks(
    KickParameters params, Fixed32*, Fixed32*, Fixed32*, TreeNode*, Vector<KickWorkitem> workitems);
#endif
int kickBlockCount();
size_t kickEstimateCudaMemoryUsage(double theta, int numberOfParticles, int checkCount);

#endif /* KICK_HPP_ */
