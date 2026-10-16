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

#include <stddef.h>
#include <string>

struct Options {
    int bucketSize;
    int nfour;
    bool cuda;
    bool closePack;
    bool useGlass;
    bool saveForce;
    bool doLc;
    bool doPower;
    bool doGroups;
    bool doTracers;
    bool doSlice;
    bool doViews;
    bool twolpt;
    bool usePowerFile;
    bool plummer;
    double plummerR;
    bool createGlass;
    int readCheck;
    int tracerCount;
    int partsDim;
    int treeCacheLineSize;
    int treeAllocLineSize;
    int partCacheLineSize;
    int checkFreq;
    int maxIter;
    int minrung;
    int viewSize;
    int lcMinGroup;
    int lcMapSize;
    int minGroup;
    int sliceRes;
    int nsteps;
    double lcB;
    double sliceSize;
    double linkLen;
    double hsoft;
    double gm;
    double eta;
    double codeToS;
    double codeToCm;
    double codeToG;
    double omegaM;
    double omegaR;
    double z0;
    double z1;
    double hubble;
    double sigma8;
    double theta;
    double omegaB;
    double omegaLam;
    double omegaC;
    double omegaGam;
    double omegaNu;
    double thetaCmb;
    double ns;
    double y0;
    double neff;
    double omegaK;
    size_t nparts;
    std::string configFile;
    std::string test;
    std::string lcDir;
    std::string gadget4Restart;

    bool htime{};
    bool sph{};
    bool yreflect{};
    int viscType{};
    bool stars{};
    bool chem{};
    bool conduction{};
    bool vsoft{};
    bool gravity{};
    int glass{};
    double kernel{};
    bool diffusion{};
    int sphBucketSize{};
    double alpha0{};
    double alpha1{};
    double beta{};
    double alphaDecay{};
    double gneighborNumber{};
    double sneighborNumber{};
    double hmin{};
    double hmax{};
    double damping{};
    double sigma8C{};
    double dmMass{};
    double sphMass{};
    double cfl{};
    double rho0B{};
    double rho0C{};
    double gamma{};
    double gy{};
    double gcentral{};
    double hcentral{};

    template <typename A>
    void serialize(A&& arc, unsigned) {
        arc & htime;
        arc & sph;
        arc & yreflect;
        arc & viscType;
        arc & stars;
        arc & chem;
        arc & conduction;
        arc & vsoft;
        arc & gravity;
        arc & glass;
        arc & kernel;
        arc & diffusion;
        arc & sphBucketSize;
        arc & alpha0;
        arc & alpha1;
        arc & beta;
        arc & alphaDecay;
        arc & gneighborNumber;
        arc & sneighborNumber;
        arc & hmin;
        arc & hmax;
        arc & damping;
        arc & sigma8C;
        arc & dmMass;
        arc & sphMass;
        arc & cfl;
        arc & rho0B;
        arc & rho0C;
        arc & gamma;
        arc & gy;
        arc & gcentral;
        arc & hcentral;
        arc & closePack;
        arc & nparts;
        arc & useGlass;
        arc & createGlass;
        arc & gadget4Restart;
        arc & minrung;
        arc & plummer;
        arc & plummerR;
        arc & omegaK;
        arc & omegaLam;
        arc & bucketSize;
        arc & nsteps;
        arc & nfour;
        arc & cuda;
        arc & doLc;
        arc & saveForce;
        arc & doPower;
        arc & doGroups;
        arc & doTracers;
        arc & doSlice;
        arc & doViews;
        arc & twolpt;
        arc & usePowerFile;
        arc & lcDir;
        arc & tracerCount;
        arc & partsDim;
        arc & treeCacheLineSize;
        arc & treeAllocLineSize;
        arc & partCacheLineSize;
        arc & readCheck;
        arc & checkFreq;
        arc & maxIter;
        arc & lcMinGroup;
        arc & minGroup;
        arc & viewSize;
        arc & sliceRes;
        arc & lcMapSize;
        arc & lcB;
        arc & theta;
        arc & sliceSize;
        arc & linkLen;
        arc & hsoft;
        arc & gm;
        arc & eta;
        arc & codeToS;
        arc & codeToCm;
        arc & codeToG;
        arc & omegaM;
        arc & omegaR;
        arc & z0;
        arc & z1;
        arc & hubble;
        arc & sigma8;
        arc & omegaB;
        arc & omegaC;
        arc & omegaGam;
        arc & omegaNu;
        arc & thetaCmb;
        arc & ns;
        arc & y0;
        arc & neff;
        arc & configFile;
        arc & test;
    }
};

bool processOptions(int argc, char* argv[]);
Options const& getOptions();
void setOptions(Options const& opts);
