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
    bool outputSilo;
    int outputSiloInterval;
    std::string outputDirectory;
    std::string outputUnits{"code"};
    std::string outputFrame{"comoving"};
    double outputXMin;
    double outputXMax;
    double outputYMin;
    double outputYMax;
    double outputZMin;
    double outputZMax;
    bool twolpt;
    bool usePowerFile;
    bool plummer;
    double plummerR;
    bool createGlass;
    int readCheck;
    int tracerCount;
    int particlesPerDimension;
    int treeCacheLineSize;
    int treeAllocLineSize;
    int particleCacheLineSize;
    int checkFreq;
    int maxIter;
    int minrung;
    int viewSize;
    int lcMinGroup;
    int lcMapSize;
    int minGroup;
    int sliceRes;
    int numberOfSteps;
    double lcB;
    double sliceSize;
    double linkingLength;
    // Internally stored compact support; public gravity.softening is h = 2 sigma.
    double hsoft;
    std::string gravityKernelFamily{"ferrers"};
    int gravityKernelDegree{8};
    std::string hydroKernelFamily{"spline"};
    int hydroKernelDegree{3};
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
    size_t numberOfParticles;
    std::string configFile;
    std::string test;
    std::string lcDir;
    std::string gadget4Restart;

    bool htime{};
    bool sph{};
    bool sphSmoothing{};
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
    double viscosityEpsilon{};
    double alphaDecay{};
    double darkMatterNeighborNumber{};
    double sphNeighborNumber{};
    double hmin{};
    double hmax{};
    double damping{};
    double sigma8C{};
    double darkMatterMass{};
    double sphMass{};
    double cfl{};
    double rho0B{};
    double rho0C{};
    double gamma{};
    double yAcceleration{};
    double centralAcceleration{};
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
        arc & viscosityEpsilon;
        arc & alphaDecay;
        arc & darkMatterNeighborNumber;
        arc & sphNeighborNumber;
        arc & hmin;
        arc & hmax;
        arc & damping;
        arc & sigma8C;
        arc & darkMatterMass;
        arc & sphMass;
        arc & cfl;
        arc & rho0B;
        arc & rho0C;
        arc & gamma;
        arc & yAcceleration;
        arc & centralAcceleration;
        arc & hcentral;
        arc & closePack;
        arc & numberOfParticles;
        arc & useGlass;
        arc & createGlass;
        arc & gadget4Restart;
        arc & minrung;
        arc & plummer;
        arc & plummerR;
        arc & omegaK;
        arc & omegaLam;
        arc & bucketSize;
        arc & numberOfSteps;
        arc & nfour;
        arc & cuda;
        arc & doLc;
        arc & saveForce;
        arc & doPower;
        arc & doGroups;
        arc & doTracers;
        arc & doSlice;
        arc & doViews;
        arc & outputSilo;
        arc & outputSiloInterval;
        arc & outputDirectory;
        arc & outputUnits;
        arc & outputFrame;
        arc & outputXMin;
        arc & outputXMax;
        arc & outputYMin;
        arc & outputYMax;
        arc & outputZMin;
        arc & outputZMax;
        arc & twolpt;
        arc & usePowerFile;
        arc & lcDir;
        arc & tracerCount;
        arc & particlesPerDimension;
        arc & treeCacheLineSize;
        arc & treeAllocLineSize;
        arc & particleCacheLineSize;
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
        arc & linkingLength;
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
        arc & sphSmoothing;
        arc & gravityKernelFamily;
        arc & gravityKernelDegree;
        arc & hydroKernelFamily;
        arc & hydroKernelDegree;
    }
};

bool processOptions(int argc, char* argv[]);
Options const& getOptions();
void setOptions(Options const& opts);
