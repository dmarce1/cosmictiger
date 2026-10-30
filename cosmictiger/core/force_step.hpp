/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#pragma once

#include <limits>

#include <cosmictiger/cuda/cuda_export.hpp>

// Force advancement diagnostics shared by hydrodynamics and gravity solvers.
struct ForceStepResult {
    int maxRung;
    double minimumTimestep;
    double pot;
    double kin;
    double dkin;
    double xmom;
    double ymom;
    double zmom;
    double nmom;
    double darkMatterKineticCorrection;
    double darkMatterXMomentumCorrection;
    double darkMatterYMomentumCorrection;
    double darkMatterZMomentumCorrection;
    double darkMatterMomentumNormCorrection;

    CUDA_EXPORT
    ForceStepResult() {
        maxRung = 0;
        minimumTimestep = std::numeric_limits<double>::max();
        pot = 0.0;
        kin = 0.0;
        xmom = ymom = zmom = nmom = 0.0;
        dkin = 0.0;
        darkMatterKineticCorrection = 0.0;
        darkMatterXMomentumCorrection = 0.0;
        darkMatterYMomentumCorrection = 0.0;
        darkMatterZMomentumCorrection = 0.0;
        darkMatterMomentumNormCorrection = 0.0;
    }
    CUDA_EXPORT
    ForceStepResult& operator+=(ForceStepResult const& other) {
        if (other.maxRung > maxRung) {
            maxRung = other.maxRung;
        }
        if (other.minimumTimestep < minimumTimestep) {
            minimumTimestep = other.minimumTimestep;
        }
        pot += other.pot;
        kin += other.kin;
        xmom += other.xmom;
        ymom += other.ymom;
        zmom += other.zmom;
        nmom += other.nmom;
        dkin += other.dkin;
        darkMatterKineticCorrection += other.darkMatterKineticCorrection;
        darkMatterXMomentumCorrection += other.darkMatterXMomentumCorrection;
        darkMatterYMomentumCorrection += other.darkMatterYMomentumCorrection;
        darkMatterZMomentumCorrection += other.darkMatterZMomentumCorrection;
        darkMatterMomentumNormCorrection += other.darkMatterMomentumNormCorrection;

        return *this;
    }
    template <class A>
    void serialize(A&& arc, unsigned) {
        arc & maxRung;
        arc & minimumTimestep;
        arc & pot;
        arc & kin;
        arc & xmom;
        arc & ymom;
        arc & zmom;
        arc & nmom;
        arc & dkin;
        arc & darkMatterKineticCorrection;
        arc & darkMatterXMomentumCorrection;
        arc & darkMatterYMomentumCorrection;
        arc & darkMatterZMomentumCorrection;
        arc & darkMatterMomentumNormCorrection;
    }
};
