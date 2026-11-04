/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

// Cullen & Dehnen (2010), equations (14)-(17), (A15), and Appendix B3's
// time-difference estimator. The committed divergence and timestamp survive
// checkpoints/migration; pending fields belong to the current density solve.
struct SphCullenDehnenState {
    float divergence;    // Physical velocity divergence in inverse code time.
    float pendingDivergence;
    float targetBulkViscosityCoefficient;
    float inverseDecayTime;      // Inverse conformal decay time.
    float limiter = 1.0f;        // Reconstructed by the density pass for Hu et al.
    double sampleTime = -1.0;    // Conformal time; negative means no committed sample.
    // Physical divergence for CD/Hu; physical divergence minus 3H for Price.
    float sampleDivergence = 0.0f;
    float pendingSampleDivergence = 0.0f;
};
