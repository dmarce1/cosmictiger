/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

// Cullen & Dehnen (2010), equations (14)-(17) and (A15). Only divergence
// survives checkpoints/migration; the density pass reconstructs the other fields.
struct SphCullenDehnenState {
    float divergence;    // Physical velocity divergence in inverse code time.
    float pendingDivergence;
    float targetAlpha;
    float inverseDecayTime;    // Inverse conformal decay time.
};
