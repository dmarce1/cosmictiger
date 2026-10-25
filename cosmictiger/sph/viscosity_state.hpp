/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

// Alpha storage shared by the Morris–Monaghan and Cullen–Dehnen evolution models.
// Morris–Monaghan (1997), equation (5), rates use conformal time.
// predictedAlpha is temporary predictor storage; checkpoints and migration omit it.
struct SphViscosityState {
    float alpha;
    float rate;
    float predictedAlpha;
};
