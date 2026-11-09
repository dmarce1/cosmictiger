/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

#include <cosmictiger/scenarios/scenario.hpp>

namespace scenarios {

// A linear adiabatic standing wave in uniform self-gravitating gas: SPH coupled to
// PMMM gravity must oscillate at the Jeans frequency, ω² = c²k² − 4πGρ₀.
ScenarioPlan initializeJeansWave();

}    // namespace scenarios
