/*
 CosmicTiger - scenario initialization and glass-output entry points.
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

#include <cosmictiger/scenarios/scenario.hpp>

struct DriverParameters;

namespace scenarios {

ScenarioPlan initializeSod();
ScenarioPlan initializeBlast();
ScenarioPlan initializeDisc();
ScenarioPlan initializePolytropicStar();
ScenarioPlan initializeHydroWave();
// A linear adiabatic standing wave in uniform self-gravitating gas: SPH coupled to
// global FMM gravity must oscillate at the Jeans frequency, ω² = c²k² − 4πGρ₀.
ScenarioPlan initializeJeansWave();
ScenarioPlan initializeKelvinHelmholtz();

void initializeCosmology(DriverParameters& state);
void initializeGlass(DriverParameters& state);
void saveGlass();
void initializePlummerParticles();
void initializePlummer(DriverParameters& state);

}    // namespace scenarios
