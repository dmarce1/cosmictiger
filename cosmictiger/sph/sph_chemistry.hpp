/* CosmicTiger - primordial chemistry/SPH boundary; GPL-2.0-or-later. */
#pragma once

#include <array>
#include <cstdint>

#include <cosmictiger/chemistry/primordial_types.hpp>

primordial::Eos sphChemistryEos(
    primordial::State const& state, float storedEnergy, double scaleFactor);
void sphChemistryInitializeParticle(long long index, double scaleFactor);
// Returns the negative change in a * total physical thermal energy (code units),
// for the existing cosmological heating ledger. All ranks participate.
double sphChemistryAdvance(
    double scaleFactor, double intervalScaleFactor, double conformalInterval);
void sphChemistrySetRungs();
std::array<double, 17> sphChemistryCheckpointParameters();
uint64_t sphChemistryRadiationChecksum();

// HI,HII,HeI,HeII,HeIII,H-,H2,H2+,e counts per H nucleus, then T,mu,gamma.
std::array<float, 12> sphChemistryOutput(
    primordial::State const& state, float storedEnergy, double scaleFactor);

// Opening source energy belongs to the next synchronized diagnostic epoch.
double& sphChemistryPendingHeating();

double sphChemistryPressureIntegral(double scaleFactor);
