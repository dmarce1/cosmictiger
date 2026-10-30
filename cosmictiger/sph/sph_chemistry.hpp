/* CosmicTiger - primordial chemistry/SPH boundary; BSL-1.0. */
#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include <cosmictiger/chemistry/primordial_types.hpp>

struct SphChemistryBatchInput {
    primordial::State state{};
    float storedEnergy{};
    float storedDensity{};
    double physicalInterval{};
};
struct SphChemistryBatchResult {
    primordial::State state{};
    float storedEnergy{};
};

// Nonmutating source transaction: all elements must finish before results return.
// Density and energy use stored code units; intervals are physical seconds.
std::vector<SphChemistryBatchResult> sphChemistryAdvanceBatch(
    std::vector<SphChemistryBatchInput> const& inputs, double scaleFactor);
primordial::State sphChemistrySynchronizeEnergy(
    primordial::State const& state, float storedEnergy, double scaleFactor);
primordial::Eos sphChemistryEos(
    primordial::State const& state, float storedEnergy, double scaleFactor);
void sphChemistryInitializeParticle(long long index, double scaleFactor);
// Close only active particles' saved source halves after the hydro corrector.
// Returns their opening plus closing changes to the cosmological heating ledger.
double sphChemistryClose(int minimumRung, double scaleFactor, double scaleDerivative,
    double baseTimestep, double explicitTimestep = -1.0);
// Open chemistry over half of each active particle's selected SPH timestep.
// The source solver subcycles internally; chemistry does not reselect timesteps.
void sphChemistryOpen(int minimumRung, double scaleFactor, double scaleDerivative,
    double baseTimestep, double explicitTimestep = -1.0, int uniformRung = -1);
std::array<double, 17> sphChemistryCheckpointParameters();
uint64_t sphChemistryRadiationChecksum();

// HI,HII,HeI,HeII,HeIII,H-,H2,H2+,e counts per H nucleus, then T,mu,gamma.
std::array<float, 12> sphChemistryOutput(
    primordial::State const& state, float storedEnergy, double scaleFactor);

double sphChemistryPressureIntegral(double scaleFactor);
