/* CosmicTiger - GADGET-4 equilibrium cooling/SPH boundary; BSL-1.0. */
#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include <cosmictiger/chemistry/gadget4_cooling.hpp>
#include <cosmictiger/particles/particles.hpp>

struct SphChemistryBatchInput {
    gadget4::State state{};
    float storedEnergy{};
    float storedDensity{};
    double physicalInterval{};
    long long particleIndex = -1;
};
struct SphChemistryBatchResult {
    gadget4::State state{};
    float storedEnergy{};
};
gadget4::Parameters sphCoolingParameters(double scaleFactor);
std::vector<SphChemistryBatchResult> sphChemistryAdvanceBatch(
    std::vector<SphChemistryBatchInput> const& inputs, double scaleFactor);
void sphChemistryInitializeParticle(long long index, double scaleFactor);
// Physical seconds of one code-time interval at a fixed scale factor, as the
// batch input expects; the time scale and cosmological factor follow the SPH source.
double sphChemistryPhysicalInterval(double interval, double scaleFactor);
// One full source at each completed SPH step, matching the pinned GADGET-4 run loop.
double sphChemistryClose(int minimumRung, double scaleFactor, double scaleDerivative,
    double baseTimestep, double explicitTimestep = -1.0);
// Rank-local cooling of the listed gas slots, each over its own completed code-time
// interval. Prepare stages the results and returns the heating diagnostic; commit
// writes them. Callers run every locality's prepare before any commit.
double sphCoolingPrepareLocal(std::vector<ParticleInteger> const& indices,
    std::vector<double> const& intervals, double scaleFactor);
void sphCoolingCommitLocal();
std::array<double, 6> sphChemistryCheckpointParameters();
// Equilibrium H,H+,He,He+,He++,e per H nucleus, then T,mu,gamma.
std::array<float, 9> sphChemistryOutput(
    gadget4::State const& state, float storedEnergy, float storedDensity, double scaleFactor);
double sphChemistryPressureIntegral(double scaleFactor);
