/* CosmicTiger - equilibrium multiphase SPH source integration; BSL-1.0. */
#pragma once

#include <cstdint>
#include <vector>

#include <cosmictiger/hydro/sph/star_formation_kernel.hpp>
#include <cosmictiger/particles/particles.hpp>

struct StarFormationUpdate {
    double heating{};
    uint64_t converted{};
    int maximumNewStarRung{};
    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & heating & converted & maximumNewStarRung;
    }
};

springel_and_hernquist::Model starFormationModel();
star_formation_backend::Parameters starFormationParameters(double scaleFactor);
void starFormationInitialize();
star_formation_backend::Output starFormationDiagnostics(ParticleInteger gas, double scaleFactor);
void starFormationRefreshEos(double scaleFactor);
double starFormationPressureIntegral(double scaleFactor);
StarFormationUpdate starFormationClose(int minimumRung, double scaleFactor, double scaleDerivative,
    double baseTimestep, double explicitTimestep = -1.0);
// Hopkins MFM: the combined cooling/star-formation source for cells whose own-step
// intervals have just closed, each over its own completed code-time interval. This
// touches no particle array; the caller folds the thermal change into its conserved state.
std::vector<star_formation_backend::Output> starFormationAdvanceBatch(
    std::vector<star_formation_backend::Input> const& inputs, std::vector<double> const& intervals,
    double scaleFactor);
// Convert one local gas particle into a star with the saved source state, and
// register it for starFormationOpenNewStars exactly as starFormationClose does.
void starFormationConvertGas(ParticleInteger gas, StarFormationState const& state);
// FMM traditional kicks defer gas gravity to SPH. Newly converted stars need
// exactly the opening gravity half kick; global/mesh finish routines own theirs.
int starFormationOpenNewStars(double baseTimestep, double scaleFactor, double maximumTimestep,
    int minimumRung, bool kickGravity);
uint64_t starFormationRandomBits(uint64_t identity, uint64_t event, uint64_t stream);
double starFormationUniform(uint64_t identity, uint64_t event, uint64_t stream);
