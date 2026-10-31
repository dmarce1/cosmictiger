/* CosmicTiger - equilibrium multiphase SPH source integration; BSL-1.0. */
#pragma once

#include <cstdint>

#include <cosmictiger/particles/particles.hpp>
#include <cosmictiger/sph/springel_and_hernquist/cooling.hpp>

namespace section_timing {
class ScopedSection;
}

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
void starFormationInitialize();
void starFormationRefreshEos(double scaleFactor);
double starFormationMaximumTimestep(int minimumRung, double scaleFactor);
double starFormationPressureIntegral(double scaleFactor);
StarFormationUpdate starFormationClose(int minimumRung, double scaleFactor, double scaleDerivative,
    double baseTimestep, double explicitTimestep = -1.0,
    section_timing::ScopedSection* sourceTimer = nullptr);
void starFormationOpen(int minimumRung, double scaleFactor, double scaleDerivative,
    double baseTimestep, double explicitTimestep = -1.0, int uniformRung = -1,
    section_timing::ScopedSection* sourceTimer = nullptr);
// FMM traditional kicks defer gas gravity to SPH. Newly converted stars need
// exactly the opening gravity half kick; global/mesh finish routines own theirs.
int starFormationOpenNewStars(double baseTimestep, double scaleFactor, double maximumTimestep,
    int minimumRung, bool kickGravity);
uint64_t starFormationRandomBits(uint64_t identity, uint64_t event, uint64_t stream);
double starFormationUniform(uint64_t identity, uint64_t event, uint64_t stream);
