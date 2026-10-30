/* CosmicTiger - persistent baryonic source metadata; BSL-1.0. */
#pragma once

#include <cstdint>

struct StarFormationState {
    uint64_t id = 0;    // Zero is reserved for an uninitialized identity.
    uint64_t eventCounter = 0;
    double formationTime = -1.0;       // Physical seconds; negative for nonstellar particles.
    double windTimeRemaining = 0.0;    // Physical seconds.
    float metallicity = 0.0f;          // Metal mass fraction.
    double sourceHalfInterval = 0.0;
    double pendingHeating = 0.0;
    double formationOpticalDepth = 0.0;
    double elapsedPhysicalTime = 0.0;

    bool operator==(StarFormationState const&) const = default;

    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & id & eventCounter & formationTime & windTimeRemaining & metallicity &
            sourceHalfInterval & pendingHeating & formationOpticalDepth & elapsedPhysicalTime;
    }
};
