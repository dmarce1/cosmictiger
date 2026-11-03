/* CosmicTiger - persistent GADGET-4 star-formation metadata; BSL-1.0. */
#pragma once

#include <cstdint>

struct StarFormationState {
    uint64_t id = 0;
    uint64_t eventCounter = 0;
    // GADGET-4 birth time: scale factor if comoving, physical seconds otherwise.
    double formationTime = -1.0;
    // Saved post-cooling SFR / gas mass, used for the next full-step event.
    double specificStarFormationRate = 0.0;
    double elapsedPhysicalTime = 0.0;

    bool operator==(StarFormationState const&) const = default;

    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & id & eventCounter & formationTime & specificStarFormationRate &
            elapsedPhysicalTime;
    }
};
