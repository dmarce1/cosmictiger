#pragma once

#include <cstddef>
#include <vector>

#include <cosmictiger/core/containers.hpp>

namespace scenarios {

struct DiagnosticSample {
    Array<double, 3> position;
    double density;
    double pressure;
    double weight;

    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & position & density & pressure & weight;
    }
};

std::vector<DiagnosticSample> diagnosticSamples(size_t maximumPerLocality);

}    // namespace scenarios
