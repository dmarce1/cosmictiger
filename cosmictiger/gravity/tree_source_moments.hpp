/* CosmicTiger - transient owned-source tree moments; BSL-1.0. */
#pragma once

#include <cosmictiger/gravity/fmm_types.hpp>

// Double arithmetic and transport during the upward build. Persistent tree
// records and all walk inputs use the narrowed MultipoleBuffer.
struct SourceMultipoleBuffer {
    Array<double, kFmmMultipoleCoefficients> coefficients{};
    /// Access a coefficient in the double-precision source buffer.
    double& operator[](int i) {
        return coefficients[i];
    }
    /// Read a coefficient in the double-precision source buffer.
    double const& operator[](int i) const {
        return coefficients[i];
    }
    /// Narrow source moments once when storing the node for force evaluation.
    MultipoleBuffer narrow() const {
        MultipoleBuffer result{};
        for (int i = 0; i < FmmOrder<kFmmOrder>::kMultipoleCoefficients_; ++i) {
            result[i] = coefficients[i];
        }
        return result;
    }
    /// Archive this record in the field order required by distributed transport.
    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        for (int i = 0; i < FmmOrder<kFmmOrder>::kMultipoleCoefficients_; ++i) {
            archive& coefficients[i];
        }
    }
};
