#pragma once

#include <array>
#include <iosfwd>
#include <string>

namespace recfast {

inline constexpr int kSampleCount = 1000;
inline constexpr double kRedshiftStep = 10.0;
inline constexpr double kMaximumRedshift = (kSampleCount - 1) * kRedshiftStep;

// The bundled RECFAST output is z = 9990, 9980, ..., 0 and xe = n_e / n_H:
// https://www.astro.ubc.ca/people/scott/recfast.readme
class IonizationHistory {
    std::array<double, kSampleCount> electronFractions_;

public:
    explicit IonizationHistory(std::istream& input);

    // Evaluate at a positive finite scale factor. Outside the tabulated redshift
    // range, retain the nearest endpoint, including the early fully ionized era.
    double operator()(double scaleFactor) const;
};

IonizationHistory loadIonizationHistory(std::string const& filename);

}    // namespace recfast
