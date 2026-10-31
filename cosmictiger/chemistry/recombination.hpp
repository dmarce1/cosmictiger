/* CosmicTiger - homogeneous primordial recombination; BSL-1.0. */
#pragma once

#include <array>
#include <cstddef>
#include <vector>

namespace primordial {
inline constexpr float kBoltzmannEv = 8.617333262145e-5f;    // eV/K for background output.

// Spatially flat matter + radiation + cosmological constant background.
// The radiation density includes photons and any massless neutrinos.
struct RecombinationParameters {
    double hubble{0.6732};    // H0 / (100 km s^-1 Mpc^-1).
    double omegaB{0.049389};
    double omegaC{0.26503};
    double omegaRadiation{9.231291330230124e-5};
    double heliumMassFraction{0.2454006};
    double cmbTemperature{2.7255};    // Present CMB temperature, K.
};
struct RecombinationOptions {
    double relativeTolerance{1.0e-7};
    double absoluteTolerance{1.0e-12};
    double maxLogStep{0.01};
    double maxScaleFactor{1.1};
    size_t maxSteps{200000};
};

// Homogeneous effective-atom specialization of primordial thermochemistry.
// History abundances use HII/H and HeII/He. This background model is independent
// of the equilibrium atomic cooling used for SPH gas.
// The rates implement RECFAST 1.5.2 Hswitch=1, Heswitch=6, with an independent
// double-precision stiff integrator. This is not the optically thin SPH model.
// No stellar reionization, molecules, deuterium, or massive neutrinos are included.
class RecombinationHistory {
    struct Sample {
        double logScaleFactor;
        std::array<double, 3> values;    // HII/H, HeII/He, Tgas/Tcmb.
        std::array<double, 3> slopes{};
    };
    RecombinationParameters parameters_;
    RecombinationOptions options_;
    double heliumRatio_;
    double hydrogenDensityToday_;    // m^-3 internally.
    double heliumSahaEnd_;
    double hydrogenSahaEnd_;
    std::vector<Sample> samples_;

    // HII/H, HeII/He, HeIII/He, temperature in K.
    std::array<double, 4> sample(double scaleFactor) const;

public:
    explicit RecombinationHistory(
        RecombinationParameters parameters, RecombinationOptions options = {});
    double electronFraction(double scaleFactor) const;    // Free electrons per H nucleus.
    double gasTemperature(double scaleFactor) const;      // K.
    double hydrogenIonizedFraction(double scaleFactor) const;
    double heliumSinglyIonizedFraction(double scaleFactor) const;
    double heliumDoublyIonizedFraction(double scaleFactor) const;
    double minScaleFactor() const;    // Nominal starting epoch; earlier epochs use equilibrium.
    double maxScaleFactor() const;
    double heliumToHydrogenNumberRatio() const;
    double hydrogenNumberDensity(double scaleFactor) const;    // Proper cm^-3.
    double pressurePerHydrogen(double scaleFactor) const;      // P/nH in eV.
};
}    // namespace primordial
