/* CosmicTiger - atomic CIE cooling for the SH03 model; BSL-1.0. */
#pragma once

#include <cosmictiger/sph/springel_and_hernquist/equations.hpp>

namespace springel_and_hernquist {

inline constexpr double kBoltzmann = 1.380649e-16;          // erg/K.
inline constexpr double kProtonMass = 1.67262192369e-24;    // g; m_He=4*m_H.
inline constexpr double kElectronVolt = 1.602176634e-12;    // erg.
inline constexpr double kMinimumTemperature = 10.0;
inline constexpr double kMaximumTemperature = 1.0e8;

struct AtomicEquilibrium {
    double temperature{};
    double specificEnergy{};        // Thermal energy only, erg/g.
    double coolingCoefficient{};    // Positive Lambda/rho^2, erg cm^3 g^-2 s^-1.
    double hydrogenNeutral{};       // H fractions normalized to H nuclei.
    double hydrogenIonized{};
    double heliumNeutral{};    // He fractions normalized to He nuclei.
    double heliumIonized{};
    double heliumDoubleIonized{};
    double electronsPerHydrogen{};
    double meanMolecularWeight{};
};

// Collisional ionization equilibrium using the existing Atomic6 reaction and
// thermal-rate primitives. No molecular chemistry, UV field, CMB exchange,
// metal cooling, density cap, or non-equilibrium chemistry integration.
// This is an explicitly CIE variant of SH03, whose original simulations also
// supplied an external UV background. It is not an exact historical cooling fit.
struct AtomicCooling {
    double hydrogenMassFraction{0.76};

    CUDA_EXPORT bool evaluateAtTemperature(double temperature, AtomicEquilibrium& result) const;
    CUDA_EXPORT bool evaluateAtEnergy(double specificEnergy, AtomicEquilibrium& result) const;
    CUDA_EXPORT double minimumEnergy() const;
    CUDA_EXPORT double maximumEnergy() const;

    // Backward-Euler isochoric loss: u_new-u_old+dt*rho*Lambda(u_new)/rho^2=0.
    // A floor only stops radiative losses: gas already below it is not heated.
    // Positive below-floor energy needs no rates, including below 10 K. For
    // rate evaluations outside the Atomic6 domain, return false unchanged.
    CUDA_EXPORT bool cool(double specificEnergy, double density, double timestep,
        double floorEnergy, double& cooledEnergy) const;
};

// Canonical temperatures with explicitly chosen molecular weights: cold 1e3 K
// and warm 1e4 K neutral, SN 1e8 K fully ionized. warmEnergy calibrates the
// threshold; ordinary gas uses the separate time-dependent chemistry solver.
CUDA_EXPORT Parameters defaultParameters(AtomicCooling const& cooling = {});
CUDA_EXPORT bool makeModel(
    Parameters const& parameters, AtomicCooling const& cooling, Model& result);
CUDA_EXPORT bool evaluateEquilibrium(
    Model const& model, AtomicCooling const& cooling, double density, Equilibrium& result);

}    // namespace springel_and_hernquist

// The host ABI retains its out-of-line definitions in cooling.cpp. CUDA's
// device pass sees the same algorithms inline, without device linking.
#ifdef __CUDA_ARCH__
#include <cosmictiger/sph/springel_and_hernquist/cooling_impl.hpp>
#endif
