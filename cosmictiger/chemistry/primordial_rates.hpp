/* CosmicTiger - primordial thermochemistry; BSL-1.0. */
#pragma once

#include <cmath>

#include <cosmictiger/chemistry/primordial_eos.hpp>
#include <cosmictiger/cuda/cuda_export.hpp>

namespace primordial {
inline constexpr int kReactionCount = 37;
inline constexpr int kCollisionCount = 29;
inline constexpr int kChemicalSpeciesCount = 9;

// Rows follow Abel et al. (1997), Appendix A: 19 collision channels and
// eight photo channels (direct and Solomon H2 dissociation are aggregated),
// followed by ten additional GA08 low-density H/He channels.
// Species columns: H, H+, He, He+, He++, e, H-, H2, H2+.
// https://arxiv.org/abs/astro-ph/9608040 ; the adopted updates are pinned in
// docs/primordial_reactions.md. Integers make nuclei/charge audits exact.
CUDA_EXPORT inline int reactionStoichiometry(int reaction, int species) {
    int const rows[kReactionCount][kChemicalSpeciesCount] = {{-1, 1, 0, 0, 0, 1, 0, 0, 0},
        {1, -1, 0, 0, 0, -1, 0, 0, 0}, {0, 0, -1, 1, 0, 1, 0, 0, 0}, {0, 0, 1, -1, 0, -1, 0, 0, 0},
        {0, 0, 0, -1, 1, 1, 0, 0, 0}, {0, 0, 0, 1, -1, -1, 0, 0, 0}, {-1, 0, 0, 0, 0, -1, 1, 0, 0},
        {-1, 0, 0, 0, 0, 1, -1, 1, 0}, {-1, -1, 0, 0, 0, 0, 0, 0, 1}, {-1, 1, 0, 0, 0, 0, 0, 1, -1},
        {1, -1, 0, 0, 0, 0, 0, -1, 1}, {2, 0, 0, 0, 0, 0, 0, -1, 0}, {2, 0, 0, 0, 0, 0, 0, -1, 0},
        {1, 0, 0, 0, 0, 1, -1, 0, 0}, {1, 0, 0, 0, 0, 1, -1, 0, 0}, {2, -1, 0, 0, 0, 0, -1, 0, 0},
        {0, -1, 0, 0, 0, 1, -1, 0, 1}, {2, 0, 0, 0, 0, -1, 0, 0, -1}, {1, 0, 0, 0, 0, 0, -1, 1, -1},
        {-1, 1, 0, 0, 0, 1, 0, 0, 0}, {0, 0, -1, 1, 0, 1, 0, 0, 0}, {0, 0, 0, -1, 1, 1, 0, 0, 0},
        {1, 0, 0, 0, 0, 1, -1, 0, 0}, {2, 0, 0, 0, 0, 0, 0, -1, 0}, {0, 0, 0, 0, 0, 1, 0, -1, 1},
        {1, 1, 0, 0, 0, 0, 0, 0, -1}, {0, 2, 0, 0, 0, 1, 0, 0, -1}, {2, 0, 0, 0, 0, 0, 0, -1, 0},
        {2, 0, 0, 0, 0, 0, 0, -1, 0}, {3, 0, 0, 0, 0, 0, -1, 0, -1}, {1, 0, 0, 0, 0, -1, 1, -1, 0},
        {1, 1, 1, -1, 0, 0, 0, -1, 0}, {0, 0, 1, -1, 0, 0, 0, -1, 1}, {-1, 1, 1, -1, 0, 0, 0, 0, 0},
        {1, -1, -1, 1, 0, 0, 0, 0, 0}, {1, 0, 1, -1, 0, 0, -1, 0, 0}, {1, 0, 0, 0, 0, 1, -1, 0, 0}};
    return rows[reaction][species];
}
struct RateCoefficients {
    float collision[kCollisionCount]{};      // cm^3 s^-1, including He+ dielectronic recombination.
    float hydrogenRecombinationCooling{};    // eV cm^3 s^-1.
    float heliumRecombinationCooling{};
    float heliumDoubleRecombinationCooling{};
};
struct ThermalRates {
    // Positive losses and gains are kept separate until the compensated sum.
    FloatFloat ionizationCooling{}, recombinationCooling{}, excitationCooling{};
    FloatFloat freeFreeCooling{}, molecularLineCooling{}, molecularIonLineCooling{},
        molecularChemicalCooling{};
    FloatFloat molecularChemicalHeating{}, photoHeating{}, comptonHeating{};
    // Upper bound on chemical binding-energy release from channels whose
    // product excitation/photon partition is not resolved by this thermal
    // model. Diagnostic only, not an invented thermal-energy source.
    FloatFloat unresolvedChemicalEnergyRelease{};
    CUDA_EXPORT FloatFloat netHeating() const {
        return ((photoHeating - ionizationCooling) + (comptonHeating - freeFreeCooling)) +
            (molecularChemicalHeating - molecularChemicalCooling) - recombinationCooling -
            excitationCooling - molecularLineCooling - molecularIonLineCooling;
    }
};
namespace detail {
    template <int count>
    CUDA_EXPORT inline float polynomial(float x, float const (&coefficients)[count]) {
        FloatFloat result(coefficients[count - 1]);
        for (int i = count - 2; i >= 0; --i) {
            result = result * FloatFloat(x) + FloatFloat(coefficients[i]);
        }
        return result.toFloat();
    }
    CUDA_EXPORT inline float coldMolecularHydrogenCollisionCooling(float temperature) {
        // Hernandez et al. 2021, A&A 647 A155, Table 2 and eqs. (5)-(7):
        // https://doi.org/10.1051/0004-6361/202039790
        // Published CDS table2.dat, J/A+A/647/A155 (retrieved 2026-10-20):
        // https://cdsarc.cds.unistra.fr/ftp/J/A+A/647/A155/table2.dat
        // Polynomial downward rates are in 1e-20 m^3/s = 1e-14 cm^3/s;
        // the fits cover 10..200 K and are used here only below 100 K.
        // Rows: (2,0)->(0,0), (2,1)->(0,1), (3,0)->(1,0), (3,1)->(1,1).
        // Eq. (6)'s identical-particle factors are already in the published
        // rates. No additional factor of two belongs in this cooling sum.
        float const downward[][6] = {
            {52.2644f, -1.30816f, 0.0325606f, -3.01884e-4f, 1.33791e-6f, -2.24735e-9f},
            {74.5105f, -1.44478f, 0.0389100f, -3.50088e-4f, 1.53233e-6f, -2.55815e-9f},
            {26.4090f, -0.530468f, 0.0141524f, -1.23414e-4f, 5.39467e-7f, -9.00632e-10f},
            {25.5303f, -0.481591f, 0.0137928f, -1.17786e-4f, 5.17233e-7f, -8.66778e-10f}};
        FloatFloat cooling(0.0f);
        for (int transition = 0; transition != 4; ++transition) {
            bool const paraTarget = transition < 2;
            float const gap = paraTarget ? 509.86f : 844.60f;
            float const degeneracy = paraTarget ? 5.0f : 7.0f / 3.0f;
            float const targetFraction = paraTarget ? 0.25f : 0.75f;
            float const colliderFraction = transition % 2 == 0 ? 0.25f : 0.75f;
            float const excitation = 1.e-14f * polynomial(temperature, downward[transition]) *
                degeneracy * ::expf(-gap / temperature);
            // Low-density ground-ladder approximation, fixed ortho:para=3:1,
            // as in Lee et al. 2008 eq. (5), https://arxiv.org/abs/0805.1623 .
            // Excitation energy subsequently escapes as line radiation.
            cooling += FloatFloat(targetFraction * colliderFraction) * FloatFloat(excitation) *
                FloatFloat(kBoltzmannEv * gap);
        }
        return cooling.toFloat();    // eV cm^3/s per H2--H2 pair density.
    }
    CUDA_EXPORT inline float matchColdCooling(
        float temperature, float cooling, float logRatio, float logSlopeDelta) {
        if (temperature <= 80.0f) {
            return cooling;
        }
        // Explicit compatibility interpolation, not a published rate fit:
        // add a cubic Hermite correction to ln(cooling) on 80..100 K.
        // Its value/slope vanish at 80 K and match the established warm fit
        // at 100 K. The input cold fits stay within their published domains.
        float const s = (temperature - 80.0f) / 20.0f;
        float const sSquared = s * s;
        float const correction =
            logRatio * sSquared * (3.0f - 2.0f * s) + 20.0f * logSlopeDelta * sSquared * (s - 1.0f);
        return cooling * ::expf(correction);
    }
    CUDA_EXPORT inline float hydrogenRecombination(float lambda) {
        // Hui & Gnedin 1997 Appendix A, case A (all recombination photons escape).
        // https://arxiv.org/abs/astro-ph/9612232
        return 1.269e-13f * ::powf(lambda, 1.503f) /
            ::powf(1.0f + ::powf(lambda / 0.522f, 0.470f), 1.923f);
    }
    CUDA_EXPORT inline float hydrogenRecombinationCooling(float temperature, float lambda) {
        return (1.778e-29f / kElectronVolt) * temperature * ::powf(lambda, 1.965f) /
            ::powf(1.0f + ::powf(lambda / 0.541f, 0.502f), 2.697f);
    }
    CUDA_EXPORT inline float heliumRecombination(float temperature) {
        // Verner & Ferland 1996 eq. (4), Table 1 He I(b), valid 3..1e9 K.
        // Use their single wide-range fit, avoiding a discontinuous 1e6 K splice.
        // https://arxiv.org/abs/astro-ph/9509083
        float const low = ::sqrtf(temperature / 0.04266f);
        float const high = ::sqrtf(temperature / 4.677e6f);
        return 9.356e-10f / (low * ::powf(1.0f + low, 0.2108f) * ::powf(1.0f + high, 1.7892f));
    }
    CUDA_EXPORT inline float molecularChargeTransfer(float temperature) {
        // Savin et al. 2004, as tabulated by Glover & Abel 2008, Appendix A.
        // https://academic.oup.com/mnras/article/388/4/1627/981339
        // This polynomial is severely cancelling. Split decimal coefficients and
        // compensated Horner preserve the fit without FP64 device arithmetic.
        FloatFloat const a[] = {FloatFloat::fromComponents(-3.323218323e-07f, 2.274828953e-15f),
            FloatFloat::fromComponents(3.373538107e-07f, 9.335646473e-15f),
            FloatFloat::fromComponents(-1.449136846e-07f, 4.607672462e-15f),
            FloatFloat::fromComponents(3.417280325e-08f, 1.746204161e-15f),
            FloatFloat::fromComponents(-4.781372187e-09f, 1.870682058e-16f),
            FloatFloat::fromComponents(3.973154306e-10f, -1.055587762e-17f),
            FloatFloat::fromComponents(-1.817141056e-11f, -4.431779881e-19f),
            FloatFloat::fromComponents(3.531193079e-13f, 1.211717088e-20f)};
        FloatFloat const x(::logf(temperature));
        FloatFloat value = a[7];
        for (int i = 6; i >= 0; --i) {
            value = value * x + a[i];
        }
        return value.toFloat() * ::expf(-21237.15f / temperature);
    }
}    // namespace detail

CUDA_EXPORT inline Status evaluateRateCoefficients(
    float temperature, Network network, RateCoefficients& result) {
    if (!finite(temperature) || temperature <= 0.0f ||
        (network != Network::Atomic6 && network != Network::Molecular9)) {
        return Status::InvalidInput;
    }
    if (temperature < minimumTemperature(network) || temperature > maximumTemperature(network)) {
        return Status::UnsupportedState;
    }
    result = RateCoefficients{};
    float const root = ::sqrtf(temperature);
    float const suppression = 1.0f + ::sqrtf(temperature / 1.0e5f);
    float* k = result.collision;
    // Cen 1992 fits as reproduced in Katz, Weinberg & Hernquist 1996 Table 1.
    // Ionization cooling below uses these very same reaction fluxes once.
    // https://arxiv.org/abs/astro-ph/9509107
    k[0] = 5.85e-11f * root * ::expf(-157809.1f / temperature) / suppression;
    k[2] = 2.38e-11f * root * ::expf(-285335.4f / temperature) / suppression;
    k[4] = 5.68e-12f * root * ::expf(-631515.0f / temperature) / suppression;
    k[1] = detail::hydrogenRecombination(315614.0f / temperature);
    k[5] = 2.0f * detail::hydrogenRecombination(1263030.0f / temperature);
    float const radiativeHelium = detail::heliumRecombination(temperature);
    float const dielectronicShape = ::powf(temperature, -1.5f) * ::expf(-470000.0f / temperature) *
        (1.0f + 0.3f * ::expf(-94000.0f / temperature));
    k[3] = radiativeHelium + 1.9e-3f * dielectronicShape;
    result.hydrogenRecombinationCooling =
        detail::hydrogenRecombinationCooling(temperature, 315614.0f / temperature);
    result.heliumDoubleRecombinationCooling =
        8.0f * detail::hydrogenRecombinationCooling(temperature, 1263030.0f / temperature);
    // Conventional He I thermal closure: kBT per radiative recombination,
    // plus the independent Cen dielectronic cooling fit. No ionization
    // threshold is subtracted here: recombination photons carry binding energy.
    result.heliumRecombinationCooling = kBoltzmannEv * temperature * radiativeHelium +
        (1.24e-13f / kElectronVolt) * dielectronicShape;
    if (network == Network::Atomic6) {
        return Status::Success;
    }
    float const logTemperature = ::log10f(temperature);
    float const electronTemperature = kBoltzmannEv * temperature;
    float const logElectronTemperature = ::logf(electronTemperature);
    // Glover & Abel 2008 Appendix A: Wishart attachment; fiducial associative
    // detachment; Galli-Palla radiative association; Savin charge transfer.
    float const attachment[] = {-17.845f, 0.762f, 0.1523f, -0.03274f};
    k[6] = ::powf(10.0f, detail::polynomial(logTemperature, attachment));
    k[7] = 1.3e-9f;
    float const association[] = {-19.38f, -1.523f, 1.118f, -0.1269f};
    k[8] = ::powf(10.0f, detail::polynomial(logTemperature, association));
    k[9] = 6.4e-10f;
    k[10] = detail::molecularChargeTransfer(temperature);
    // Glover & Abel 2008 Table A1 vibrational-ground-state dissociation.
    // evaluateDensityDependentRates supplies the prescribed LTE interpolation.
    k[11] = 4.49e-9f * ::powf(temperature, 0.11f) * ::expf(-101858.0f / temperature);
    k[12] = 6.67e-12f * root * ::expf(-(1.0f + 63593.0f / temperature));
    float const electronDetachment[] = {-18.01849334273f, 2.360852208681f, -0.2827443061704f,
        0.01623316639567f, -0.03365012031363f, 0.01178329782711f, -0.001656194699504f,
        0.0001068275202678f, -2.631285809207e-6f};
    k[13] = ::expf(detail::polynomial(logElectronTemperature, electronDetachment));
    float const atomDetachment[] = {-20.3726089653f, 1.13944933584f, -0.142101352155f,
        0.00846445538663f, -0.0014327641213f, 0.00020122502848f, 0.00008663963243f,
        -0.0000258500968026f, 2.455501197e-6f, -8.06838246118e-8f};
    k[14] = electronTemperature <= 0.1f ?
        2.5634e-9f * ::powf(electronTemperature, 1.78186f) :
        ::expf(detail::polynomial(logElectronTemperature, atomDetachment));
    // Croft et al. 1999 mutual neutralization and GA08 Appendix A channels
    // for H2+ formation, dissociative recombination, and H- destruction.
    k[15] = 2.4e-6f * (1.0f + temperature / 20000.0f) / root;
    k[16] = 6.9e-9f * ::powf(temperature, -0.35f);
    k[17] = temperature <= 617.0f ? 1.0e-8f : 1.32e-6f * ::powf(temperature, -0.76f);
    // GA08 Table A1: split H- + H2+ products, molecular dissociation by H2
    // and He, dissociative attachment, and minor H/He charge-transfer channels.
    k[18] = 1.4e-7f * ::sqrtf(300.0f / temperature);
    k[19] = 5.996e-30f * ::powf(temperature, 4.1881f) /
        ::powf(1.0f + 6.761e-6f * temperature, 5.6881f) * ::expf(-54657.4f / temperature);
    k[20] = ::powf(10.0f, -27.029f + 3.801f * logTemperature - 29487.0f / temperature);
    k[21] = k[18];
    k[22] = 2.7e-8f * ::powf(temperature, -1.27f) * ::expf(-43000.0f / temperature);
    k[23] = 3.7e-14f * ::expf(35.0f / temperature);
    k[24] = 7.2e-15f;
    k[25] = 1.2e-15f * ::powf(temperature / 300.0f, 0.25f);
    k[26] = 1.26e-9f * ::powf(temperature, -0.75f) * ::expf(-127500.0f / temperature);
    k[27] = 2.32e-7f * ::powf(temperature / 300.0f, -0.52f) * ::expf(temperature / 22400.0f);
    k[28] = 4.1e-17f * temperature * temperature * ::expf(-19870.0f / temperature);
    for (int i = 0; i != kCollisionCount; ++i) {
        if (!finite(k[i]) || k[i] < 0.0f) {
            return Status::UnsupportedState;
        }
    }
    return Status::Success;
}

// GA08 eqs. (13)-(17). Interpolate logarithms analytically so cold rates
// can underflow naturally only after interpolation; never log a rounded zero.
// Its critical-density closure assumes electron excitation is subdominant at
// high density, and is not a state-resolved rovibrational population model.
CUDA_EXPORT inline void evaluateDensityDependentRates(Composition const& c,
    Environment const& environment, float temperature, RateCoefficients& result) {
    if (environment.network != Network::Molecular9) {
        return;
    }
    float const logTemperature = ::log10f(temperature);
    float const t4 = logTemperature - 4.0f;
    float const criticalH = ::powf(10.0f, 3.0f - 0.416f * t4 - 0.327f * t4 * t4);
    float const criticalH2 = ::powf(10.0f, 4.845f - 1.3f * t4 + 1.62f * t4 * t4);
    float const criticalHe = ::powf(10.0f, 5.0792f * (1.0f - 1.23e-5f * (temperature - 2000.0f)));
    FloatFloat const densityRatio = FloatFloat(environment.nHydrogen) *
        (c.hydrogen / FloatFloat(criticalH) +
            FloatFloat(2.0f) * c.molecularHydrogen / FloatFloat(criticalH2) +
            c.helium / FloatFloat(criticalHe));
    float const lteFraction = (densityRatio / (FloatFloat(1.0f) + densityRatio)).toFloat();
    float const naturalLog = ::logf(temperature);
    float const logLow[] = {::logf(4.49e-9f) + 0.11f * naturalLog - 101858.0f / temperature,
        ::logf(6.67e-12f) + 0.5f * naturalLog - 1.0f - 63593.0f / temperature,
        ::logf(5.996e-30f) + 4.1881f * naturalLog - 5.6881f * ::log1pf(6.761e-6f * temperature) -
            54657.4f / temperature,
        2.302585092994046f * (-27.029f + 3.801f * logTemperature - 29487.0f / temperature)};
    float const logLte[] = {::logf(1.91e-9f) + 0.136f * naturalLog - 53407.1f / temperature,
        ::logf(3.52e-9f) - 43900.0f / temperature, ::logf(1.3e-9f) - 53300.0f / temperature,
        2.302585092994046f * (-2.729f - 1.75f * logTemperature - 23474.0f / temperature)};
    int const indices[] = {11, 12, 19, 20};
    for (int i = 0; i != 4; ++i) {
        result.collision[indices[i]] = ::expf(logLow[i] + lteFraction * (logLte[i] - logLow[i]));
    }
}

CUDA_EXPORT inline Status validateRadiation(Radiation const& r) {
    float const values[] = {r.hydrogenIonization, r.heliumIonization, r.heliumPlusIonization,
        r.hydrogenMinusDetachment, r.molecularHydrogenDissociation, r.molecularHydrogenIonization,
        r.molecularHydrogenPlusDissociation, r.molecularHydrogenPlusIonization,
        r.heatingPerHydrogen, r.heatingPerHelium, r.heatingPerHeliumPlus, r.heatingPerHydrogenMinus,
        r.heatingPerMolecularHydrogen, r.heatingPerMolecularHydrogenPlus};
    for (float value : values) {
        if (!finite(value) || value < 0.0f) {
            return Status::InvalidInput;
        }
    }
    return Status::Success;
}
CUDA_EXPORT inline void reactionFluxes(Composition const& c, Environment const& environment,
    RateCoefficients const& rates, FloatFloat (&flux)[kReactionCount]) {
    FloatFloat const reactantA[] = {c.hydrogen, c.hydrogenPlus, c.helium, c.heliumPlus,
        c.heliumPlus, c.heliumDoublePlus, c.hydrogen, c.hydrogenMinus, c.hydrogen,
        c.molecularHydrogenPlus, c.molecularHydrogen, c.molecularHydrogen, c.molecularHydrogen,
        c.hydrogenMinus, c.hydrogenMinus, c.hydrogenMinus, c.hydrogenMinus, c.molecularHydrogenPlus,
        c.molecularHydrogenPlus, c.molecularHydrogen, c.molecularHydrogen, c.hydrogenMinus,
        c.molecularHydrogen, c.molecularHydrogen, c.molecularHydrogen, c.heliumPlus, c.helium,
        c.heliumPlus, c.helium};
    FloatFloat const reactantB[] = {c.electrons, c.electrons, c.electrons, c.electrons, c.electrons,
        c.electrons, c.electrons, c.hydrogen, c.hydrogenPlus, c.hydrogen, c.hydrogenPlus,
        c.electrons, c.hydrogen, c.electrons, c.hydrogen, c.hydrogenPlus, c.hydrogenPlus,
        c.electrons, c.hydrogenMinus, c.molecularHydrogen, c.helium, c.molecularHydrogenPlus,
        c.electrons, c.heliumPlus, c.heliumPlus, c.hydrogen, c.hydrogenPlus, c.hydrogenMinus,
        c.hydrogenMinus};
    for (int i = 0; i != kCollisionCount; ++i) {
        // k*n first keeps the largest physical factors together before trace
        // abundances, and FF products/source sums retain cancellation residuals.
        flux[i < 19 ? i : i + 8] =
            (FloatFloat(rates.collision[i]) * FloatFloat(environment.nHydrogen)) * reactantA[i] *
            reactantB[i];
    }
    Radiation const& r = environment.radiation;
    flux[19] = FloatFloat(r.hydrogenIonization) * c.hydrogen;
    flux[20] = FloatFloat(r.heliumIonization) * c.helium;
    flux[21] = FloatFloat(r.heliumPlusIonization) * c.heliumPlus;
    flux[22] = FloatFloat(r.hydrogenMinusDetachment) * c.hydrogenMinus;
    flux[23] = FloatFloat(r.molecularHydrogenDissociation) * c.molecularHydrogen;
    flux[24] = FloatFloat(r.molecularHydrogenIonization) * c.molecularHydrogen;
    flux[25] = FloatFloat(r.molecularHydrogenPlusDissociation) * c.molecularHydrogenPlus;
    flux[26] = FloatFloat(r.molecularHydrogenPlusIonization) * c.molecularHydrogenPlus;
}

// Low-density H2 cooling coefficients (eV cm^3/s), fixed ortho:para=3:1.
// Glover & Abel 2008 Table 8, https://arxiv.org/abs/0803.1768 .
// Original warm fits are retained. Below 100 K, H2 collisions use the
// Hernandez et al. data above; all other partners use GA08's cold fits.
// Order of collision partners: H, H2, He, H+, e. All five are retained.
// This remains spontaneous line cooling with no molecular CMB absorption or
// evolved spin populations. Compton exchange is a separate thermal process.
CUDA_EXPORT inline Status molecularCoolingCoefficients(
    float temperature, float (&cooling)[5], float& lte) {
    if (!finite(temperature) || temperature < 10.0f || temperature > 6000.0f) {
        return Status::UnsupportedState;
    }
    float const x = ::log10f(temperature / 1000.0f);
    float const atomLow[] = {
        -24.311209f, 3.5692468f, -11.332860f, -27.850082f, -21.328264f, -4.2519023f};
    float const atomHigh[] = {
        -24.311209f, 4.6450521f, -3.7209846f, 5.9369081f, -5.5108047f, 1.5538288f};
    float const molecule[] = {
        -23.962112f, 2.09433740f, -0.77151436f, 0.43693353f, -0.14913216f, -0.033638326f};
    float const helium[] = {
        -23.689237f, 2.1892372f, -0.81520438f, 0.29036281f, -0.16596184f, 0.19191375f};
    float const proton[] = {
        -21.716699f, 1.3865783f, -0.37915285f, 0.11453688f, -0.23214154f, 0.058538864f};
    float const electronLow[] = {
        -34.286155f, -48.537163f, -77.121176f, -51.352459f, -15.169160f, -0.98120322f};
    float const electronHigh[] = {
        -22.190316f, 1.5728955f, -0.213351f, 0.96149759f, -0.91023195f, 0.13749749f};
    if (temperature < 100.0f) {
        // GA08 Table 8 / eq. (40), 10..100 K H branch. GA08 explicitly
        // describes its underlying low-T H-collision continuation in eqs.
        // (28)-(29); the He, proton and electron fits also extend to 10 K.
        float const atomCold[] = {
            -16.818342f, 37.383713f, 58.145166f, 48.656103f, 20.159831f, 3.8479610f};
        // Convert erg to eV before exponentiating: at 10 K the H and electron
        // coefficients are representable in eV but underflow in FP32 ergs.
        float const logElectronVolt = ::log10f(kElectronVolt);
        cooling[0] = ::powf(10.0f, detail::polynomial(x, atomCold) - logElectronVolt);
        cooling[1] = detail::coldMolecularHydrogenCollisionCooling(temperature);
        cooling[2] = ::powf(10.0f, detail::polynomial(x, helium) - logElectronVolt);
        cooling[3] = ::powf(10.0f, detail::polynomial(x, proton) - logElectronVolt);
        cooling[4] = ::powf(10.0f, detail::polynomial(x, electronLow) - logElectronVolt);
        // ln(warm/cold) and d ln(warm/cold)/dT at 100 K, independently
        // evaluated from the cited decimal coefficients. Raw discrepancies
        // are 9.26% (H) and 15.92% (H2); only 80..100 K is adjusted.
        cooling[0] = detail::matchColdCooling(
            temperature, cooling[0], -0.08858850757531106f, 0.01076094300000003f);
        cooling[1] = detail::matchColdCooling(
            temperature, cooling[1], -0.14771090012072186f, -0.0055088563359083076f);
    } else {
        cooling[0] = ::powf(10.0f,
                         temperature < 1000.0f ? detail::polynomial(x, atomLow) :
                                                 detail::polynomial(x, atomHigh)) /
            kElectronVolt;
        cooling[1] = ::powf(10.0f, detail::polynomial(x, molecule)) / kElectronVolt;
        cooling[2] = ::powf(10.0f, detail::polynomial(x, helium)) / kElectronVolt;
        cooling[3] = ::powf(10.0f, detail::polynomial(x, proton)) / kElectronVolt;
        cooling[4] = ::powf(10.0f,
                         temperature < 200.0f ? detail::polynomial(x, electronLow) :
                                                detail::polynomial(x, electronHigh)) /
            kElectronVolt;
    }
    // Hollenbach & McKee LTE asymptote, reproduced by Yoshida et al. 2006
    // eqs. (18)-(20), https://arxiv.org/abs/astro-ph/0606106 . Units eV/s/H2.
    float const t = temperature / 1000.0f;
    lte = (9.5e-22f / kElectronVolt) * ::powf(t, 3.76f) / (1.0f + 0.12f * ::powf(t, 2.1f)) *
            ::expf(-::powf(0.13f / t, 3.0f)) +
        (3.0e-24f / kElectronVolt) * ::expf(-0.51f / t) +
        (6.7e-19f / kElectronVolt) * ::expf(-5.86f / t) +
        (1.6e-18f / kElectronVolt) * ::expf(-11.7f / t);
    return Status::Success;
}

CUDA_EXPORT inline Status evaluateThermalRates(Composition const& c, Environment const& environment,
    float temperature, RateCoefficients const& rates, FloatFloat const (&flux)[kReactionCount],
    ThermalRates& result) {
    result = ThermalRates{};
    FloatFloat const electronDensity = FloatFloat(environment.nHydrogen) * c.electrons;
    float const root = ::sqrtf(temperature);
    float const suppression = 1.0f + ::sqrtf(temperature / 1.0e5f);
    result.ionizationCooling = FloatFloat(13.5984f) * flux[0] + FloatFloat(24.5874f) * flux[2] +
        FloatFloat(54.4178f) * flux[4];
    result.recombinationCooling = electronDensity *
        (c.hydrogenPlus * FloatFloat(rates.hydrogenRecombinationCooling) +
            c.heliumPlus * FloatFloat(rates.heliumRecombinationCooling) +
            c.heliumDoublePlus * FloatFloat(rates.heliumDoubleRecombinationCooling));
    // Katz et al. 1996 Table 1: HI and HeII excitation and free-free cooling.
    float const hydrogenExcitation =
        (7.5e-19f / kElectronVolt) * ::expf(-118348.0f / temperature) / suppression;
    float const heliumPlusExcitation = (5.54e-17f / kElectronVolt) * ::powf(temperature, -0.397f) *
        ::expf(-473638.0f / temperature) / suppression;
    // Ground-state neutral He excitation: Glover & Jappsen 2007,
    // Table 7 (Bray et al. 2000), https://arxiv.org/abs/0705.0182 .
    // The metastable-He contribution is omitted from this ground-state closure.
    float const heliumExcitation =
        (1.1e-19f / kElectronVolt) * ::powf(temperature, 0.082f) * ::expf(-230000.0f / temperature);
    result.excitationCooling = electronDensity *
        (c.hydrogen * FloatFloat(hydrogenExcitation) + c.helium * FloatFloat(heliumExcitation) +
            c.heliumPlus * FloatFloat(heliumPlusExcitation));
    float const logTemperature = ::log10f(temperature);
    float const gaunt =
        1.1f + 0.34f * ::expf(-(5.5f - logTemperature) * (5.5f - logTemperature) / 3.0f);
    result.freeFreeCooling = electronDensity *
        (c.hydrogenPlus + c.heliumPlus + FloatFloat(4.0f) * c.heliumDoublePlus) *
        FloatFloat((1.42e-27f / kElectronVolt) * root * gaunt);    // ionic Z^2, including He++.
    Radiation const& r = environment.radiation;
    result.photoHeating = c.hydrogen * FloatFloat(r.heatingPerHydrogen) +
        c.helium * FloatFloat(r.heatingPerHelium) +
        c.heliumPlus * FloatFloat(r.heatingPerHeliumPlus) +
        c.hydrogenMinus * FloatFloat(r.heatingPerHydrogenMinus) +
        c.molecularHydrogen * FloatFloat(r.heatingPerMolecularHydrogen) +
        c.molecularHydrogenPlus * FloatFloat(r.heatingPerMolecularHydrogenPlus);
    // 4 sigma_T a_r kB/(m_e c), in eV s^-1 K^-5. Thomson, nonrelativistic
    // Compton exchange (Hui & Gnedin Appendix A); its sign is not clipped.
    float const cmbSquared = environment.cmbTemperature * environment.cmbTemperature;
    result.comptonHeating = c.electrons * FloatFloat(6.35264354e-26f * cmbSquared * cmbSquared) *
        FloatFloat(environment.cmbTemperature - temperature);
    if (environment.network == Network::Molecular9) {
        float cooling[5], lte;
        Status const status = molecularCoolingCoefficients(temperature, cooling, lte);
        if (status != Status::Success) {
            return status;
        }
        FloatFloat const low = FloatFloat(environment.nHydrogen) *
            (c.hydrogen * FloatFloat(cooling[0]) + c.molecularHydrogen * FloatFloat(cooling[1]) +
                c.helium * FloatFloat(cooling[2]) + c.hydrogenPlus * FloatFloat(cooling[3]) +
                c.electrons * FloatFloat(cooling[4]));
        // Stable low-density/LTE bridge; neither product nor ratio of two tiny
        // cooling functions is needed. LTE is an asymptote, not a density gate.
        // Preserve the zero-emission limit if an exponentially small LTE rate
        // underflows. This also avoids 0/0 when the collider density is zero.
        FloatFloat const emission =
            lte > 0.0f ? low / (FloatFloat(1.0f) + low / FloatFloat(lte)) : FloatFloat(0.0f);
        result.molecularLineCooling = c.molecularHydrogen * emission;
        // GA08 eqs. (44)-(50), vibrational H2+ cooling by H and e, with LTE
        // saturation. The published low-T H fit is itself an extrapolation.
        float const ionElectron =
            (temperature <= 2000.0f ? (1.1e-19f / kElectronVolt) * ::powf(temperature, -0.34f) :
                                      (3.35e-21f / kElectronVolt) * ::powf(temperature, 0.12f)) *
            ::expf(-3025.0f / temperature);
        float const ionAtom = temperature <= 1000.0f ?
            (1.36e-22f / kElectronVolt) * ::expf(-3152.0f / temperature) :
            ::powf(10.0f,
                -36.42f + 5.95f * logTemperature - 0.526f * logTemperature * logTemperature) /
                kElectronVolt;
        float const ionLte =
            (2.0e-19f / kElectronVolt) * ::powf(temperature, 0.1f) * ::expf(-3125.0f / temperature);
        FloatFloat const ionLow = FloatFloat(environment.nHydrogen) *
            (c.hydrogen * FloatFloat(ionAtom) + c.electrons * FloatFloat(ionElectron));
        if (ionLte > 0.0f) {
            result.molecularIonLineCooling =
                c.molecularHydrogenPlus * ionLow / (FloatFloat(1.0f) + ionLow / FloatFloat(ionLte));
        }
        // Yoshida et al. 2006 eqs. (24)-(26): only the collisionally quenched
        // part of formation excitation heats the gas; the rest escapes.
        // https://arxiv.org/abs/astro-ph/0606106 . H2 is a molecule count here.
        float const ratio = 400.0f / temperature;
        FloatFloat const colliders = c.hydrogen * FloatFloat(1.6f * ::expf(-ratio * ratio)) +
            c.molecularHydrogen * FloatFloat(1.4f * ::expf(-12000.0f / (temperature + 1200.0f)));
        FloatFloat const weightedDensity = FloatFloat(environment.nHydrogen) * colliders;
        FloatFloat const quenched = weightedDensity / (weightedDensity + FloatFloat(1.0e6f / root));
        result.molecularChemicalHeating =
            quenched * (FloatFloat(3.53f) * flux[7] + FloatFloat(1.83f) * flux[9]);
        result.molecularChemicalCooling =
            FloatFloat(4.48f) * (flux[11] + flux[12] + flux[27] + flux[28]) +
            FloatFloat(1.83f) * flux[10] + FloatFloat(0.7542f) * (flux[13] + flux[14] + flux[36]) +
            FloatFloat(4.48f - 0.7542f) * flux[30] + FloatFloat(24.5874f - 13.5984f) * flux[34];
        // GA08 section 2.3.9 retains the dominant chemical thermal processes,
        // not state-specific heating from minor neutralization/transfer paths.
        // Bound their available binding energy instead of assuming every
        // exothermic eV becomes heat. H2+ binding relative to H+ + H is 2.65 eV.
        result.unresolvedChemicalEnergyRelease = FloatFloat(13.5984f - 0.7542f) * flux[15] +
            FloatFloat(2.65f - 0.7542f) * flux[16] +
            FloatFloat(13.5984f - 2.65f) * (flux[17] + flux[29]) +
            FloatFloat(13.5984f - 2.65f - 0.7542f + 4.48f) * flux[18] +
            FloatFloat(24.5874f - 4.48f - 13.5984f) * flux[31] +
            FloatFloat(24.5874f - 4.48f - 13.5984f + 2.65f) * flux[32] +
            FloatFloat(24.5874f - 13.5984f) * flux[33] + FloatFloat(24.5874f - 0.7542f) * flux[35];
    }
    return result.netHeating().isFinite() ? Status::Success : Status::InvalidInput;
}

CUDA_EXPORT inline Status evaluateRhsAtTemperature(WorkingState const& state,
    Environment const& environment, float temperature, Derivative& derivative) {
    Status status = validateComposition(state, environment);
    if (status != Status::Success) {
        return status;
    }
    status = validateRadiation(environment.radiation);
    if (status != Status::Success) {
        return status;
    }
    RateCoefficients rates;
    status = evaluateRateCoefficients(temperature, environment.network, rates);
    if (status != Status::Success) {
        return status;
    }
    Composition const c = composition(state, environment);
    evaluateDensityDependentRates(c, environment, temperature, rates);
    FloatFloat flux[kReactionCount];
    reactionFluxes(c, environment, rates, flux);
    FloatFloat source[kChemicalSpeciesCount]{};
    for (int reaction = 0; reaction != kReactionCount; ++reaction) {
        for (int species = 0; species != kChemicalSpeciesCount; ++species) {
            int const change = reactionStoichiometry(reaction, species);
            if (change != 0) {
                source[species] += FloatFloat(static_cast<float>(change)) * flux[reaction];
            }
        }
    }
    Derivative result;
    result.abundances[0] = source[1];
    if (environment.heliumToHydrogenNumberRatio > 0.0f) {
        result.abundances[1] = source[3] / FloatFloat(environment.heliumToHydrogenNumberRatio);
        result.abundances[2] = source[4] / FloatFloat(environment.heliumToHydrogenNumberRatio);
    }
    result.abundances[3] = source[6];
    result.abundances[4] = FloatFloat(2.0f) * source[7];
    result.abundances[5] = FloatFloat(2.0f) * source[8];
    ThermalRates thermal;
    status = evaluateThermalRates(c, environment, temperature, rates, flux, thermal);
    if (status != Status::Success) {
        return status;
    }
    result.thermalEnergy = thermal.netHeating();
    for (int i = 0; i != kAbundanceCount; ++i) {
        if (!result.abundances[i].isFinite()) {
            return Status::InvalidInput;
        }
    }
    derivative = result;
    return Status::Success;
}
CUDA_EXPORT inline Status evaluateRhs(
    WorkingState const& state, Environment const& environment, Derivative& derivative) {
    Eos eos;
    Status const status = evaluateEos(state, environment, eos);
    return status == Status::Success ?
        evaluateRhsAtTemperature(state, environment, eos.temperature, derivative) :
        status;
}
CUDA_EXPORT inline Status evaluateRhs(
    State const& state, Environment const& environment, Derivative& derivative) {
    Status const status = validateState(state, environment);
    return status == Status::Success ? evaluateRhs(workingState(state), environment, derivative) :
                                       status;
}
}    // namespace primordial
