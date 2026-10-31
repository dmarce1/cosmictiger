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
template <class Real>
struct RateCoefficientsT {
    Real collision[kCollisionCount]{};      // cm^3 s^-1, including He+ dielectronic recombination.
    Real hydrogenRecombinationCooling{};    // eV cm^3 s^-1.
    Real heliumRecombinationCooling{};
    Real heliumDoubleRecombinationCooling{};
};
template <class Real>
struct ThermalRatesT {
    // Positive losses and gains are kept separate until the final source sum.
    Real ionizationCooling{}, recombinationCooling{}, excitationCooling{};
    Real freeFreeCooling{}, molecularLineCooling{}, molecularIonLineCooling{},
        molecularChemicalCooling{};
    Real molecularChemicalHeating{}, photoHeating{}, comptonHeating{};
    // Upper bound on chemical binding-energy release from channels whose
    // product excitation/photon partition is not resolved by this thermal
    // model. Diagnostic only, not an invented thermal-energy source.
    Real unresolvedChemicalEnergyRelease{};
    CUDA_EXPORT Real netHeating() const {
        return ((photoHeating - ionizationCooling) + (comptonHeating - freeFreeCooling)) +
            (molecularChemicalHeating - molecularChemicalCooling) - recombinationCooling -
            excitationCooling - molecularLineCooling - molecularIonLineCooling;
    }
};
using RateCoefficients = RateCoefficientsT<float>;
using ThermalRates = ThermalRatesT<float>;
namespace detail {
    // Molecular9 above 6000 K: see docs/molecular9_hot.md. Fits with a
    // published upper temperature, and fits whose published form diverges
    // (exponential or steep power-law growth far above any collisional rate),
    // are evaluated at min(T, freeze). Slowly growing forms that stay below
    // 3.5e-8 cm^3/s at 1e8 K are left as published. Published high-temperature
    // branches are joined to the low branch by the binary64 ratio of the two
    // published expressions at the splice, so no coefficient jumps there.
    template <class Real>
    inline constexpr Real kMolecularFitFreezeTemperature = Real(1.0e4);    // GA08 model range.
    template <class Real>
    inline constexpr Real kAssociationFreezeTemperature = Real(32000.0);    // GP98 Table 1, H8.
    template <class Real>
    inline constexpr Real kChargeTransferFreezeTemperature = Real(30000.0);    // Savin 2004.
    template <class Real>
    inline constexpr Real kHeliumTransferFreezeTemperature = Real(1.0e5);    // GA08 A1/27.
    template <class Real>
    inline constexpr Real kAttachmentContinuity = Real(1.0056449265332643);    // 6000 K.
    template <class Real>
    inline constexpr Real kIonPairContinuity = Real(1.0075749241800007);    // 8000 K.
    template <class Real>
    inline constexpr Real kHeliumTransferContinuity = Real(1.0024342795865604);    // 1e4 K.
    template <class Real>
    CUDA_EXPORT inline Real molecularFitTemperature(Real temperature) {
        return temperature < kMolecularFitFreezeTemperature<Real> ?
            temperature :
            kMolecularFitFreezeTemperature<Real>;
    }
    template <class Real, int count>
    CUDA_EXPORT inline Real polynomial(Real x, Real const (&coefficients)[count]) {
        Real result(coefficients[count - 1]);
        for (int i = count - 2; i >= 0; --i) {
            result = result * Real(x) + Real(coefficients[i]);
        }
        return result;
    }
    template <class Real>
    CUDA_EXPORT inline Real coldMolecularHydrogenCollisionCooling(Real temperature) {
        // Hernandez et al. 2021, A&A 647 A155, Table 2 and eqs. (5)-(7):
        // https://doi.org/10.1051/0004-6361/202039790
        // Published CDS table2.dat, J/A+A/647/A155 (retrieved 2026-10-20):
        // https://cdsarc.cds.unistra.fr/ftp/J/A+A/647/A155/table2.dat
        // Polynomial downward rates are in 1e-20 m^3/s = 1e-14 cm^3/s;
        // the fits cover 10..200 K and are used here only below 100 K.
        // Rows: (2,0)->(0,0), (2,1)->(0,1), (3,0)->(1,0), (3,1)->(1,1).
        // Eq. (6)'s identical-particle factors are already in the published
        // rates. No additional factor of two belongs in this cooling sum.
        Real const downward[][6] = {{Real(52.2644), -Real(1.30816), Real(0.0325606),
                                        -Real(3.01884e-4), Real(1.33791e-6), -Real(2.24735e-9)},
            {Real(74.5105), -Real(1.44478), Real(0.0389100), -Real(3.50088e-4), Real(1.53233e-6),
                -Real(2.55815e-9)},
            {Real(26.4090), -Real(0.530468), Real(0.0141524), -Real(1.23414e-4), Real(5.39467e-7),
                -Real(9.00632e-10)},
            {Real(25.5303), -Real(0.481591), Real(0.0137928), -Real(1.17786e-4), Real(5.17233e-7),
                -Real(8.66778e-10)}};
        Real cooling(Real(0.0));
        for (int transition = 0; transition != 4; ++transition) {
            bool const paraTarget = transition < 2;
            Real const gap = paraTarget ? Real(509.86) : Real(844.60);
            Real const degeneracy = paraTarget ? Real(5.0) : Real(7.0) / Real(3.0);
            Real const targetFraction = paraTarget ? Real(0.25) : Real(0.75);
            Real const colliderFraction = transition % 2 == 0 ? Real(0.25) : Real(0.75);
            Real const excitation = Real(1.e-14) * polynomial(temperature, downward[transition]) *
                degeneracy * std::exp(-gap / temperature);
            // Low-density ground-ladder approximation, fixed ortho:para=3:1,
            // as in Lee et al. 2008 eq. (5), https://arxiv.org/abs/0805.1623 .
            // Excitation energy subsequently escapes as line radiation.
            cooling += Real(targetFraction * colliderFraction) * Real(excitation) *
                Real(kBoltzmannEvT<Real> * gap);
        }
        return cooling;    // eV cm^3/s per H2--H2 pair density.
    }
    template <class Real>
    CUDA_EXPORT inline Real matchColdCooling(
        Real temperature, Real cooling, Real logRatio, Real logSlopeDelta) {
        if (temperature <= Real(80.0)) {
            return cooling;
        }
        // Explicit compatibility interpolation, not a published rate fit:
        // add a cubic Hermite correction to ln(cooling) on 80..100 K.
        // Its value/slope vanish at 80 K and match the established warm fit
        // at 100 K. The input cold fits stay within their published domains.
        Real const s = (temperature - Real(80.0)) / Real(20.0);
        Real const sSquared = s * s;
        Real const correction = logRatio * sSquared * (Real(3.0) - Real(2.0) * s) +
            Real(20.0) * logSlopeDelta * sSquared * (s - Real(1.0));
        return cooling * std::exp(correction);
    }
    template <class Real>
    CUDA_EXPORT inline Real hydrogenRecombination(Real lambda) {
        // Hui & Gnedin 1997 Appendix A, case A (all recombination photons escape).
        // https://arxiv.org/abs/astro-ph/9612232
        return Real(1.269e-13) * std::pow(lambda, Real(1.503)) /
            std::pow(Real(1.0) + std::pow(lambda / Real(0.522), Real(0.470)), Real(1.923));
    }
    template <class Real>
    CUDA_EXPORT inline Real hydrogenRecombinationCooling(Real temperature, Real lambda) {
        return (Real(1.778e-29) / kElectronVoltT<Real>) *temperature *
            std::pow(lambda, Real(1.965)) /
            std::pow(Real(1.0) + std::pow(lambda / Real(0.541), Real(0.502)), Real(2.697));
    }
    template <class Real>
    CUDA_EXPORT inline Real heliumRecombination(Real temperature) {
        // Verner & Ferland 1996 eq. (4), Table 1 He I(b), valid 3..1e9 K.
        // Use their single wide-range fit, avoiding a discontinuous 1e6 K splice.
        // https://arxiv.org/abs/astro-ph/9509083
        Real const low = std::sqrt(temperature / Real(0.04266));
        Real const high = std::sqrt(temperature / Real(4.677e6));
        return Real(9.356e-10) /
            (low * std::pow(Real(1.0) + low, Real(0.2108)) *
                std::pow(Real(1.0) + high, Real(1.7892)));
    }
    template <class Real>
    CUDA_EXPORT inline Real molecularChargeTransfer(Real temperature) {
        // Savin et al. 2004, as tabulated by Glover & Abel 2008, Appendix A.
        // https://academic.oup.com/mnras/article/388/4/1627/981339
        // Algebraically shift the published degree-seven polynomial to y=ln(T)-8.
        // The original powers of ln(T) strongly cancel in single precision.
        // b[k] = sum(j=k..7) a[j]*binomial(j,k)*8^(j-k), evaluated once in exact
        // decimal arithmetic from the published coefficients. Ordinary Horner
        // below uses only the selected Real type; the physical fit is unchanged.
        Real const coefficients[] = {Real(2.5986355264E-10), Real(1.544886656E-11),
            Real(-1.845800704E-11), Real(-1.29312480E-12), Real(-5.41154560E-12),
            Real(-3.1994192E-13), Real(1.60327092E-12), Real(3.5311932E-13)};
        Real const shiftedLog = std::log(temperature) - Real(8);
        Real value = coefficients[7];
        for (int i = 6; i >= 0; --i) {
            value = value * shiftedLog + coefficients[i];
        }
        // Savin's fit starts at 100 K. Its extrapolated polynomial can be
        // negative below that range, where the exponential suppressed the
        // rate to zero in float/double; retain a nonnegative rate for wider types.
        return std::fmax(Real(0), value) * std::exp(-Real(21237.15) / temperature);
    }
}    // namespace detail

template <class Real>
CUDA_EXPORT inline Status evaluateRateCoefficients(
    Real temperature, Network network, RateCoefficientsT<Real>& result) {
    if (!finite(temperature) || temperature <= Real(0.0) ||
        (network != Network::Atomic6 && network != Network::Molecular9)) {
        return Status::InvalidInput;
    }
    if (temperature < minimumTemperature<Real>(network) ||
        temperature > maximumTemperature<Real>(network)) {
        return Status::UnsupportedState;
    }
    result = RateCoefficientsT<Real>{};
    Real const root = std::sqrt(temperature);
    Real const suppression = Real(1.0) + std::sqrt(temperature / Real(1.0e5));
    Real* k = result.collision;
    // Cen 1992 fits as reproduced in Katz, Weinberg & Hernquist 1996 Table 1.
    // Ionization cooling below uses these very same reaction fluxes once.
    // https://arxiv.org/abs/astro-ph/9509107
    k[0] = Real(5.85e-11) * root * std::exp(-Real(157809.1) / temperature) / suppression;
    k[2] = Real(2.38e-11) * root * std::exp(-Real(285335.4) / temperature) / suppression;
    k[4] = Real(5.68e-12) * root * std::exp(-Real(631515.0) / temperature) / suppression;
    k[1] = detail::hydrogenRecombination(Real(315614.0) / temperature);
    k[5] = Real(2.0) * detail::hydrogenRecombination(Real(1263030.0) / temperature);
    Real const radiativeHelium = detail::heliumRecombination(temperature);
    Real const dielectronicShape = std::pow(temperature, -Real(1.5)) *
        std::exp(-Real(470000.0) / temperature) *
        (Real(1.0) + Real(0.3) * std::exp(-Real(94000.0) / temperature));
    k[3] = radiativeHelium + Real(1.9e-3) * dielectronicShape;
    result.hydrogenRecombinationCooling =
        detail::hydrogenRecombinationCooling(temperature, Real(315614.0) / temperature);
    result.heliumDoubleRecombinationCooling = Real(8.0) *
        detail::hydrogenRecombinationCooling(temperature, Real(1263030.0) / temperature);
    // Conventional He I thermal closure: kBT per radiative recombination,
    // plus the independent Cen dielectronic cooling fit. No ionization
    // threshold is subtracted here: recombination photons carry binding energy.
    result.heliumRecombinationCooling = kBoltzmannEvT<Real> * temperature * radiativeHelium +
        (Real(1.24e-13) / kElectronVoltT<Real>) *dielectronicShape;
    if (network == Network::Atomic6) {
        return Status::Success;
    }
    Real const logTemperature = std::log10(temperature);
    Real const electronTemperature = kBoltzmannEvT<Real> * temperature;
    Real const logElectronTemperature = std::log(electronTemperature);
    Real const frozenTemperature = detail::molecularFitTemperature(temperature);
    // Glover & Abel 2008 Appendix A: Wishart attachment; fiducial associative
    // detachment; Galli-Palla radiative association; Savin charge transfer.
    // Attachment has both published branches (GA08 A1/1; Abel97 fit 1..1e8 K).
    if (temperature <= Real(6000.0)) {
        Real const attachment[] = {-Real(17.845), Real(0.762), Real(0.1523), -Real(0.03274)};
        k[6] = std::pow(Real(10.0), detail::polynomial(logTemperature, attachment));
    } else {
        Real const logSquared = logTemperature * logTemperature;
        Real const hotAttachment[] = {
            -Real(16.4199), Real(0.1998), -Real(5.447e-3), Real(4.0415e-5)};
        k[6] = detail::kAttachmentContinuity<Real> *
            std::pow(Real(10.0), detail::polynomial(logSquared, hotAttachment));
    }
    k[7] = Real(1.3e-9);
    // GP98 Table 1 H8 is published for 1 <= T <= 32000 K and held above.
    Real const association[] = {-Real(19.38), -Real(1.523), Real(1.118), -Real(0.1269)};
    k[8] = std::pow(Real(10.0),
        detail::polynomial(
            std::log10(std::fmin(temperature, detail::kAssociationFreezeTemperature<Real>)),
            association));
    k[9] = Real(6.4e-10);
    // Savin et al. 2004 computed 100..30000 K; the polynomial is held above.
    k[10] = detail::molecularChargeTransfer(
        std::fmin(temperature, detail::kChargeTransferFreezeTemperature<Real>));
    // Glover & Abel 2008 Table A1 vibrational-ground-state dissociation.
    // evaluateDensityDependentRates supplies the prescribed LTE interpolation.
    k[11] =
        Real(4.49e-9) * std::pow(temperature, Real(0.11)) * std::exp(-Real(101858.0) / temperature);
    k[12] = Real(6.67e-12) * root * std::exp(-(Real(1.0) + Real(63593.0) / temperature));
    Real const electronDetachment[] = {-Real(18.01849334273), Real(2.360852208681),
        -Real(0.2827443061704), Real(0.01623316639567), -Real(0.03365012031363),
        Real(0.01178329782711), -Real(0.001656194699504), Real(0.0001068275202678),
        -Real(2.631285809207e-6)};
    k[13] = std::exp(detail::polynomial(logElectronTemperature, electronDetachment));
    Real const atomDetachment[] = {-Real(20.3726089653), Real(1.13944933584), -Real(0.142101352155),
        Real(0.00846445538663), -Real(0.0014327641213), Real(0.00020122502848),
        Real(0.00008663963243), -Real(0.0000258500968026), Real(2.455501197e-6),
        -Real(8.06838246118e-8)};
    k[14] = electronTemperature <= Real(0.1) ?
        Real(2.5634e-9) * std::pow(electronTemperature, Real(1.78186)) :
        std::exp(detail::polynomial(logElectronTemperature, atomDetachment));
    // Croft et al. 1999 mutual neutralization and GA08 Appendix A channels
    // for H2+ formation, dissociative recombination, and H- destruction.
    // The Croft form grows as sqrt(T) and is held at the molecular freeze
    // temperature; H- + H+ -> H2+ + e has both published branches (GA08 A1/16).
    k[15] = Real(2.4e-6) * (Real(1.0) + frozenTemperature / Real(20000.0)) /
        std::sqrt(frozenTemperature);
    k[16] = temperature <= Real(8000.0) ?
        Real(6.9e-9) * std::pow(temperature, -Real(0.35)) :
        detail::kIonPairContinuity<Real> * Real(9.6e-7) * std::pow(temperature, -Real(0.9));
    k[17] = temperature <= Real(617.0) ? Real(1.0e-8) :
                                         Real(1.32e-6) * std::pow(temperature, -Real(0.76));
    // GA08 Table A1: split H- + H2+ products, molecular dissociation by H2
    // and He, dissociative attachment, and minor H/He charge-transfer channels.
    k[18] = Real(1.4e-7) * std::sqrt(Real(300.0) / temperature);
    k[19] = Real(5.996e-30) * std::pow(temperature, Real(4.1881)) /
        std::pow(Real(1.0) + Real(6.761e-6) * temperature, Real(5.6881)) *
        std::exp(-Real(54657.4) / temperature);
    // The He ground-state dissociation fit grows as T^3.8 and is held at the
    // molecular freeze temperature; evaluateDensityDependentRates agrees.
    k[20] = std::pow(Real(10.0),
        -Real(27.029) + Real(3.801) * std::log10(frozenTemperature) -
            Real(29487.0) / frozenTemperature);
    k[21] = k[18];
    k[22] =
        Real(2.7e-8) * std::pow(temperature, -Real(1.27)) * std::exp(-Real(43000.0) / temperature);
    k[23] = Real(3.7e-14) * std::exp(Real(35.0) / temperature);
    k[24] = Real(7.2e-15);
    k[25] = Real(1.2e-15) * std::pow(temperature / Real(300.0), Real(0.25));
    // He + H+ -> He+ + H: both GA08 A1/27 branches (Kimura et al. 1993). The
    // open-ended T^4.74 branch is held at 1e5 K, where it is already three
    // orders of magnitude below electron-impact ionization of He.
    if (temperature <= Real(1.0e4)) {
        k[26] = Real(1.26e-9) * std::pow(temperature, -Real(0.75)) *
            std::exp(-Real(127500.0) / temperature);
    } else {
        k[26] = detail::kHeliumTransferContinuity<Real> * Real(4.0e-37) *
            std::pow(
                std::fmin(temperature, detail::kHeliumTransferFreezeTemperature<Real>), Real(4.74));
    }
    // exp(+T/22400) is a low-temperature fit artefact and is held at the
    // molecular freeze temperature, like the other mutual neutralization fit.
    k[27] = Real(2.32e-7) * std::pow(frozenTemperature / Real(300.0), -Real(0.52)) *
        std::exp(frozenTemperature / Real(22400.0));
    // He + H- detachment grows as T^2 and is held like the other diverging fits.
    k[28] = Real(4.1e-17) * frozenTemperature * frozenTemperature *
        std::exp(-Real(19870.0) / frozenTemperature);
    for (int i = 0; i != kCollisionCount; ++i) {
        if (!finite(k[i]) || k[i] < Real(0.0)) {
            return Status::UnsupportedState;
        }
    }
    return Status::Success;
}

// GA08 eqs. (13)-(17). Interpolate logarithms analytically so cold rates
// can underflow naturally only after interpolation; never log a rounded zero.
// Its critical-density closure assumes electron excitation is subdominant at
// high density, and is not a state-resolved rovibrational population model.
template <class Real>
CUDA_EXPORT inline void evaluateDensityDependentRates(CompositionT<Real> const& c,
    EnvironmentT<Real> const& environment, Real temperature, RateCoefficientsT<Real>& result) {
    if (environment.network != Network::Molecular9) {
        return;
    }
    // The critical-density fits (GA08 eqs. 15-17) and the He ground-state
    // dissociation fit are held at the molecular freeze temperature; the
    // linear He expression would otherwise underflow to zero near 7e5 K.
    Real const frozenTemperature = detail::molecularFitTemperature(temperature);
    Real const logTemperature = std::log10(frozenTemperature);
    Real const t4 = logTemperature - Real(4.0);
    Real const criticalH =
        std::pow(Real(10.0), Real(3.0) - Real(0.416) * t4 - Real(0.327) * t4 * t4);
    Real const criticalH2 =
        std::pow(Real(10.0), Real(4.845) - Real(1.3) * t4 + Real(1.62) * t4 * t4);
    Real const criticalHe = std::pow(Real(10.0),
        Real(5.0792) * (Real(1.0) - Real(1.23e-5) * (frozenTemperature - Real(2000.0))));
    Real const densityRatio = Real(environment.nHydrogen) *
        (c.hydrogen / Real(criticalH) + Real(2.0) * c.molecularHydrogen / Real(criticalH2) +
            c.helium / Real(criticalHe));
    Real const lteFraction = (densityRatio / (Real(1.0) + densityRatio));
    Real const naturalLog = std::log(temperature);
    Real const logLow[] = {
        std::log(Real(4.49e-9)) + Real(0.11) * naturalLog - Real(101858.0) / temperature,
        std::log(Real(6.67e-12)) + Real(0.5) * naturalLog - Real(1.0) - Real(63593.0) / temperature,
        std::log(Real(5.996e-30)) + Real(4.1881) * naturalLog -
            Real(5.6881) * std::log1p(Real(6.761e-6) * temperature) - Real(54657.4) / temperature,
        Real(2.302585092994046) *
            (-Real(27.029) + Real(3.801) * logTemperature - Real(29487.0) / frozenTemperature)};
    Real const logLte[] = {
        std::log(Real(1.91e-9)) + Real(0.136) * naturalLog - Real(53407.1) / temperature,
        std::log(Real(3.52e-9)) - Real(43900.0) / temperature,
        std::log(Real(1.3e-9)) - Real(53300.0) / temperature,
        Real(2.302585092994046) *
            (-Real(2.729) - Real(1.75) * std::log10(temperature) - Real(23474.0) / temperature)};
    int const indices[] = {11, 12, 19, 20};
    for (int i = 0; i != 4; ++i) {
        result.collision[indices[i]] = std::exp(logLow[i] + lteFraction * (logLte[i] - logLow[i]));
    }
}

template <class Real>
CUDA_EXPORT inline Status validateRadiation(RadiationT<Real> const& r) {
    Real const values[] = {r.hydrogenIonization, r.heliumIonization, r.heliumPlusIonization,
        r.hydrogenMinusDetachment, r.molecularHydrogenDissociation, r.molecularHydrogenIonization,
        r.molecularHydrogenPlusDissociation, r.molecularHydrogenPlusIonization,
        r.heatingPerHydrogen, r.heatingPerHelium, r.heatingPerHeliumPlus, r.heatingPerHydrogenMinus,
        r.heatingPerMolecularHydrogen, r.heatingPerMolecularHydrogenPlus};
    for (Real value : values) {
        if (!finite(value) || value < Real(0.0)) {
            return Status::InvalidInput;
        }
    }
    return Status::Success;
}
// Reactant indices use the same H, H+, He, He+, He++, e, H-, H2, H2+ order
// as the stoichiometric table. A missing second reactant denotes a photo channel.
CUDA_EXPORT inline void reactionReactants(int reaction, int& first, int& second) {
    int const collisionFirst[kCollisionCount] = {
        0, 1, 2, 3, 3, 4, 0, 6, 0, 8, 7, 7, 7, 6, 6, 6, 6, 8, 8, 7, 7, 6, 7, 7, 7, 3, 2, 3, 2};
    int const collisionSecond[kCollisionCount] = {
        5, 5, 5, 5, 5, 5, 5, 0, 1, 0, 1, 5, 0, 5, 0, 1, 1, 5, 6, 7, 2, 8, 5, 3, 3, 0, 1, 6, 6};
    if (reaction >= 19 && reaction < 27) {
        int const photoFirst[8] = {0, 2, 3, 6, 7, 7, 8, 8};
        first = photoFirst[reaction - 19];
        second = -1;
    } else {
        int const collision = reaction < 19 ? reaction : reaction - 8;
        first = collisionFirst[collision];
        second = collisionSecond[collision];
    }
}

template <class Real>
CUDA_EXPORT inline Real chemicalSpeciesAbundance(CompositionT<Real> const& c, int species) {
    Real const values[kChemicalSpeciesCount] = {c.hydrogen, c.hydrogenPlus, c.helium, c.heliumPlus,
        c.heliumDoublePlus, c.electrons, c.hydrogenMinus, c.molecularHydrogen,
        c.molecularHydrogenPlus};
    return values[species];
}

template <class Real>
CUDA_EXPORT inline Real reactionCoefficient(
    EnvironmentT<Real> const& environment, RateCoefficientsT<Real> const& rates, int reaction) {
    if (reaction >= 19 && reaction < 27) {
        RadiationT<Real> const& r = environment.radiation;
        Real const photo[8] = {r.hydrogenIonization, r.heliumIonization, r.heliumPlusIonization,
            r.hydrogenMinusDetachment, r.molecularHydrogenDissociation,
            r.molecularHydrogenIonization, r.molecularHydrogenPlusDissociation,
            r.molecularHydrogenPlusIonization};
        return photo[reaction - 19];
    }
    int const collision = reaction < 19 ? reaction : reaction - 8;
    return rates.collision[collision] * environment.nHydrogen;
}

template <class Real>
CUDA_EXPORT inline void reactionFluxes(CompositionT<Real> const& c,
    EnvironmentT<Real> const& environment, RateCoefficientsT<Real> const& rates,
    Real (&flux)[kReactionCount]) {
    for (int reaction = 0; reaction != kReactionCount; ++reaction) {
        int first, second;
        reactionReactants(reaction, first, second);
        Real value =
            reactionCoefficient(environment, rates, reaction) * chemicalSpeciesAbundance(c, first);
        if (second >= 0) {
            value *= chemicalSpeciesAbundance(c, second);
        }
        flux[reaction] = value;
    }
}

// dn_s/dt = creation - destruction*n_s, with n_s measured per H nucleus.
// Remove one reactant factor algebraically, rather than dividing a flux by
// n_s. This remains defined for absent species and for identical reactants.
template <class Real>
CUDA_EXPORT inline void reactionCreationDestruction(CompositionT<Real> const& c,
    EnvironmentT<Real> const& environment, RateCoefficientsT<Real> const& rates, int species,
    Real& creation, Real& destruction) {
    creation = destruction = Real(0);
    for (int reaction = 0; reaction != kReactionCount; ++reaction) {
        int const change = reactionStoichiometry(reaction, species);
        if (change == 0) {
            continue;
        }
        int first, second;
        reactionReactants(reaction, first, second);
        Real value = reactionCoefficient(environment, rates, reaction);
        if (change > 0) {
            value *= chemicalSpeciesAbundance(c, first);
            if (second >= 0) {
                value *= chemicalSpeciesAbundance(c, second);
            }
            creation += Real(change) * value;
        } else {
            if (first == species) {
                if (second >= 0) {
                    value *= chemicalSpeciesAbundance(c, second);
                }
            } else {
                value *= chemicalSpeciesAbundance(c, first);
            }
            destruction -= Real(change) * value;
        }
    }
}

// Low-density H2 cooling coefficients (eV cm^3/s), fixed ortho:para=3:1.
// Glover & Abel 2008 Table 8, https://arxiv.org/abs/0803.1768 .
// Original warm fits are retained. Below 100 K, H2 collisions use the
// Hernandez et al. data above; all other partners use GA08's cold fits.
// Order of collision partners: H, H2, He, H+, e. All five are retained.
// This remains spontaneous line cooling with no molecular CMB absorption or
// evolved spin populations. Compton exchange is a separate thermal process.
// Above 6000 K the H, H2 and He polynomials are evaluated beyond their
// published range up to 1e4 K, and every coefficient (including the LTE
// asymptote) is held at its 1e4 K value above that. This follows Grackle's
// GAHI/GAH2/GAHe/GAHp/GAel/H2LTE clamps (rate_functions.c, revision
// 01261db); it is a documented continuation, not a published fit.
template <class Real>
CUDA_EXPORT inline Status molecularCoolingCoefficients(
    Real gasTemperature, Real (&cooling)[5], Real& lte) {
    if (!finite(gasTemperature) || gasTemperature < Real(10.0) || gasTemperature > Real(1.0e8)) {
        return Status::UnsupportedState;
    }
    Real const temperature = detail::molecularFitTemperature(gasTemperature);
    Real const x = std::log10(temperature / Real(1000.0));
    Real const atomLow[] = {-Real(24.311209), Real(3.5692468), -Real(11.332860), -Real(27.850082),
        -Real(21.328264), -Real(4.2519023)};
    Real const atomHigh[] = {-Real(24.311209), Real(4.6450521), -Real(3.7209846), Real(5.9369081),
        -Real(5.5108047), Real(1.5538288)};
    Real const molecule[] = {-Real(23.962112), Real(2.09433740), -Real(0.77151436),
        Real(0.43693353), -Real(0.14913216), -Real(0.033638326)};
    Real const helium[] = {-Real(23.689237), Real(2.1892372), -Real(0.81520438), Real(0.29036281),
        -Real(0.16596184), Real(0.19191375)};
    Real const proton[] = {-Real(21.716699), Real(1.3865783), -Real(0.37915285), Real(0.11453688),
        -Real(0.23214154), Real(0.058538864)};
    Real const electronLow[] = {-Real(34.286155), -Real(48.537163), -Real(77.121176),
        -Real(51.352459), -Real(15.169160), -Real(0.98120322)};
    Real const electronHigh[] = {-Real(22.190316), Real(1.5728955), -Real(0.213351),
        Real(0.96149759), -Real(0.91023195), Real(0.13749749)};
    if (temperature < Real(100.0)) {
        // GA08 Table 8 / eq. (40), 10..100 K H branch. GA08 explicitly
        // describes its underlying low-T H-collision continuation in eqs.
        // (28)-(29); the He, proton and electron fits also extend to 10 K.
        Real const atomCold[] = {-Real(16.818342), Real(37.383713), Real(58.145166),
            Real(48.656103), Real(20.159831), Real(3.8479610)};
        // Convert erg to eV before exponentiating: at 10 K the H and electron
        // coefficients are representable in eV but underflow in FP32 ergs.
        Real const logElectronVolt = std::log10(kElectronVoltT<Real>);
        cooling[0] = std::pow(Real(10.0), detail::polynomial(x, atomCold) - logElectronVolt);
        cooling[1] = detail::coldMolecularHydrogenCollisionCooling(temperature);
        cooling[2] = std::pow(Real(10.0), detail::polynomial(x, helium) - logElectronVolt);
        cooling[3] = std::pow(Real(10.0), detail::polynomial(x, proton) - logElectronVolt);
        cooling[4] = std::pow(Real(10.0), detail::polynomial(x, electronLow) - logElectronVolt);
        // ln(warm/cold) and d ln(warm/cold)/dT at 100 K, independently
        // evaluated from the cited decimal coefficients. Raw discrepancies
        // are 9.26% (H) and 15.92% (H2); only 80..100 K is adjusted.
        cooling[0] = detail::matchColdCooling(
            temperature, cooling[0], -Real(0.08858850757531106), Real(0.01076094300000003));
        cooling[1] = detail::matchColdCooling(
            temperature, cooling[1], -Real(0.14771090012072186), -Real(0.0055088563359083076));
    } else {
        cooling[0] = std::pow(Real(10.0),
                         temperature < Real(1000.0) ? detail::polynomial(x, atomLow) :
                                                      detail::polynomial(x, atomHigh)) /
            kElectronVoltT<Real>;
        cooling[1] = std::pow(Real(10.0), detail::polynomial(x, molecule)) / kElectronVoltT<Real>;
        cooling[2] = std::pow(Real(10.0), detail::polynomial(x, helium)) / kElectronVoltT<Real>;
        cooling[3] = std::pow(Real(10.0), detail::polynomial(x, proton)) / kElectronVoltT<Real>;
        cooling[4] = std::pow(Real(10.0),
                         temperature < Real(200.0) ? detail::polynomial(x, electronLow) :
                                                     detail::polynomial(x, electronHigh)) /
            kElectronVoltT<Real>;
    }
    // Hollenbach & McKee LTE asymptote, reproduced by Yoshida et al. 2006
    // eqs. (18)-(20), https://arxiv.org/abs/astro-ph/0606106 . Units eV/s/H2.
    Real const t = temperature / Real(1000.0);
    lte = (Real(9.5e-22) / kElectronVoltT<Real>) *std::pow(t, Real(3.76)) /
            (Real(1.0) + Real(0.12) * std::pow(t, Real(2.1))) *
            std::exp(-std::pow(Real(0.13) / t, Real(3.0))) +
        (Real(3.0e-24) / kElectronVoltT<Real>) *std::exp(-Real(0.51) / t) +
        (Real(6.7e-19) / kElectronVoltT<Real>) *std::exp(-Real(5.86) / t) +
        (Real(1.6e-18) / kElectronVoltT<Real>) *std::exp(-Real(11.7) / t);
    return Status::Success;
}

template <class Real>
CUDA_EXPORT inline Status evaluateThermalRates(CompositionT<Real> const& c,
    EnvironmentT<Real> const& environment, Real temperature, RateCoefficientsT<Real> const& rates,
    Real const (&flux)[kReactionCount], ThermalRatesT<Real>& result) {
    result = ThermalRatesT<Real>{};
    Real const electronDensity = Real(environment.nHydrogen) * c.electrons;
    Real const root = std::sqrt(temperature);
    Real const suppression = Real(1.0) + std::sqrt(temperature / Real(1.0e5));
    result.ionizationCooling =
        Real(13.5984) * flux[0] + Real(24.5874) * flux[2] + Real(54.4178) * flux[4];
    result.recombinationCooling = electronDensity *
        (c.hydrogenPlus * Real(rates.hydrogenRecombinationCooling) +
            c.heliumPlus * Real(rates.heliumRecombinationCooling) +
            c.heliumDoublePlus * Real(rates.heliumDoubleRecombinationCooling));
    // Katz et al. 1996 Table 1: HI and HeII excitation and free-free cooling.
    Real const hydrogenExcitation =
        (Real(7.5e-19) / kElectronVoltT<Real>) *std::exp(-Real(118348.0) / temperature) /
        suppression;
    Real const heliumPlusExcitation =
        (Real(5.54e-17) / kElectronVoltT<Real>) *std::pow(temperature, -Real(0.397)) *
        std::exp(-Real(473638.0) / temperature) / suppression;
    // Ground-state neutral He excitation: Glover & Jappsen 2007,
    // Table 7 (Bray et al. 2000), https://arxiv.org/abs/0705.0182 .
    // The metastable-He contribution is omitted from this ground-state closure.
    Real const heliumExcitation =
        (Real(1.1e-19) / kElectronVoltT<Real>) *std::pow(temperature, Real(0.082)) *
        std::exp(-Real(230000.0) / temperature);
    result.excitationCooling = electronDensity *
        (c.hydrogen * Real(hydrogenExcitation) + c.helium * Real(heliumExcitation) +
            c.heliumPlus * Real(heliumPlusExcitation));
    Real const logTemperature = std::log10(temperature);
    Real const gaunt = Real(1.1) +
        Real(0.34) *
            std::exp(-(Real(5.5) - logTemperature) * (Real(5.5) - logTemperature) / Real(3.0));
    result.freeFreeCooling = electronDensity *
        (c.hydrogenPlus + c.heliumPlus + Real(4.0) * c.heliumDoublePlus) *
        Real(
            (Real(1.42e-27) / kElectronVoltT<Real>) *root * gaunt);    // ionic Z^2, including He++.
    RadiationT<Real> const& r = environment.radiation;
    result.photoHeating = c.hydrogen * Real(r.heatingPerHydrogen) +
        c.helium * Real(r.heatingPerHelium) + c.heliumPlus * Real(r.heatingPerHeliumPlus) +
        c.hydrogenMinus * Real(r.heatingPerHydrogenMinus) +
        c.molecularHydrogen * Real(r.heatingPerMolecularHydrogen) +
        c.molecularHydrogenPlus * Real(r.heatingPerMolecularHydrogenPlus);
    // 4 sigma_T a_r kB/(m_e c), in eV s^-1 K^-5. Thomson, nonrelativistic
    // Compton exchange (Hui & Gnedin Appendix A); its sign is not clipped.
    Real const cmbSquared = environment.cmbTemperature * environment.cmbTemperature;
    result.comptonHeating = c.electrons * Real(Real(6.35264354e-26) * cmbSquared * cmbSquared) *
        Real(environment.cmbTemperature - temperature);
    if (environment.network == Network::Molecular9) {
        Real cooling[5], lte;
        Status const status = molecularCoolingCoefficients(temperature, cooling, lte);
        if (status != Status::Success) {
            return status;
        }
        Real const low = Real(environment.nHydrogen) *
            (c.hydrogen * Real(cooling[0]) + c.molecularHydrogen * Real(cooling[1]) +
                c.helium * Real(cooling[2]) + c.hydrogenPlus * Real(cooling[3]) +
                c.electrons * Real(cooling[4]));
        // Stable low-density/LTE bridge; neither product nor ratio of two tiny
        // cooling functions is needed. LTE is an asymptote, not a density gate.
        // Preserve the zero-emission limit if an exponentially small LTE rate
        // underflows. This also avoids 0/0 when the collider density is zero.
        Real const emission = lte > Real(0.0) ? low / (Real(1.0) + low / Real(lte)) : Real(0.0);
        result.molecularLineCooling = c.molecularHydrogen * emission;
        // GA08 eqs. (44)-(50), vibrational H2+ cooling by H and e, with LTE
        // saturation. The published low-T H fit is itself an extrapolation.
        // GA08 computed these to 1e4 K; all three are held at that value above.
        Real const ionTemperature = detail::molecularFitTemperature(temperature);
        Real const logIonTemperature = std::log10(ionTemperature);
        Real const ionElectron =
            (ionTemperature <= Real(2000.0) ?
                    (Real(1.1e-19) / kElectronVoltT<Real>) *std::pow(ionTemperature, -Real(0.34)) :
                    (Real(3.35e-21) / kElectronVoltT<Real>) *std::pow(ionTemperature, Real(0.12))) *
            std::exp(-Real(3025.0) / ionTemperature);
        Real const ionAtom = ionTemperature <= Real(1000.0) ?
            (Real(1.36e-22) / kElectronVoltT<Real>) *std::exp(-Real(3152.0) / ionTemperature) :
            std::pow(Real(10.0),
                -Real(36.42) + Real(5.95) * logIonTemperature -
                    Real(0.526) * logIonTemperature * logIonTemperature) /
                kElectronVoltT<Real>;
        Real const ionLte =
            (Real(2.0e-19) / kElectronVoltT<Real>) *std::pow(ionTemperature, Real(0.1)) *
            std::exp(-Real(3125.0) / ionTemperature);
        Real const ionLow = Real(environment.nHydrogen) *
            (c.hydrogen * Real(ionAtom) + c.electrons * Real(ionElectron));
        if (ionLte > Real(0.0)) {
            result.molecularIonLineCooling =
                c.molecularHydrogenPlus * ionLow / (Real(1.0) + ionLow / Real(ionLte));
        }
        // Yoshida et al. 2006 eqs. (24)-(26): only the collisionally quenched
        // part of formation excitation heats the gas; the rest escapes.
        // https://arxiv.org/abs/astro-ph/0606106 . H2 is a molecule count here.
        Real const ratio = Real(400.0) / temperature;
        Real const colliders = c.hydrogen * Real(Real(1.6) * std::exp(-ratio * ratio)) +
            c.molecularHydrogen *
                Real(Real(1.4) * std::exp(-Real(12000.0) / (temperature + Real(1200.0))));
        Real const weightedDensity = Real(environment.nHydrogen) * colliders;
        Real const quenched = weightedDensity / (weightedDensity + Real(Real(1.0e6) / root));
        result.molecularChemicalHeating = quenched * (Real(3.53) * flux[7] + Real(1.83) * flux[9]);
        result.molecularChemicalCooling = Real(4.48) * (flux[11] + flux[12] + flux[27] + flux[28]) +
            Real(1.83) * flux[10] + Real(0.7542) * (flux[13] + flux[14] + flux[36]) +
            Real(Real(4.48) - Real(0.7542)) * flux[30] +
            Real(Real(24.5874) - Real(13.5984)) * flux[34];
        // GA08 section 2.3.9 retains the dominant chemical thermal processes,
        // not state-specific heating from minor neutralization/transfer paths.
        // Bound their available binding energy instead of assuming every
        // exothermic eV becomes heat. H2+ binding relative to H+ + H is 2.65 eV.
        result.unresolvedChemicalEnergyRelease = Real(Real(13.5984) - Real(0.7542)) * flux[15] +
            Real(Real(2.65) - Real(0.7542)) * flux[16] +
            Real(Real(13.5984) - Real(2.65)) * (flux[17] + flux[29]) +
            Real(Real(13.5984) - Real(2.65) - Real(0.7542) + Real(4.48)) * flux[18] +
            Real(Real(24.5874) - Real(4.48) - Real(13.5984)) * flux[31] +
            Real(Real(24.5874) - Real(4.48) - Real(13.5984) + Real(2.65)) * flux[32] +
            Real(Real(24.5874) - Real(13.5984)) * flux[33] +
            Real(Real(24.5874) - Real(0.7542)) * flux[35];
    }
    return finite(result.netHeating()) ? Status::Success : Status::InvalidInput;
}

template <class Real>
CUDA_EXPORT inline Status evaluateRhsAtTemperature(StateT<Real> const& state,
    EnvironmentT<Real> const& environment, Real temperature, DerivativeT<Real>& derivative) {
    Status status = validateComposition(state, environment);
    if (status != Status::Success) {
        return status;
    }
    status = validateRadiation(environment.radiation);
    if (status != Status::Success) {
        return status;
    }
    RateCoefficientsT<Real> rates;
    status = evaluateRateCoefficients(temperature, environment.network, rates);
    if (status != Status::Success) {
        return status;
    }
    CompositionT<Real> const c = composition(state, environment);
    evaluateDensityDependentRates(c, environment, temperature, rates);
    Real flux[kReactionCount];
    reactionFluxes(c, environment, rates, flux);
    Real source[kChemicalSpeciesCount]{};
    for (int reaction = 0; reaction != kReactionCount; ++reaction) {
        for (int species = 0; species != kChemicalSpeciesCount; ++species) {
            int const change = reactionStoichiometry(reaction, species);
            if (change != 0) {
                source[species] += Real(static_cast<Real>(change)) * flux[reaction];
            }
        }
    }
    DerivativeT<Real> result;
    for (int species = 0; species != kChemicalSpeciesCount; ++species) {
        result.abundances[species] = source[species];
    }
    ThermalRatesT<Real> thermal;
    status = evaluateThermalRates(c, environment, temperature, rates, flux, thermal);
    if (status != Status::Success) {
        return status;
    }
    result.thermalEnergy = thermal.netHeating();
    for (int i = 0; i != kAbundanceCount; ++i) {
        if (!finite(result.abundances[i])) {
            return Status::InvalidInput;
        }
    }
    derivative = result;
    return Status::Success;
}
template <class Real>
CUDA_EXPORT inline Status evaluateRhs(StateT<Real> const& state,
    EnvironmentT<Real> const& environment, DerivativeT<Real>& derivative) {
    EosT<Real> eos;
    Status const status = evaluateEos(state, environment, eos);
    return status == Status::Success ?
        evaluateRhsAtTemperature(state, environment, eos.temperature, derivative) :
        status;
}

}    // namespace primordial
