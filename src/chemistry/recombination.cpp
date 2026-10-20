/* CosmicTiger - homogeneous primordial recombination; GPL-2.0-or-later.
 *
 * Effective-atom physics adapted from RECFAST 1.5.2, developed by Douglas Scott,
 * based on Seager, Sasselov & Scott (1999, 2000), Wong, Moss & Scott (2008),
 * and Scott & Moss (2009). The integration and interpolation below are new.
 * https://www.astro.ubc.ca/people/scott/recfast.html
 *
 * Permission to use, copy, modify and distribute without fee or royalty at
 * any tier, this software and its documentation, for any purpose and without
 * fee or royalty is hereby granted, provided that you agree to comply with
 * the following copyright notice and statements, including the disclaimer,
 * and that the same appear on ALL copies of the software and documentation,
 * including modifications that you make for internal use or for distribution:
 *
 * Copyright 1999-2010 by University of British Columbia. All rights reserved.
 *
 * THIS SOFTWARE IS PROVIDED "AS IS", AND U.B.C. MAKES NO
 * REPRESENTATIONS OR WARRANTIES, EXPRESS OR IMPLIED.
 * BY WAY OF EXAMPLE, BUT NOT LIMITATION,
 * U.B.C. MAKES NO REPRESENTATIONS OR WARRANTIES OF
 * MERCHANTABILITY OR FITNESS FOR ANY PARTICULAR PURPOSE OR THAT
 * THE USE OF THE LICENSED SOFTWARE OR DOCUMENTATION WILL NOT INFRINGE
 * ANY THIRD PARTY PATENTS, COPYRIGHTS, TRADEMARKS OR OTHER RIGHTS.
 */
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#include <cosmictiger/chemistry/primordial_eos.hpp>
#include <cosmictiger/chemistry/recombination.hpp>

// The cosmological stiff solve relies on finite checks and cancellation near
// equilibrium, independently of the application's optional -ffast-math.
#if defined(__clang__)
#pragma float_control(precise, on, push)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC optimize("no-fast-math")
#endif

namespace primordial {
namespace detail {
    bool isFiniteRecombination(double value) {
        static_assert(std::numeric_limits<double>::is_iec559);
        constexpr std::uint64_t kExponentMask = UINT64_C(0x7ff0000000000000);
        return (std::bit_cast<std::uint64_t>(value) & kExponentMask) != kExponentMask;
    }

    using RecombinationVector = std::array<double, 3>;
    constexpr double kPi = 3.141592653589;
    constexpr double kSpeedOfLight = 2.99792458e8;
    constexpr double kBoltzmann = 1.380658e-23;
    constexpr double kPlanck = 6.6260755e-34;
    constexpr double kElectronMass = 9.1093897e-31;
    constexpr double kHydrogenMass = 1.673575e-27;
    constexpr double kHeliumMassRatio = 3.9715;
    constexpr double kGravity = 6.6742e-11;
    constexpr double kThomson = 6.6524616e-29;
    constexpr double kRadiation = 7.565914e-16;
    constexpr double kHubbleUnit = 1.0e5 / 3.0856775807e22;
    constexpr double kHydrogenIonization = 1.096787737e7;
    constexpr double kHydrogenAlpha = 8.225916453e6;
    constexpr double kHeliumIonization = 1.98310772e7;
    constexpr double kHeliumPlusIonization = 4.389088863e7;
    constexpr double kHeliumSingletS = 1.66277434e7;
    constexpr double kHeliumSingletP = 1.71134891e7;
    constexpr double kHeliumTripletP = 1.690871466e7;
    constexpr double kHeliumTripletS = 1.5985597526e7;
    constexpr double kHeliumTripletIonization = 3.8454693845e6;
    constexpr double kSingletEinstein = 1.798287e9;
    constexpr double kTripletEinstein = 177.58;
    constexpr double kSingletCrossSection = 1.436289e-22;
    constexpr double kTripletCrossSection = 1.484872e-22;
    constexpr double kHydrogenTwoPhoton = 8.2245809;
    constexpr double kHeliumTwoPhoton = 51.3;
    constexpr double kEnergyToKelvin = kPlanck * kSpeedOfLight / kBoltzmann;
    constexpr double kSahaCoefficient =
        2.0 * kPi * (kElectronMass / kPlanck) * (kBoltzmann / kPlanck);
    constexpr double kCompton =
        (8.0 / 3.0) * kThomson / (kElectronMass * kSpeedOfLight) * kRadiation;
    constexpr double kInitialScaleFactor = 1.0 / 10001.0;

    double square(double value) {
        return value * value;
    }
    double escapeProbability(double opticalDepth) {
        return opticalDepth > 1.0e-8 ? -std::expm1(-opticalDepth) / opticalDepth :
                                       1.0 - opticalDepth * 0.5;
    }

    // Positive solution of a*x^2 + b*x = c, avoiding subtractive cancellation.
    double positiveRoot(double quadratic, double linear, double rhs) {
        if (rhs == 0.0) {
            return 0.0;
        }
        if (quadratic == 0.0) {
            return rhs / linear;
        }
        double const ratio = rhs / linear;
        return 2.0 * ratio / (1.0 + std::sqrt(1.0 + 4.0 * quadratic * ratio / linear));
    }

    struct RecombinationModel {
        RecombinationParameters parameters;
        double heliumRatio;
        double hydrogenDensityToday;
        double hydrogenSahaEnd;

        double sahaRatio(double scaleFactor, double wavenumber, double degeneracy) const {
            double const temperature = parameters.cmbTemperature / scaleFactor;
            double const logarithm = std::log(degeneracy / hydrogenDensityToday) +
                1.5 * std::log(kSahaCoefficient * parameters.cmbTemperature * scaleFactor) -
                kEnergyToKelvin * wavenumber / temperature;
            return std::exp(std::min(690.0, logarithm));
        }
        double hydrogenSaha(double scaleFactor) const {
            double const ratio = sahaRatio(scaleFactor, kHydrogenIonization, 1.0);
            return ratio > 1.0 ? 2.0 / (1.0 + std::sqrt(1.0 + 4.0 / ratio)) :
                                 positiveRoot(1.0, ratio, ratio);
        }
        double heliumSaha(double scaleFactor) const {
            double const ratio = sahaRatio(scaleFactor, kHeliumIonization, 4.0);
            // (1+fHe*x)*x/(1-x) = ratio, H is fully ionized here.
            return positiveRoot(heliumRatio, 1.0 + ratio, ratio);
        }
        double heliumDoubleSaha(double scaleFactor) const {
            double const ratio = sahaRatio(scaleFactor, kHeliumPlusIonization, 1.0);
            // (1+fHe+fHe*x)*x/(1-x) = ratio.
            return positiveRoot(heliumRatio, 1.0 + heliumRatio + ratio, ratio);
        }
        double hubble(double scaleFactor) const {
            double const inverse = 1.0 / scaleFactor;
            double const omegaMatter = parameters.omegaB + parameters.omegaC;
            double const omegaLambda = 1.0 - omegaMatter - parameters.omegaRadiation;
            return parameters.hubble * kHubbleUnit *
                std::sqrt(parameters.omegaRadiation * square(square(inverse)) +
                    omegaMatter * inverse * square(inverse) + omegaLambda);
        }
        RecombinationVector derivative(double logScaleFactor, RecombinationVector const& state,
            bool hydrogenEquilibrium) const {
            double const scaleFactor = std::exp(logScaleFactor);
            double const hydrogen = hydrogenEquilibrium ? hydrogenSaha(scaleFactor) : state[0];
            double const helium = state[1];
            double const neutralHydrogen = std::max(0.0, 1.0 - hydrogen);
            double const neutralHelium = std::max(0.0, 1.0 - helium);
            double const electrons = hydrogen + heliumRatio * helium;
            double const radiationTemperature = parameters.cmbTemperature / scaleFactor;
            double const temperature = state[2] * radiationTemperature;
            double const density = hydrogenDensityToday / (scaleFactor * scaleFactor * scaleFactor);
            double const heliumDensity = heliumRatio * density;
            double const expansion = hubble(scaleFactor);
            double const phaseSpace = std::pow(kSahaCoefficient * temperature, 1.5);
            double const alphaHydrogen = 4.309e-19 * std::pow(temperature / 1.0e4, -0.6166) /
                (1.0 + 0.6703 * std::pow(temperature / 1.0e4, 0.5300));
            double const betaHydrogen = alphaHydrogen * phaseSpace *
                std::exp(-kEnergyToKelvin * (kHydrogenIonization - kHydrogenAlpha) / temperature);
            double const hydrogenNet = alphaHydrogen * electrons * hydrogen * density -
                betaHydrogen * neutralHydrogen *
                    std::exp(-kEnergyToKelvin * kHydrogenAlpha / temperature);
            double const hydrogenEscape =
                (1.0 - 0.14 * std::exp(-square((-logScaleFactor - 7.28) / 0.18)) +
                    0.079 * std::exp(-square((-logScaleFactor - 6.73) / 0.33))) /
                (8.0 * kPi * std::pow(kHydrogenAlpha, 3) * expansion);
            double const hydrogenOptical = hydrogenEscape * density * neutralHydrogen;
            double const hydrogenC = hydrogen > 0.985 ?
                1.0 :
                (1.0 + hydrogenOptical * kHydrogenTwoPhoton) /
                    ((1.0 + hydrogenOptical * kHydrogenTwoPhoton) / 1.125 +
                        hydrogenOptical * betaHydrogen);
            RecombinationVector result{
                hydrogenEquilibrium ? 0.0 : -hydrogenNet * hydrogenC / expansion, 0.0, 0.0};

            if (heliumRatio > 0.0 && helium >= 1.0e-15) {
                double const sqrtLow = std::sqrt(temperature / 3.0);
                double const sqrtHigh = std::sqrt(temperature / std::pow(10.0, 5.114));
                double const alphaHelium = std::pow(10.0, -16.744) /
                    (sqrtLow * std::pow(1.0 + sqrtLow, 0.289) * std::pow(1.0 + sqrtHigh, 1.711));
                double const betaHelium = 4.0 * alphaHelium * phaseSpace *
                    std::exp(
                        -kEnergyToKelvin * (kHeliumIonization - kHeliumSingletS) / temperature);
                double heliumEscape = 1.0 / (8.0 * kPi * std::pow(kHeliumSingletP, 3) * expansion);
                bool const heliumCorrection = helium >= 5.0e-9 && helium <= 0.980;
                double tripletC = 0.0;
                double alphaTriplet = 0.0;
                double betaTriplet = 0.0;
                if (heliumCorrection) {
                    double const singletTau =
                        kSingletEinstein * heliumEscape * 3.0 * heliumDensity * neutralHelium;
                    double singletEscapeRate = kSingletEinstein * escapeProbability(singletTau);
                    auto const continuumRate = [&](double einstein, double wavenumber,
                                                   double crossSection, double coefficient,
                                                   double exponent) {
                        double const doppler = kSpeedOfLight * wavenumber *
                            std::sqrt(2.0 * kBoltzmann * temperature /
                                (kHydrogenMass * kHeliumMassRatio * square(kSpeedOfLight)));
                        double const opacityRatio = 3.0 * einstein * heliumRatio * neutralHelium /
                            (std::sqrt(kPi) * crossSection * 8.0 * kPi * doppler * neutralHydrogen *
                                square(wavenumber));
                        return einstein / (1.0 + coefficient * std::pow(opacityRatio, exponent));
                    };
                    if (hydrogen < 0.999999) {
                        singletEscapeRate += continuumRate(
                            kSingletEinstein, kHeliumSingletP, kSingletCrossSection, 0.36, 0.86);
                    }
                    heliumEscape = 1.0 / (singletEscapeRate * 3.0 * heliumDensity * neutralHelium);
                    alphaTriplet = std::pow(10.0, -16.306) /
                        (sqrtLow * std::pow(1.0 + sqrtLow, 0.239) *
                            std::pow(1.0 + sqrtHigh, 1.761));
                    betaTriplet = alphaTriplet * phaseSpace * (4.0 / 3.0) *
                        std::exp(-kEnergyToKelvin * kHeliumTripletIonization / temperature);
                    double const tripletTau = 3.0 * kTripletEinstein * heliumDensity *
                        neutralHelium / (8.0 * kPi * expansion * std::pow(kHeliumTripletP, 3));
                    double tripletEscapeRate = kTripletEinstein * escapeProbability(tripletTau);
                    if (hydrogen <= 0.99999) {
                        tripletEscapeRate += continuumRate(kTripletEinstein, kHeliumTripletP,
                                                 kTripletCrossSection, 0.66, 0.9) /
                            3.0;
                    }
                    // Divide out the common P-to-S Boltzmann factor before
                    // taking the ratio. Both rates can otherwise underflow
                    // together in cold gas, producing a spurious 0/0.
                    double const effectiveBetaTriplet = alphaTriplet * phaseSpace * (4.0 / 3.0) *
                        std::exp(-kEnergyToKelvin *
                            (kHeliumTripletIonization - kHeliumTripletP + kHeliumTripletS) /
                            temperature);
                    tripletC = tripletEscapeRate / (effectiveBetaTriplet + tripletEscapeRate);
                }
                // Divide both numerator and denominator by the He Boltzmann
                // factor to avoid exp(680) overflow in a cold neutral gas.
                double const inverseBoltzmann =
                    std::exp(-kEnergyToKelvin * (kHeliumSingletP - kHeliumSingletS) / temperature);
                double const heliumOptical = heliumEscape * heliumDensity * neutralHelium;
                double const heliumC = (inverseBoltzmann + heliumOptical * kHeliumTwoPhoton) /
                    (inverseBoltzmann + heliumOptical * (kHeliumTwoPhoton + betaHelium));
                double const heliumNet = electrons * helium * density * alphaHelium -
                    betaHelium * neutralHelium *
                        std::exp(-kEnergyToKelvin * kHeliumSingletS / temperature);
                double const tripletNet = electrons * helium * density * alphaTriplet -
                    neutralHelium * 3.0 * betaTriplet *
                        std::exp(-kEnergyToKelvin * kHeliumTripletS / temperature);
                result[1] = -(heliumNet * heliumC + tripletNet * tripletC) / expansion;
            }
            double const comptonRatio = kCompton * square(square(radiationTemperature)) *
                electrons / ((1.0 + heliumRatio + electrons) * expansion);
            // q=Tgas/Tgamma; Tgamma'=-Tgamma in log(a).
            result[2] = -state[2] + comptonRatio * (1.0 - state[2]);
            return result;
        }
    };

    double findSahaEnd(RecombinationModel const& model, bool hydrogen) {
        double lower = std::log(kInitialScaleFactor);
        double upper = std::log(0.01);
        for (int i = 0; i < 80; ++i) {
            double const middle = 0.5 * (lower + upper);
            double const fraction = hydrogen ? model.hydrogenSaha(std::exp(middle)) :
                                               model.heliumSaha(std::exp(middle));
            if (fraction > 0.99) {
                lower = middle;
            } else {
                upper = middle;
            }
        }
        return std::exp(0.5 * (lower + upper));
    }

    bool solveLinear(std::array<RecombinationVector, 3> matrix, RecombinationVector rhs,
        RecombinationVector& result) {
        for (int row = 0; row < 3; ++row) {
            if (!detail::isFiniteRecombination(rhs[row])) {
                return false;
            }
            for (int column = 0; column < 3; ++column) {
                if (!detail::isFiniteRecombination(matrix[row][column])) {
                    return false;
                }
            }
        }
        for (int column = 0; column < 3; ++column) {
            int pivot = column;
            for (int row = column + 1; row < 3; ++row) {
                if (std::abs(matrix[row][column]) > std::abs(matrix[pivot][column])) {
                    pivot = row;
                }
            }
            if (!detail::isFiniteRecombination(matrix[pivot][column]) ||
                !(std::abs(matrix[pivot][column]) > 1.0e-200)) {
                return false;
            }
            std::swap(matrix[column], matrix[pivot]);
            std::swap(rhs[column], rhs[pivot]);
            for (int row = column + 1; row < 3; ++row) {
                double const factor = matrix[row][column] / matrix[column][column];
                for (int j = column + 1; j < 3; ++j) {
                    matrix[row][j] -= factor * matrix[column][j];
                }
                rhs[row] -= factor * rhs[column];
            }
        }
        for (int row = 2; row >= 0; --row) {
            double value = rhs[row];
            for (int column = row + 1; column < 3; ++column) {
                value -= matrix[row][column] * result[column];
            }
            result[row] = value / matrix[row][row];
            if (!detail::isFiniteRecombination(result[row])) {
                return false;
            }
        }
        return true;
    }

    bool implicitStage(RecombinationModel const& model, RecombinationOptions const& options,
        double time, double diagonalStep, RecombinationVector const& base, bool hydrogenEquilibrium,
        RecombinationVector& result) {
        result = base;
        result[0] = std::clamp(result[0], 0.0, 1.0);
        result[1] = std::clamp(result[1], 0.0, 1.0);
        result[2] = std::clamp(result[2], 1.0e-10, 1.0);
        for (int iteration = 0; iteration < 20; ++iteration) {
            auto const derivative = model.derivative(time, result, hydrogenEquilibrium);
            RecombinationVector residual;
            double residualNorm = 0.0;
            for (int i = 0; i < 3; ++i) {
                residual[i] = result[i] - base[i] - diagonalStep * derivative[i];
                if (!detail::isFiniteRecombination(derivative[i]) ||
                    !detail::isFiniteRecombination(residual[i])) {
                    return false;
                }
                double const scale = options.absoluteTolerance +
                    options.relativeTolerance * std::max(std::abs(base[i]), std::abs(result[i]));
                residualNorm = std::max(residualNorm, std::abs(residual[i]) / scale);
            }
            if (residualNorm < 0.02) {
                return true;
            }
            std::array<RecombinationVector, 3> matrix{};
            for (int column = 0; column < 3; ++column) {
                double const offset = 1.0e-6 * std::max(std::abs(result[column]), 1.0e-6);
                auto plus = result;
                auto minus = result;
                plus[column] += offset;
                minus[column] -= offset;
                if (column == 2 && minus[column] <= 0.0) {
                    minus[column] = result[column];
                }
                auto const plusDerivative = model.derivative(time, plus, hydrogenEquilibrium);
                auto const minusDerivative = model.derivative(time, minus, hydrogenEquilibrium);
                for (int row = 0; row < 3; ++row) {
                    if (!detail::isFiniteRecombination(plusDerivative[row]) ||
                        !detail::isFiniteRecombination(minusDerivative[row])) {
                        return false;
                    }
                    matrix[row][column] = (row == column ? 1.0 : 0.0) -
                        diagonalStep * (plusDerivative[row] - minusDerivative[row]) /
                            (plus[column] - minus[column]);
                }
            }
            RecombinationVector correction{};
            if (!solveLinear(matrix, residual, correction)) {
                return false;
            }
            double damping = 1.0;
            for (int i = 0; i < 3; ++i) {
                double const lower = i == 2 ? 1.0e-12 : -options.absoluteTolerance;
                double const upper = 1.0 + options.absoluteTolerance;
                if (correction[i] > 0.0 && result[i] - correction[i] < lower) {
                    damping = std::min(damping, 0.8 * (result[i] - lower) / correction[i]);
                } else if (correction[i] < 0.0 && result[i] - correction[i] > upper) {
                    damping = std::min(damping, 0.8 * (upper - result[i]) / -correction[i]);
                }
            }
            for (int i = 0; i < 3; ++i) {
                result[i] -= damping * correction[i];
                if (!detail::isFiniteRecombination(result[i])) {
                    return false;
                }
            }
        }
        return false;
    }

    bool integrateStep(RecombinationModel const& model, RecombinationOptions const& options,
        double time, double step, RecombinationVector const& initial, bool hydrogenEquilibrium,
        RecombinationVector& result) {
        // Alexander's L-stable, stiffly accurate three-stage SDIRK, order three.
        constexpr double kGamma = 0.43586652150845899942;
        constexpr double kA21 = (1.0 - kGamma) * 0.5;
        constexpr double kA31 = -1.5 * kGamma * kGamma + 4.0 * kGamma - 0.25;
        constexpr double kA32 = 1.5 * kGamma * kGamma - 5.0 * kGamma + 1.25;
        RecombinationVector first;
        if (!implicitStage(model, options, time + kGamma * step, kGamma * step, initial,
                hydrogenEquilibrium, first)) {
            return false;
        }
        auto const firstRate = model.derivative(time + kGamma * step, first, hydrogenEquilibrium);
        auto base = initial;
        for (int i = 0; i < 3; ++i) {
            base[i] += step * kA21 * firstRate[i];
        }
        RecombinationVector second;
        if (!implicitStage(model, options, time + (kA21 + kGamma) * step, kGamma * step, base,
                hydrogenEquilibrium, second)) {
            return false;
        }
        auto const secondRate =
            model.derivative(time + (kA21 + kGamma) * step, second, hydrogenEquilibrium);
        for (int i = 0; i < 3; ++i) {
            base[i] = initial[i] + step * (kA31 * firstRate[i] + kA32 * secondRate[i]);
        }
        return implicitStage(
            model, options, time + step, kGamma * step, base, hydrogenEquilibrium, result);
    }

    double monotoneSlope(double leftStep, double rightStep, double leftSlope, double rightSlope) {
        if (leftSlope * rightSlope <= 0.0) {
            return 0.0;
        }
        double const weightLeft = 2.0 * rightStep + leftStep;
        double const weightRight = rightStep + 2.0 * leftStep;
        return (weightLeft + weightRight) / (weightLeft / leftSlope + weightRight / rightSlope);
    }
}    // namespace detail

RecombinationHistory::RecombinationHistory(
    RecombinationParameters parameters, RecombinationOptions options)
  : parameters_(parameters)
  , options_(options) {
    auto const validPositive = [](double value) {
        return detail::isFiniteRecombination(value) && value > 0.0;
    };
    if (!validPositive(parameters.hubble) || !validPositive(parameters.omegaB) ||
        !detail::isFiniteRecombination(parameters.omegaC) || parameters.omegaC < 0.0 ||
        !detail::isFiniteRecombination(parameters.omegaRadiation) ||
        parameters.omegaRadiation < 0.0 ||
        !detail::isFiniteRecombination(parameters.heliumMassFraction) ||
        parameters.heliumMassFraction < 0.0 || parameters.heliumMassFraction >= 1.0 ||
        !validPositive(parameters.cmbTemperature) || !validPositive(options.relativeTolerance) ||
        options.relativeTolerance < 1.0e-12 || options.relativeTolerance > 0.01 ||
        !validPositive(options.absoluteTolerance) || options.absoluteTolerance > 1.0e-4 ||
        !validPositive(options.maxLogStep) || !validPositive(options.maxScaleFactor) ||
        options.maxScaleFactor < detail::kInitialScaleFactor || options.maxSteps == 0) {
        throw std::invalid_argument("Invalid homogeneous recombination parameters or tolerances");
    }
    heliumRatio_ = parameters.heliumMassFraction /
        (detail::kHeliumMassRatio * (1.0 - parameters.heliumMassFraction));
    hydrogenDensityToday_ = 3.0 * detail::square(parameters.hubble * detail::kHubbleUnit) *
        parameters.omegaB * (1.0 - parameters.heliumMassFraction) /
        (8.0 * detail::kPi * detail::kGravity * detail::kHydrogenMass);
    detail::RecombinationModel model{parameters, heliumRatio_, hydrogenDensityToday_, 0.0};
    if (!validPositive(hydrogenDensityToday_) ||
        !validPositive(model.hubble(options.maxScaleFactor))) {
        throw std::invalid_argument(
            "Nonpositive or nonfinite homogeneous recombination background");
    }
    heliumSahaEnd_ = detail::findSahaEnd(model, false);
    hydrogenSahaEnd_ = detail::findSahaEnd(model, true);
    model.hydrogenSahaEnd = hydrogenSahaEnd_;
    double time = std::log(heliumRatio_ > 0.0 ? heliumSahaEnd_ : hydrogenSahaEnd_);
    double const finalTime = std::log(options.maxScaleFactor);
    detail::RecombinationVector state{
        model.hydrogenSaha(std::exp(time)), heliumRatio_ > 0.0 ? 0.99 : 0.0, 1.0};
    samples_.push_back({time, state});
    double step = std::min(1.0e-4, options.maxLogStep);
    size_t attempts = 0;
    while (time < finalTime) {
        if (++attempts > options.maxSteps) {
            throw std::runtime_error("Homogeneous recombination integration exceeded maxSteps");
        }
        bool const hydrogenEquilibrium = time < std::log(hydrogenSahaEnd_);
        double const endpoint =
            hydrogenEquilibrium ? std::min(finalTime, std::log(hydrogenSahaEnd_)) : finalTime;
        step = std::min({step, options.maxLogStep, endpoint - time});
        if (!(step >
                32.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, std::abs(time)))) {
            throw std::runtime_error("Homogeneous recombination step underflow near a=" +
                std::to_string(std::exp(time)));
        }
        detail::RecombinationVector full, half, refined;
        bool const converged =
            detail::integrateStep(model, options, time, step, state, hydrogenEquilibrium, full) &&
            detail::integrateStep(
                model, options, time, 0.5 * step, state, hydrogenEquilibrium, half) &&
            detail::integrateStep(
                model, options, time + 0.5 * step, 0.5 * step, half, hydrogenEquilibrium, refined);
        double error = std::numeric_limits<double>::infinity();
        if (converged) {
            error = 0.0;
            for (int i = 0; i < 3; ++i) {
                if (!detail::isFiniteRecombination(full[i]) ||
                    !detail::isFiniteRecombination(half[i]) ||
                    !detail::isFiniteRecombination(refined[i])) {
                    error = std::numeric_limits<double>::infinity();
                    break;
                }
                double const scale = options.absoluteTolerance +
                    options.relativeTolerance * std::max(std::abs(state[i]), std::abs(refined[i]));
                error = std::max(error, std::abs(refined[i] - full[i]) / (7.0 * scale));
                if (!detail::isFiniteRecombination(refined[i]) ||
                    refined[i] < -options.absoluteTolerance ||
                    refined[i] > 1.0 + options.absoluteTolerance || (i == 2 && refined[i] <= 0.0)) {
                    error = std::numeric_limits<double>::infinity();
                }
            }
        }
        if (error <= 1.0) {
            time += step;
            if (endpoint - time < 16.0 * std::numeric_limits<double>::epsilon()) {
                time = endpoint;
            }
            state = refined;
            state[1] = std::max(0.0, state[1]);
            if (hydrogenEquilibrium) {
                state[0] = model.hydrogenSaha(std::exp(time));
            }
            samples_.push_back({time, state});
        }
        double const factor = error > 0.0 && detail::isFiniteRecombination(error) ?
            std::clamp(0.85 * std::pow(error, -0.25), 0.2, 3.0) :
            (error == 0.0 ? 3.0 : 0.25);
        step *= factor;
    }
    // Shape-preserving cubic interpolation avoids negative minority species
    // and controls interpolation accuracy by the same maximum log(a) spacing.
    if (samples_.size() > 1) {
        for (size_t i = 0; i < samples_.size(); ++i) {
            size_t const left = i == 0 ? 0 : i - 1;
            size_t const right = i + 1 == samples_.size() ? i : i + 1;
            for (int component = 0; component < 3; ++component) {
                double const leftStep = samples_[i].logScaleFactor - samples_[left].logScaleFactor;
                double const rightStep =
                    samples_[right].logScaleFactor - samples_[i].logScaleFactor;
                double const leftSlope = leftStep > 0.0 ?
                    (samples_[i].values[component] - samples_[left].values[component]) / leftStep :
                    0.0;
                double const rightSlope = rightStep > 0.0 ?
                    (samples_[right].values[component] - samples_[i].values[component]) /
                        rightStep :
                    0.0;
                samples_[i].slopes[component] = i == 0 ?
                    rightSlope :
                    (i + 1 == samples_.size() ?
                            leftSlope :
                            detail::monotoneSlope(leftStep, rightStep, leftSlope, rightSlope));
            }
        }
    }
}

std::array<double, 4> RecombinationHistory::sample(double scaleFactor) const {
    if (!detail::isFiniteRecombination(scaleFactor) || scaleFactor <= 0.0 ||
        scaleFactor >
            options_.maxScaleFactor * (1.0 + 16.0 * std::numeric_limits<double>::epsilon()) ||
        !detail::isFiniteRecombination(parameters_.cmbTemperature / scaleFactor)) {
        throw std::invalid_argument("Scale factor outside homogeneous recombination history");
    }
    scaleFactor = std::min(scaleFactor, options_.maxScaleFactor);
    double const radiationTemperature = parameters_.cmbTemperature / scaleFactor;
    detail::RecombinationModel const model{
        parameters_, heliumRatio_, hydrogenDensityToday_, hydrogenSahaEnd_};
    if (scaleFactor < std::exp(samples_.front().logScaleFactor)) {
        double const redshift = 1.0 / scaleFactor - 1.0;
        if (redshift > 8000.0) {
            return {1.0, 0.0, heliumRatio_ > 0.0 ? 1.0 : 0.0, radiationTemperature};
        }
        if (redshift > 5000.0) {
            double const doubleHelium =
                heliumRatio_ > 0.0 ? model.heliumDoubleSaha(scaleFactor) : 0.0;
            return {1.0, heliumRatio_ > 0.0 ? 1.0 - doubleHelium : 0.0, doubleHelium,
                radiationTemperature};
        }
        if (redshift > 3500.0 && heliumRatio_ > 0.0) {
            return {1.0, 1.0, 0.0, radiationTemperature};
        }
        return {heliumRatio_ > 0.0 ? 1.0 : model.hydrogenSaha(scaleFactor),
            heliumRatio_ > 0.0 ? model.heliumSaha(scaleFactor) : 0.0, 0.0, radiationTemperature};
    }
    double const time = std::log(scaleFactor);
    if (time <= samples_.front().logScaleFactor) {
        auto const& state = samples_.front().values;
        return {state[0], state[1], 0.0, state[2] * radiationTemperature};
    }
    auto upper = std::upper_bound(samples_.begin(), samples_.end(), time,
        [](double value, Sample const& entry) { return value < entry.logScaleFactor; });
    if (upper == samples_.end()) {
        auto const& state = samples_.back().values;
        return {state[0], state[1], 0.0, state[2] * radiationTemperature};
    }
    auto const& right = *upper;
    auto const& left = *(upper - 1);
    double const interval = right.logScaleFactor - left.logScaleFactor;
    double const fraction = (time - left.logScaleFactor) / interval;
    double const squared = fraction * fraction;
    double const cubed = squared * fraction;
    detail::RecombinationVector state;
    for (int i = 0; i < 3; ++i) {
        state[i] = (2.0 * cubed - 3.0 * squared + 1.0) * left.values[i] +
            (cubed - 2.0 * squared + fraction) * interval * left.slopes[i] +
            (-2.0 * cubed + 3.0 * squared) * right.values[i] +
            (cubed - squared) * interval * right.slopes[i];
    }
    if (scaleFactor < hydrogenSahaEnd_) {
        state[0] = model.hydrogenSaha(scaleFactor);
    }
    return {state[0], state[1], 0.0, state[2] * radiationTemperature};
}

double RecombinationHistory::electronFraction(double scaleFactor) const {
    auto const value = sample(scaleFactor);
    return value[0] + heliumRatio_ * (value[1] + 2.0 * value[2]);
}
double RecombinationHistory::gasTemperature(double scaleFactor) const {
    return sample(scaleFactor)[3];
}
double RecombinationHistory::hydrogenIonizedFraction(double scaleFactor) const {
    return sample(scaleFactor)[0];
}
double RecombinationHistory::heliumSinglyIonizedFraction(double scaleFactor) const {
    return sample(scaleFactor)[1];
}
double RecombinationHistory::heliumDoublyIonizedFraction(double scaleFactor) const {
    return sample(scaleFactor)[2];
}
double RecombinationHistory::minScaleFactor() const {
    return detail::kInitialScaleFactor;
}
double RecombinationHistory::maxScaleFactor() const {
    return options_.maxScaleFactor;
}
double RecombinationHistory::heliumToHydrogenNumberRatio() const {
    return heliumRatio_;
}
double RecombinationHistory::hydrogenNumberDensity(double scaleFactor) const {
    sample(scaleFactor);
    double const result =
        1.0e-6 * hydrogenDensityToday_ / (scaleFactor * scaleFactor * scaleFactor);
    if (!detail::isFiniteRecombination(result)) {
        throw std::invalid_argument("Hydrogen number density overflows at requested scale factor");
    }
    return result;
}
double RecombinationHistory::pressurePerHydrogen(double scaleFactor) const {
    return (1.0 + heliumRatio_ + electronFraction(scaleFactor)) *
        static_cast<double>(kBoltzmannEv) * gasTemperature(scaleFactor);
}
Status RecombinationHistory::chemistryState(
    double scaleFactor, State& state, Environment& environment) const {
    auto const value = sample(scaleFactor);
    double const density = hydrogenNumberDensity(scaleFactor);
    double const radiationTemperature = parameters_.cmbTemperature / scaleFactor;
    if (density > 1.0e4 || value[3] < minimumTemperature(Network::Atomic6) ||
        value[3] > maximumTemperature(Network::Atomic6) ||
        radiationTemperature > std::numeric_limits<float>::max()) {
        return Status::UnsupportedState;
    }
    State converted;
    for (int i = 0; i < 3; ++i) {
        bool const upper = value[i] > 0.5;
        converted.abundances[i] = SignedAbundance::fromMinority(
            static_cast<float>(upper ? 1.0 - value[i] : value[i]), upper);
    }
    Environment convertedEnvironment;
    convertedEnvironment.nHydrogen = static_cast<float>(density);
    convertedEnvironment.heliumToHydrogenNumberRatio = static_cast<float>(heliumRatio_);
    convertedEnvironment.cmbTemperature = static_cast<float>(radiationTemperature);
    auto const status = thermalEnergyAtTemperature(
        converted, convertedEnvironment, static_cast<float>(value[3]), converted.thermalEnergy);
    if (status != Status::Success) {
        return status;
    }
    state = converted;
    environment = convertedEnvironment;
    return Status::Success;
}
}    // namespace primordial

#if defined(__clang__)
#pragma float_control(pop)
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
