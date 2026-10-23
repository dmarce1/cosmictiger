/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 This program is free software; you can redistribute it and/or
 modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation; either version 2
 of the License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */

// Adapted from dmarce1/experimental, commit
// 7ba8f1829fdede7348a1418b7cb36c0e7a955302; extended with quadratures and dense output.

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include <cosmictiger/cosmology/friedmann.hpp>

namespace cosmology {
namespace detail {

    constexpr size_t kFriedmannStages = 7;
    constexpr double kMinimumRelativeTolerance = 32.0 * std::numeric_limits<double>::epsilon();
    using FriedmannStages = std::array<FriedmannVariables, kFriedmannStages>;

    struct FriedmannPolynomial {
        double value;
        double amplitude;
    };

    void require(bool condition, char const* message) {
        if (!condition) {
            throw std::domain_error(message);
        }
    }

    FriedmannPolynomial friedmannPolynomial(
        BackgroundCosmologyParameters const& parameters, double a, uint64_t& flops) {
        double const matter = parameters.omegaMatter * a;
        double const curvature = parameters.omegaCurvature * a * a;
        double const darkEnergy = parameters.omegaDarkEnergy * a * a * a * a;
        double const value = parameters.omegaRadiation + matter + curvature + darkEnergy;
        double const amplitude =
            parameters.omegaRadiation + matter + std::abs(curvature) + std::abs(darkEnergy);
        flops += 15;
        if (!(amplitude > 0.0) || !std::isfinite(amplitude) || !std::isfinite(value)) {
            throw std::overflow_error(
                "Friedmann components are outside the double-precision range");
        }
        return {value, amplitude};
    }

    // u=a*adot/H0, s=H0*t or H0*eta. The last three variables are H0 times
    // physical time, conformal time, and integral(d eta/a), respectively.
    bool friedmannRhs(BackgroundCosmologyParameters const& parameters, CosmologicalTime time,
        FriedmannVariables const& value, FriedmannVariables& result, uint64_t& flops) {
        double const a = value[0];
        if (!(a > 0.0) || !std::isfinite(a)) {
            return false;
        }
        if (time == CosmologicalTime::physical) {
            result = {value[1] / a,
                0.5 * parameters.omegaMatter / a + parameters.omegaCurvature +
                    2.0 * parameters.omegaDarkEnergy * a * a,
                1.0, 1.0 / a, 1.0 / (a * a)};
            flops += 11;
        } else {
            result = {value[1],
                0.5 * parameters.omegaMatter + parameters.omegaCurvature * a +
                    2.0 * parameters.omegaDarkEnergy * a * a * a,
                a, 1.0, 1.0 / a};
            flops += 9;
        }
        for (double const component : result) {
            if (!std::isfinite(component)) {
                return false;
            }
        }
        return true;
    }

    FriedmannVariables friedmannStage(FriedmannVariables const& value, double step,
        FriedmannStages const& stages, std::array<double, kFriedmannStages> const& weights,
        uint64_t& flops) {
        auto result = value;
        for (size_t component = 0; component < kFriedmannVariables; component++) {
            double sum = 0.0;
            for (size_t i = 0; i < kFriedmannStages; i++) {
                if (weights[i] != 0.0) {
                    sum += weights[i] * stages[i][component];
                    flops += 2;
                }
            }
            result[component] += step * sum;
            flops += 2;
        }
        return result;
    }

    bool friedmannStep(BackgroundCosmologyParameters const& parameters, CosmologicalTime time,
        FriedmannVariables const& value, double step, FriedmannVariables& next,
        FriedmannVariables& error, FriedmannStages& stages, uint64_t& flops) {
        constexpr std::array<std::array<double, kFriedmannStages>, kFriedmannStages - 1>
            kCoefficients{{
                {1.0 / 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                {3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                {44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0.0, 0.0, 0.0, 0.0},
                {19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0, 0.0, 0.0,
                    0.0},
                {9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0,
                    0.0, 0.0},
                {35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0,
                    0.0},
            }};
        constexpr std::array<double, kFriedmannStages> kErrorWeights{
            35.0 / 384.0 - 5179.0 / 57600.0, 0.0, 500.0 / 1113.0 - 7571.0 / 16695.0,
            125.0 / 192.0 - 393.0 / 640.0, -2187.0 / 6784.0 + 92097.0 / 339200.0,
            11.0 / 84.0 - 187.0 / 2100.0, -1.0 / 40.0};
        if (!friedmannRhs(parameters, time, value, stages[0], flops)) {
            return false;
        }
        for (size_t i = 1; i < kFriedmannStages; i++) {
            next = friedmannStage(value, step, stages, kCoefficients[i - 1], flops);
            if (!friedmannRhs(parameters, time, next, stages[i], flops)) {
                return false;
            }
        }
        error = friedmannStage({}, step, stages, kErrorWeights, flops);
        return true;
    }

    FriedmannDenseSegment denseSegment(double begin, double step, FriedmannVariables const& initial,
        FriedmannStages const& stages, uint64_t& flops) {
        // Dormand-Prince/Shampine quartic continuous extension. Each row
        // multiplies one RK derivative; columns multiply x,x^2,x^3,x^4.
        constexpr std::array<std::array<double, 4>, kFriedmannStages> kDenseWeights{{
            {1.0, -8048581381.0 / 2820520608.0, 8663915743.0 / 2820520608.0,
                -12715105075.0 / 11282082432.0},
            {0.0, 0.0, 0.0, 0.0},
            {0.0, 131558114200.0 / 32700410799.0, -68118460800.0 / 10900136933.0,
                87487479700.0 / 32700410799.0},
            {0.0, -1754552775.0 / 470086768.0, 14199869525.0 / 1410260304.0,
                -10690763975.0 / 1880347072.0},
            {0.0, 127303824393.0 / 49829197408.0, -318862633887.0 / 49829197408.0,
                701980252875.0 / 199316789632.0},
            {0.0, -282668133.0 / 205662961.0, 2019193451.0 / 616988883.0,
                -1453857185.0 / 822651844.0},
            {0.0, 40617522.0 / 29380423.0, -110615467.0 / 29380423.0, 69997945.0 / 29380423.0},
        }};
        FriedmannDenseSegment segment{begin, step, initial, {}};
        for (size_t component = 0; component < kFriedmannVariables; component++) {
            for (size_t power = 0; power < 4; power++) {
                for (size_t stage = 0; stage < kFriedmannStages; stage++) {
                    if (kDenseWeights[stage][power] != 0.0) {
                        segment.coefficients[component][power] +=
                            kDenseWeights[stage][power] * stages[stage][component];
                        flops += 2;
                    }
                }
            }
        }
        return segment;
    }

    double constraintError(
        FriedmannVariables const& value, FriedmannPolynomial const& polynomial, uint64_t& flops) {
        double const normalizedU = value[1] / std::sqrt(polynomial.amplitude);
        flops += 6;
        return std::abs(normalizedU * normalizedU - polynomial.value / polynomial.amplitude);
    }

    double constraintTolerance(FriedmannOptions const& options) {
        return 256.0 *
            (options.relativeTolerance + options.absoluteTolerance +
                std::numeric_limits<double>::epsilon());
    }

    FriedmannAdvance result(BackgroundCosmologyParameters const& parameters,
        FriedmannVariables const& value, uint64_t flops) {
        double const conformalDerivative = parameters.hubbleConstant * value[1];
        double const physicalDerivative = conformalDerivative / value[0];
        if (!std::isfinite(conformalDerivative) || !std::isfinite(physicalDerivative) ||
            (value[1] != 0.0 && (conformalDerivative == 0.0 || physicalDerivative == 0.0))) {
            throw std::overflow_error("Friedmann derivative is outside the double-precision range");
        }
        FriedmannAdvance answer{{value[0], physicalDerivative, conformalDerivative},
            value[2] / parameters.hubbleConstant, value[3] / parameters.hubbleConstant,
            value[4] / parameters.hubbleConstant, flops + 5};
        if (!std::isfinite(answer.physicalTime) || !std::isfinite(answer.conformalTime) ||
            !std::isfinite(answer.driftIntegral)) {
            throw std::overflow_error("Friedmann quadrature is outside the double-precision range");
        }
        return answer;
    }

}    // namespace detail

FriedmannCosmology::FriedmannCosmology(
    BackgroundCosmologyParameters parameters, FriedmannOptions options)
  : parameters_(parameters)
  , options_(options) {
    detail::require(std::isfinite(parameters.hubbleConstant) && parameters.hubbleConstant > 0.0,
        "Friedmann H0 must be positive and finite");
    detail::require(std::isfinite(parameters.omegaRadiation) && parameters.omegaRadiation >= 0.0 &&
            std::isfinite(parameters.omegaMatter) && parameters.omegaMatter >= 0.0 &&
            std::isfinite(parameters.omegaDarkEnergy) && std::isfinite(parameters.omegaCurvature),
        "Friedmann densities must be finite; matter and radiation must be nonnegative");
    double const sum = parameters.omegaRadiation + parameters.omegaMatter +
        parameters.omegaDarkEnergy + parameters.omegaCurvature;
    detail::require(std::abs(sum - 1.0) <= 1.0e-10, "Friedmann density parameters must sum to one");
    detail::require(std::isfinite(options.relativeTolerance) &&
            options.relativeTolerance >= detail::kMinimumRelativeTolerance &&
            options.relativeTolerance < 1.0 && std::isfinite(options.absoluteTolerance) &&
            options.absoluteTolerance >= 0.0 && options.absoluteTolerance < 1.0 &&
            options.maxSteps > 0,
        "Invalid Friedmann integration tolerances or maximum step count");
}

BackgroundCosmologyParameters const& FriedmannCosmology::parameters() const noexcept {
    return parameters_;
}

FriedmannOptions const& FriedmannCosmology::options() const noexcept {
    return options_;
}

FriedmannState FriedmannCosmology::state(double scaleFactor, ExpansionBranch branch) const {
    detail::require(std::isfinite(scaleFactor) && scaleFactor > 0.0,
        "Friedmann scale factor must be positive and finite");
    detail::require(branch == ExpansionBranch::expanding || branch == ExpansionBranch::contracting,
        "Invalid Friedmann expansion branch");
    uint64_t flops = 0;
    auto const polynomial = detail::friedmannPolynomial(parameters_, scaleFactor, flops);
    detail::require(
        polynomial.value >= -64.0 * std::numeric_limits<double>::epsilon() * polynomial.amplitude,
        "Friedmann H squared is negative at the requested scale factor");
    double const u = std::sqrt(std::max(0.0, polynomial.value)) *
        (branch == ExpansionBranch::expanding ? 1.0 : -1.0);
    return detail::result(parameters_, {scaleFactor, u, 0.0, 0.0, 0.0}, flops).state;
}

FriedmannAdvance FriedmannCosmology::integrate(FriedmannState const& initialState,
    double elapsedTime, CosmologicalTime time,
    std::vector<detail::FriedmannDenseSegment>* segments) const {
    detail::require(time == CosmologicalTime::physical || time == CosmologicalTime::conformal,
        "Invalid Friedmann time coordinate");
    detail::require(std::isfinite(elapsedTime), "Friedmann elapsed time must be finite");
    detail::require(std::isfinite(initialState.scaleFactor) && initialState.scaleFactor > 0.0 &&
            std::isfinite(initialState.physicalDerivative) &&
            std::isfinite(initialState.conformalDerivative),
        "Invalid initial Friedmann state");
    uint64_t flops = 7;
    detail::FriedmannVariables value{initialState.scaleFactor,
        initialState.conformalDerivative / parameters_.hubbleConstant, 0.0, 0.0, 0.0};
    auto polynomial = detail::friedmannPolynomial(parameters_, value[0], flops);
    double const impliedU = initialState.physicalDerivative * value[0] / parameters_.hubbleConstant;
    double const consistency = std::abs(impliedU - value[1]) / std::sqrt(polynomial.amplitude);
    double const constraintTolerance = detail::constraintTolerance(options_);
    detail::require(consistency <= constraintTolerance &&
            detail::constraintError(value, polynomial, flops) <= constraintTolerance,
        "Initial Friedmann state violates the derivative relation or density constraint");
    if (elapsedTime == 0.0) {
        return {initialState, 0.0, 0.0, 0.0, flops};
    }
    double const interval = parameters_.hubbleConstant * elapsedTime;
    if (interval == 0.0 || !std::isfinite(interval)) {
        throw std::overflow_error("Friedmann dimensionless time interval is not representable");
    }
    detail::FriedmannVariables rhs{};
    if (!detail::friedmannRhs(parameters_, time, value, rhs, flops)) {
        throw std::overflow_error("Initial Friedmann acceleration is not representable");
    }
    double const amplitude = std::sqrt(polynomial.amplitude);
    double const frequency = std::max(std::abs(rhs[0]) / value[0], std::abs(rhs[1]) / amplitude);
    double const initialStep =
        frequency == 0.0 ? std::abs(interval) : std::min(std::abs(interval), 0.05 / frequency);
    double step = std::copysign(initialStep, interval);
    double position = 0.0;
    flops += 14;
    for (size_t attempt = 0; attempt < options_.maxSteps; attempt++) {
        double const remaining = interval - position;
        flops += 3;
        if (std::abs(step) > std::abs(remaining)) {
            step = remaining;
        }
        if (step == 0.0 || position + step == position) {
            throw std::domain_error(
                "Friedmann evolution reached a singular boundary or time resolution limit");
        }
        detail::FriedmannVariables next{};
        detail::FriedmannVariables error{};
        detail::FriedmannStages stages{};
        if (!detail::friedmannStep(parameters_, time, value, step, next, error, stages, flops)) {
            step *= 0.25;
            flops++;
            continue;
        }
        detail::FriedmannPolynomial nextPolynomial{};
        try {
            nextPolynomial = detail::friedmannPolynomial(parameters_, next[0], flops);
        } catch (std::overflow_error const&) {
            step *= 0.25;
            flops++;
            continue;
        }
        double const scaleA = options_.relativeTolerance * std::max(value[0], next[0]);
        double const scaleU =
            options_.relativeTolerance * std::max(std::abs(value[1]), std::abs(next[1])) +
            options_.absoluteTolerance *
                std::sqrt(std::max(polynomial.amplitude, nextPolynomial.amplitude));
        double errorNorm = std::max(error[0] == 0.0 ? 0.0 : std::abs(error[0]) / scaleA,
            error[1] == 0.0 ? 0.0 : std::abs(error[1]) / scaleU);
        flops += 16;
        for (size_t component = 2; component < detail::kFriedmannVariables; component++) {
            double const scale = options_.relativeTolerance *
                    std::max(std::abs(value[component]), std::abs(next[component])) +
                options_.absoluteTolerance * std::abs(step) *
                    std::max(std::abs(stages[0][component]), std::abs(stages[6][component]));
            errorNorm = std::max(
                errorNorm, error[component] == 0.0 ? 0.0 : std::abs(error[component]) / scale);
            flops += 13;
        }
        if (!std::isfinite(errorNorm)) {
            step *= 0.25;
            flops++;
            continue;
        }
        if (errorNorm <= 1.0) {
            double projection = 0.0;
            if (nextPolynomial.value > 0.0 && next[1] != 0.0) {
                // The acceleration system conserves u^2=P(a). Tiny local
                // invariant errors otherwise accumulate near a~1 and become
                // large relative errors during contraction. Correct u only
                // when that correction PLUS the embedded error fits the
                // existing local tolerance. Preserve the evolved sign and
                // skip the ill-conditioned square root near a turnaround.
                double const constrainedU = std::copysign(std::sqrt(nextPolynomial.value), next[1]);
                double const correction = constrainedU - next[1];
                double const correctedError = std::abs(error[1]) + std::abs(correction);
                flops += 6;
                if (correctedError <= scaleU) {
                    projection = correction;
                    next[1] = constrainedU;
                    errorNorm = std::max(errorNorm, correctedError / scaleU);
                    flops += 2;
                }
            }
            if (detail::constraintError(next, nextPolynomial, flops) > constraintTolerance) {
                throw std::runtime_error(
                    "Friedmann density constraint lost accuracy; tighten tolerances");
            }
            if (segments) {
                auto segment = detail::denseSegment(position, step, value, stages, flops);
                // The x^4 correction vanishes at the initial endpoint and
                // joins the projected next state continuously at x=1.
                segment.coefficients[1][3] += projection / step;
                flops += 2;
                segments->push_back(std::move(segment));
            }
            value = next;
            polynomial = nextPolynomial;
            if (step == remaining) {
                auto answer = detail::result(parameters_, value, flops);
                if (time == CosmologicalTime::physical) {
                    answer.physicalTime = elapsedTime;
                } else {
                    answer.conformalTime = elapsedTime;
                }
                return answer;
            }
            position += step;
            flops++;
        }
        double const factor =
            errorNorm == 0.0 ? 5.0 : std::min(5.0, std::max(0.1, 0.9 * std::pow(errorNorm, -0.2)));
        step *= factor;
        flops += 5;
    }
    throw std::runtime_error(
        "Friedmann evolution exceeded maxSteps; check singular boundaries and tolerances");
}

FriedmannState FriedmannCosmology::advance(
    FriedmannState const& initialState, double elapsedTime, CosmologicalTime time) const {
    return advanceWithIntegrals(initialState, elapsedTime, time).state;
}

FriedmannAdvance FriedmannCosmology::advanceWithIntegrals(
    FriedmannState const& initialState, double elapsedTime, CosmologicalTime time) const {
    return integrate(initialState, elapsedTime, time, nullptr);
}

FriedmannState FriedmannCosmology::evolve(double initialScaleFactor, double elapsedTime,
    CosmologicalTime time, ExpansionBranch branch) const {
    return advance(state(initialScaleFactor, branch), elapsedTime, time);
}

FriedmannTrajectory FriedmannCosmology::trajectory(
    FriedmannState const& initialState, double elapsedTime, CosmologicalTime time) const {
    FriedmannTrajectory answer;
    answer.parameters_ = parameters_;
    answer.initial_ = initialState;
    answer.elapsedTime_ = elapsedTime;
    answer.time_ = time;
    answer.final_ = integrate(initialState, elapsedTime, time, &answer.segments_);
    return answer;
}

FriedmannAdvance const& FriedmannTrajectory::final() const noexcept {
    return final_;
}

FriedmannAdvance FriedmannTrajectory::sample(double elapsedTime) const {
    detail::require(std::isfinite(elapsedTime) && elapsedTime >= std::min(0.0, elapsedTime_) &&
            elapsedTime <= std::max(0.0, elapsedTime_),
        "Friedmann trajectory sample is outside its integrated time interval");
    if (elapsedTime == 0.0) {
        return {initial_, 0.0, 0.0, 0.0, 0};
    }
    if (elapsedTime == elapsedTime_) {
        auto answer = final_;
        answer.flops = 0;
        return answer;
    }
    double const position = elapsedTime * parameters_.hubbleConstant;
    double const sign = elapsedTime_ > 0.0 ? 1.0 : -1.0;
    auto iterator = std::upper_bound(segments_.begin(), segments_.end(), sign * position,
        [sign](double location, detail::FriedmannDenseSegment const& segment) {
            return location < sign * segment.begin;
        });
    if (iterator != segments_.begin()) {
        --iterator;
    }
    auto const& segment = *iterator;
    double const fraction = (position - segment.begin) / segment.step;
    auto value = segment.initial;
    for (size_t component = 0; component < detail::kFriedmannVariables; component++) {
        auto const& coefficients = segment.coefficients[component];
        value[component] += segment.step * fraction *
            (coefficients[0] +
                fraction *
                    (coefficients[1] + fraction * (coefficients[2] + fraction * coefficients[3])));
    }
    auto answer = detail::result(parameters_, value, 4 + 9 * detail::kFriedmannVariables);
    if (time_ == CosmologicalTime::physical) {
        answer.physicalTime = elapsedTime;
    } else {
        answer.conformalTime = elapsedTime;
    }
    return answer;
}

namespace detail {

    double polynomialDerivative(BackgroundCosmologyParameters const& parameters, double a) {
        return parameters.omegaMatter + 2.0 * parameters.omegaCurvature * a +
            4.0 * parameters.omegaDarkEnergy * a * a * a;
    }

    // Check the extrema as well as the endpoints, so a narrow forbidden
    // region cannot be missed by quadrature nodes on either side of it.
    void requireAccessibleInterval(BackgroundCosmologyParameters const& parameters, double lower,
        double upper, uint64_t& flops) {
        std::vector<double> boundaries{lower};
        if (parameters.omegaDarkEnergy != 0.0 &&
            parameters.omegaCurvature / parameters.omegaDarkEnergy < 0.0) {
            double const extremum =
                std::sqrt(-parameters.omegaCurvature / (6.0 * parameters.omegaDarkEnergy));
            if (extremum > lower && extremum < upper) {
                boundaries.push_back(extremum);
            }
            flops += 5;
        }
        boundaries.push_back(upper);
        for (size_t interval = 1; interval < boundaries.size(); interval++) {
            double left = boundaries[interval - 1];
            double right = boundaries[interval];
            double leftDerivative = polynomialDerivative(parameters, left);
            double const rightDerivative = polynomialDerivative(parameters, right);
            flops += 16;
            if (leftDerivative == 0.0 && left > lower) {
                auto const value = friedmannPolynomial(parameters, left, flops);
                require(
                    value.value > 64.0 * std::numeric_limits<double>::epsilon() * value.amplitude,
                    "Scale interval crosses an inaccessible or infinite-time Friedmann root");
            }
            if ((leftDerivative < 0.0) == (rightDerivative < 0.0) || leftDerivative == 0.0 ||
                rightDerivative == 0.0) {
                continue;
            }
            for (size_t iteration = 0; iteration < 80; iteration++) {
                double const middle = left + 0.5 * (right - left);
                if (middle == left || middle == right) {
                    break;
                }
                double const derivative = polynomialDerivative(parameters, middle);
                flops += 11;
                if ((derivative < 0.0) == (leftDerivative < 0.0)) {
                    left = middle;
                    leftDerivative = derivative;
                } else {
                    right = middle;
                }
            }
            auto const value = friedmannPolynomial(parameters, left + 0.5 * (right - left), flops);
            require(value.value > 64.0 * std::numeric_limits<double>::epsilon() * value.amplitude,
                "Scale interval crosses an inaccessible or infinite-time Friedmann root");
        }
    }

    using FriedmannIntegrals = std::array<double, 3>;

    struct ScaleIntegrand {
        BackgroundCosmologyParameters const& parameters;
        double lower;
        double upper;
        double logRatio;
        bool lowerRoot;
        bool upperRoot;

        FriedmannIntegrals evaluate(double angle, uint64_t& flops) const {
            double const sine = std::sin(angle);
            double const cosine = std::cos(angle);
            bool const nearerLower = sine < cosine;
            double const exponent =
                nearerLower ? logRatio * sine * sine : -logRatio * cosine * cosine;
            double const endpoint = nearerLower ? lower : upper;
            // Form a multiplicatively: upper-(upper-a) loses relative digits
            // across the many decades used for cosmological age integrals.
            double const a = endpoint * std::exp(exponent);
            double polynomial;
            if ((lowerRoot && sine < cosine) || (upperRoot && sine >= cosine)) {
                double const root = endpoint;
                double const distance = endpoint * std::expm1(exponent);
                // Factor P(a)-P(root) exactly near a simple endpoint root;
                // direct subtraction loses all precision as a approaches it.
                double const darkEnergy = parameters.omegaDarkEnergy == 0.0 ?
                    0.0 :
                    parameters.omegaDarkEnergy * (a + root) * (a * a + root * root);
                polynomial = distance *
                    (parameters.omegaMatter + parameters.omegaCurvature * (a + root) + darkEnergy);
                flops += parameters.omegaDarkEnergy == 0.0 ? 7 : 13;
            } else {
                polynomial = friedmannPolynomial(parameters, a, flops).value;
            }
            require(polynomial > 0.0 && std::isfinite(polynomial),
                "Scale interval enters an inaccessible Friedmann region");
            double const conformal = 2.0 * logRatio * a * sine * cosine / std::sqrt(polynomial);
            flops += 15;
            return {a * conformal, conformal, conformal / a};
        }
    };

    std::pair<FriedmannIntegrals, FriedmannIntegrals> gaussKronrod(
        ScaleIntegrand const& integrand, double lower, double upper, uint64_t& flops) {
        constexpr std::array<double, 8> kAbscissae{0.99145537112081263921, 0.94910791234275852453,
            0.86486442335976907279, 0.74153118559939443986, 0.58608723546769113029,
            0.40584515137739716691, 0.20778495500789846760, 0.0};
        constexpr std::array<double, 8> kKronrodWeights{0.02293532201052922496,
            0.06309209262997855329, 0.10479001032225018384, 0.14065325971552591875,
            0.16900472663926790283, 0.19035057806478540991, 0.20443294007529889241,
            0.20948214108472782801};
        constexpr std::array<double, 4> kGaussWeights{0.12948496616886969327,
            0.27970539148927666790, 0.38183005050511894495, 0.41795918367346938776};
        double const middle = lower + 0.5 * (upper - lower);
        double const half = 0.5 * (upper - lower);
        auto const center = integrand.evaluate(middle, flops);
        FriedmannIntegrals kronrod{};
        FriedmannIntegrals gauss{};
        flops += 5;
        for (size_t component = 0; component < kronrod.size(); component++) {
            kronrod[component] = kKronrodWeights.back() * center[component];
            gauss[component] = kGaussWeights.back() * center[component];
            flops += 2;
        }
        for (size_t node = 0; node < 7; node++) {
            double const offset = half * kAbscissae[node];
            auto const left = integrand.evaluate(middle - offset, flops);
            auto const right = integrand.evaluate(middle + offset, flops);
            flops += 3;
            for (size_t component = 0; component < kronrod.size(); component++) {
                double const sum = left[component] + right[component];
                kronrod[component] += kKronrodWeights[node] * sum;
                flops += 3;
                if (node % 2 == 1) {
                    gauss[component] += kGaussWeights[node / 2] * sum;
                    flops += 2;
                }
            }
        }
        FriedmannIntegrals error{};
        for (size_t component = 0; component < kronrod.size(); component++) {
            error[component] = std::abs((kronrod[component] - gauss[component]) * half);
            kronrod[component] *= half;
            flops += 4;
        }
        return {kronrod, error};
    }

}    // namespace detail

FriedmannAdvance FriedmannCosmology::intervalsBetweenScales(
    double initialScale, double finalScale) const {
    auto const finalState = state(finalScale);
    state(initialScale);
    if (initialScale == finalScale) {
        return {finalState, 0.0, 0.0, 0.0, 0};
    }
    uint64_t flops = 0;
    double const lower = std::min(initialScale, finalScale);
    double const upper = std::max(initialScale, finalScale);
    auto const lowerPolynomial = detail::friedmannPolynomial(parameters_, lower, flops);
    auto const upperPolynomial = detail::friedmannPolynomial(parameters_, upper, flops);
    double const rootTolerance = 64.0 * std::numeric_limits<double>::epsilon();
    bool const lowerRoot =
        std::abs(lowerPolynomial.value) <= rootTolerance * lowerPolynomial.amplitude;
    bool const upperRoot =
        std::abs(upperPolynomial.value) <= rootTolerance * upperPolynomial.amplitude;
    detail::requireAccessibleInterval(parameters_, lower, upper, flops);
    for (auto const endpoint :
        {std::pair<double, bool>{lower, lowerRoot}, std::pair<double, bool>{upper, upperRoot}}) {
        if (endpoint.second) {
            auto const polynomial = detail::friedmannPolynomial(parameters_, endpoint.first, flops);
            detail::require(std::abs(detail::polynomialDerivative(parameters_, endpoint.first)) *
                        endpoint.first >
                    rootTolerance * polynomial.amplitude,
                "Scale interval ends at an infinite-time Friedmann root");
        }
    }
    double const ratio = (upper - lower) / lower;
    double const logRatio =
        std::isfinite(ratio) ? std::log1p(ratio) : std::log(upper) - std::log(lower);
    detail::ScaleIntegrand const integrand{
        parameters_, lower, upper, logRatio, lowerRoot, upperRoot};
    std::vector<std::pair<double, double>> intervals{{0.0, 0.5 * std::acos(-1.0)}};
    detail::FriedmannIntegrals integral{};
    size_t attempts = 0;
    while (!intervals.empty()) {
        if (++attempts > options_.maxSteps) {
            throw std::runtime_error("Friedmann scale quadrature exceeded maxSteps");
        }
        auto const interval = intervals.back();
        intervals.pop_back();
        auto const estimate =
            detail::gaussKronrod(integrand, interval.first, interval.second, flops);
        bool accepted = true;
        for (size_t component = 0; component < integral.size(); component++) {
            accepted = accepted &&
                estimate.second[component] <=
                    options_.relativeTolerance * std::abs(estimate.first[component]);
            flops += 2;
        }
        if (accepted) {
            for (size_t component = 0; component < integral.size(); component++) {
                integral[component] += estimate.first[component];
                flops++;
            }
        } else {
            double const middle = interval.first + 0.5 * (interval.second - interval.first);
            detail::require(middle != interval.first && middle != interval.second,
                "Friedmann scale quadrature cannot resolve the requested interval");
            intervals.push_back({middle, interval.second});
            intervals.push_back({interval.first, middle});
            flops += 3;
        }
    }
    double const factor = (finalScale > initialScale ? 1.0 : -1.0) / parameters_.hubbleConstant;
    return {
        finalState, factor * integral[0], factor * integral[1], factor * integral[2], flops + 4};
}

}    // namespace cosmology
