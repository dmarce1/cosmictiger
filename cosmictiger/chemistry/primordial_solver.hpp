/* CosmicTiger - shared primordial source solver; BSL-1.0. */
#pragma once

#include <cmath>

#include <cosmictiger/chemistry/primordial_rates.hpp>
#include <cosmictiger/cuda/cuda_export.hpp>

namespace primordial {

using SolverStatus = Status;

struct SolverOptions {
    float relativeTolerance{1.e-4f};
    float abundanceAbsoluteTolerance[kAbundanceCount]{
        1.e-16f, 1.e-16f, 1.e-16f, 1.e-20f, 1.e-16f, 1.e-20f};
    float energyAbsoluteTolerance{1.e-6f};    // eV per H nucleus.
    unsigned maxSubsteps{4096};
    unsigned maxAttempts{8192};
    unsigned maxNewtonIterations{20};
    float minStepFraction{1.e-12f};
    float initialStepFraction{1.0f};
};

struct StepResult {
    State state{};
    SolverStatus status{SolverStatus::InvalidInput};
    float acceptedTime{};    // Physical seconds; interval exactly on success.
    float maxError{};        // Largest finite step-doubling norm, including rejected trials.
    unsigned acceptedSubsteps{};
    unsigned rejectedSubsteps{};
    unsigned newtonIterations{};
};

namespace detail {
    inline constexpr int kSolverVariables = kAbundanceCount + 1;

    CUDA_EXPORT inline FloatFloat solverAbs(FloatFloat value) {
        return value < FloatFloat(0.0f) ? -value : value;
    }
    CUDA_EXPORT inline float solverMaximum(float left, float right) {
        return left > right ? left : right;
    }
    CUDA_EXPORT inline float solverMinimum(float left, float right) {
        return left < right ? left : right;
    }
    CUDA_EXPORT inline FloatFloat solverValue(WorkingState const& state, int index) {
        return index < kAbundanceCount ? state.abundances[index] : state.thermalEnergy;
    }
    CUDA_EXPORT inline void solverSetValue(WorkingState& state, int index, FloatFloat value) {
        if (index < kAbundanceCount) {
            state.abundances[index] = value;
        } else {
            state.thermalEnergy = value;
        }
    }
    CUDA_EXPORT inline FloatFloat solverDerivative(Derivative const& derivative, int index) {
        return index < kAbundanceCount ? derivative.abundances[index] : derivative.thermalEnergy;
    }
    CUDA_EXPORT inline float solverMinority(FloatFloat value) {
        return (value > FloatFloat(0.5f) ? FloatFloat(1.0f) - value : value).toFloat();
    }
    CUDA_EXPORT inline float solverWeight(WorkingState const& start, WorkingState const& trial,
        SolverOptions const& options, int index) {
        if (index < kAbundanceCount) {
            return options.abundanceAbsoluteTolerance[index] +
                options.relativeTolerance *
                solverMaximum(solverMinority(start.abundances[index]),
                    solverMinority(trial.abundances[index]));
        }
        return options.energyAbsoluteTolerance +
            options.relativeTolerance *
            solverMaximum(start.thermalEnergy.toFloat(), trial.thermalEnergy.toFloat());
    }
    CUDA_EXPORT inline bool solverOptionsValid(SolverOptions const& options) {
        // Coefficients and EOS temperature are FP32. Tighter requests need a
        // separately validated coefficient/Jacobian path, not a smaller number here.
        if (!finite(options.relativeTolerance) || options.relativeTolerance < 1.e-5f ||
            options.relativeTolerance > 0.1f || !finite(options.energyAbsoluteTolerance) ||
            options.energyAbsoluteTolerance <= 0.0f || options.maxSubsteps == 0 ||
            options.maxAttempts == 0 || options.maxNewtonIterations == 0 ||
            !finite(options.minStepFraction) || options.minStepFraction <= 0.0f ||
            options.minStepFraction > 1.0f || !finite(options.initialStepFraction) ||
            options.initialStepFraction <= 0.0f || options.initialStepFraction > 1.0f) {
            return false;
        }
        for (int index = 0; index < kAbundanceCount; ++index) {
            if (!finite(options.abundanceAbsoluteTolerance[index]) ||
                options.abundanceAbsoluteTolerance[index] <= 0.0f) {
                return false;
            }
        }
        return true;
    }

    struct SolverCoordinates {
        bool upper[kAbundanceCount]{};
        FloatFloat values[kSolverVariables]{};

        CUDA_EXPORT explicit SolverCoordinates(WorkingState const& state) {
            for (int index = 0; index < kAbundanceCount; ++index) {
                upper[index] = state.abundances[index] > FloatFloat(0.5f);
                values[index] = state.abundances[index] - FloatFloat(upper[index] ? 1.0f : 0.0f);
            }
            values[kAbundanceCount] = state.thermalEnergy;
        }
        CUDA_EXPORT WorkingState state() const {
            WorkingState result;
            for (int index = 0; index < kAbundanceCount; ++index) {
                result.abundances[index] = values[index] + FloatFloat(upper[index] ? 1.0f : 0.0f);
            }
            result.thermalEnergy = values[kAbundanceCount];
            return result;
        }
    };

    // Clamp each fraction to its branch's feasible half-line: fraction >= 0 on
    // the lower branch (stored value >= 0) and fraction <= 1 on the upper branch
    // (stored value <= 0). A species at exactly zero (He++ below 2e4 K, say)
    // can receive a noise-level correction of the wrong sign from the
    // numerical Jacobian; left unprojected, every damped candidate stays
    // infeasible, the composition check rejects all of them, and the substep
    // is rejected at every useful step size. The far bounds and the 0.5
    // branch split are still left to the composition check.
    CUDA_EXPORT inline void solverProject(SolverCoordinates& coordinates) {
        for (int index = 0; index < kAbundanceCount; ++index) {
            FloatFloat& value = coordinates.values[index];
            if (coordinates.upper[index] ? value > FloatFloat(0.0f) : value < FloatFloat(0.0f)) {
                value = FloatFloat(0.0f);
            }
        }
    }

    // Partial-pivoted dense elimination in FloatFloat. Independent elemental
    // abundances remove redundant conservation rows before this solve.
    CUDA_EXPORT inline bool solverLinearSolve(FloatFloat matrix[kSolverVariables][kSolverVariables],
        FloatFloat right[kSolverVariables], FloatFloat solution[kSolverVariables]) {
        for (int column = 0; column < kSolverVariables; ++column) {
            int pivot = column;
            for (int row = column + 1; row < kSolverVariables; ++row) {
                if (solverAbs(matrix[row][column]) > solverAbs(matrix[pivot][column])) {
                    pivot = row;
                }
            }
            if (!matrix[pivot][column].isFinite() || matrix[pivot][column] == FloatFloat(0.0f)) {
                return false;
            }
            if (pivot != column) {
                for (int j = column; j < kSolverVariables; ++j) {
                    FloatFloat const saved = matrix[pivot][j];
                    matrix[pivot][j] = matrix[column][j];
                    matrix[column][j] = saved;
                }
                FloatFloat const saved = right[pivot];
                right[pivot] = right[column];
                right[column] = saved;
            }
            for (int row = column + 1; row < kSolverVariables; ++row) {
                FloatFloat const multiplier = matrix[row][column] / matrix[column][column];
                for (int j = column + 1; j < kSolverVariables; ++j) {
                    matrix[row][j] -= multiplier * matrix[column][j];
                }
                right[row] -= multiplier * right[column];
            }
        }
        for (int row = kSolverVariables - 1; row >= 0; --row) {
            FloatFloat value = right[row];
            for (int j = row + 1; j < kSolverVariables; ++j) {
                value -= matrix[row][j] * solution[j];
            }
            solution[row] = value / matrix[row][row];
            if (!solution[row].isFinite()) {
                return false;
            }
        }
        return true;
    }

    CUDA_EXPORT inline WorkingState solverInteriorAnchor(
        WorkingState const& state, Environment const& environment) {
        WorkingState anchor = state;
        float const interior[kAbundanceCount] = {0.2f, 0.2f, 0.2f, 0.01f, 0.1f, 0.01f};
        for (int index = 0; index < kAbundanceCount; ++index) {
            float const target =
                environment.network == Network::Atomic6 && index >= 3 ? 0.0f : interior[index];
            anchor.abundances[index] = state.abundances[index] +
                FloatFloat(0.0001f) * (FloatFloat(target) - state.abundances[index]);
        }
        return anchor;
    }

    // Numerical Jacobian for the initial verification solver. Composition products
    // are FloatFloat; temperature-dependent coefficients are FP32, so perturbations
    // use sqrt(epsilon_float), not a double-precision finite-difference step.
    // At a simplex corner, a nearby valid interior point supplies otherwise blocked
    // coordinate derivatives. This changes only the Jacobian, never accepted state.
    // Analytic/AD Jacobians and a broader stiffness campaign remain separate gates.
    template <class Model>
    CUDA_EXPORT bool solverJacobian(WorkingState const& state, Environment const& environment,
        Derivative const& baseDerivative, SolverOptions const& options, Model const& model,
        FloatFloat jacobian[kSolverVariables][kSolverVariables]) {
        for (int column = 0; column < kSolverVariables; ++column) {
            bool const inactive = column < kAbundanceCount &&
                ((environment.network == Network::Atomic6 && column >= 3) ||
                    (environment.heliumToHydrogenNumberRatio == 0.0f &&
                        (column == 1 || column == 2)));
            if (inactive) {
                for (int row = 0; row < kSolverVariables; ++row) {
                    jacobian[row][column] = FloatFloat(0.0f);
                }
                continue;
            }
            WorkingState base = state;
            Derivative derivativeAtBase = baseDerivative;
            bool found = false;
            for (int anchorAttempt = 0; anchorAttempt < 4 && !found; ++anchorAttempt) {
                if (anchorAttempt != 0) {
                    base = solverInteriorAnchor(state, environment);
                    // At an exact thermal endpoint, the composition anchor or
                    // its one-sided coordinate probe can leave the EOS domain.
                    // Search a small energy neighbourhood for a feasible
                    // Jacobian base. Only trial derivative evaluations move;
                    // the residual and every accepted state remain unchanged.
                    if (anchorAttempt > 1) {
                        FloatFloat const displacement = base.thermalEnergy * FloatFloat(0.002f);
                        base.thermalEnergy += anchorAttempt == 2 ? -displacement : displacement;
                    }
                    if (model(base, environment, derivativeAtBase) != Status::Success) {
                        continue;
                    }
                }
                // Reaction products are evaluated in FloatFloat. Resolve the
                // minority, not a fixed abundance floor: an oversized secant for
                // x'=-k*x*x otherwise invents a large derivative at trace x and
                // makes Newton's correction falsely small. The absolute-error
                // budget supplies the scale only below meaningful state accuracy.
                float const scale = column == kAbundanceCount ?
                    solverMaximum(base.thermalEnergy.toFloat(), 1.e-6f) :
                    solverMaximum(solverMinority(base.abundances[column]),
                        solverMinimum(1.0f,
                            options.abundanceAbsoluteTolerance[column] /
                                options.relativeTolerance));
                FloatFloat const perturbation(3.4526698e-4f * scale);

                for (int direction = 0; direction < 2 && !found; ++direction) {
                    FloatFloat const delta = direction == 0 ? perturbation : -perturbation;
                    WorkingState perturbed = base;
                    solverSetValue(perturbed, column, solverValue(base, column) + delta);
                    Derivative shifted;
                    if (model(perturbed, environment, shifted) == Status::Success) {
                        for (int row = 0; row < kSolverVariables; ++row) {
                            jacobian[row][column] = (solverDerivative(shifted, row) -
                                                        solverDerivative(derivativeAtBase, row)) /
                                delta;
                            if (!jacobian[row][column].isFinite()) {
                                return false;
                            }
                        }
                        found = true;
                    }
                }
            }
            if (!found) {
                return false;
            }
        }
        return true;
    }

    template <class Model>
    CUDA_EXPORT Status solverResidual(SolverCoordinates const& trial,
        SolverCoordinates const& startCoordinates, WorkingState const& start,
        Environment const& environment, FloatFloat interval, SolverOptions const& options,
        Model const& model, FloatFloat residual[kSolverVariables], Derivative& derivative,
        float& norm) {
        WorkingState const state = trial.state();
        Status const status = model(state, environment, derivative);
        if (status != Status::Success) {
            return status;
        }
        norm = 0.0f;
        for (int index = 0; index < kSolverVariables; ++index) {
            residual[index] = trial.values[index] - startCoordinates.values[index] -
                interval * solverDerivative(derivative, index);
            float const scaled = (solverAbs(residual[index]) /
                FloatFloat(solverWeight(start, state, options, index)))
                                     .toFloat();
            if (!finite(scaled)) {
                return Status::NonConvergence;
            }
            norm = solverMaximum(norm, scaled);
        }
        return Status::Success;
    }

    template <class Model>
    CUDA_EXPORT Status backwardEuler(WorkingState const& start, Environment const& environment,
        FloatFloat interval, SolverOptions const& options, Model const& model, WorkingState& result,
        unsigned& iterations) {
        SolverCoordinates const startCoordinates(start);
        SolverCoordinates trial = startCoordinates;
        for (unsigned iteration = 0; iteration < options.maxNewtonIterations; ++iteration) {
            ++iterations;
            FloatFloat residual[kSolverVariables]{};
            Derivative derivative;
            float norm;
            Status const status = solverResidual(trial, startCoordinates, start, environment,
                interval, options, model, residual, derivative, norm);
            if (status != Status::Success) {
                return status;
            }
            if (norm <= 0.05f) {
                result = trial.state();
                return Status::Success;
            }
            WorkingState const state = trial.state();
            FloatFloat jacobian[kSolverVariables][kSolverVariables]{};
            if (!solverJacobian(state, environment, derivative, options, model, jacobian)) {
                return Status::NonConvergence;
            }
            float weights[kSolverVariables]{};
            for (int index = 0; index < kSolverVariables; ++index) {
                weights[index] = solverWeight(start, state, options, index);
            }
            FloatFloat right[kSolverVariables]{};
            for (int row = 0; row < kSolverVariables; ++row) {
                right[row] = -residual[row] / FloatFloat(weights[row]);
                for (int column = 0; column < kSolverVariables; ++column) {
                    jacobian[row][column] = (FloatFloat(row == column ? 1.0f : 0.0f) -
                                                interval * jacobian[row][column]) *
                        FloatFloat(weights[column]) / FloatFloat(weights[row]);
                }
            }
            FloatFloat correction[kSolverVariables]{};
            if (!solverLinearSolve(jacobian, right, correction)) {
                return Status::NonConvergence;
            }
            float correctionNorm = 0.0f;
            for (int index = 0; index < kSolverVariables; ++index) {
                correctionNorm =
                    solverMaximum(correctionNorm, solverAbs(correction[index]).toFloat());
                correction[index] *= FloatFloat(weights[index]);
            }
            // A small correction from a numerical Jacobian is not evidence of
            // convergence: an inaccurate derivative can make it arbitrarily
            // small while the residual is large. The only residual exception
            // is actual representation stagnation below the state tolerance.
            // This can occur for a stiff upper-branch trace whose decoded
            // FloatFloat value has exhausted the minority word's precision.
            if (finite(correctionNorm) && correctionNorm <= 0.05f) {
                SolverCoordinates corrected = trial;
                for (int index = 0; index < kSolverVariables; ++index) {
                    corrected.values[index] += correction[index];
                }
                // Project before decoding: a representable noise correction on
                // an exact-zero species must not disguise stagnation.
                solverProject(corrected);
                WorkingState const decodedCorrection = corrected.state();
                bool unchanged = true;
                for (int index = 0; index < kSolverVariables; ++index) {
                    unchanged = unchanged &&
                        solverValue(decodedCorrection, index) == solverValue(state, index);
                }
                if (unchanged) {
                    result = state;
                    return Status::Success;
                }
            }
            bool accepted = false;
            float damping = 1.0f;
            for (int search = 0; search < 20; ++search) {
                SolverCoordinates candidate = trial;
                for (int index = 0; index < kSolverVariables; ++index) {
                    candidate.values[index] += FloatFloat(damping) * correction[index];
                }
                // The residual is evaluated at the projected point.
                solverProject(candidate);
                FloatFloat candidateResidual[kSolverVariables]{};
                Derivative candidateDerivative;
                float candidateNorm;
                Status const candidateStatus =
                    solverResidual(candidate, startCoordinates, start, environment, interval,
                        options, model, candidateResidual, candidateDerivative, candidateNorm);
                if (candidateStatus == Status::Success &&
                    (candidateNorm <= 0.05f || candidateNorm < norm * (1.0f - 0.001f * damping))) {
                    trial = candidate;
                    accepted = true;
                    break;
                }
                damping *= 0.5f;
            }
            if (!accepted) {
                return Status::NonConvergence;
            }
        }
        return Status::NonConvergence;
    }

    struct PhysicalModel {
        CUDA_EXPORT Status operator()(WorkingState const& state, Environment const& environment,
            Derivative& derivative) const {
            return evaluateRhs(state, environment, derivative);
        }
    };
}    // namespace detail

// Adaptive backward Euler with two half-steps as the accepted solution.
// Local time is normalized to [0,1]; all physical seconds enter as interval*h.
// The accepted internal state and clock retain FloatFloat increments. Saved
// fractions change signed-complement branch only after an accepted substep.
// This is a first-order verification baseline, not the planned SDIRK release.
template <class Model>
CUDA_EXPORT StepResult advanceWithModel(State const& initial, Environment const& environment,
    float interval, SolverOptions const& options, Model const& model) {
    StepResult result;
    result.state = initial;
    if (!finite(interval) || interval < 0.0f || !detail::solverOptionsValid(options)) {
        return result;
    }
    for (int index = 0; index < kAbundanceCount; ++index) {
        if (!finite(initial.abundances[index].raw()) ||
            ::fabsf(initial.abundances[index].raw()) > 0.5f) {
            return result;
        }
    }
    WorkingState accepted = workingState(initial);
    Derivative initialDerivative;
    result.status = model(accepted, environment, initialDerivative);
    if (result.status != Status::Success || interval == 0.0f) {
        return result;
    }
    FloatFloat acceptedFraction(0.0f);
    float step = options.initialStepFraction;
    Status lastFailure = Status::NonConvergence;
    for (unsigned attempt = 0; attempt < options.maxAttempts; ++attempt) {
        if (result.acceptedSubsteps >= options.maxSubsteps) {
            result.status = Status::StepLimit;
            return result;
        }
        FloatFloat const remaining = FloatFloat(1.0f) - acceptedFraction;
        FloatFloat const fraction = remaining < FloatFloat(step) ? remaining : FloatFloat(step);
        FloatFloat const physicalStep = FloatFloat(interval) * fraction;
        if (acceptedFraction + fraction == acceptedFraction || !physicalStep.isFinite() ||
            physicalStep <= FloatFloat(0.0f)) {
            result.status = Status::NonConvergence;
            return result;
        }
        WorkingState coarse, halfway, fine;
        Status status = detail::backwardEuler(
            accepted, environment, physicalStep, options, model, coarse, result.newtonIterations);
        if (status == Status::Success) {
            status = detail::backwardEuler(accepted, environment, physicalStep * FloatFloat(0.5f),
                options, model, halfway, result.newtonIterations);
        }
        if (status == Status::Success) {
            status = detail::backwardEuler(halfway, environment, physicalStep * FloatFloat(0.5f),
                options, model, fine, result.newtonIterations);
        }
        float error = 0.0f;
        State packed;
        if (status == Status::Success) {
            for (int index = 0; index < detail::kSolverVariables; ++index) {
                float const componentError = (detail::solverAbs(detail::solverValue(fine, index) -
                                                  detail::solverValue(coarse, index)) /
                    FloatFloat(detail::solverWeight(accepted, fine, options, index)))
                                                 .toFloat();
                if (!finite(componentError)) {
                    status = Status::NonConvergence;
                    break;
                }
                error = detail::solverMaximum(error, componentError);
            }
            if (status == Status::Success) {
                result.maxError = detail::solverMaximum(result.maxError, error);
                status = composeState(fine, packed);
                if (status == Status::Success) {
                    Derivative packedDerivative;
                    status = model(workingState(packed), environment, packedDerivative);
                }
            }
        }
        if (status == Status::Success && error <= 1.0f) {
            accepted = fine;
            result.state = packed;
            acceptedFraction += fraction;
            ++result.acceptedSubsteps;
            result.acceptedTime = (FloatFloat(interval) * acceptedFraction).toFloat();
            if (acceptedFraction >= FloatFloat(1.0f)) {
                result.acceptedTime = interval;
                result.status = Status::Success;
                return result;
            }
            float const factor =
                error > 0.0f ? detail::solverMinimum(2.0f, 0.9f / ::sqrtf(error)) : 2.0f;
            step = fraction.toFloat() * detail::solverMaximum(0.5f, factor);
        } else {
            ++result.rejectedSubsteps;
            lastFailure = status == Status::UnsupportedState ? status : Status::NonConvergence;
            float const factor = status == Status::Success && error > 0.0f ?
                detail::solverMaximum(0.1f, detail::solverMinimum(0.5f, 0.8f / ::sqrtf(error))) :
                0.25f;
            step = fraction.toFloat() * factor;
            if (step < options.minStepFraction || !finite(step) || step <= 0.0f) {
                result.status = lastFailure;
                return result;
            }
        }
    }
    result.status = Status::StepLimit;
    return result;
}

CUDA_EXPORT inline StepResult advance(State const& initial, Environment const& environment,
    float interval, SolverOptions const& options = SolverOptions{}) {
    return advanceWithModel(initial, environment, interval, options, detail::PhysicalModel{});
}

}    // namespace primordial
