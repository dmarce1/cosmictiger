/* CosmicTiger - Anninos et al. primordial source solver; BSL-1.0. */
#pragma once

#include <cmath>

#include <cosmictiger/chemistry/primordial_rates.hpp>
#include <cosmictiger/cuda/cuda_export.hpp>

namespace primordial {

using SolverStatus = Status;

template <class Real>
struct SolverOptionsT {
    // Anninos et al. (1997), section 3.2: ten percent per source substep.
    Real fractionalChangeLimit{Real(0.1)};
    // Minimum electron scale for timestep selection near zero; it never
    // changes a species abundance or an equilibrium concentration.
    Real electronFractionFloor{Real(1.e-20)};
    unsigned maxSubsteps{4096};
};
using SolverOptions = SolverOptionsT<float>;

template <class Real>
struct StepResultT {
    StateT<Real> state{};
    SolverStatus status{SolverStatus::InvalidInput};
    Real acceptedTime{};
    unsigned acceptedSubsteps{};    // Cooling and chemical substeps together.
};
using StepResult = StepResultT<float>;

namespace detail {
    template <class Real>
    CUDA_EXPORT inline Real& anninosSpecies(CompositionT<Real>& c, int species) {
        switch (species) {
        case 0:
            return c.hydrogen;
        case 1:
            return c.hydrogenPlus;
        case 2:
            return c.helium;
        case 3:
            return c.heliumPlus;
        case 4:
            return c.heliumDoublePlus;
        case 5:
            return c.electrons;
        case 6:
            return c.hydrogenMinus;
        case 7:
            return c.molecularHydrogen;
        default:
            return c.molecularHydrogenPlus;
        }
    }

    // Anninos et al. (1997), section 3.1, equations (10)-(12): put the
    // conservation correction in the dominant component, protecting traces.
    template <class Real>
    CUDA_EXPORT inline Status anninosConserve(
        CompositionT<Real>& c, EnvironmentT<Real> const& environment) {
        int const hydrogenSpecies[] = {0, 1, 6, 7, 8};
        int dominant = 0;
        Real largest = c.hydrogen;
        for (int index = 1; index < 5; ++index) {
            int const species = hydrogenSpecies[index];
            Real const nuclei = anninosSpecies(c, species) * Real(species >= 7 ? 2 : 1);
            if (nuclei > largest) {
                largest = nuclei;
                dominant = species;
            }
        }
        Real otherHydrogen = Real(0);
        for (int species : hydrogenSpecies) {
            if (species != dominant) {
                otherHydrogen += anninosSpecies(c, species) * Real(species >= 7 ? 2 : 1);
            }
        }
        anninosSpecies(c, dominant) = (Real(1) - otherHydrogen) / Real(dominant >= 7 ? 2 : 1);
        dominant = 2;
        for (int species = 3; species <= 4; ++species) {
            if (anninosSpecies(c, species) > anninosSpecies(c, dominant)) {
                dominant = species;
            }
        }
        Real otherHelium = Real(0);
        for (int species = 2; species <= 4; ++species) {
            if (species != dominant) {
                otherHelium += anninosSpecies(c, species);
            }
        }
        anninosSpecies(c, dominant) = environment.heliumToHydrogenNumberRatio - otherHelium;
        c.electrons = c.hydrogenPlus + c.heliumPlus + Real(2) * c.heliumDoublePlus +
            c.molecularHydrogenPlus - c.hydrogenMinus;
        c.particles = Real(0);
        for (int species = 0; species < kChemicalSpeciesCount; ++species) {
            Real const value = anninosSpecies(c, species);
            if (!finite(value) || value < Real(0)) {
                return Status::NonConvergence;
            }
            c.particles += value;
        }
        return finite(c.particles) && c.particles > Real(0) ? Status::Success :
                                                              Status::NonConvergence;
    }

    template <class Real>
    CUDA_EXPORT inline void anninosStore(CompositionT<Real> const& c, StateT<Real>& state) {
        for (int species = 0; species < kChemicalSpeciesCount; ++species) {
            state.abundances[species] = chemicalSpeciesAbundance(c, species);
        }
    }

    template <class Real>
    CUDA_EXPORT inline Status anninosEquilibrium(Real creation, Real destruction, Real& value) {
        if (!finite(creation) || !finite(destruction) || creation < Real(0) ||
            destruction < Real(0)) {
            return Status::NonConvergence;
        }
        if (destruction > Real(0)) {
            value = creation / destruction;
        } else if (creation > Real(0)) {
            // No finite equilibrium exists for a species with only production.
            return Status::UnsupportedState;
        }
        return finite(value) ? Status::Success : Status::NonConvergence;
    }

    template <class Real>
    CUDA_EXPORT inline Status anninosChemicalStep(CompositionT<Real>& c,
        EnvironmentT<Real> const& environment, RateCoefficientsT<Real> const& rates, Real step) {
        // Original paper order, section 3.2. Each C/D evaluation sees the
        // preceding species' new abundance, with temperature fixed in this sweep.
        for (int species = 0; species < 6; ++species) {
            Real creation, destruction;
            reactionCreationDestruction(c, environment, rates, species, creation, destruction);
            Real& abundance = anninosSpecies(c, species);
            abundance = (abundance + step * creation) / (Real(1) + step * destruction);
            if (!finite(abundance) || abundance < Real(0)) {
                return Status::NonConvergence;
            }
        }
        if (environment.network == Network::Molecular9) {
            // Fast intermediates use algebraic equilibrium, equations (32)-(33).
            // C and D use the adopted reaction catalogue and coefficient fits.
            int const intermediates[] = {6, 8};
            for (int species : intermediates) {
                Real creation, destruction;
                CompositionT<Real> sources = c;
                if (species == 6) {
                    // Equation (32) neglects collisions between the two trace
                    // intermediates, making H- equilibrium independent of H2+.
                    // This also drops the analogous added GA08 collision channel.
                    sources.molecularHydrogenPlus = Real(0);
                }
                reactionCreationDestruction(
                    sources, environment, rates, species, creation, destruction);
                Status const status =
                    anninosEquilibrium(creation, destruction, anninosSpecies(c, species));
                if (status != Status::Success) {
                    return status;
                }
            }
            Real creation, destruction;
            reactionCreationDestruction(c, environment, rates, 7, creation, destruction);
            c.molecularHydrogen =
                (c.molecularHydrogen + step * creation) / (Real(1) + step * destruction);
        }
        // Conservation is the separate sixth stage in section 3.1. Preserve
        // the independently integrated electron value between subcycles.
        c.particles = Real(0);
        for (int species = 0; species < kChemicalSpeciesCount; ++species) {
            Real const value = anninosSpecies(c, species);
            if (!finite(value) || value < Real(0)) {
                return Status::NonConvergence;
            }
            c.particles += value;
        }
        return finite(c.particles) && c.particles > Real(0) ? Status::Success :
                                                              Status::NonConvergence;
    }
}    // namespace detail

// Anninos, Zhang, Abel & Norman (1997), sections 3.1-3.2:
// https://arxiv.org/abs/astro-ph/9608041 . Explicit thermal subcycling precedes
// sequential backward-difference chemistry. Both retain the supplied outer
// source interval; there is no Newton solve, Jacobian, step doubling, precision
// promotion, compensated state or signed-complement encoding.
// Source application is transactional: failure returns the initial state/time.
template <class Real>
CUDA_EXPORT StepResultT<Real> advance(StateT<Real> const& initial,
    EnvironmentT<Real> const& environment, Real interval,
    SolverOptionsT<Real> const& options = SolverOptionsT<Real>{}) {
    using std::abs;
    StepResultT<Real> result;
    result.state = initial;
    if (!finite(interval) || interval < Real(0) || !finite(options.fractionalChangeLimit) ||
        options.fractionalChangeLimit <= Real(0) || options.fractionalChangeLimit > Real(0.1) ||
        !finite(options.electronFractionFloor) || options.electronFractionFloor <= Real(0) ||
        options.electronFractionFloor > Real(1) || options.maxSubsteps == 0) {
        return result;
    }
    result.status = validateState(initial, environment);
    if (result.status == Status::Success) {
        result.status = validateRadiation(environment.radiation);
    }
    if (result.status != Status::Success || interval == Real(0)) {
        return result;
    }
    StateT<Real> state = initial;
    Real elapsed = Real(0);
    // Equation (20), explicitly subcycled with epsilon*energy/abs(net heating).
    while (elapsed < interval) {
        DerivativeT<Real> derivative;
        result.status = evaluateRhs(state, environment, derivative);
        if (result.status != Status::Success) {
            return result;
        }
        Real const remaining = interval - elapsed;
        Real step = remaining;
        if (derivative.thermalEnergy != Real(0)) {
            Real const bound =
                options.fractionalChangeLimit * state.thermalEnergy / abs(derivative.thermalEnergy);
            step = bound < step ? bound : step;
        }
        if (!finite(step) || step <= Real(0) || elapsed + step == elapsed) {
            result.status = Status::NonConvergence;
            return result;
        }
        if (result.acceptedSubsteps == options.maxSubsteps) {
            result.status = Status::StepLimit;
            return result;
        }
        state.thermalEnergy += step * derivative.thermalEnergy;
        result.status = validateState(state, environment);
        if (result.status != Status::Success) {
            return result;
        }
        ++result.acceptedSubsteps;
        elapsed = step == remaining ? interval : elapsed + step;
    }
    elapsed = Real(0);
    CompositionT<Real> c = composition(state, environment);
    EosT<Real> eos;
    result.status = evaluateEos(state, environment, eos);
    if (result.status != Status::Success) {
        return result;
    }
    // Section 3.2 permits retarded, centered or advanced temperatures. Use
    // the advanced (post-thermal) temperature throughout the chemical stage.
    RateCoefficientsT<Real> thermalRates;
    result.status = evaluateRateCoefficients(eos.temperature, environment.network, thermalRates);
    if (result.status != Status::Success) {
        return result;
    }
    while (elapsed < interval) {
        auto rates = thermalRates;
        evaluateDensityDependentRates(c, environment, eos.temperature, rates);
        Real creation, destruction;
        reactionCreationDestruction(c, environment, rates, 5, creation, destruction);
        Real const electronDerivative = creation - destruction * c.electrons;
        Real const remaining = interval - elapsed;
        Real step = remaining;
        if (electronDerivative != Real(0)) {
            Real const electrons = c.electrons > options.electronFractionFloor ?
                c.electrons :
                options.electronFractionFloor;
            Real const bound = options.fractionalChangeLimit * electrons / abs(electronDerivative);
            step = bound < step ? bound : step;
        }
        if (!finite(step) || step <= Real(0) || elapsed + step == elapsed) {
            result.status = Status::NonConvergence;
            return result;
        }
        if (result.acceptedSubsteps == options.maxSubsteps) {
            result.status = Status::StepLimit;
            return result;
        }
        result.status = detail::anninosChemicalStep(c, environment, rates, step);
        if (result.status != Status::Success) {
            return result;
        }
        ++result.acceptedSubsteps;
        elapsed = step == remaining ? interval : elapsed + step;
    }
    result.status = detail::anninosConserve(c, environment);
    if (result.status != Status::Success) {
        return result;
    }
    detail::anninosStore(c, state);
    result.status = validateState(state, environment);
    if (result.status != Status::Success) {
        return result;
    }
    result.state = state;
    result.acceptedTime = interval;
    result.status = Status::Success;
    return result;
}

}    // namespace primordial
