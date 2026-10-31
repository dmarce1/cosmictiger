/* CosmicTiger - scalar GADGET-4 cooling equations; BSL-1.0. */
#pragma once

#include <cmath>
#include <cstdint>

// Mathematical implementation of Katz, Weinberg & Hernquist (1996), ApJS 105,
// 19, equations 33-38 and Table 1, with the constants, interpolation and iteration
// conventions of official GADGET-4 cooling.cc at commit
// 6fb393b5e25907f2ff06211f67d341c5e40e90d1. It intentionally does not use Anninos,
// molecular chemistry, external UV, a Newton solve, or a non-equilibrium abundance history.

namespace gadget4 {
namespace detail {
    CUDA_EXPORT inline bool finite(double value) {
        // Fast-math must not remove invalid-input checks.
        union Bits {
            double value;
            uint64_t bits;
        } representation{value};
        return (representation.bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
    }
    CUDA_EXPORT inline double maximum(double first, double second) {
        return first < second ? second : first;
    }
    CUDA_EXPORT inline bool validParameters(Parameters const& parameters) {
        if (parameters.rates == nullptr || !finite(parameters.redshift) ||
            (parameters.comoving && parameters.redshift <= -1.0)) {
            return false;
        }
        return true;
    }
    CUDA_EXPORT inline bool validState(double energy, double density, double electrons) {
        return finite(energy) && energy > 0.0 && finite(density) && density > 0.0 &&
            finite(electrons) && electrons >= 0.0;
    }
    CUDA_EXPORT inline void finishGas(GasState& gas, double temperature, double electrons) {
        gas.temperature = temperature;
        gas.electronFraction = electrons;
        gas.meanMolecularWeight =
            (1.0 + 4.0 * kHeliumNumberRatio) / (1.0 + kHeliumNumberRatio + electrons);
    }
    CUDA_EXPORT inline RateRow interpolate(Parameters const& parameters, double logTemperature) {
        double const coordinate =
            (logTemperature - kMinimumLogTemperature) / kLogTemperatureSpacing;
        size_t const index = size_t(coordinate);
        double const upperWeight = coordinate - double(index);
        double const lowerWeight = 1.0 - upperWeight;
        RateRow const& lower = parameters.rates[index];
        RateRow const& upper = parameters.rates[index + 1];
        RateRow result;
        result.hydrogenExcitation =
            lowerWeight * lower.hydrogenExcitation + upperWeight * upper.hydrogenExcitation;
        result.heliumIonExcitation =
            lowerWeight * lower.heliumIonExcitation + upperWeight * upper.heliumIonExcitation;
        result.freeFree = lowerWeight * lower.freeFree + upperWeight * upper.freeFree;
        result.hydrogenRecombination =
            lowerWeight * lower.hydrogenRecombination + upperWeight * upper.hydrogenRecombination;
        result.heliumRecombination =
            lowerWeight * lower.heliumRecombination + upperWeight * upper.heliumRecombination;
        result.heliumDoubleRecombination = lowerWeight * lower.heliumDoubleRecombination +
            upperWeight * upper.heliumDoubleRecombination;
        result.heliumDielectronicRecombination =
            lowerWeight * lower.heliumDielectronicRecombination +
            upperWeight * upper.heliumDielectronicRecombination;
        result.hydrogenIonization =
            lowerWeight * lower.hydrogenIonization + upperWeight * upper.hydrogenIonization;
        result.heliumIonization =
            lowerWeight * lower.heliumIonization + upperWeight * upper.heliumIonization;
        result.heliumIonIonization =
            lowerWeight * lower.heliumIonIonization + upperWeight * upper.heliumIonIonization;
        return result;
    }
    CUDA_EXPORT inline Status findAbundances(
        double logTemperature, Parameters const& parameters, double& electrons, GasState& gas) {
        if (!finite(logTemperature)) {
            return Status::InvalidInput;
        }
        gas = GasState{};
        if (logTemperature <= kMinimumLogTemperature) {
            electrons = 0.0;
            finishGas(gas, std::pow(10.0, logTemperature), electrons);
            return Status::Success;
        }
        if (logTemperature >= kMaximumLogTemperature) {
            gas.hydrogenNeutral = gas.heliumNeutral = 0.0;
            gas.hydrogenIonized = 1.0;
            gas.heliumDoubleIonized = kHeliumNumberRatio;
            electrons = 1.0 + 2.0 * kHeliumNumberRatio;
            finishGas(gas, std::pow(10.0, logTemperature), electrons);
            return Status::Success;
        }
        gas.coefficients = interpolate(parameters, logTemperature);
        RateRow const& rates = gas.coefficients;
        gas.hydrogenNeutral =
            rates.hydrogenRecombination / (rates.hydrogenRecombination + rates.hydrogenIonization);
        gas.hydrogenIonized = 1.0 - gas.hydrogenNeutral;
        if (rates.heliumIonization <= 1.0e-60) {
            gas.heliumNeutral = kHeliumNumberRatio;
            gas.heliumIonized = gas.heliumDoubleIonized = 0.0;
        } else {
            gas.heliumIonized = kHeliumNumberRatio /
                (1.0 +
                    (rates.heliumRecombination + rates.heliumDielectronicRecombination) /
                        rates.heliumIonization +
                    rates.heliumIonIonization / rates.heliumDoubleRecombination);
            gas.heliumNeutral = gas.heliumIonized *
                (rates.heliumRecombination + rates.heliumDielectronicRecombination) /
                rates.heliumIonization;
            gas.heliumDoubleIonized =
                gas.heliumIonized * rates.heliumIonIonization / rates.heliumDoubleRecombination;
        }
        electrons = gas.hydrogenIonized + gas.heliumIonized + 2.0 * gas.heliumDoubleIonized;
        finishGas(gas, std::pow(10.0, logTemperature), electrons);
        return finite(electrons) ? Status::Success : Status::NonConvergence;
    }
    CUDA_EXPORT inline Status convertEnergy(double energy, double density,
        Parameters const& parameters, double& electrons, double& temperature) {
        double molecularWeight =
            (1.0 + 4.0 * kHeliumNumberRatio) / (1.0 + kHeliumNumberRatio + electrons);
        temperature = kGammaMinusOne / kBoltzmann * energy * kProtonMass * molecularWeight;
        double damping = 0.0;
        for (int iteration = 1; iteration <= kMaximumIterations; ++iteration) {
            double const previousElectrons = electrons;
            GasState gas;
            Status const status =
                findAbundances(std::log10(temperature), parameters, electrons, gas);
            if (status != Status::Success) {
                return status;
            }
            double const previousTemperature = temperature;
            molecularWeight =
                (1.0 + 4.0 * kHeliumNumberRatio) / (1.0 + kHeliumNumberRatio + electrons);
            double const nextTemperature =
                kGammaMinusOne / kBoltzmann * energy * kProtonMass * molecularWeight;
            damping = maximum(damping,
                nextTemperature / (1.0 + kHeliumNumberRatio + electrons) *
                    std::fabs((electrons - previousElectrons) /
                        (nextTemperature - previousTemperature + 1.0)));
            temperature =
                previousTemperature + (nextTemperature - previousTemperature) / (1.0 + damping);
            if (!finite(temperature) || temperature <= 0.0) {
                return Status::NonConvergence;
            }
            if (std::fabs(temperature - previousTemperature) <= 1.0e-3 * temperature &&
                iteration < kMaximumIterations) {
                return Status::Success;
            }
        }
        return Status::NonConvergence;
    }
    CUDA_EXPORT inline Status rateAtTemperature(double logTemperature, double density,
        Parameters const& parameters, double& electrons, double& netRate) {
        if (logTemperature <= kMinimumLogTemperature) {
            logTemperature = kMinimumLogTemperature + 0.5 * kLogTemperatureSpacing;
        }
        double const hydrogenDensity = kHydrogenMassFraction * density / kProtonMass;
        double const temperature = std::pow(10.0, logTemperature);
        double loss;
        if (logTemperature < kMaximumLogTemperature) {
            GasState gas;
            Status const status = findAbundances(logTemperature, parameters, electrons, gas);
            if (status != Status::Success) {
                return status;
            }
            RateRow const& r = gas.coefficients;
            double const excitation = r.hydrogenExcitation * electrons * gas.hydrogenNeutral +
                r.heliumIonExcitation * electrons * gas.heliumIonized;
            double const ionization =
                2.18e-11 * r.hydrogenIonization * electrons * gas.hydrogenNeutral +
                3.94e-11 * r.heliumIonization * electrons * gas.heliumNeutral +
                8.72e-11 * r.heliumIonIonization * electrons * gas.heliumIonized;
            double const recombination = 1.036e-16 * temperature * electrons *
                    (r.hydrogenRecombination * gas.hydrogenIonized) +
                1.036e-16 * temperature * electrons * (r.heliumRecombination * gas.heliumIonized) +
                1.036e-16 * temperature * electrons *
                    (r.heliumDoubleRecombination * gas.heliumDoubleIonized) +
                6.526e-11 * r.heliumDielectronicRecombination * electrons * gas.heliumIonized;
            double const freeFree = r.freeFree *
                (gas.hydrogenIonized + gas.heliumIonized + 4.0 * gas.heliumDoubleIonized) *
                electrons;
            loss = excitation + ionization + recombination + freeFree;
        } else {
            electrons = 1.0 + 2.0 * kHeliumNumberRatio;
            loss = 1.42e-27 * std::sqrt(temperature) *
                (1.1 + 0.34 * std::exp(-(5.5 - logTemperature) * (5.5 - logTemperature) / 3.0)) *
                (1.0 + 4.0 * kHeliumNumberRatio) * electrons;
        }
        if (parameters.comoving) {
            double const onePlusRedshift = 1.0 + parameters.redshift;
            loss += 5.65e-36 * electrons * (temperature - 2.73 * onePlusRedshift) *
                std::pow(onePlusRedshift, 4.0) / hydrogenDensity;
        }
        netRate = 0.0 - loss;
        return finite(netRate) ? Status::Success : Status::NonConvergence;
    }
    CUDA_EXPORT inline Status rateFromEnergy(double energy, double density,
        Parameters const& parameters, double& electrons, double& netRate) {
        double temperature;
        Status const status = convertEnergy(energy, density, parameters, electrons, temperature);
        if (status != Status::Success) {
            return status;
        }
        return rateAtTemperature(std::log10(temperature), density, parameters, electrons, netRate);
    }
}    // namespace detail

CUDA_EXPORT inline Status equilibriumAtTemperature(double temperature, double density,
    Parameters const& parameters, double& electronFraction, GasState& gas) {
    if (!detail::validState(temperature, density, electronFraction) ||
        !detail::validParameters(parameters)) {
        return Status::InvalidInput;
    }
    double electrons = electronFraction;
    GasState candidate;
    Status const status =
        detail::findAbundances(std::log10(temperature), parameters, electrons, candidate);
    if (status == Status::Success) {
        electronFraction = electrons;
        gas = candidate;
    }
    return status;
}

CUDA_EXPORT inline Status temperatureFromEnergy(double specificEnergy, double density,
    Parameters const& parameters, double& electronFraction, double& temperature) {
    if (!detail::validState(specificEnergy, density, electronFraction) ||
        !detail::validParameters(parameters)) {
        return Status::InvalidInput;
    }
    double electrons = electronFraction;
    double candidate;
    Status const status =
        detail::convertEnergy(specificEnergy, density, parameters, electrons, candidate);
    if (status == Status::Success) {
        electronFraction = electrons;
        temperature = candidate;
    }
    return status;
}

CUDA_EXPORT inline Status coolingRate(double specificEnergy, double density,
    Parameters const& parameters, double& electronFraction, double& netRatePerHydrogenSquared) {
    if (!detail::validState(specificEnergy, density, electronFraction) ||
        !detail::validParameters(parameters)) {
        return Status::InvalidInput;
    }
    double electrons = electronFraction;
    double candidate;
    Status const status =
        detail::rateFromEnergy(specificEnergy, density, parameters, electrons, candidate);
    if (status == Status::Success) {
        electronFraction = electrons;
        netRatePerHydrogenSquared = candidate;
    }
    return status;
}

CUDA_EXPORT inline Status coolingTime(double specificEnergy, double density,
    Parameters const& parameters, double& electronFraction, double& seconds) {
    double electrons = electronFraction;
    double netRate;
    Status const status = coolingRate(specificEnergy, density, parameters, electrons, netRate);
    if (status != Status::Success) {
        return status;
    }
    double candidate = 0.0;
    if (netRate < 0.0) {
        double const hydrogenDensity = kHydrogenMassFraction * density / kProtonMass;
        double const rateFactor = hydrogenDensity * hydrogenDensity / density;
        candidate = specificEnergy / (-rateFactor * netRate);
        if (!detail::finite(candidate) || candidate < 0.0) {
            return Status::NonConvergence;
        }
    }
    electronFraction = electrons;
    seconds = candidate;
    return Status::Success;
}

CUDA_EXPORT inline Status doCooling(double specificEnergy, double density, double interval,
    Parameters const& parameters, double& electronFraction, double& newSpecificEnergy) {
    if (!detail::validState(specificEnergy, density, electronFraction) ||
        !detail::validParameters(parameters) || !detail::finite(interval) || interval < 0.0) {
        return Status::InvalidInput;
    }
    double const hydrogenDensity = kHydrogenMassFraction * density / kProtonMass;
    double const rateFactor = hydrogenDensity * hydrogenDensity / density;
    if (!detail::finite(rateFactor)) {
        return Status::InvalidInput;
    }
    double electrons = electronFraction;
    double initialRate;
    Status status =
        detail::rateFromEnergy(specificEnergy, density, parameters, electrons, initialRate);
    if (status != Status::Success) {
        return status;
    }
    double lower = specificEnergy;
    double upper = specificEnergy;
    double const initialResidual = -rateFactor * initialRate * interval;
    if (initialResidual != 0.0) {
        bool const heating = initialResidual < 0.0;
        upper *= std::sqrt(1.1);
        lower /= std::sqrt(1.1);
        bool bracketed = false;
        // GADGET-4's multiplicative bracket search has no iteration guard. The
        // guard only reports failure for a nonterminating/invalid search.
        for (int iteration = 0; iteration < kMaximumIterations; ++iteration) {
            double const trial = heating ? upper : lower;
            double rate;
            status = detail::rateFromEnergy(trial, density, parameters, electrons, rate);
            if (status != Status::Success) {
                return status;
            }
            double const residual = trial - specificEnergy - rateFactor * rate * interval;
            if (!detail::finite(residual)) {
                return Status::NonConvergence;
            }
            if ((heating && residual >= 0.0) || (!heating && residual <= 0.0)) {
                bracketed = true;
                break;
            }
            if (heating) {
                upper *= 1.1;
                lower *= 1.1;
            } else {
                upper /= 1.1;
                lower /= 1.1;
            }
        }
        if (!bracketed) {
            return Status::NonConvergence;
        }
    }
    for (int iteration = 1; iteration <= kMaximumIterations; ++iteration) {
        double const energy = 0.5 * (lower + upper);
        double rate;
        status = detail::rateFromEnergy(energy, density, parameters, electrons, rate);
        if (status != Status::Success) {
            return status;
        }
        double const residual = energy - specificEnergy - rateFactor * rate * interval;
        if (!detail::finite(residual) || !(energy > 0.0)) {
            return Status::NonConvergence;
        }
        if (residual > 0.0) {
            upper = energy;
        } else {
            lower = energy;
        }
        if (std::fabs((upper - lower) / energy) <= 1.0e-6 && iteration < kMaximumIterations) {
            electronFraction = electrons;
            newSpecificEnergy = energy;
            return Status::Success;
        }
    }
    return Status::NonConvergence;
}

CUDA_EXPORT inline Result advance(Input const& input, Parameters const& parameters) {
    Result result;
    result.state = input.state;
    result.specificEnergy = input.specificEnergy;
    if (!detail::finite(input.specificEnergy) || input.specificEnergy < 0.0 ||
        !detail::finite(parameters.minimumSpecificEnergy) ||
        parameters.minimumSpecificEnergy < 0.0) {
        return result;
    }
    double electrons = input.state.electronFraction;
    double energy;
    result.status =
        doCooling(detail::maximum(parameters.minimumSpecificEnergy, input.specificEnergy),
            input.density, input.interval, parameters, electrons, energy);
    if (result.status == Status::Success) {
        result.specificEnergy = detail::maximum(energy, parameters.minimumSpecificEnergy);
        result.state.electronFraction = float(electrons);
    }
    return result;
}

}    // namespace gadget4
