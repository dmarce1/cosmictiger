// Host-only adapter for an independent SciPy Radau integration of the exact
// production RHS. FP64 appears only in this reference boundary, not the solver.
#include <cosmictiger/primordial_solver.hpp>

namespace {
primordial::Environment environment(int caseIndex) {
    primordial::Environment result;
    result.network =
        caseIndex != 0 ? primordial::Network::Molecular9 : primordial::Network::Atomic6;
    result.nHydrogen = 1.0f;    // Proper H nuclei cm^-3, fixed during integration.
    result.cmbTemperature = 30.0f;
    return result;
}
primordial::WorkingState unpack(double const* values) {
    primordial::WorkingState state;
    for (int i = 0; i < primordial::kAbundanceCount; ++i) {
        state.abundances[i] = FloatFloat(values[i]);
    }
    state.thermalEnergy = FloatFloat(values[primordial::kAbundanceCount]);
    return state;
}
}    // namespace
extern "C" int primordialReferenceInitial(int caseIndex, double* values) {
    primordial::State state;
    state.abundances[0] = primordial::SignedAbundance::fromFraction(1.e-4f);
    if (caseIndex != 0) {
        // Case 2 has substantial H2 cooling and rotational/vibrational energy;
        // case 1 isolates the formation/relaxation of trace intermediates.
        state.abundances[4] =
            primordial::SignedAbundance::fromFraction(caseIndex == 2 ? 0.2f : 2.e-6f);
    }
    if (caseIndex != 1) {
        // Positive trace helium ions keep Radau's internal error probes inside
        // the physical domain; both integrators receive this same initial state.
        state.abundances[1] = primordial::SignedAbundance::fromFraction(1.e-8f);
        state.abundances[2] = primordial::SignedAbundance::fromFraction(1.e-20f);
    }
    auto const status = primordial::thermalEnergyAtTemperature(state, environment(caseIndex),
        caseIndex == 2 ? 2000.0f : (caseIndex == 1 ? 1000.0f : 20000.0f), state.thermalEnergy);
    for (int i = 0; i < primordial::kAbundanceCount; ++i) {
        values[i] = state.abundances[i].value().toDouble();
    }
    values[primordial::kAbundanceCount] = state.thermalEnergy;
    return static_cast<int>(status);
}
extern "C" int primordialReferenceRhs(int caseIndex, double const* values, double* derivative) {
    primordial::Derivative rates;
    auto const status = primordial::evaluateRhs(unpack(values), environment(caseIndex), rates);
    for (int i = 0; i < primordial::kAbundanceCount; ++i) {
        derivative[i] = rates.abundances[i].toDouble();
    }
    derivative[primordial::kAbundanceCount] = rates.thermalEnergy.toDouble();
    return static_cast<int>(status);
}
extern "C" int primordialReferenceAdvance(
    int caseIndex, float tolerance, double* values, float* acceptedTime, unsigned* substeps) {
    primordial::State initial;
    auto status = primordial::composeState(unpack(values), initial);
    if (status != primordial::Status::Success) {
        return static_cast<int>(status);
    }
    primordial::SolverOptions options;
    options.relativeTolerance = tolerance;
    auto const result = primordial::advance(initial, environment(caseIndex), 1.e12f, options);
    for (int i = 0; i < primordial::kAbundanceCount; ++i) {
        values[i] = result.state.abundances[i].value().toDouble();
    }
    values[primordial::kAbundanceCount] = result.state.thermalEnergy;
    *acceptedTime = result.acceptedTime;
    *substeps = result.acceptedSubsteps;
    return static_cast<int>(result.status);
}

extern "C" int primordialReferenceTemperature(
    int caseIndex, double const* values, float* temperature) {
    primordial::Eos eos;
    auto const status = primordial::evaluateEos(unpack(values), environment(caseIndex), eos);
    *temperature = eos.temperature;
    return static_cast<int>(status);
}
