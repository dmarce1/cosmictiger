// Host-only adapter for an independent SciPy Radau integration of the exact
// production RHS. FP64 appears only in this reference boundary, not the solver.
#include <cosmictiger/chemistry/primordial_solver.hpp>

namespace {
primordial::Environment environment(int caseIndex) {
    primordial::Environment result;
    result.network =
        caseIndex != 0 ? primordial::Network::Molecular9 : primordial::Network::Atomic6;
    // Proper H nuclei cm^-3, fixed during integration. The hot ionized case
    // is denser so that cooling, recombination and H2 formation all fit in
    // the common 1e12 s interval.
    result.nHydrogen = caseIndex == 4 ? 10.0f : 1.0f;
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
    float temperature = 20000.0f;
    if (caseIndex == 3) {
        // Hot molecular gas above the former 6000 K ceiling: partially ionized
        // with a trace of H2 that dissociates while the gas cools through 1e4 K.
        state.abundances[0] = primordial::SignedAbundance::fromFraction(0.5f);
        state.abundances[1] = primordial::SignedAbundance::fromFraction(1.e-2f);
        state.abundances[2] = primordial::SignedAbundance::fromFraction(1.e-4f);
        state.abundances[4] = primordial::SignedAbundance::fromFraction(2.e-3f);
        temperature = 30000.0f;
    } else if (caseIndex == 4) {
        // Fully ionized hot gas that cools from 1e5 K to 7400 K, recombines
        // and forms H2 through H- without any reset of its nonequilibrium
        // abundances. Traces keep Radau's error probes inside the domain; the
        // helium fractions leave room for the adapter's binary64 round trip.
        state.abundances[0] = primordial::SignedAbundance::fromMinority(1.e-6f, true);
        state.abundances[1] = primordial::SignedAbundance::fromFraction(1.e-3f);
        state.abundances[2] = primordial::SignedAbundance::fromMinority(2.e-3f, true);
        temperature = 1.0e5f;
    } else {
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
        temperature = caseIndex == 2 ? 2000.0f : (caseIndex == 1 ? 1000.0f : 20000.0f);
    }
    auto const status = primordial::thermalEnergyAtTemperature(
        state, environment(caseIndex), temperature, state.thermalEnergy);
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
    // The hot cases integrate through cooling, recombination and H2 formation
    // at the test default trace tolerances (1e-16 and 1e-20); allow them the
    // substeps that needs. The limit is a work bound, not an accuracy setting.
    options.maxSubsteps = 32768;
    options.maxAttempts = 65536;
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
