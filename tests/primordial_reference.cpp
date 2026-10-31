/* CosmicTiger - double scalar boundary for analytic references; BSL-1.0. */
#include <cosmictiger/chemistry/primordial_solver.hpp>

namespace {
using namespace primordial;
EnvironmentT<double> environment(int caseIndex) {
    EnvironmentT<double> result;
    result.cmbTemperature = caseIndex == 1 ? 100.0 : 0.0;
    result.network = caseIndex == 2 ? Network::Molecular9 : Network::Atomic6;
    if (caseIndex == 0 || caseIndex == 3) {
        result.radiation.hydrogenIonization = 1.e-12;
    }
    if (caseIndex == 2) {
        result.radiation.molecularHydrogenDissociation = 1.e-12;
    }
    if (caseIndex == 3) {
        result.radiation.heatingPerHydrogen = 2.e-12;
    }
    return result;
}
StateT<double> unpack(double const* values) {
    StateT<double> state;
    for (int species = 0; species < kAbundanceCount; ++species) {
        state.abundances[species] = values[species];
    }
    state.thermalEnergy = values[kAbundanceCount];
    return state;
}
void pack(StateT<double> const& state, double* values) {
    for (int species = 0; species < kAbundanceCount; ++species) {
        values[species] = state.abundances[species];
    }
    values[kAbundanceCount] = state.thermalEnergy;
}
}    // namespace
extern "C" int primordialReferenceInitial(int caseIndex, double* values) {
    StateT<double> state;
    auto const env = environment(caseIndex);
    double const fractions[6] = {
        caseIndex == 1 ? 1.0 : 0.2, 0.0, 0.0, 0.0, caseIndex == 2 ? 0.2 : 0.0, 0.0};
    auto status = initializeState(state, env, fractions);
    if (status == Status::Success) {
        status = thermalEnergyAtTemperature(state, env, 10000.0, state.thermalEnergy);
    }
    pack(state, values);
    return static_cast<int>(status);
}
extern "C" int primordialReferenceAdvance(int caseIndex, double interval, double limit,
    double* values, double* acceptedTime, unsigned* substeps) {
    SolverOptionsT<double> options;
    options.fractionalChangeLimit = limit;
    auto const result = advance(unpack(values), environment(caseIndex), interval, options);
    pack(result.state, values);
    *acceptedTime = result.acceptedTime;
    *substeps = result.acceptedSubsteps;
    return static_cast<int>(result.status);
}
extern "C" int primordialReferenceTemperature(
    int caseIndex, double const* values, double* temperature) {
    EosT<double> eos;
    auto const status = evaluateEos(unpack(values), environment(caseIndex), eos);
    *temperature = eos.temperature;
    return static_cast<int>(status);
}
