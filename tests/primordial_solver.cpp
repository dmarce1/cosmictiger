/* CosmicTiger - Anninos source solver regression; BSL-1.0. */
#include <algorithm>
#include <cfenv>
#include <cmath>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include <cosmictiger/chemistry/primordial_chemistry.hpp>

namespace {
using namespace primordial;

template <class Real>
StateT<Real> initialState(EnvironmentT<Real> const& environment, Real temperature,
    Real ionized = Real(0.2), Real molecular = Real(0)) {
    StateT<Real> state;
    Real const fractions[6] = {ionized, Real(0), Real(0), Real(0), molecular, Real(0)};
    EXPECT_EQ(initializeState(state, environment, fractions), Status::Success);
    EXPECT_EQ(thermalEnergyAtTemperature(state, environment, temperature, state.thermalEnergy),
        Status::Success);
    return state;
}

template <class Real>
void expectConserved(StateT<Real> const& state, EnvironmentT<Real> const& environment) {
    auto const* x = state.abundances;
    Real const tolerance = Real(16) * std::numeric_limits<Real>::epsilon();
    for (Real value : state.abundances) {
        EXPECT_TRUE(finite(value));
        EXPECT_GE(value, Real(0));
    }
    EXPECT_NEAR(x[0] + x[1] + x[6] + Real(2) * (x[7] + x[8]), Real(1), tolerance);
    EXPECT_NEAR(x[2] + x[3] + x[4], environment.heliumToHydrogenNumberRatio, tolerance);
    EXPECT_NEAR(x[1] + x[3] + Real(2) * x[4] + x[8], x[5] + x[6], tolerance);
    EXPECT_EQ(validateState(state, environment), Status::Success);
}

template <class Real>
void checkPhotoSweep() {
    EnvironmentT<Real> environment;
    environment.cmbTemperature = Real(0);
    environment.radiation.hydrogenIonization = Real(1.e-12);
    environment.radiation.heliumIonization = Real(2.e-12);
    environment.radiation.heliumPlusIonization = Real(3.e-12);
    auto state = initialState(environment, Real(10000));
    Real const interval = Real(1.e9);
    auto const result = advance(state, environment, interval);
    ASSERT_EQ(result.status, Status::Success);
    ASSERT_EQ(result.acceptedSubsteps, 2U);    // One thermal and one chemical substep.
    EXPECT_EQ(result.acceptedTime, interval);
    Real const h =
        state.abundances[0] / (Real(1) + interval * environment.radiation.hydrogenIonization);
    Real const hp = state.abundances[1] + interval * environment.radiation.hydrogenIonization * h;
    Real const he =
        state.abundances[2] / (Real(1) + interval * environment.radiation.heliumIonization);
    Real const hep = interval * environment.radiation.heliumIonization * he /
        (Real(1) + interval * environment.radiation.heliumPlusIonization);
    Real const hepp = interval * environment.radiation.heliumPlusIonization * hep;
    Real const tolerance = Real(8) * std::numeric_limits<Real>::epsilon();
    EXPECT_NEAR(result.state.abundances[1], hp, tolerance * hp);
    EXPECT_NEAR(result.state.abundances[3], hep, tolerance * hep);
    EXPECT_NEAR(result.state.abundances[4], hepp, tolerance * hepp);
    EXPECT_EQ(result.state.thermalEnergy, state.thermalEnergy);
    expectConserved(result.state, environment);
}

template <class Real>
void checkIntermediateEquilibria() {
    EnvironmentT<Real> environment;
    environment.network = Network::Molecular9;
    environment.nHydrogen = Real(1);
    auto state = initialState(environment, Real(1000));
    auto c = composition(state, environment);
    c.molecularHydrogenPlus = Real(1.e-4);
    RateCoefficientsT<Real> rates;
    rates.collision[6] = Real(2.e-8);      // H + e -> H-.
    rates.collision[7] = Real(3.e-8);      // H- + H -> H2 + e.
    rates.collision[8] = Real(5.e-9);      // H + H+ -> H2+.
    rates.collision[9] = Real(7.e-8);      // H2+ + H -> H2 + H+.
    rates.collision[16] = Real(11.e-8);    // H- + H+ -> H2+ + e.
    rates.collision[18] = Real(13.e-8);    // H- + H2+ -> H2 + H.
    rates.collision[21] = Real(17.e-8);    // Added H- + H2+ -> 3H channel.
    Real const hminus = rates.collision[6] * c.hydrogen * c.electrons /
        (rates.collision[7] * c.hydrogen + rates.collision[16] * c.hydrogenPlus);
    Real const h2plus = (rates.collision[8] * c.hydrogen * c.hydrogenPlus +
                            rates.collision[16] * hminus * c.hydrogenPlus) /
        (rates.collision[9] * c.hydrogen + (rates.collision[18] + rates.collision[21]) * hminus);
    // A zero-duration sweep isolates paper equations (32)-(33). H- ignores
    // collisions between the trace intermediates; H2+ sees the new H-.
    ASSERT_EQ(detail::anninosChemicalStep(c, environment, rates, Real(0)), Status::Success);
    Real const tolerance = Real(8) * std::numeric_limits<Real>::epsilon();
    EXPECT_NEAR(c.hydrogenMinus, hminus, tolerance * hminus);
    EXPECT_NEAR(c.molecularHydrogenPlus, h2plus, tolerance * h2plus);
    ASSERT_EQ(detail::anninosConserve(c, environment), Status::Success);
    detail::anninosStore(c, state);
    expectConserved(state, environment);
}

template <class Real>
void checkIndependentElectrons() {
    EnvironmentT<Real> environment;
    environment.nHydrogen = Real(1);
    auto state = initialState(environment, Real(10000));
    auto c = composition(state, environment);
    RateCoefficientsT<Real> rates;
    rates.collision[1] = Real(1);    // Isolated H+ + e -> H recombination.
    Real h = c.hydrogen;
    Real hp = c.hydrogenPlus;
    Real electrons = c.electrons;
    Real const step = Real(0.5);
    Real const tolerance = Real(8) * std::numeric_limits<Real>::epsilon();
    for (int substep = 0; substep < 2; ++substep) {
        // Paper order H, H+, ..., e: electron destruction sees the new H+,
        // while H+ destruction sees the independently advanced old electrons.
        h += step * hp * electrons;
        hp /= Real(1) + step * electrons;
        electrons /= Real(1) + step * hp;
        ASSERT_EQ(detail::anninosChemicalStep(c, environment, rates, step), Status::Success);
        EXPECT_NEAR(c.hydrogen, h, tolerance * h);
        EXPECT_NEAR(c.hydrogenPlus, hp, tolerance * hp);
        EXPECT_NEAR(c.electrons, electrons, tolerance * electrons);
        EXPECT_GT(c.electrons - c.hydrogenPlus, Real(0.001));
    }
    // Conservation is a separate final stage, not a per-subcycle projection.
    ASSERT_EQ(detail::anninosConserve(c, environment), Status::Success);
    EXPECT_EQ(c.electrons, c.hydrogenPlus);
    detail::anninosStore(c, state);
    expectConserved(state, environment);
}

template <class Real>
void checkTraceStorage() {
    EnvironmentT<Real> environment;
    environment.cmbTemperature = Real(0);
    environment.radiation.hydrogenIonization = Real(1.e-12);
    auto state = initialState(environment, Real(10000), Real(1));
    state.abundances[0] = Real(1.e-30);
    state.abundances[2] = Real(1.e-30);
    state.abundances[3] = Real(1.e-25);
    state.abundances[4] = environment.heliumToHydrogenNumberRatio;
    state.abundances[5] = state.abundances[1] + state.abundances[3] + Real(2) * state.abundances[4];
    auto const result = advance(state, environment, Real(1.e9));
    ASSERT_EQ(result.status, Status::Success);
    Real const expected = state.abundances[0] / (Real(1) + Real(1.e-3));
    EXPECT_NEAR(result.state.abundances[0], expected,
        Real(8) * std::numeric_limits<Real>::epsilon() * expected);
    EXPECT_GT(result.state.abundances[0], Real(0));
    EXPECT_EQ(result.state.abundances[2], state.abundances[2]);
    EXPECT_EQ(result.state.abundances[3], state.abundances[3]);
    expectConserved(result.state, environment);
    environment.radiation = {};
    state = initialState(environment, Real(100), Real(0));
    SolverOptionsT<Real> options;
    options.electronFractionFloor = Real(0.01);
    auto const neutral = advance(state, environment, Real(1.e12), options);
    ASSERT_EQ(neutral.status, Status::Success);
    EXPECT_EQ(neutral.state.abundances[1], Real(0));
    EXPECT_EQ(neutral.state.abundances[5], Real(0));
}

template <class Real>
void checkPhotoConvergence() {
    EnvironmentT<Real> environment;
    environment.cmbTemperature = Real(0);
    environment.radiation.hydrogenIonization = Real(1.e-12);
    auto const state = initialState(environment, Real(10000));
    Real const interval = Real(1.e12);
    Real const exact = Real(1) - state.abundances[0] * std::exp(Real(-1));
    Real previousError = Real(1);
    for (Real limit : {Real(0.1), Real(0.05), Real(0.025), Real(0.0125), Real(0.00625)}) {
        SolverOptionsT<Real> options;
        options.fractionalChangeLimit = limit;
        auto const result = advance(state, environment, interval, options);
        ASSERT_EQ(result.status, Status::Success);
        Real const error = std::abs(result.state.abundances[1] - exact);
        EXPECT_LT(error, previousError * Real(0.65));
        previousError = error;
        expectConserved(result.state, environment);
    }
    EXPECT_LT(previousError, Real(0.002));
}

template <class Real>
void checkThermal() {
    EnvironmentT<Real> environment;
    environment.cmbTemperature = Real(0);
    environment.radiation.heatingPerHydrogen = Real(1.e-12);
    auto const state = initialState(environment, Real(10000));
    Real const interval = Real(1.e12);
    auto const heated = advance(state, environment, interval);
    ASSERT_EQ(heated.status, Status::Success);
    Real const expected = state.thermalEnergy +
        state.abundances[0] * environment.radiation.heatingPerHydrogen * interval;
    EXPECT_NEAR(heated.state.thermalEnergy, expected,
        Real(16) * std::numeric_limits<Real>::epsilon() * expected);
    environment.radiation = {};
    environment.cmbTemperature = Real(100);
    auto const ionized = initialState(environment, Real(10000), Real(1));
    Real const capacity =
        Real(1.5) * kBoltzmannEvT<Real> * composition(ionized, environment).particles;
    Real const decay = Real(6.35264354e-26) * Real(1.e8) / capacity;
    Real const duration = Real(1) / decay;
    Real const exactTemperature = Real(100) + Real(9900) * std::exp(Real(-1));
    Real previousError = Real(10000);
    for (Real limit : {Real(0.1), Real(0.05), Real(0.025), Real(0.0125), Real(0.00625)}) {
        SolverOptionsT<Real> options;
        options.fractionalChangeLimit = limit;
        auto const result = advance(ionized, environment, duration, options);
        ASSERT_EQ(result.status, Status::Success);
        EosT<Real> eos;
        ASSERT_EQ(evaluateEos(result.state, environment, eos), Status::Success);
        Real const error = std::abs(eos.temperature - exactTemperature);
        EXPECT_LT(error, previousError * Real(0.65));
        previousError = error;
    }
    EXPECT_LT(previousError, Real(25));
}

TEST(PrimordialSolver, PaperSequentialPhotoFloatAndDouble) {
    checkPhotoSweep<float>();
    checkPhotoSweep<double>();
}
TEST(PrimordialSolver, PaperIntermediateEquilibriumFloatAndDouble) {
    checkIntermediateEquilibria<float>();
    checkIntermediateEquilibria<double>();
}
TEST(PrimordialSolver, IndependentElectronsBetweenSubcycles) {
    checkIndependentElectrons<float>();
    checkIndependentElectrons<double>();
}
TEST(PrimordialSolver, DirectTraceStorageFloatAndDouble) {
    checkTraceStorage<float>();
    checkTraceStorage<double>();
}
TEST(PrimordialSolver, PhotoionizationFirstOrderConvergence) {
    checkPhotoConvergence<float>();
    checkPhotoConvergence<double>();
}
TEST(PrimordialSolver, ExplicitHeatingAndComptonConvergence) {
    checkThermal<float>();
    checkThermal<double>();
}
TEST(PrimordialSolver, PhysicalNetworksAndThermalDomain) {
    for (Network network : {Network::Atomic6, Network::Molecular9}) {
        for (float temperature : {10.0f, 100.0f, 1000.0f, 30000.0f, 1.e5f}) {
            Environment environment;
            environment.network = network;
            environment.nHydrogen = 0.1f;
            environment.cmbTemperature = 30.0f;
            auto const state = initialState(
                environment, temperature, 1.e-3f, network == Network::Molecular9 ? 2.e-3f : 0.0f);
            auto const result = advance(state, environment, 1.e7f);
            ASSERT_EQ(result.status, Status::Success) << temperature << ' ' << int(network);
            expectConserved(result.state, environment);
        }
    }
}
TEST(PrimordialSolver, InvalidInputAndWorkLimitAreTransactional) {
    Environment environment;
    environment.cmbTemperature = 0;
    environment.radiation.hydrogenIonization = 1.e-12f;
    auto const state = initialState(environment, 10000.0f);
    auto expectUnchanged = [&](StepResult const& result) {
        EXPECT_NE(result.status, Status::Success);
        EXPECT_EQ(result.acceptedTime, 0.0f);
        EXPECT_EQ(result.state.thermalEnergy, state.thermalEnergy);
        for (int species = 0; species < kAbundanceCount; ++species) {
            EXPECT_EQ(result.state.abundances[species], state.abundances[species]);
        }
    };
    SolverOptions options;
    options.maxSubsteps = 1;
    auto const limited = advance(state, environment, 1.e12f, options);
    EXPECT_EQ(limited.status, Status::StepLimit);
    expectUnchanged(limited);
    for (float invalid :
        {-1.0f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
        expectUnchanged(advance(state, environment, invalid));
    }
    options = {};
    options.fractionalChangeLimit = 0.2f;
    expectUnchanged(advance(state, environment, 1.0f, options));
    auto invalidState = state;
    invalidState.abundances[7] = 1.e-6f;
    EXPECT_NE(advance(invalidState, environment, 0.0f).status, Status::Success);
    auto const unchanged = advance(state, environment, 0.0f);
    EXPECT_EQ(unchanged.status, Status::Success);
    EXPECT_EQ(unchanged.acceptedSubsteps, 0U);
}
TEST(PrimordialSolver, CpuBatchWorkerDeterminismAndFloatingEnvironment) {
    std::vector<BatchInput> inputs(17);
    for (size_t index = 0; index < inputs.size(); ++index) {
        auto& input = inputs[index];
        input.environment.cmbTemperature = 0;
        input.environment.radiation.hydrogenIonization = 1.e-12f;
        input.state = initialState(input.environment, 10000.0f, 0.1f + float(index) * 0.01f);
        input.interval = 1.e11f;
    }
    std::vector<StepResult> serial(inputs.size()), parallel(inputs.size());
    ASSERT_TRUE(integrateBatchCpu(inputs.data(), serial.data(), inputs.size(), {}, 1));
    int const original = std::fegetround();
    ASSERT_EQ(std::fesetround(FE_DOWNWARD), 0);
    bool const dispatched = integrateBatchCpu(inputs.data(), parallel.data(), inputs.size(), {}, 4);
    int const restored = std::fegetround();
    std::fesetround(original);
    ASSERT_TRUE(dispatched);
    EXPECT_EQ(restored, FE_DOWNWARD);
    for (size_t index = 0; index < inputs.size(); ++index) {
        ASSERT_EQ(serial[index].status, Status::Success);
        EXPECT_EQ(parallel[index].status, serial[index].status);
        EXPECT_EQ(parallel[index].acceptedSubsteps, serial[index].acceptedSubsteps);
        EXPECT_EQ(parallel[index].state.thermalEnergy, serial[index].state.thermalEnergy);
        for (int species = 0; species < kAbundanceCount; ++species) {
            EXPECT_EQ(
                parallel[index].state.abundances[species], serial[index].state.abundances[species]);
        }
    }
    EXPECT_TRUE(integrateBatchCpu(nullptr, nullptr, 0));
    EXPECT_FALSE(integrateBatchCpu(nullptr, parallel.data(), 1));
}
}    // namespace
