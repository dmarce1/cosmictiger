/* CosmicTiger - SH03 host/device parity and source invariants; BSL-1.0. */
#include <cmath>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cosmictiger/chemistry/primordial_float_environment.hpp>
#include <cosmictiger/sph/star_formation_cuda.hpp>

namespace {
using star_formation_backend::Input;
using star_formation_backend::Operation;
using star_formation_backend::Output;
using star_formation_backend::Parameters;

Parameters parameters() {
    Parameters p;
    p.cooling.hydrogenMassFraction = 0.76;
    auto const settings = springel_and_hernquist::defaultParameters(p.cooling);
    EXPECT_TRUE(springel_and_hernquist::makeModel(settings, p.cooling, p.model));
    p.environment.network = primordial::Network::Atomic6;
    p.environment.heliumToHydrogenNumberRatio = float(0.24 / (4.0 * 0.76));
    p.environment.cmbTemperature = 2.73f;
    p.densityScale = 1.0;
    p.energyScale = 1.0;
    p.hydrogenMass = 1.67262192369e-24;
    p.scaleFactor = 1.0;
    p.timeSeconds = 1.e14;
    p.lengthCm = 1.e20;
    p.gasMass = 0.16;
    p.baseTimestep = 0.001;
    p.explicitTimestep = 0.001;
    p.minimumRung = 0;
    p.uniformRung = -1;
    p.metalYield = 0.02;
    p.windMassLoading = 2.0;
    p.windEnergyFraction = 0.25;
    p.windDecouplingSeconds = 50.e6 * 31557600.0;
    p.windRecoupleDensityFraction = 0.1;
    p.seed = 42;
    p.opening = true;
    p.winds = true;
    return p;
}
Input input(Parameters const& p, double densityRatio, uint64_t identity = 1) {
    Input result;
    result.state.id = identity;
    result.energy = 1.e12f;
    result.density = float(densityRatio * p.model.thresholdDensity);
    float const fractions[] = {2.e-4f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    EXPECT_EQ(primordial::initializeState(result.chemistry, p.environment, fractions),
        primordial::Status::Success);
    result.rung = 3;
    result.pressureRatio = float(2.0 / 3.0);
    result.velocity[0] = 0.01f;
    result.velocity[1] = -0.02f;
    result.velocity[2] = 0.03f;
    return result;
}
void close(double actual, double expected, double relative = 3.e-5) {
    EXPECT_NEAR(actual, expected, relative * std::fmax(std::fabs(expected), 1.e-30));
}
void compare(Output const& actual, Output const& expected) {
    ASSERT_EQ(actual.status, expected.status);
    EXPECT_EQ(actual.project, expected.project);
    EXPECT_EQ(actual.convert, expected.convert);
    EXPECT_EQ(actual.rung, expected.rung);
    EXPECT_EQ(actual.state.id, expected.state.id);
    EXPECT_EQ(actual.state.eventCounter, expected.state.eventCounter);
    close(actual.state.formationTime, expected.state.formationTime);
    close(actual.state.windTimeRemaining, expected.state.windTimeRemaining);
    close(actual.state.metallicity, expected.state.metallicity);
    close(actual.state.sourceHalfInterval, expected.state.sourceHalfInterval);
    close(actual.state.pendingHeating, expected.state.pendingHeating, 1.e-4);
    close(actual.state.formationOpticalDepth, expected.state.formationOpticalDepth);
    close(actual.state.elapsedPhysicalTime, expected.state.elapsedPhysicalTime);
    close(actual.chemistryInterval, expected.chemistryInterval);
    close(actual.value, expected.value, 1.e-4);
    close(actual.energy, expected.energy);
    close(actual.pressureRatio, expected.pressureRatio);
    close(actual.signalIndex, expected.signalIndex);
    close(actual.chemistry.thermalEnergy, expected.chemistry.thermalEnergy);
    for (int species = 0; species < primordial::kAbundanceCount; ++species) {
        close(actual.chemistry.abundances[species], expected.chemistry.abundances[species]);
    }
    for (int dim = 0; dim < 3; ++dim) {
        close(actual.velocity[dim], expected.velocity[dim]);
    }
}
std::vector<Output> parity(Operation operation, std::vector<Input> const& inputs,
    Parameters const& p, std::vector<Output> outputs = {}) {
    primordial::StrictFloatEnvironment floatingEnvironment;
    EXPECT_TRUE(floatingEnvironment.valid());
    if (outputs.empty()) {
        outputs.resize(inputs.size());
    }
    auto expected = outputs;
    for (size_t index = 0; index < inputs.size(); ++index) {
        expected[index] =
            star_formation_backend::evaluate(operation, inputs[index], outputs[index], p);
        EXPECT_EQ(expected[index].status, 0);
    }
    star_formation_backend::runCuda(operation, inputs.data(), outputs.data(), inputs.size(), p);
    for (size_t index = 0; index < inputs.size(); ++index) {
        compare(outputs[index], expected[index]);
    }
    return outputs;
}
bool haveDevice() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}
}    // namespace

TEST(StarFormationCuda, EosTimestepAndPressureAcrossBranches) {
    if (!haveDevice()) {
        GTEST_SKIP() << "CUDA device unavailable";
    }
    auto p = parameters();
    std::vector<Input> inputs{input(p, 0.01), input(p, 0.999), input(p, 1.001), input(p, 100.0)};
    auto wind = input(p, 10.0, 5);
    wind.state.windTimeRemaining = 5.e12;
    inputs.push_back(wind);
    auto inactive = input(p, 10.0, 6);
    inactive.rung = 0;
    inputs.push_back(inactive);
    auto const eos = parity(Operation::eos, inputs, p);
    close(eos[0].signalIndex, 5.0 / 3.0);
    close(eos[4].signalIndex, 5.0 / 3.0);
    p.minimumRung = 2;
    auto const timestep = parity(Operation::timestep, inputs, p);
    EXPECT_DOUBLE_EQ(timestep.back().value, std::numeric_limits<double>::max());
    EXPECT_DOUBLE_EQ(
        timestep[4].value, wind.state.windTimeRemaining / p.scaleFactor / p.timeSeconds);
    parity(Operation::pressure, inputs, p);
    p.scaleFactor = 0.2;
    p.energyScale = 3.0;
    p.densityScale = 2.e-3;
    parity(Operation::eos, inputs, p);
}
TEST(StarFormationCuda, SourcePreparationAndOpeningHeating) {
    if (!haveDevice()) {
        GTEST_SKIP() << "CUDA device unavailable";
    }
    auto p = parameters();
    std::vector<Input> inputs{input(p, 0.01), input(p, 2.0), input(p, 5.0)};
    inputs[2].state.windTimeRemaining = 0.25 * p.explicitTimestep * p.timeSeconds;
    auto outputs = parity(Operation::prepare, inputs, p);
    EXPECT_FALSE(outputs[0].project);
    EXPECT_TRUE(outputs[1].project);
    EXPECT_TRUE(outputs[2].project);
    EXPECT_EQ(outputs[1].chemistryInterval, 0.0);
    EXPECT_GT(outputs[2].chemistryInterval, 0.0);
    // Substitute an ordinary-gas chemistry result to test the SH03 boundary.
    outputs[0].energy *= 0.9f;
    auto const finished = parity(Operation::finish, inputs, p, outputs);
    for (size_t i = 0; i < inputs.size(); ++i) {
        EXPECT_FALSE(finished[i].convert);
        EXPECT_EQ(finished[i].state.eventCounter, inputs[i].state.eventCounter);
        close(finished[i].state.pendingHeating,
            -p.gasMass * (double(finished[i].energy) - inputs[i].energy) / p.scaleFactor);
        EXPECT_EQ(finished[i].state.sourceHalfInterval, 0.5 * p.explicitTimestep);
    }
}
TEST(StarFormationCuda, ClosingConversionWindAndLedger) {
    if (!haveDevice()) {
        GTEST_SKIP() << "CUDA device unavailable";
    }
    auto p = parameters();
    p.opening = false;
    p.windMassLoading = 100.0;
    uint64_t windIdentity = 1;
    while (!(star_formation_backend::detail::uniform(windIdentity, 0, 0, p.seed) > 0.1 &&
        star_formation_backend::detail::uniform(windIdentity, 0, 1, p.seed) < 0.5)) {
        ++windIdentity;
    }
    std::vector<Input> inputs{
        input(p, 0.01, 600), input(p, 0.01, windIdentity), input(p, 0.01, 601)};
    for (auto& item : inputs) {
        item.state.sourceHalfInterval = 0.5 * p.explicitTimestep;
        item.state.pendingHeating = 1.e8;
    }
    inputs[0].state.formationOpticalDepth = 100.0;
    inputs[1].state.formationOpticalDepth = 0.01;
    auto outputs = parity(Operation::prepare, inputs, p);
    outputs = parity(Operation::finish, inputs, p, outputs);
    EXPECT_TRUE(outputs[0].convert);
    EXPECT_EQ(outputs[0].energy, 0.0f);
    EXPECT_FALSE(outputs[1].convert);
    EXPECT_EQ(outputs[1].state.windTimeRemaining, p.windDecouplingSeconds);
    EXPECT_FALSE(outputs[2].convert);
    for (size_t i = 0; i < inputs.size(); ++i) {
        EXPECT_EQ(outputs[i].state.eventCounter, 1);
        EXPECT_EQ(outputs[i].state.sourceHalfInterval, 0.0);
        EXPECT_EQ(outputs[i].state.pendingHeating, 0.0);
        EXPECT_EQ(outputs[i].state.formationOpticalDepth, 0.0);
        double kinetic = 0.0;
        for (int dim = 0; dim < 3; ++dim) {
            kinetic += 0.5 *
                (double(outputs[i].velocity[dim]) * outputs[i].velocity[dim] -
                    double(inputs[i].velocity[dim]) * inputs[i].velocity[dim]);
        }
        close(outputs[i].value,
            inputs[i].state.pendingHeating -
                p.gasMass * (double(outputs[i].energy) - inputs[i].energy + kinetic) /
                    p.scaleFactor);
    }
}
TEST(StarFormationCuda, RejectsInvalidSourceBeforeCommit) {
    if (!haveDevice()) {
        GTEST_SKIP() << "CUDA device unavailable";
    }
    auto const p = parameters();
    std::vector<Input> inputs{input(p, 0.01), input(p, 2.0)};
    inputs[1].state.id = 0;
    std::vector<Output> outputs(inputs.size());
    EXPECT_THROW(star_formation_backend::runCuda(
                     Operation::prepare, inputs.data(), outputs.data(), inputs.size(), p),
        std::runtime_error);
    EXPECT_EQ(inputs[0].state.sourceHalfInterval, 0.0);
    EXPECT_EQ(inputs[0].state.pendingHeating, 0.0);
    EXPECT_NO_THROW(star_formation_backend::runCuda(Operation::prepare, nullptr, nullptr, 0, p));
}

TEST(StarFormationCuda, MolecularOrdinaryGasEos) {
    if (!haveDevice()) {
        GTEST_SKIP() << "CUDA device unavailable";
    }
    auto p = parameters();
    p.environment.network = primordial::Network::Molecular9;
    std::vector<Input> inputs{input(p, 0.01), input(p, 10.0)};
    inputs[1].state.windTimeRemaining = 1.e10;
    for (auto& item : inputs) {
        item.energy = 1.e11f;
        float const fractions[] = {0.01f, 0.0f, 0.0f, 1.e-6f, 0.4f, 1.e-3f};
        ASSERT_EQ(primordial::initializeState(item.chemistry, p.environment, fractions),
            primordial::Status::Success);
    }
    auto const outputs = parity(Operation::eos, inputs, p);
    for (auto const& item : outputs) {
        EXPECT_GT(item.signalIndex, 1.0f);
        EXPECT_LT(item.signalIndex, 5.0f / 3.0f);
        EXPECT_GT(item.pressureRatio, 0.0f);
        EXPECT_LT(item.pressureRatio, 2.0f / 3.0f);
    }
}
TEST(StarFormationCuda, NewStarRungAndOpeningKick) {
    if (!haveDevice()) {
        GTEST_SKIP() << "CUDA device unavailable";
    }
    Parameters p;
    p.baseTimestep = 1.0;
    p.timestepBase = 1.0f;
    p.timestepCap = 0.25f;
    p.effectiveRadius = 0.02f;
    p.timestepFactor = 0.05f;
    p.minimumRung = 2;
    p.rungLimit = 64;
    p.kickSign = 1.0f;
    std::vector<Input> inputs(3);
    for (size_t index = 0; index < inputs.size(); ++index) {
        inputs[index].rung = 3 + int(index);
        for (int dim = 0; dim < 3; ++dim) {
            inputs[index].gravity[dim] = float(index) * float(dim - 1);
            inputs[index].velocity[dim] = 0.125f * float(dim + 1);
        }
    }
    for (float sign : {1.0f, -1.0f}) {
        p.kickSign = sign;
        auto const outputs = parity(Operation::openNewStar, inputs, p);
        for (size_t index = 0; index < inputs.size(); ++index) {
            EXPECT_GE(outputs[index].rung, p.minimumRung);
            EXPECT_GE(outputs[index].rung, inputs[index].rung - 1);
            double const half = sign * std::ldexp(0.5 * p.baseTimestep, -outputs[index].rung);
            for (int dim = 0; dim < 3; ++dim) {
                EXPECT_FLOAT_EQ(outputs[index].velocity[dim],
                    float(std::fma(double(inputs[index].gravity[dim]), half,
                        double(inputs[index].velocity[dim]))));
            }
        }
    }
    p.timestepCap = 0.0f;
    std::vector<Output> outputs(inputs.size());
    EXPECT_THROW(star_formation_backend::runCuda(
                     Operation::openNewStar, inputs.data(), outputs.data(), inputs.size(), p),
        std::runtime_error);
}
