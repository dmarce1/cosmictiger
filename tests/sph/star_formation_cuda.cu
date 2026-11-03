/* CosmicTiger - GADGET-4 SF host/device parity; BSL-1.0. */
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cosmictiger/stars/star_formation_cuda.hpp>

namespace {
namespace sf = star_formation_backend;

cudaError_t deviceStatus(int& count) {
    return cudaGetDeviceCount(&count);
}
sf::Parameters parameters() {
    sf::Parameters p;
    p.model = {springel_and_hernquist::defaultParameters(), 1.e-24, 0.0};
    p.cooling = gadget4::makeParameters(1.0, false, 0.0);
    p.densityScale = p.energyScale = p.scaleFactor = 1.0;
    p.timeSeconds = 1.e14;
    p.gasMass = 0.16;
    p.baseTimestep = p.explicitTimestep = 0.001;
    p.seed = 42;
    return p;
}
sf::Input input(sf::Parameters const& p, double factor, uint64_t identity) {
    sf::Input result;
    result.state.id = identity;
    result.energy = 1.e12f;
    result.density = float(factor * p.model.thresholdDensity);
    result.rung = 3;
    result.pressureRatio = float(gadget4::kGammaMinusOne);
    result.velocity[0] = 0.01f;
    result.velocity[1] = -0.02f;
    result.velocity[2] = 0.03f;
    return result;
}
void close(double actual, double expected, double relative = 3.e-5) {
    EXPECT_NEAR(actual, expected, relative * std::fmax(std::fabs(expected), 1.e-30));
}
void compare(sf::Output const& actual, sf::Output const& expected) {
    ASSERT_EQ(actual.status, expected.status);
    EXPECT_EQ(actual.starForming, expected.starForming);
    EXPECT_EQ(actual.convert, expected.convert);
    EXPECT_EQ(actual.rung, expected.rung);
    EXPECT_EQ(actual.state.id, expected.state.id);
    EXPECT_EQ(actual.state.eventCounter, expected.state.eventCounter);
    close(actual.state.formationTime, expected.state.formationTime);
    close(actual.state.specificStarFormationRate, expected.state.specificStarFormationRate);
    close(actual.state.elapsedPhysicalTime, expected.state.elapsedPhysicalTime);
    close(actual.physicalInterval, expected.physicalInterval);
    close(actual.value, expected.value, 1.e-4);
    close(actual.energy, expected.energy);
    close(actual.pressureRatio, expected.pressureRatio);
    close(actual.signalIndex, expected.signalIndex);
    close(actual.electronFraction, expected.electronFraction);
    close(actual.coldFraction, expected.coldFraction);
    close(actual.hotEnergy, expected.hotEnergy);
    close(actual.effectiveEnergy, expected.effectiveEnergy);
    for (int dim = 0; dim < 3; ++dim) {
        close(actual.velocity[dim], expected.velocity[dim]);
    }
}
void parity(
    sf::Operation operation, std::vector<sf::Input> const& inputs, sf::Parameters const& p) {
    std::vector<sf::Output> outputs(inputs.size());
    sf::runCuda(operation, inputs.data(), outputs.data(), inputs.size(), p);
    for (size_t index = 0; index < inputs.size(); ++index) {
        auto const expected = sf::evaluate(operation, inputs[index], {}, p);
        ASSERT_EQ(expected.status, 0);
        compare(outputs[index], expected);
    }
}
}    // namespace

TEST(StarFormationCuda, OrdinaryDenseRelaxationAndConversion) {
    int deviceCount = 0;
    auto const deviceResult = deviceStatus(deviceCount);
    if (deviceResult == cudaErrorNoDevice || (deviceResult == cudaSuccess && deviceCount == 0)) {
        GTEST_SKIP() << "CUDA device unavailable";
    }
    ASSERT_EQ(deviceResult, cudaSuccess) << cudaGetErrorString(deviceResult);
    auto p = parameters();
    std::vector<sf::Input> inputs{
        input(p, 0.01, 1), input(p, 1.0, 2), input(p, 100.0, 3), input(p, 10.0, 4)};
    inputs[2].energy = 1.e16f;
    inputs[3].state.specificStarFormationRate = 1.0;
    parity(sf::Operation::advance, inputs, p);
    parity(sf::Operation::eos, inputs, p);
    parity(sf::Operation::diagnostics, inputs, p);
    parity(sf::Operation::pressure, inputs, p);
    p.cooling.comoving = true;
    p.model.overdensityThreshold = 5.e-23;
    parity(sf::Operation::advance, inputs, p);
}

TEST(StarFormationCuda, RejectsInvalidSourceBeforeCommit) {
    int deviceCount = 0;
    auto const deviceResult = deviceStatus(deviceCount);
    if (deviceResult == cudaErrorNoDevice || (deviceResult == cudaSuccess && deviceCount == 0)) {
        GTEST_SKIP() << "CUDA device unavailable";
    }
    ASSERT_EQ(deviceResult, cudaSuccess) << cudaGetErrorString(deviceResult);
    auto const p = parameters();
    std::vector<sf::Input> inputs{input(p, 10.0, 1), input(p, 10.0, 0)};
    std::vector<sf::Output> outputs(inputs.size());
    for (size_t index = 0; index < outputs.size(); ++index) {
        outputs[index].state.id = 700 + index;
        outputs[index].state.eventCounter = 900 + index;
        outputs[index].energy = -100.0f - float(index);
        outputs[index].electronFraction = -200.0f - float(index);
        outputs[index].status = -17;
    }
    auto const original = outputs;
    try {
        sf::runCuda(sf::Operation::advance, inputs.data(), outputs.data(), inputs.size(), p);
        FAIL() << "Expected the second particle to reject the batch";
    } catch (std::runtime_error const& error) {
        std::string const message = error.what();
        EXPECT_NE(message.find("particle_id=0"), std::string::npos);
        EXPECT_NE(message.find("batch_index=1"), std::string::npos);
        EXPECT_NE(message.find("status=invalid_source_state_or_interval"), std::string::npos);
    }
    // A successful first lane must not leak through when the second lane fails.
    for (size_t index = 0; index < outputs.size(); ++index) {
        EXPECT_EQ(outputs[index].state, original[index].state);
        EXPECT_EQ(outputs[index].energy, original[index].energy);
        EXPECT_EQ(outputs[index].electronFraction, original[index].electronFraction);
        EXPECT_EQ(outputs[index].status, original[index].status);
    }
}

TEST(StarFormationCuda, NewStarRungAndOpeningKick) {
    int deviceCount = 0;
    auto const deviceResult = deviceStatus(deviceCount);
    if (deviceResult == cudaErrorNoDevice || (deviceResult == cudaSuccess && deviceCount == 0)) {
        GTEST_SKIP() << "CUDA device unavailable";
    }
    ASSERT_EQ(deviceResult, cudaSuccess) << cudaGetErrorString(deviceResult);
    auto p = parameters();
    p.effectiveRadius = 0.125f;
    p.timestepBase = 1.0f;
    p.timestepCap = 0.125f;
    p.timestepFactor = 0.2f;
    p.baseTimestep = 1.0;
    p.minimumRung = 2;
    p.rungLimit = 32;
    auto star = input(p, 1.0, 1);
    star.gravity[0] = 1.0f;
    star.gravity[1] = -2.0f;
    parity(sf::Operation::openNewStar, {star}, p);
}
