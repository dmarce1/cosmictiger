#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <limits>
#include <stdexcept>

#include <gtest/gtest.h>

#include <cosmictiger/chemistry/primordial_eos.hpp>
#include <cosmictiger/chemistry/recombination.hpp>
#include <cosmictiger/cosmology/native_recombination.hpp>

namespace detail {

bool isFinite(double value) {
    constexpr std::uint64_t kExponentMask = UINT64_C(0x7ff0000000000000);
    return (std::bit_cast<std::uint64_t>(value) & kExponentMask) != kExponentMask;
}

void require(bool condition, char const* label) {
    if (!condition) {
        std::fprintf(stderr, "%s\n", label);
        throw std::runtime_error(label);
    }
}

void requireClose(double actual, double expected, double relativeTolerance,
    double absoluteTolerance, char const* label) {
    require(isFinite(actual) && isFinite(expected), "Nonfinite recombination result");
    if (std::abs(actual - expected) >
        absoluteTolerance + relativeTolerance * std::max(std::abs(actual), std::abs(expected))) {
        std::fprintf(stderr, "%s: actual %.17g, expected %.17g\n", label, actual, expected);
        throw std::runtime_error(label);
    }
}

template <class Function>
void requireInvalid(Function function, char const* label) {
    try {
        function();
    } catch (std::invalid_argument const&) {
        return;
    } catch (std::out_of_range const&) {
        return;
    }
    require(false, label);
}

CosmicParameters fiducialCosmology() {
    using namespace constants;
    CosmicParameters parameters{};
    parameters.omegaB = 0.049389;
    parameters.omegaC = 0.26503;
    parameters.neff = 3.046;
    parameters.y = 0.2454006;
    parameters.theta = 2.7255 / kTcmb;
    parameters.hubble = 0.6732;
    double const hubbleToday = parameters.hubble * kH0;
    double const criticalDensity = 3.0 * hubbleToday * hubbleToday / (8.0 * std::acos(-1.0) * kG);
    double const radiationConstant = 4.0 * kSigma / kC;
    parameters.omegaGam =
        radiationConstant * std::pow(parameters.theta * kTcmb, 4) / (criticalDensity * kC * kC);
    parameters.omegaNu =
        parameters.omegaGam * parameters.neff * (7.0 / 8.0) * std::pow(4.0 / 11.0, 4.0 / 3.0);
    return parameters;
}

void testPhysicalLimits(primordial::RecombinationHistory const& history,
    primordial::RecombinationParameters const& parameters) {
    double const helium = history.heliumToHydrogenNumberRatio();
    double previousElectrons = 1.0 + 2.0 * helium;
    double previousTemperature = parameters.cmbTemperature / 1.0e-6;
    for (int sample = 0; sample <= 1200; ++sample) {
        double const scale = std::exp(
            std::log(1.0e-6) + sample / 1200.0 * std::log(history.maxScaleFactor() / 1.0e-6));
        double const hydrogenIonized = history.hydrogenIonizedFraction(scale);
        double const heliumSinglyIonized = history.heliumSinglyIonizedFraction(scale);
        double const heliumDoublyIonized = history.heliumDoublyIonizedFraction(scale);
        double const electrons = history.electronFraction(scale);
        double const temperature = history.gasTemperature(scale);
        for (double const fraction : {hydrogenIonized, heliumSinglyIonized, heliumDoublyIonized}) {
            require(isFinite(fraction) && fraction >= 0.0 && fraction <= 1.0,
                "Ionization fraction outside the physical domain");
        }
        require(heliumSinglyIonized + heliumDoublyIonized <= 1.0 + 1.0e-12,
            "Helium populations exceed the number of nuclei");
        requireClose(electrons,
            hydrogenIonized + helium * (heliumSinglyIonized + 2.0 * heliumDoublyIonized), 2.0e-12,
            1.0e-14, "Electron number agrees with ionic charges");
        require(isFinite(temperature) && temperature > 0.0 &&
                temperature <= parameters.cmbTemperature / scale * (1.0 + 1.0e-10),
            "Gas temperature must be positive and no warmer than the unheated CMB");
        require(electrons <= previousElectrons * (1.0 + 1.0e-5) + 1.0e-12,
            "Unheated primordial gas must not develop a reionization episode");
        require(temperature <= previousTemperature * (1.0 + 1.0e-10),
            "Unheated homogeneous gas temperature must decrease");
        require(isFinite(history.pressurePerHydrogen(scale)) &&
                history.pressurePerHydrogen(scale) > 0.0,
            "Native chemistry pressure must be positive");
        previousElectrons = electrons;
        previousTemperature = temperature;
    }

    requireClose(history.electronFraction(1.0e-6), 1.0 + 2.0 * helium, 1.0e-10, 1.0e-12,
        "Early ionized hydrogen and doubly ionized helium limit");
    requireClose(history.gasTemperature(1.0e-6), parameters.cmbTemperature / 1.0e-6, 1.0e-8, 0.0,
        "Early tightly coupled matter temperature");
    require(history.heliumSinglyIonizedFraction(1.0 / 4001.0) > 0.99,
        "Singly ionized helium plateau before HeI recombination");
    require(history.heliumDoublyIonizedFraction(1.0 / 4001.0) < 1.0e-4,
        "HeIII recombination precedes HeII recombination");
    require(history.heliumSinglyIonizedFraction(1.0 / 1501.0) < 0.01,
        "Helium recombines before hydrogen");
    require(history.hydrogenIonizedFraction(1.0 / 1501.0) > 0.8,
        "Hydrogen remains ionized after helium recombines");
    require(history.electronFraction(1.0) > 1.0e-5 && history.electronFraction(1.0) < 1.0e-3,
        "Residual nonequilibrium hydrogen survives after recombination");
    require(history.gasTemperature(1.0 / 51.0) > 40.0 && history.gasTemperature(1.0 / 51.0) < 60.0,
        "Matter cools below the radiation temperature after thermal decoupling");
    double const lateCoolingRatio = history.gasTemperature(0.5) / history.gasTemperature(1.0);
    require(lateCoolingRatio > 3.9 && lateCoolingRatio < 4.01,
        "Late neutral gas approaches the adiabatic T proportional to a^-2 limit");
    requireClose(history.hydrogenNumberDensity(0.01) / history.hydrogenNumberDensity(1.0), 1.0e6,
        2.0e-13, 0.0, "Conserved hydrogen nuclei dilute as a^-3");

    // Scott & Moss (2009), arXiv:0902.3438, equations (2) and (6):
    // the early fractional temperature lag is H*t_Compton. Setting dT/dt=0
    // would give twice this lag, because the radiation also cools with expansion.
    // Evaluate after the model leaves its early Saha/Tgas=Tcmb initialization.
    double const earlyScale = 1.0 / 2501.0;
    double const radiationTemperature = parameters.cmbTemperature / earlyScale;
    double const electronFraction = history.electronFraction(earlyScale);
    double const hubble = parameters.hubble * constants::kH0 *
        std::sqrt(parameters.omegaRadiation / std::pow(earlyScale, 4) +
            (parameters.omegaB + parameters.omegaC) / std::pow(earlyScale, 3) + 1.0 -
            parameters.omegaB - parameters.omegaC - parameters.omegaRadiation);
    double const radiationConstant = 4.0 * constants::kSigma / constants::kC;
    double const comptonTime = 3.0 * constants::kMe * constants::kC /
        (8.0 * constants::kSigmaT * radiationConstant * std::pow(radiationTemperature, 4)) *
        (1.0 + helium + electronFraction) / electronFraction;
    double const lag = 1.0 - history.gasTemperature(earlyScale) / radiationTemperature;
    requireClose(lag, hubble * comptonTime, 0.1, 0.0,
        "Tightly coupled gas follows the evolving radiation temperature");
}

void testConvergence(primordial::RecombinationHistory const& baseline,
    primordial::RecombinationParameters const& parameters) {
    primordial::RecombinationOptions options;
    options.relativeTolerance = 1.0e-9;
    options.absoluteTolerance = 1.0e-14;
    options.maxLogStep = 0.005;
    primordial::RecombinationHistory const refined(parameters, options);
    double maximumElectronError = 0.0;
    double maximumTemperatureError = 0.0;
    for (double const redshift : {7000.0, 6000.0, 4000.0, 3000.0, 2500.0, 2200.0, 1800.0, 1500.0,
             1300.0, 1100.0, 1000.0, 850.0, 500.0, 200.0, 100.0, 50.0, 10.0, 0.0}) {
        double const scale = 1.0 / (1.0 + redshift);
        double const electrons = baseline.electronFraction(scale);
        double const refinedElectrons = refined.electronFraction(scale);
        double const temperature = baseline.gasTemperature(scale);
        double const refinedTemperature = refined.gasTemperature(scale);
        maximumElectronError =
            std::max(maximumElectronError, std::abs(electrons / refinedElectrons - 1.0));
        maximumTemperatureError =
            std::max(maximumTemperatureError, std::abs(temperature / refinedTemperature - 1.0));
        requireClose(electrons, refinedElectrons, 1.0e-5, 2.0e-10,
            "Native ionization history converges with tolerance and log-step refinement");
        requireClose(temperature, refinedTemperature, 1.0e-6, 1.0e-9,
            "Native temperature history converges with tolerance and log-step refinement");
    }
    std::printf("Native recombination convergence: max relative xe error %.9g, T error %.9g\n",
        maximumElectronError, maximumTemperatureError);
}

void testDomains(primordial::RecombinationHistory const& history,
    primordial::RecombinationParameters const& parameters) {
    double const nan = std::numeric_limits<double>::quiet_NaN();
    double const infinity = std::numeric_limits<double>::infinity();
    for (double const invalid : {-1.0, 0.0, nan, infinity, 1.01 * history.maxScaleFactor()}) {
        requireInvalid(
            [&] { history.electronFraction(invalid); }, "Invalid electron query accepted");
        requireInvalid(
            [&] { history.gasTemperature(invalid); }, "Invalid temperature query accepted");
    }
    auto const invalidParameters = [&](primordial::RecombinationParameters const& invalid) {
        requireInvalid([&] { primordial::RecombinationHistory rejected(invalid); },
            "Invalid recombination cosmology accepted");
    };
    auto invalid = parameters;
    invalid.hubble = nan;
    invalidParameters(invalid);
    invalid = parameters;
    invalid.omegaB = 0.0;
    invalidParameters(invalid);
    invalid = parameters;
    invalid.omegaC = -0.1;
    invalidParameters(invalid);
    invalid = parameters;
    invalid.omegaRadiation = -0.1;
    invalidParameters(invalid);
    invalid = parameters;
    invalid.heliumMassFraction = 1.0;
    invalidParameters(invalid);
    invalid = parameters;
    invalid.cmbTemperature = 0.0;
    invalidParameters(invalid);
    auto const invalidOptions = [&](primordial::RecombinationOptions const& options) {
        requireInvalid([&] { primordial::RecombinationHistory rejected(parameters, options); },
            "Invalid recombination solver options accepted");
    };
    primordial::RecombinationOptions options;
    options.relativeTolerance = nan;
    invalidOptions(options);
    options = {};
    options.absoluteTolerance = -1.0;
    invalidOptions(options);
    options = {};
    options.maxLogStep = 0.0;
    invalidOptions(options);
    options = {};
    options.maxScaleFactor = 0.0;
    invalidOptions(options);
    options = {};
    options.maxSteps = 0;
    invalidOptions(options);
    options = {};
    options.maxSteps = 1;
    bool budgetEnforced = false;
    try {
        primordial::RecombinationHistory rejected(parameters, options);
    } catch (std::runtime_error const&) {
        budgetEnforced = true;
    }
    require(budgetEnforced, "Native recombination enforces its finite integration attempt budget");
}

void testNoHelium(primordial::RecombinationParameters parameters) {
    parameters.heliumMassFraction = 0.0;
    primordial::RecombinationHistory const history(parameters);
    require(history.heliumToHydrogenNumberRatio() == 0.0, "Helium-free composition is supported");
    for (double const scale : {1.0e-6, 1.0 / 3001.0, 1.0 / 1101.0, 0.01, 1.0}) {
        requireClose(history.electronFraction(scale), history.hydrogenIonizedFraction(scale),
            1.0e-13, 0.0, "Only hydrogen contributes electrons in the helium-free limit");
        require(isFinite(history.gasTemperature(scale)) && history.gasTemperature(scale) > 0.0,
            "Helium-free gas temperature remains physical");
    }
}

void testChemistryComposition(primordial::RecombinationHistory const& history) {
    // The homogeneous effective-atom rates differ from the optically thin SPH
    // network, but both must use identical species charges and ideal-gas EOS
    // within the existing atomic framework's temperature and density domain.
    for (double const scale : {1.0 / 3001.0, 0.001, 0.002, 0.01}) {
        primordial::State state;
        primordial::Environment environment;
        auto status = history.chemistryState(scale, state, environment);
        require(status == primordial::Status::Success,
            "Native history converts within the existing chemistry EOS domain");
        require(environment.network == primordial::Network::Atomic6,
            "Homogeneous atomic history maps to the existing atomic network");
        requireClose(environment.nHydrogen, history.hydrogenNumberDensity(scale), 1.0e-7, 0.0,
            "Chemistry conversion preserves hydrogen nuclei density");
        primordial::Eos eos;
        status = primordial::evaluateEos(state, environment, eos);
        require(status == primordial::Status::Success, "Existing chemistry evaluates native state");
        requireClose(eos.temperature, history.gasTemperature(scale), 2.0e-6, 0.0,
            "Chemistry conversion preserves temperature");
        requireClose(primordial::composition(state, environment).electrons.toDouble(),
            history.electronFraction(scale), 2.0e-6, 1.0e-12,
            "Homogeneous ion charges agree with the existing chemistry composition");
        requireClose(eos.pressurePerHydrogen, history.pressurePerHydrogen(scale), 2.0e-6, 0.0,
            "Homogeneous pressure agrees with the existing chemistry ideal-gas EOS");
    }
    for (double const scale : {1.0e-6, 1.0}) {
        primordial::State state;
        state.thermalEnergy = 123.0f;
        primordial::Environment environment;
        environment.nHydrogen = 456.0f;
        auto const status = history.chemistryState(scale, state, environment);
        require(status == primordial::Status::UnsupportedState,
            "Cosmic epochs outside the existing SPH domain remain unsupported by that network");
        require(state.thermalEnergy == 123.0f && environment.nHydrogen == 456.0f,
            "Unsupported chemistry conversion preserves output objects");
    }
}

void testBackgroundAdapter(
    primordial::RecombinationHistory const& history, CosmicParameters const& parameters) {
    using namespace constants;
    ZeroOrderUniverse universe;
    createNativeZeroOrderUniverse(&universe, history.maxScaleFactor(), parameters);
    double const hubbleToday = parameters.hubble * kH0;
    double const criticalDensity = 3.0 * hubbleToday * hubbleToday / (8.0 * std::acos(-1.0) * kG);
    for (size_t index = 0; index < universe.k.values.size(); index += 29) {
        double const scale = std::exp(universe.k.minloga + index * universe.k.dloga);
        double const density = criticalDensity * parameters.omegaB / std::pow(scale, 3);
        double const hydrogen = history.hydrogenNumberDensity(scale);
        double const electrons = hydrogen * history.electronFraction(scale);
        double const pressure = universe.k.values[index] * std::pow(density, 5.0 / 3.0);
        constexpr double kElectronVoltErg = 1.602176634e-12;
        requireClose(pressure, hydrogen * history.pressurePerHydrogen(scale) * kElectronVoltErg,
            2.0e-11, 0.0, "Native background preserves the coupled chemistry pressure");
        double const particles = hydrogen *
            (1.0 + history.heliumToHydrogenNumberRatio() + history.electronFraction(scale));
        double const inferredTemperature =
            pressure / (primordial::kBoltzmannEv * kElectronVoltErg * particles);
        requireClose(inferredTemperature, history.gasTemperature(scale), 2.0e-6, 0.0,
            "Native background pressure uses the coupled chemistry temperature");
        requireClose(universe.xe(scale), history.electronFraction(scale), 2.0e-13, 0.0,
            "Native background preserves the chemistry electron fraction");
        double const hubble =
            hubbleFunction(scale, parameters.hubble, parameters.omegaB + parameters.omegaC,
                parameters.omegaGam + parameters.omegaNu) *
            kH0 / cosmic_constants::kH0;
        requireClose(universe.sigmaT.values[index], kC * kSigmaT * electrons / hubble, 2.0e-11, 0.0,
            "Native Thomson opacity uses the same electron history");
        require(isFinite(universe.cs2.values[index]) && universe.cs2.values[index] > 0.0,
            "Native background sound speed is positive");
    }
}

}    // namespace detail

TEST(NativeRecombination, PhysicalLimits) {
    CosmicParameters const cosmology = detail::fiducialCosmology();
    auto const parameters = recombinationParameters(cosmology);
    primordial::RecombinationHistory const history(parameters);
    detail::testPhysicalLimits(history, parameters);
}

TEST(NativeRecombination, Convergence) {
    CosmicParameters const cosmology = detail::fiducialCosmology();
    auto const parameters = recombinationParameters(cosmology);
    primordial::RecombinationHistory const history(parameters);
    detail::testConvergence(history, parameters);
}

TEST(NativeRecombination, Domains) {
    CosmicParameters const cosmology = detail::fiducialCosmology();
    auto const parameters = recombinationParameters(cosmology);
    primordial::RecombinationHistory const history(parameters);
    detail::testDomains(history, parameters);
}

TEST(NativeRecombination, NoHelium) {
    CosmicParameters const cosmology = detail::fiducialCosmology();
    auto const parameters = recombinationParameters(cosmology);
    detail::testNoHelium(parameters);
}

TEST(NativeRecombination, ChemistryComposition) {
    CosmicParameters const cosmology = detail::fiducialCosmology();
    auto const parameters = recombinationParameters(cosmology);
    primordial::RecombinationHistory const history(parameters);
    detail::testChemistryComposition(history);
}

TEST(NativeRecombination, BackgroundAdapter) {
    CosmicParameters const cosmology = detail::fiducialCosmology();
    auto const parameters = recombinationParameters(cosmology);
    primordial::RecombinationHistory const history(parameters);
    detail::testBackgroundAdapter(history, cosmology);
}
