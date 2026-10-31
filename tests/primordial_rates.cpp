#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <limits>
#include <stdexcept>

#include <gtest/gtest.h>

#include <cosmictiger/chemistry/primordial_rates.hpp>

namespace {
void require(bool condition, char const* message) {
    if (!condition) {
        std::fprintf(stderr, "primordial rates: %s\n", message);
        throw std::runtime_error(message);
    }
}
void close(double actual, double expected, double tolerance, char const* message) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance * std::abs(expected)) {
        std::fprintf(stderr, "%s: actual %.17g expected %.17g tolerance %.3g\n", message, actual,
            expected, tolerance);
        throw std::runtime_error(message);
    }
}
void setInitialFraction(
    primordial::State& state, primordial::Environment const& environment, int index, float value) {
    auto const& x = state.abundances;
    float const helium = x[2] + x[3] + x[4];
    float fractions[primordial::kInitialFractionCount] = {x[1],
        helium > 0.0f ? x[3] / helium : 0.0f, helium > 0.0f ? x[4] / helium : 0.0f, x[6],
        2.0f * x[7], 2.0f * x[8]};
    fractions[index] = value;
    // Some fixtures intentionally construct invalid compositions for validation.
    primordial::initializeState(state, environment, fractions);
}
primordial::State atTemperature(
    primordial::State state, primordial::Environment const& environment, float temperature) {
    require(primordial::thermalEnergyAtTemperature(state, environment, temperature,
                state.thermalEnergy) == primordial::Status::Success,
        "temperature initialization");
    return state;
}
void checkStoichiometry() {
    int const hydrogen[] = {1, 1, 0, 0, 0, 0, 1, 2, 2};
    int const helium[] = {0, 0, 1, 1, 1, 0, 0, 0, 0};
    int const charge[] = {0, 1, 0, 1, 2, -1, -1, 0, 1};
    for (int reaction = 0; reaction < primordial::kReactionCount; ++reaction) {
        int hydrogenChange = 0, heliumChange = 0, chargeChange = 0;
        for (int species = 0; species < primordial::kChemicalSpeciesCount; ++species) {
            int const change = primordial::reactionStoichiometry(reaction, species);
            hydrogenChange += hydrogen[species] * change;
            heliumChange += helium[species] * change;
            chargeChange += charge[species] * change;
        }
        require(hydrogenChange == 0 && heliumChange == 0 && chargeChange == 0,
            "reaction violates exact nuclei or charge conservation");
    }
}
template <class Real>
void checkCreationDestruction() {
    using namespace primordial;
    EnvironmentT<Real> environment;
    environment.network = Network::Molecular9;
    environment.nHydrogen = Real(8);
    // Exercise every isolated channel and the full catalogue, with all species
    // present, each species absent, and every species absent. These synthetic
    // populations test the source algebra without imposing conserved totals.
    for (int absent = -2; absent < kChemicalSpeciesCount; ++absent) {
        StateT<Real> state;
        for (int species = 0; species < kChemicalSpeciesCount; ++species) {
            state.abundances[species] =
                absent == -2 || absent == species ? Real(0) : std::ldexp(Real(1), -species - 2);
        }
        auto const c = composition(state, environment);
        for (int active = -1; active < kReactionCount; ++active) {
            SCOPED_TRACE(::testing::Message() << "scalar bytes=" << sizeof(Real)
                                              << " absent=" << absent << " reaction=" << active);
            RateCoefficientsT<Real> rates;
            environment.radiation = {};
            Real* photo[] = {&environment.radiation.hydrogenIonization,
                &environment.radiation.heliumIonization,
                &environment.radiation.heliumPlusIonization,
                &environment.radiation.hydrogenMinusDetachment,
                &environment.radiation.molecularHydrogenDissociation,
                &environment.radiation.molecularHydrogenIonization,
                &environment.radiation.molecularHydrogenPlusDissociation,
                &environment.radiation.molecularHydrogenPlusIonization};
            for (int reaction = 0; reaction < kReactionCount; ++reaction) {
                if (active < 0 || active == reaction) {
                    Real const coefficient = Real(reaction + 1) / Real(128);
                    if (reaction >= 19 && reaction < 27) {
                        *photo[reaction - 19] = coefficient;
                    } else {
                        rates.collision[reaction < 19 ? reaction : reaction - 8] = coefficient;
                    }
                }
            }
            Real flux[kReactionCount];
            reactionFluxes(c, environment, rates, flux);
            for (int species = 0; species < kChemicalSpeciesCount; ++species) {
                Real expected = Real(0);
                for (int reaction = 0; reaction < kReactionCount; ++reaction) {
                    expected += Real(reactionStoichiometry(reaction, species)) * flux[reaction];
                }
                Real creation, destruction;
                reactionCreationDestruction(c, environment, rates, species, creation, destruction);
                ASSERT_TRUE(finite(creation));
                ASSERT_TRUE(finite(destruction));
                EXPECT_GE(creation, Real(0));
                EXPECT_GE(destruction, Real(0));
                Real const sink = destruction * state.abundances[species];
                Real const tolerance =
                    Real(32) * std::numeric_limits<Real>::epsilon() * (creation + sink);
                EXPECT_NEAR(creation - sink, expected, tolerance) << "species=" << species;
            }
        }
    }
    // Source equality alone cannot constrain D when its species is absent.
    // H+ + e -> H must still give D_H+ = k*n_H*x_e at x_H+ = 0.
    environment.radiation = {};
    CompositionT<Real> c;
    c.electrons = Real(0.25);
    RateCoefficientsT<Real> rates;
    rates.collision[1] = Real(0.125);
    Real creation, destruction;
    reactionCreationDestruction(c, environment, rates, 1, creation, destruction);
    EXPECT_EQ(creation, Real(0));
    EXPECT_EQ(destruction, Real(0.25));
    // H2 + H2 -> H2 + 2H consumes one net H2; removing one reactant
    // factor leaves D_H2 = k*n_H*x_H2, including its zero-abundance limit.
    rates = {};
    rates.collision[19] = Real(0.125);
    for (Real molecular : {Real(0), Real(0.25)}) {
        c.molecularHydrogen = molecular;
        reactionCreationDestruction(c, environment, rates, 7, creation, destruction);
        EXPECT_EQ(creation, Real(0));
        EXPECT_EQ(destruction, molecular);
    }
}
void checkAbundances() {
    using namespace primordial;
    Environment environment;
    for (float minority : {1.0e-8f, 1.0e-20f, 1.0e-30f}) {
        State state;
        state.abundances[0] = minority;
        state.abundances[1] = 1.0f;
        state.abundances[5] = 1.0f;
        state.thermalEnergy = 1.0f;
        Composition const c = composition(state, environment);
        close(c.hydrogen, minority, 1.0e-6, "direct neutral trace survives composition access");
        State const copied = state;
        require(copied.abundances[0] == minority && copied.abundances[1] == 1.0f,
            "direct state copy retains both neutral trace and dominant ion");
        require(validateComposition(state, environment) == Status::Success,
            "direct populations conserve within ordinary rounding");
    }
    State invalid;
    environment.network = Network::Molecular9;
    setInitialFraction(invalid, environment, 0, 0.7f);
    setInitialFraction(invalid, environment, 4, 0.4f);
    require(validateComposition(invalid, environment) == Status::InvalidInput,
        "reject excess hydrogen nuclei");
    invalid = State{};
    invalid.abundances[5] = 0.1f;
    require(validateComposition(invalid, environment) == Status::InvalidInput,
        "reject charge imbalance");
    invalid = State{};
    invalid.abundances[2] *= 2.0f;
    require(validateComposition(invalid, environment) == Status::InvalidInput,
        "reject excess helium nuclei");
}
void checkRates() {
    using namespace primordial;
    RateCoefficients rates;
    require(evaluateRateCoefficients(10000.0f, Network::Atomic6, rates) == Status::Success,
        "atomic coefficient evaluation");
    // Independent binary64 evaluations of Hui-Gnedin Appendix A, Verner-Ferland
    // Table 1(b), and GA08 Table 8; these literals are not generated by the code.
    close(rates.collision[1], 4.296953615610385e-13, 3.0e-6, "H case-A recombination");
    close(rates.collision[3], 4.839402053866274e-13, 3.0e-6, "He case-A recombination");
    close(rates.hydrogenRecombinationCooling, 2.8650394162666563e-13, 3.0e-6,
        "H recombination thermal coefficient");
    for (int i = 0; i <= 140; ++i) {
        float const temperature = std::pow(10.0f, 1.0f + i / 20.0f);
        require(evaluateRateCoefficients(temperature, Network::Atomic6, rates) == Status::Success,
            "atomic coefficient domain scan");
        for (float rate : rates.collision) {
            require(std::isfinite(rate) && rate >= 0.0f, "invalid atomic coefficient");
        }
    }
    require(evaluateRateCoefficients(1000.0f, Network::Molecular9, rates) == Status::Success,
        "molecular coefficient evaluation");
    close(rates.collision[6], 8.466813624857995e-16, 4.0e-6, "radiative attachment anchor");
    close(rates.collision[8], 4.860713235201269e-18, 8.0e-6, "H2+ radiative association anchor");
    close(rates.collision[7], 1.3e-9, 1.0e-7, "associative detachment");
    close(rates.collision[9], 6.4e-10, 1.0e-7, "H2+ charge exchange");
    close(rates.collision[10], 1.3011163950992776e-19, 3.0e-5, "cancelling Savin polynomial");
    float cooling[5], lte;
    require(molecularCoolingCoefficients(1000.0f, cooling, lte) == Status::Success,
        "H2 cooling coefficients");
    double const constant[] = {-24.311209, -23.962112, -23.689237, -21.716699, -22.190316};
    for (int i = 0; i < 5; ++i) {
        close(cooling[i], std::pow(10.0, constant[i]) / 1.602176634e-12, 8.0e-6,
            "H2 cooling collision partner anchor");
    }
    close(lte, 1.7299950645163523e-9, 2.0e-6, "H2 LTE cooling anchor");
    require(evaluateRateCoefficients(6000.0f, Network::Molecular9, rates) == Status::Success,
        "upper molecular rates");
    close(rates.collision[19], 3.521039924814333e-18, 5.0e-6, "H2-H2 dissociation anchor");
    close(rates.collision[20], 2.6136829373249576e-18, 8.0e-6, "H2-He dissociation anchor");
    for (int i = 0; i <= 200; ++i) {
        float const temperature = 100.0f * std::pow(60.0f, i / 200.0f);
        require(
            evaluateRateCoefficients(temperature, Network::Molecular9, rates) == Status::Success,
            "molecular coefficient domain scan");
        for (float rate : rates.collision) {
            require(std::isfinite(rate) && rate >= 0.0f, "invalid rate coefficient");
        }
    }
    for (Network network : {Network::Molecular9, Network::Atomic6}) {
        require(evaluateRateCoefficients(1.0e8f, network, rates) == Status::Success &&
                evaluateRateCoefficients(1.5e8f, network, rates) == Status::UnsupportedState,
            "both networks stop at the shared 1e8 K ceiling");
    }
}
void checkEos() {
    using namespace primordial;
    Environment environment;
    environment.nHydrogen = 1.0f;
    State neutral;
    neutral = atTemperature(neutral, environment, 1000.0f);
    Eos eos;
    require(evaluateEos(neutral, environment, eos) == Status::Success, "neutral EOS");
    close(eos.adiabaticIndex, 5.0 / 3.0, 3.0e-7, "monatomic gamma");
    environment.network = Network::Molecular9;
    State molecule;
    setInitialFraction(molecule, environment, 4, (1.0f - 0.0f));
    for (float temperature :
        {100.0f, 300.0f, 1000.0f, 3000.0f, 6000.0f, 8000.0f, 1.0e4f, 3.0e4f, 1.0e6f, 1.0e8f}) {
        molecule = atTemperature(molecule, environment, temperature);
        require(
            evaluateEos(molecule, environment, eos) == Status::Success, "molecular EOS inversion");
        close(eos.temperature, temperature, 1.0e-6, "EOS temperature round trip");
        require(eos.heatCapacity > 0.0f && eos.adiabaticIndex > 1.0f && eos.adiabaticIndex < 1.67f,
            "physical molecular heat capacity");
        if (temperature > 100.0f && temperature < 1.0e8f) {
            float lower, upper;
            float capacity;
            float const step = 0.005f * temperature;
            require(thermalEnergyAtTemperature(molecule, environment, temperature - step, lower,
                        capacity) == Status::Success,
                "lower EOS sample");
            require(thermalEnergyAtTemperature(molecule, environment, temperature + step, upper,
                        capacity) == Status::Success,
                "upper EOS sample");
            close(eos.heatCapacity, (upper - lower) / (2.0 * step), 5.0e-5,
                "cv is the derivative of energy");
        }
    }
    environment.heliumToHydrogenNumberRatio = 0.0f;
    setInitialFraction(molecule, environment, 4, 1.0f);
    double const referenceEnergy[] = {
        0.006635103227285061, 0.025635344015995497, 0.10161292585928484, 0.7901953105218633};
    double const referenceCapacity[] = {
        7.36423966206575e-5, 1.0593396721564733e-4, 1.1162197626272923e-4, 1.4739969716322487e-4};
    float const referenceTemperature[] = {100.0f, 300.0f, 1000.0f, 6000.0f};
    // Independent binary64 partition sums through J=201, not the runtime's
    // 65-level float sums. Zero-temperature ortho offsets are removed.
    for (int i = 0; i < 4; ++i) {
        molecule = atTemperature(molecule, environment, referenceTemperature[i]);
        require(
            evaluateEos(molecule, environment, eos) == Status::Success, "reference molecular EOS");
        close(molecule.thermalEnergy, referenceEnergy[i], 8.0e-7,
            "molecular partition energy anchor");
        close(eos.heatCapacity, referenceCapacity[i], 8.0e-7,
            "molecular partition heat capacity anchor");
    }
    environment.nHydrogen = 10001.0f;
    require(validateState(molecule, environment) == Status::UnsupportedState,
        "explicit high density rejection");
}
void checkColdMolecularPhysics() {
    using namespace primordial;
    // Independent anchors from the published upward rates in CDS Table 1,
    // J/A+A/647/A155, not the downward polynomial coefficients used at runtime:
    // https://cdsarc.cds.unistra.fr/ftp/J/A+A/647/A155/table1.dat
    // Rows 0020, 0121, 1030, 1131; multiply m^3/s by 1e6, excitation
    // energy kB*DeltaT, and fixed 3:1 pair weights 1/16,3/16,3/16,9/16.
    // The 3.5% allowance covers the independent printed rates versus Table 2 fits.
    double const tableCooling[] = {2.3330194081667853e-36, 2.4019446499887801e-25,
        1.1854452296577507e-21, 8.7509890777940519e-20, 1.2028808870639384e-18,
        7.1258981523484905e-18, 2.6077273613845835e-17, 7.0590055823064442e-17};
    for (int i = 0; i != 8; ++i) {
        float cooling[5], lte;
        float const temperature = static_cast<float>(10 * (i + 1));
        require(molecularCoolingCoefficients(temperature, cooling, lte) == Status::Success,
            "cold molecular cooling domain");
        close(cooling[1], tableCooling[i], 0.035, "independent CDS upward-rate cooling anchor");
    }
    double const polynomialTemperature[] = {10.0, 20.0, 30.0, 50.0, 80.0};
    double const polynomialCooling[] = {2.3019867366788452e-36, 2.4779944150853826e-25,
        1.1932265300740437e-21, 1.1880812413988798e-18, 7.093909511871683e-17};
    // Separate binary64 evaluation of the published Table 2 polynomials and
    // detailed balance checks implementation precision independently of the
    // several-percent fitting error against Table 1 above.
    for (int i = 0; i != 5; ++i) {
        float cooling[5], lte;
        require(molecularCoolingCoefficients(
                    static_cast<float>(polynomialTemperature[i]), cooling, lte) == Status::Success,
            "cold polynomial reference domain");
        close(cooling[1], polynomialCooling[i], 5.0e-6,
            "CDS polynomial and detailed balance binary64 reference");
    }
    // Independent binary64 GA08 Table 8 evaluation. H has its published
    // 10..100 K branch; He,H+,e fits already include this temperature range.
    // https://arxiv.org/abs/0803.1768
    double const coefficients[][6] = {
        {-16.818342, 37.383713, 58.145166, 48.656103, 20.159831, 3.8479610},
        {-23.689237, 2.1892372, -0.81520438, 0.29036281, -0.16596184, 0.19191375},
        {-21.716699, 1.3865783, -0.37915285, 0.11453688, -0.23214154, 0.058538864},
        {-34.286155, -48.537163, -77.121176, -51.352459, -15.169160, -0.98120322}};
    int const partners[] = {0, 2, 3, 4};
    for (float temperature : {10.0f, 20.0f, 49.0f, 80.0f}) {
        float cooling[5], lte;
        require(molecularCoolingCoefficients(temperature, cooling, lte) == Status::Success,
            "cold GA08 coefficients");
        double const x = std::log10(static_cast<double>(temperature) / 1000.0);
        for (int partner = 0; partner != 4; ++partner) {
            double value = 0.0;
            for (int power = 0; power != 6; ++power) {
                value += coefficients[partner][power] * std::pow(x, power);
            }
            double const expected = std::pow(10.0, value) / 1.602176634e-12;
            close(cooling[partners[partner]], expected, 1.0e-3,
                "cold GA08 independent binary64 reference");
            require(cooling[partners[partner]] > 0.0f,
                "cold cooling avoids premature erg-unit underflow");
        }
    }
    for (float boundary : {80.0f, 100.0f}) {
        float center[5], left[5], right[5], lte;
        require(molecularCoolingCoefficients(boundary, center, lte) == Status::Success &&
                molecularCoolingCoefficients(boundary - 0.05f, left, lte) == Status::Success &&
                molecularCoolingCoefficients(boundary + 0.05f, right, lte) == Status::Success,
            "cold compatibility join samples");
        for (int partner = 0; partner != 2; ++partner) {
            double const leftSlope = std::log(static_cast<double>(center[partner]) / left[partner]);
            double const rightSlope =
                std::log(static_cast<double>(right[partner]) / center[partner]);
            close(leftSlope, rightSlope, 0.025, "cold compatibility join first derivative");
            require(std::abs(leftSlope) < 0.005 && std::abs(rightSlope) < 0.005,
                "cold compatibility join continuous value");
        }
    }
    Environment environment;
    environment.network = Network::Molecular9;
    // Every kelvin resolves the H2+ LTE transition from zero through subnormal
    // values around 35..45 K. Exact-zero H2+ must not create a 0/0 source.
    for (int temperature = 10; temperature <= 100; ++temperature) {
        RateCoefficients rates;
        require(evaluateRateCoefficients(
                    static_cast<float>(temperature), Network::Molecular9, rates) == Status::Success,
            "cold reaction coefficient grid");
        for (float rate : rates.collision) {
            require(std::isfinite(rate) && rate >= 0.0f, "nonnegative finite cold reaction rate");
        }
        for (float density : {0.0f, 1.0e-8f, 1.0f, 1.0e4f}) {
            environment.nHydrogen = density;
            for (float cmb : {0.0f, 20.0f, 136.275f}) {
                environment.cmbTemperature = cmb;
                for (int mixture = 0; mixture != 6; ++mixture) {
                    State state;
                    if (mixture == 1) {
                        setInitialFraction(state, environment, 4, 1.0f);
                    } else if (mixture == 2) {
                        setInitialFraction(state, environment, 0, 1.0f);
                        setInitialFraction(state, environment, 2, 1.0f);
                    } else if (mixture >= 3) {
                        setInitialFraction(state, environment, 0, 0.2f);
                        setInitialFraction(state, environment, 1, 0.1f);
                        setInitialFraction(state, environment, 4, 0.1f);
                        setInitialFraction(state, environment, 5,
                            mixture == 3 ? 0.0f : (mixture == 4 ? 1.0e-20f : 0.1f));
                    }
                    state = atTemperature(state, environment, static_cast<float>(temperature));
                    Eos eos;
                    Derivative derivative;
                    require(evaluateEos(state, environment, eos) == Status::Success,
                        "cold mixture EOS inversion");
                    close(
                        eos.temperature, temperature, 1.5e-6, "cold mixture temperature roundtrip");
                    require(evaluateRhs(state, environment, derivative) == Status::Success &&
                            finite(derivative.thermalEnergy),
                        "finite cold mixture thermal RHS");
                    for (auto const& abundance : derivative.abundances) {
                        require(finite(abundance), "finite cold mixture species RHS");
                    }
                }
            }
        }
    }
    // Binary64 evaluation of GA08 eqs. (44)-(50) isolates the H2+ LTE
    // bridge through its binary32 subnormal interval. Absolute error of a
    // few subnormal ulps is allowed when the physical result is unresolvable.
    environment.nHydrogen = 1.0e4f;
    environment.cmbTemperature = 0.0f;
    State ionMixture;
    setInitialFraction(ionMixture, environment, 0, 0.2f);
    setInitialFraction(ionMixture, environment, 4, 0.1f);
    setInitialFraction(ionMixture, environment, 5, 0.1f);
    auto const ionComposition = composition(ionMixture, environment);
    for (int temperature = 30; temperature <= 70; ++temperature) {
        RateCoefficients rates;
        require(evaluateRateCoefficients(
                    static_cast<float>(temperature), Network::Molecular9, rates) == Status::Success,
            "H2+ bridge coefficient setup");
        float flux[kReactionCount]{};
        ThermalRates thermal;
        require(evaluateThermalRates(ionComposition, environment, static_cast<float>(temperature),
                    rates, flux, thermal) == Status::Success,
            "H2+ subnormal bridge finite");
        double const electron = (1.1e-19 / 1.602176634e-12) *
            std::pow(static_cast<double>(temperature), -0.34) * std::exp(-3025.0 / temperature);
        double const atom = (1.36e-22 / 1.602176634e-12) * std::exp(-3152.0 / temperature);
        double const lte = (2.0e-19 / 1.602176634e-12) *
            std::pow(static_cast<double>(temperature), 0.1) * std::exp(-3125.0 / temperature);
        double const low = environment.nHydrogen *
            (ionComposition.hydrogen * atom + ionComposition.electrons * electron);
        double const expected = ionComposition.molecularHydrogenPlus * low / (1.0 + low / lte);
        require(std::abs(thermal.molecularIonLineCooling - expected) <=
                5.0 * std::numeric_limits<float>::denorm_min() + 3.0e-5 * expected,
            "H2+ bridge binary64 reference across subnormal LTE");
    }
    environment.heliumToHydrogenNumberRatio = 0.0f;
    State molecule;
    setInitialFraction(molecule, environment, 4, 1.0f);
    for (float temperature : {10.0f, 20.0f, 49.0f, 80.0f, 100.0f}) {
        // Independent double partition sums through J=201 for fixed ortho:para
        // 3:1. This checks the thermal energy and derivative, not an E=cv*T model.
        double rotationEnergy = 0.0, rotationCapacity = 0.0;
        for (int parity = 0; parity != 2; ++parity) {
            double partition = 0.0, first = 0.0, second = 0.0;
            for (int j = parity; j <= 201; j += 2) {
                double const level = 85.4 * (j * (j + 1) - 2 * parity);
                double const weight = (2 * j + 1) * std::exp(-level / temperature);
                partition += weight;
                first += weight * level;
                second += weight * level * level;
            }
            double const fraction = parity == 0 ? 0.25 : 0.75;
            rotationEnergy += fraction * first / partition;
            rotationCapacity += fraction * (second / partition - std::pow(first / partition, 2)) /
                (static_cast<double>(temperature) * temperature);
        }
        double const occupation = std::exp(-5987.0 / temperature);
        double const vibrationEnergy = 5987.0 * occupation / (1.0 - occupation);
        double const vibrationCapacity =
            std::pow(5987.0 / temperature, 2) * occupation / std::pow(1.0 - occupation, 2);
        double const expectedEnergy =
            0.5 * 8.617333262145e-5 * (1.5 * temperature + rotationEnergy + vibrationEnergy);
        double const expectedCapacity =
            0.5 * 8.617333262145e-5 * (1.5 + rotationCapacity + vibrationCapacity);
        molecule = atTemperature(molecule, environment, temperature);
        Eos eos;
        require(evaluateEos(molecule, environment, eos) == Status::Success,
            "cold molecular partition EOS");
        close(molecule.thermalEnergy, expectedEnergy, 8.0e-7, "cold independent partition energy");
        close(eos.heatCapacity, expectedCapacity, 8.0e-7, "cold independent partition derivative");
    }
    float energy;
    float capacity, cooling[5], lte;
    RateCoefficients rates;
    require(thermalEnergyAtTemperature(molecule, environment, 9.99f, energy, capacity) ==
                Status::UnsupportedState &&
            molecularCoolingCoefficients(9.99f, cooling, lte) == Status::UnsupportedState &&
            evaluateRateCoefficients(9.99f, Network::Molecular9, rates) == Status::UnsupportedState,
        "no silent extrapolation below 10 K");
}

// Independent binary64 molecular partition sums, converged to exp(-60) per ladder.
struct ReferenceModes {
    double energy{}, capacity{};
};
ReferenceModes referenceModes(double temperature, double rotation, double vibration) {
    ReferenceModes result;
    for (int parity = 0; parity != 2; ++parity) {
        double partition = 0.0, first = 0.0, second = 0.0;
        for (int j = parity;; j += 2) {
            double const level = rotation * (j * (j + 1.0) - 2.0 * parity);
            double const weight = (2 * j + 1) * std::exp(-level / temperature);
            partition += weight;
            first += weight * level;
            second += weight * level * level;
            if (level > 60.0 * temperature) {
                break;
            }
        }
        double const fraction = parity == 0 ? 0.25 : 0.75;
        double const mean = first / partition;
        result.energy += fraction * mean;
        result.capacity +=
            fraction * (second / partition - mean * mean) / (temperature * temperature);
    }
    double const occupation = std::exp(-vibration / temperature);
    result.energy += vibration * occupation / (1.0 - occupation);
    result.capacity +=
        std::pow(vibration / temperature, 2) * occupation / std::pow(1.0 - occupation, 2);
    return result;
}

void checkHotMolecularPhysics() {
    using namespace primordial;
    // Molecular9 above 6000 K: docs/molecular9_hot.md. Published high-temperature
    // branches, held fits and the Grackle-style cooling clamp are anchored by
    // binary64 evaluations of the cited expressions; the literals below are not
    // generated by the code under test.
    RateCoefficients rates;
    float cooling[5], lte;
    float energy;
    float capacity;
    Environment environment;
    environment.network = Network::Molecular9;
    State molecule;
    setInitialFraction(molecule, environment, 4, 1.0f);
    require(evaluateRateCoefficients(1.0e8f, Network::Molecular9, rates) == Status::Success &&
            molecularCoolingCoefficients(1.0e8f, cooling, lte) == Status::Success &&
            thermalEnergyAtTemperature(molecule, environment, 1.0e8f, energy, capacity) ==
                Status::Success,
        "molecular ceiling is 1e8 K");
    require(molecularCoolingCoefficients(1.5e8f, cooling, lte) == Status::UnsupportedState &&
            thermalEnergyAtTemperature(molecule, environment, 1.5e8f, energy, capacity) ==
                Status::UnsupportedState,
        "no silent extrapolation above 1e8 K");
    // GA08 A1/1 (Wishart) hot branch times the 6000 K continuity ratio; GP98 H8
    // held at 32000 K; Savin held at 30000 K; Croft held at 1e4 K; GA08 A1/16
    // hot branch times its 8000 K ratio; He v=0 dissociation held at 1e4 K;
    // GA08 A1/27 (Kimura) hot branch times its 1e4 K ratio, held at 1e5 K;
    // He+ + H- held at 1e4 K.
    struct Anchor {
        int index;
        float temperature;
        double value;
    };
    Anchor const anchors[] = {{6, 1.0e4f, 3.5518327464369794e-15},
        {6, 1.0e6f, 3.9960867434996674e-15}, {6, 1.0e8f, 4.50176092223626e-16},
        {8, 5.0e4f, 7.022557093244852e-16}, {8, 1.0e8f, 7.022557093244852e-16},
        {10, 1.0e5f, 1.8371411012969387e-10}, {15, 1.0e5f, 3.6e-8}, {15, 1.0e8f, 3.6e-8},
        {16, 1.0e4f, 2.429677229545956e-10}, {16, 1.0e6f, 3.850778900985164e-12},
        {20, 1.0e5f, 1.6838368126872296e-15}, {26, 5.0e4f, 7.520274988276722e-15},
        {26, 1.0e6f, 2.0096290539165483e-13}, {27, 1.0e5f, 5.854266686360125e-8},
        {27, 1.0e8f, 5.854266686360125e-8}};
    for (auto const& anchor : anchors) {
        require(evaluateRateCoefficients(anchor.temperature, Network::Molecular9, rates) ==
                Status::Success,
            "hot anchor evaluation");
        close(rates.collision[anchor.index], anchor.value, 2.0e-5, "hot molecular rate anchor");
    }
    // Published branch pairs are joined without a value jump; the binary64
    // continuity ratios are 1.00564 (6000 K), 1.00757 (8000 K), 1.00243 (1e4 K).
    int const splicedIndex[] = {6, 16, 26};
    float const spliceTemperature[] = {6000.0f, 8000.0f, 1.0e4f};
    for (int i = 0; i != 3; ++i) {
        RateCoefficients below, above;
        // One part per million either side: the published jumps without the
        // continuity ratios are 0.56 %, 0.76 % and 0.24 %, far above this.
        require(evaluateRateCoefficients(spliceTemperature[i] * (1.0f - 1.0e-6f),
                    Network::Molecular9, below) == Status::Success &&
                evaluateRateCoefficients(spliceTemperature[i] * (1.0f + 1.0e-6f),
                    Network::Molecular9, above) == Status::Success,
            "splice samples");
        close(above.collision[splicedIndex[i]], below.collision[splicedIndex[i]], 2.0e-4,
            "published branch pair joins continuously");
    }
    // Every fit is finite and nonnegative over the whole domain, and held fits
    // are exactly constant above their freeze temperature.
    RateCoefficients frozen, frozenAssociation, frozenTransfer, frozenHelium;
    require(evaluateRateCoefficients(1.0e4f, Network::Molecular9, frozen) == Status::Success &&
            evaluateRateCoefficients(32000.0f, Network::Molecular9, frozenAssociation) ==
                Status::Success &&
            evaluateRateCoefficients(30000.0f, Network::Molecular9, frozenTransfer) ==
                Status::Success &&
            evaluateRateCoefficients(1.0e5f, Network::Molecular9, frozenHelium) == Status::Success,
        "freeze references");
    for (int i = 0; i <= 140; ++i) {
        float const temperature = std::pow(10.0f, 1.0f + i / 20.0f);
        require(
            evaluateRateCoefficients(temperature, Network::Molecular9, rates) == Status::Success,
            "hot molecular coefficient domain scan");
        for (float rate : rates.collision) {
            require(std::isfinite(rate) && rate >= 0.0f, "invalid hot molecular coefficient");
        }
        require(molecularCoolingCoefficients(temperature, cooling, lte) == Status::Success,
            "hot cooling domain scan");
        for (float value : cooling) {
            require(std::isfinite(value) && value >= 0.0f, "invalid hot cooling coefficient");
        }
        require(std::isfinite(lte) && lte >= 0.0f, "invalid hot LTE coefficient");
        if (temperature > 1.0e4f) {
            for (int index : {15, 20, 27, 28}) {
                require(rates.collision[index] == frozen.collision[index],
                    "held fit is constant above its freeze temperature");
            }
            require(
                (temperature <= 32000.0f || rates.collision[8] == frozenAssociation.collision[8]) &&
                    (temperature <= 30000.0f ||
                        rates.collision[10] == frozenTransfer.collision[10]) &&
                    (temperature <= 1.0e5f || rates.collision[26] == frozenHelium.collision[26]),
                "association, charge transfer and He+H+ are constant above their freezes");
            float frozenCooling[5], frozenLte;
            require(
                molecularCoolingCoefficients(1.0e4f, frozenCooling, frozenLte) == Status::Success,
                "frozen cooling reference");
            for (int partner = 0; partner != 5; ++partner) {
                require(cooling[partner] == frozenCooling[partner] && lte == frozenLte,
                    "molecular cooling is held at its 1e4 K value");
            }
        }
    }
    // GA08 Table 8 polynomials evaluated at 8000 K (H, H2, He beyond their
    // published 6000 K) and the Hollenbach-McKee LTE asymptote at 1e4 K.
    double const extrapolated[] = {1.9267209878922416e-10, 1.9849130851057365e-11,
        4.3353763502813e-11, 9.698023271863455e-10, 1.088328750841478e-9};
    require(molecularCoolingCoefficients(8000.0f, cooling, lte) == Status::Success,
        "extrapolated cooling evaluation");
    for (int partner = 0; partner != 5; ++partner) {
        close(cooling[partner], extrapolated[partner], 1.0e-5, "8000 K cooling polynomial anchor");
    }
    close(lte, 5.733910528087682e-7, 1.0e-5, "8000 K LTE anchor");
    require(molecularCoolingCoefficients(1.0e4f, cooling, lte) == Status::Success,
        "clamp cooling evaluation");
    close(lte, 7.545189261663554e-7, 1.0e-5, "1e4 K LTE anchor");
    close(cooling[0], 2.443882414121584e-10, 1.0e-5, "1e4 K H collider anchor");
    // Molecular thermodynamics: the 129-level ladder between 6000 K and 1e4 K
    // and the Mulholland asymptote above agree with converged binary64 sums.
    environment.heliumToHydrogenNumberRatio = 0.0f;
    for (int ion = 0; ion != 2; ++ion) {
        State state;
        setInitialFraction(state, environment, ion == 0 ? 4 : 5, 1.0f);
        double const rotation = ion == 0 ? 85.4 : (30.21 - 0.8425) * 1.438776877;
        double const vibration = ion == 0 ? 5987.0 : (2321.7 - 132.4) * 1.438776877;
        for (float temperature :
            {6000.0f, 8000.0f, 9999.0f, 1.0e4f, 10001.0f, 2.0e4f, 1.0e5f, 1.0e6f, 1.0e8f}) {
            ReferenceModes const reference = referenceModes(temperature, rotation, vibration);
            // Half a molecule per H nucleus; the ion also frees half an electron.
            double const particles = ion == 0 ? 0.5 : 1.0;
            double const expectedEnergy =
                8.617333262145e-5 * (1.5 * particles * temperature + 0.5 * reference.energy);
            double const expectedCapacity =
                8.617333262145e-5 * (1.5 * particles + 0.5 * reference.capacity);
            state = atTemperature(state, environment, temperature);
            Eos eos;
            require(evaluateEos(state, environment, eos) == Status::Success, "hot molecular EOS");
            close(state.thermalEnergy, expectedEnergy, 4.0e-7, "hot partition energy anchor");
            close(eos.heatCapacity, expectedCapacity, 4.0e-7, "hot partition derivative anchor");
            close(eos.temperature, temperature, 1.0e-6, "hot molecular inversion");
        }
    }
    // Mixed hot states invert and have finite sources at every density.
    environment.heliumToHydrogenNumberRatio = 0.24f / (4.0f * 0.76f);
    for (int i = 0; i <= 44; ++i) {
        float const temperature = 6000.0f * std::pow(10.0f, i / 10.0f);
        if (temperature > 1.0e8f) {
            break;
        }
        for (float density : {0.0f, 1.0f, 1.0e4f}) {
            environment.nHydrogen = density;
            for (int mixture = 0; mixture != 4; ++mixture) {
                State state;
                if (mixture == 1) {
                    setInitialFraction(state, environment, 4, 1.0f);
                } else if (mixture == 2) {
                    setInitialFraction(state, environment, 0, 0.9f);
                    setInitialFraction(state, environment, 1, 0.5f);
                    setInitialFraction(state, environment, 2, 0.5f);
                    setInitialFraction(state, environment, 3, 1.0e-8f);
                    setInitialFraction(state, environment, 4, 1.0e-3f);
                    setInitialFraction(state, environment, 5, 1.0e-6f);
                } else if (mixture == 3) {
                    setInitialFraction(state, environment, 0, 0.3f);
                    setInitialFraction(state, environment, 1, 0.1f);
                    setInitialFraction(state, environment, 4, 0.2f);
                    setInitialFraction(state, environment, 5, 0.1f);
                }
                state = atTemperature(state, environment, temperature);
                Eos eos;
                Derivative derivative;
                require(evaluateEos(state, environment, eos) == Status::Success,
                    "hot mixture EOS inversion");
                close(eos.temperature, temperature, 1.5e-6, "hot mixture temperature roundtrip");
                require(evaluateRhs(state, environment, derivative) == Status::Success &&
                        finite(derivative.thermalEnergy),
                    "finite hot mixture thermal RHS");
                for (auto const& abundance : derivative.abundances) {
                    require(finite(abundance), "finite hot mixture species RHS");
                }
                if (mixture == 1 && density > 0.0f) {
                    require(derivative.abundances[7] < float(0.0f) &&
                            derivative.thermalEnergy < float(0.0f),
                        "hot molecular gas dissociates and loses binding energy");
                }
            }
        }
    }
    // Negligible-molecule atomic limit. For a molecule-free state the two
    // networks differ exactly by the molecular9-only channels C07 (H- formation,
    // row 6), C09 (H2+ radiative association, row 8), C26 (row 33) and C27
    // (row 34); only C27 enters the thermal ledger (its 10.99 eV charge-transfer
    // cost). The remainder is identical, and the differing channels stay far
    // below the atomic rates in hot gas.
    environment.nHydrogen = 1.0f;
    environment.cmbTemperature = 30.0f;
    for (float temperature : {2.0e4f, 1.0e5f, 1.0e6f, 1.0e7f, 1.0e8f}) {
        State state;
        setInitialFraction(state, environment, 0, 0.5f);
        setInitialFraction(state, environment, 1, 0.3f);
        setInitialFraction(state, environment, 2, 0.2f);
        Environment atomic = environment;
        atomic.network = Network::Atomic6;
        State molecularState = atTemperature(state, environment, temperature);
        State atomicState = atTemperature(state, atomic, temperature);
        require(molecularState.thermalEnergy == atomicState.thermalEnergy,
            "molecule-free energy is network independent");
        // Compare at the common explicit temperature: the inversion start differs
        // between the networks and a one-ulp temperature difference would leak
        // into the steep exponentials.
        Derivative molecularRhs, atomicRhs;
        require(evaluateRhsAtTemperature(molecularState, environment, temperature, molecularRhs) ==
                    Status::Success &&
                evaluateRhsAtTemperature(atomicState, atomic, temperature, atomicRhs) ==
                    Status::Success,
            "atomic limit RHS");
        require(
            evaluateRateCoefficients(temperature, Network::Molecular9, rates) == Status::Success,
            "atomic limit rates");
        Composition const c = composition(molecularState, environment);
        evaluateDensityDependentRates(c, environment, temperature, rates);
        float flux[kReactionCount];
        reactionFluxes(c, environment, rates, flux);
        double const transfer = (flux[33] - flux[34]);
        double const association = flux[8];
        double const differences[] = {
            (molecularRhs.abundances[1] - atomicRhs.abundances[1]) - transfer + association,
            (molecularRhs.abundances[3] - atomicRhs.abundances[3]) + transfer,
            (molecularRhs.abundances[4] - atomicRhs.abundances[4]),
            (molecularRhs.thermalEnergy - atomicRhs.thermalEnergy) +
                (24.5874 - 13.5984) * flux[34]};
        double const scales[] = {std::abs(atomicRhs.abundances[1]),
            std::abs(atomicRhs.abundances[3]), std::abs(atomicRhs.abundances[4]),
            std::abs(atomicRhs.thermalEnergy)};
        for (int i = 0; i != 4; ++i) {
            if (std::abs(differences[i]) > 2.0e-6 * scales[i] + 1.0e-30) {
                std::fprintf(stderr, "atomic limit T %g component %d residual %g scale %g\n",
                    temperature, i, differences[i], scales[i]);
                require(false, "networks differ only by the identified charge-transfer channels");
            }
        }
        // Compare the differing channels with the H ionization plus recombination
        // traffic rather than their near-cancelling net, and the C27 cost with
        // the net thermal rate.
        double const traffic = (flux[0] + flux[1]);
        double const cost = (24.5874 - 13.5984) * flux[34];
        std::printf("atomic limit T %.3g: |transfer|/traffic %.3g, association/traffic %.3g, "
                    "H- formation/traffic %.3g, C27 cost/|thermal| %.3g\n",
            temperature, std::abs(transfer) / traffic, association / traffic, flux[6] / traffic,
            cost / scales[3]);
        // Measured: 6e-5, 1.7e-4 and 2e-3 at 2e4 K, all falling with temperature;
        // the C27 share of the net thermal rate is 8e-8 at 2e4 K and at most 2e-6.
        require(std::abs(transfer) < 5.0e-3 * traffic && association < 5.0e-3 * traffic &&
                flux[6] < 5.0e-3 * traffic && cost < 5.0e-3 * scales[3],
            "molecular9-only channels are negligible in molecule-free hot gas");
    }
    // Hot thermal ledger: every dissociation partner still consumes 4.48 eV and
    // the H2+ bridge uses the 1e4 K-held GA08 coefficients.
    for (float temperature : {2.0e4f, 1.0e5f, 1.0e7f}) {
        environment.nHydrogen = 100.0f;
        State state;
        setInitialFraction(state, environment, 0, 0.5f);
        setInitialFraction(state, environment, 3, 1.0e-6f);
        setInitialFraction(state, environment, 4, 0.01f);
        setInitialFraction(state, environment, 5, 1.0e-4f);
        state = atTemperature(state, environment, temperature);
        Composition const c = composition(state, environment);
        require(
            evaluateRateCoefficients(temperature, Network::Molecular9, rates) == Status::Success,
            "hot ledger rates");
        evaluateDensityDependentRates(c, environment, temperature, rates);
        float flux[kReactionCount];
        reactionFluxes(c, environment, rates, flux);
        ThermalRates thermal;
        require(evaluateThermalRates(c, environment, temperature, rates, flux, thermal) ==
                Status::Success,
            "hot ledger evaluation");
        // The full chemical-cooling ledger: 4.48 eV per dissociated molecule
        // for all four partners, 1.83 eV reverse charge transfer, 0.7542 eV
        // per collisional detachment, dissociative attachment and the 10.99 eV
        // He/H charge-transfer cost. An equality catches a dropped partner.
        double const dissociation = 4.48 * (flux[11] + flux[12] + flux[27] + flux[28]);
        double const ledger = dissociation + 1.83 * flux[10] +
            0.7542 * (flux[13] + flux[14] + flux[36]) + (4.48 - 0.7542) * flux[30] +
            (24.5874 - 13.5984) * flux[34];
        require(dissociation > 0.0, "hot dissociation is active");
        close(thermal.molecularChemicalCooling, ledger, 1.0e-6, "hot chemical cooling ledger");
        double const ionElectron = 4.666167586504443e-9, ionAtom = 5.744994354680924e-10,
                     ionLte = 2.2940483994493419e-7;
        double const low =
            environment.nHydrogen * (c.hydrogen * ionAtom + c.electrons * ionElectron);
        close(thermal.molecularIonLineCooling, c.molecularHydrogenPlus * low / (1.0 + low / ionLte),
            3.0e-5, "hot H2+ cooling uses the held GA08 coefficients");
        Derivative rhs;
        require(evaluateRhs(state, environment, rhs) == Status::Success, "hot ledger RHS");
        close(rhs.thermalEnergy, thermal.netHeating(), 2.0e-5,
            "hot RHS shares the single thermal ledger");
    }
}

void checkSources() {
    using namespace primordial;
    Environment environment;
    environment.nHydrogen = 0.0f;
    environment.heliumToHydrogenNumberRatio = 0.0f;
    environment.cmbTemperature = 1000.0f;
    State ion;
    setInitialFraction(ion, environment, 0, (1.0f - 0.0f));
    Derivative rhs;
    ion = atTemperature(ion, environment, 500.0f);
    require(
        evaluateRhs(ion, environment, rhs) == Status::Success && rhs.thermalEnergy > float(0.0f),
        "Compton heats below CMB");
    ion = atTemperature(ion, environment, 2000.0f);
    require(
        evaluateRhs(ion, environment, rhs) == Status::Success && rhs.thermalEnergy < float(0.0f),
        "Compton cools above CMB");
    ion = atTemperature(ion, environment, 1000.0f);
    require(
        evaluateRhs(ion, environment, rhs) == Status::Success && rhs.thermalEnergy == float(0.0f),
        "Compton balance at CMB");
    environment.cmbTemperature = 0.0f;
    environment.radiation.hydrogenIonization = 1.0e-12f;
    environment.radiation.heatingPerHydrogen = 3.0e-12f;
    State neutral;
    require(initializeState(neutral, environment) == Status::Success, "neutral input conversion");
    neutral = atTemperature(neutral, environment, 1000.0f);
    require(evaluateRhs(neutral, environment, rhs) == Status::Success, "photo source");
    close(rhs.abundances[1], 1.0e-12, 1.0e-7, "photo ionization per absorber");
    close(rhs.thermalEnergy, 3.0e-12, 1.0e-7, "photo threshold is not charged twice");
    environment.radiation.hydrogenIonization = -1.0f;
    require(evaluateRhs(neutral, environment, rhs) == Status::InvalidInput,
        "negative radiation is rejected");
}

void checkThermalLedger() {
    using namespace primordial;
    Environment environment;
    environment.nHydrogen = 10.0f;
    environment.heliumToHydrogenNumberRatio = 0.1f;
    State state;
    setInitialFraction(state, environment, 2, (1.0f - 0.0f));
    Composition c = composition(state, environment);
    RateCoefficients rates;
    require(evaluateRateCoefficients(10000.0f, Network::Atomic6, rates) == Status::Success,
        "free-free setup");
    float flux[kReactionCount]{};
    reactionFluxes(c, environment, rates, flux);
    ThermalRates thermal;
    require(evaluateThermalRates(c, environment, 10000.0f, rates, flux, thermal) == Status::Success,
        "thermal evaluation");
    double const gaunt = 1.1 + .34 * std::exp(-2.25 / 3.0);
    double const expected = 10.0 * .2 * (4.0 * .1) * (1.42e-27 / 1.602176634e-12) * 100.0 * gaunt;
    close(thermal.freeFreeCooling, expected, 3.0e-7, "He++ free-free uses Z squared");
    close(thermal.ionizationCooling, 13.5984 * flux[0], 1.0e-7,
        "H ionization threshold uses the reaction flux once");
    environment.network = Network::Molecular9;
    environment.nHydrogen = 10000.0f;
    state = State{};
    setInitialFraction(state, environment, 0, 0.1f);
    setInitialFraction(state, environment, 3, 0.01f);
    setInitialFraction(state, environment, 4, 0.2f);
    setInitialFraction(state, environment, 5, 0.1f);
    state = atTemperature(state, environment, 1000.0f);
    c = composition(state, environment);
    require(evaluateRateCoefficients(1000.0f, Network::Molecular9, rates) == Status::Success,
        "molecular thermal setup");
    evaluateDensityDependentRates(c, environment, 1000.0f, rates);
    reactionFluxes(c, environment, rates, flux);
    require(evaluateThermalRates(c, environment, 1000.0f, rates, flux, thermal) == Status::Success,
        "molecular thermal evaluation");
    require(
        thermal.molecularLineCooling > float(0.0f) && thermal.molecularIonLineCooling > float(0.0f),
        "both molecular ions and molecules radiate");
    require(thermal.unresolvedChemicalEnergyRelease > float(0.0f),
        "unresolved chemical energy is disclosed");
    double const dissociation = 4.48 * (flux[11] + flux[12] + flux[27] + flux[28]);
    double const ledger = dissociation + 1.83 * flux[10] +
        0.7542 * (flux[13] + flux[14] + flux[36]) + (4.48 - 0.7542) * flux[30] +
        (24.5874 - 13.5984) * flux[34];
    require(dissociation > 0.0, "dissociation partners are active");
    close(thermal.molecularChemicalCooling, ledger, 1.0e-6,
        "all four dissociation partners and the other chemical costs form the ledger");
    Derivative rhs;
    require(evaluateRhs(state, environment, rhs) == Status::Success, "extended molecular RHS");
    close(rhs.thermalEnergy, thermal.netHeating(), 2.0e-5, "RHS shares the single thermal ledger");
    // Each externally specified photo channel gets its own stoichiometry test
    // without collision or CMB sources; a nonzero net heating field is separate.
    environment.nHydrogen = 0.0f;
    environment.cmbTemperature = 0.0f;
    setInitialFraction(state, environment, 1, 0.2f);
    state = atTemperature(state, environment, 1000.0f);
    c = composition(state, environment);
    float const absorber[] = {c.hydrogen, c.helium, c.heliumPlus, c.hydrogenMinus,
        c.molecularHydrogen, c.molecularHydrogen, c.molecularHydrogenPlus, c.molecularHydrogenPlus};
    for (int photo = 0; photo < 8; ++photo) {
        environment.radiation = Radiation{};
        float* fields[] = {&environment.radiation.hydrogenIonization,
            &environment.radiation.heliumIonization, &environment.radiation.heliumPlusIonization,
            &environment.radiation.hydrogenMinusDetachment,
            &environment.radiation.molecularHydrogenDissociation,
            &environment.radiation.molecularHydrogenIonization,
            &environment.radiation.molecularHydrogenPlusDissociation,
            &environment.radiation.molecularHydrogenPlusIonization};
        *fields[photo] = 1.0e-10f;
        require(evaluateRhs(state, environment, rhs) == Status::Success, "photo channel RHS");
        for (int species = 0; species < kAbundanceCount; ++species) {
            double const expectedSource = reactionStoichiometry(photo + 19, species) *
                double(absorber[photo]) * static_cast<double>(*fields[photo]);
            close(rhs.abundances[species], expectedSource, 1.0e-6,
                "photo stoichiometry source for every direct population");
        }
        require(rhs.thermalEnergy == float(0.0f), "photo threshold is not thermal cooling");
    }
}
void checkScalarTemplates() {
    using namespace primordial;
    long double const coefficients[] = {-3.3232183e-7L, 3.3735382e-7L, -1.4491368e-7L,
        3.4172805e-8L, -4.781372e-9L, 3.9731542e-10L, -1.8171411e-11L, 3.5311932e-13L};
    for (double temperature : {1000.0, 3000.0, 10000.0, 30000.0}) {
        // Independent unshifted polynomial, evaluated with the wider host scalar.
        long double const logarithm = std::log(static_cast<long double>(temperature));
        long double polynomial = 0;
        for (int power = 0; power != 8; ++power) {
            polynomial +=
                static_cast<long double>(coefficients[power]) * std::pow(logarithm, power);
        }
        long double const expected = polynomial * std::exp(-21237.15L / temperature);
        RateCoefficientsT<double> rates;
        require(
            evaluateRateCoefficients(temperature, Network::Molecular9, rates) == Status::Success,
            "double coefficient evaluation");
        close(rates.collision[10], static_cast<double>(expected), 1.0e-12,
            "double shifted Savin fit agrees with original polynomial");
        RateCoefficients single;
        require(evaluateRateCoefficients(static_cast<float>(temperature), Network::Molecular9,
                    single) == Status::Success,
            "float coefficient evaluation");
        close(single.collision[10], static_cast<double>(expected), 3.0e-6,
            "float shifted Savin fit preserves published coefficient accuracy");
    }
    EnvironmentT<double> environment;
    environment.network = Network::Molecular9;
    environment.nHydrogen = 1.0;
    StateT<double> state;
    double const fractions[6] = {0.2, 0.1, 0.2, 1.0e-8, 0.01, 1.0e-5};
    require(
        initializeState(state, environment, fractions) == Status::Success, "double initial state");
    require(thermalEnergyAtTemperature(state, environment, 1234.5, state.thermalEnergy) ==
            Status::Success,
        "double thermal state");
    EosT<double> eos;
    require(evaluateEos(state, environment, eos) == Status::Success, "double EOS");
    close(eos.temperature, 1234.5, 1.0e-12, "double EOS retains double precision");
    DerivativeT<double> derivative;
    require(evaluateRhs(state, environment, derivative) == Status::Success, "double RHS");
}
}    // namespace
TEST(PrimordialRates, Stoichiometry) {
    checkStoichiometry();
}

TEST(PrimordialRates, Abundances) {
    checkAbundances();
}

TEST(PrimordialRates, Rates) {
    checkRates();
}

TEST(PrimordialRates, Eos) {
    checkEos();
}

TEST(PrimordialRates, ColdMolecularPhysics) {
    checkColdMolecularPhysics();
}

TEST(PrimordialRates, HotMolecularPhysics) {
    checkHotMolecularPhysics();
}

TEST(PrimordialRates, Sources) {
    checkSources();
}

TEST(PrimordialRates, ThermalLedger) {
    checkThermalLedger();
}

TEST(PrimordialRates, ScalarTemplates) {
    checkScalarTemplates();
}

TEST(PrimordialRates, CreationDestructionFloatAndDouble) {
    checkCreationDestruction<float>();
    checkCreationDestruction<double>();
}
