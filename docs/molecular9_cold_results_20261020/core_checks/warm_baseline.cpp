#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include <cosmictiger/chemistry/primordial_rates.hpp>

void dump(float value) {
    auto const bits = std::bit_cast<std::uint32_t>(value);
    std::fwrite(&bits, sizeof(bits), 1, stdout);
}
void dump(FloatFloat value) {
    dump(value.high());
    dump(value.low());
}
void dump(primordial::Status value) {
    dump(static_cast<float>(value));
}
int main() {
    using namespace primordial;
    std::vector<float> temperatures;
    for (int index = 0; index <= 400; ++index) {
        temperatures.push_back(static_cast<float>(100.0 * std::pow(60.0, index / 400.0)));
    }
    for (float const temperature : {100.0f, 200.0f, 617.0f, 1000.0f, 2000.0f, 6000.0f}) {
        if (temperature > 100.0f) {
            temperatures.push_back(std::nextafter(temperature, 0.0f));
        }
        temperatures.push_back(temperature);
        if (temperature < 6000.0f) {
            temperatures.push_back(std::nextafter(temperature, 10000.0f));
        }
    }
    std::sort(temperatures.begin(), temperatures.end());
    temperatures.erase(std::unique(temperatures.begin(), temperatures.end()), temperatures.end());
    std::array<std::array<float, kAbundanceCount>, 6> const mixtures = {{
        {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f},
        {0.2f,0.1f,0.01f,0.0f,0.0f,0.0f},
        {0.0002f,0.0f,0.0f,0.0f,0.0f,0.0f},
        {0.0002f,0.0001f,1.e-6f,1.e-8f,0.1f,1.e-7f},
        {0.0f,0.0f,0.0f,0.0f,1.0f,0.0f},
        {1.e-20f,1.e-20f,1.e-30f,1.e-30f,1.e-10f,1.e-20f}
    }};
    for (float const temperature : temperatures) {
        for (size_t mixture = 0; mixture != mixtures.size(); ++mixture) {
            for (float const density : {0.0f,1.e-4f,1.0f,1.e4f}) {
                for (float const cmb : {2.7255f,136.275f}) {
                    Environment environment;
                    environment.network = mixture < 2 ? Network::Atomic6 : Network::Molecular9;
                    environment.nHydrogen = density;
                    environment.cmbTemperature = cmb;
                    WorkingState state;
                    for (int i = 0; i != kAbundanceCount; ++i) {
                        state.abundances[i] = FloatFloat(mixtures[mixture][i]);
                    }
                    dump(temperature);dump(static_cast<float>(mixture));dump(density);dump(cmb);
                    float capacity = 0.0f;
                    dump(thermalEnergyAtTemperature(state,environment,temperature,state.thermalEnergy,capacity));
                    dump(state.thermalEnergy);dump(capacity);
                    Eos eos;
                    dump(evaluateEos(state,environment,eos));
                    for (float const value : {eos.temperature,eos.pressurePerHydrogen,eos.heatCapacity,eos.adiabaticIndex,eos.meanMolecularWeight,eos.pressureEnergyRatio}) {
                        dump(value);
                    }
                    RateCoefficients rates;
                    dump(evaluateRateCoefficients(temperature,environment.network,rates));
                    auto const c = composition(state,environment);
                    evaluateDensityDependentRates(c,environment,temperature,rates);
                    for (float const value : rates.collision) {
                        dump(value);
                    }
                    dump(rates.hydrogenRecombinationCooling);dump(rates.heliumRecombinationCooling);dump(rates.heliumDoubleRecombinationCooling);
                    FloatFloat flux[kReactionCount];reactionFluxes(c,environment,rates,flux);
                    for (auto const value : flux) {
                        dump(value);
                    }
                    ThermalRates thermal;
                    dump(evaluateThermalRates(c,environment,temperature,rates,flux,thermal));
                    for (auto const value : {thermal.ionizationCooling,thermal.recombinationCooling,thermal.excitationCooling,thermal.freeFreeCooling,thermal.molecularLineCooling,thermal.molecularIonLineCooling,thermal.molecularChemicalCooling,thermal.molecularChemicalHeating,thermal.photoHeating,thermal.comptonHeating,thermal.unresolvedChemicalEnergyRelease,thermal.netHeating()}) {
                        dump(value);
                    }
                    Derivative derivative;
                    dump(evaluateRhsAtTemperature(state,environment,temperature,derivative));
                    for (auto const value : derivative.abundances) {
                        dump(value);
                    }
                    dump(derivative.thermalEnergy);
                }
            }
        }
    }
    std::fprintf(stderr,"temperatures=%zu mixtures=%zu densities=4 cmbs=2\n",temperatures.size(),mixtures.size());
}
