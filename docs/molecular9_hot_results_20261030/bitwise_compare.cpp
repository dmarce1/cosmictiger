// Bit-for-bit comparison of the molecular9 headers over the previously
// supported domain, 10-6000 K. Build once against the headers of the commit
// before the hot extension (extract them with `git show <commit>:<path>` into a
// directory placed first on the include path) and once against the current
// tree, then compare the two printed hashes:
//   fixed-temperature hash: rate coefficients, cooling coefficients, the thermal
//     energy function and the RHS evaluated at the grid temperature. These must
//     be identical.
//   inversion hash: the EOS temperature, heat capacity, pressure and gamma
//     recovered from the energy by Newton iteration. These may differ by a few
//     binary32 ulp for molecule-rich states above about 3200 K because the
//     Newton start is no longer clipped at 6000 K.
// Compile with: g++ -std=c++20 -O2 -fno-fast-math -ffp-contract=off -I<headers> -I<repo>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <initializer_list>

#include <cosmictiger/chemistry/primordial_rates.hpp>

namespace {
struct Hash {
    unsigned long long value = 1469598103934665603ULL;
    void mix(float v) {
        unsigned bits;
        std::memcpy(&bits, &v, 4);
        value = (value ^ bits) * 1099511628211ULL;
    }
};
}    // namespace

int main() {
    using namespace primordial;
    Hash fixed, inversion;
    int states = 0;
    for (int i = 0; i <= 1660; ++i) {
        float const temperature = static_cast<float>(10.0 * std::pow(600.0, i / 1660.0));
        if (temperature > 6000.0f) {
            break;
        }
        RateCoefficients rates;
        if (evaluateRateCoefficients(temperature, Network::Molecular9, rates) != Status::Success) {
            std::printf("rate failure at %g K\n", temperature);
            return 1;
        }
        for (float k : rates.collision) {
            fixed.mix(k);
        }
        fixed.mix(rates.hydrogenRecombinationCooling);
        fixed.mix(rates.heliumRecombinationCooling);
        fixed.mix(rates.heliumDoubleRecombinationCooling);
        float cooling[5], lte;
        if (molecularCoolingCoefficients(temperature, cooling, lte) != Status::Success) {
            std::printf("cooling failure at %g K\n", temperature);
            return 1;
        }
        for (float c : cooling) {
            fixed.mix(c);
        }
        fixed.mix(lte);
        if (i % 20 != 0) {
            continue;
        }
        for (int mixture = 0; mixture != 4; ++mixture) {
            for (float density : {0.0f, 1.0f, 1.0e4f}) {
                Environment environment;
                environment.network = Network::Molecular9;
                environment.nHydrogen = density;
                environment.cmbTemperature = 30.0f;
                State state;
                if (mixture == 1) {
                    state.abundances[4] = SignedAbundance::fromFraction(1.0f);
                } else if (mixture == 2) {
                    state.abundances[0] = SignedAbundance::fromFraction(0.3f);
                    state.abundances[1] = SignedAbundance::fromFraction(0.1f);
                    state.abundances[3] = SignedAbundance::fromFraction(1.0e-6f);
                    state.abundances[4] = SignedAbundance::fromFraction(0.2f);
                    state.abundances[5] = SignedAbundance::fromFraction(1.0e-5f);
                } else if (mixture == 3) {
                    state.abundances[0] = SignedAbundance::fromFraction(0.99f);
                    state.abundances[2] = SignedAbundance::fromFraction(0.5f);
                    state.abundances[5] = SignedAbundance::fromFraction(1.0e-3f);
                }
                if (thermalEnergyAtTemperature(
                        state, environment, temperature, state.thermalEnergy) != Status::Success) {
                    std::printf("energy failure at %g K mixture %d\n", temperature, mixture);
                    return 1;
                }
                fixed.mix(state.thermalEnergy);
                Derivative rhs;
                if (evaluateRhsAtTemperature(workingState(state), environment, temperature, rhs) !=
                    Status::Success) {
                    std::printf("RHS failure at %g K mixture %d\n", temperature, mixture);
                    return 1;
                }
                for (auto const& a : rhs.abundances) {
                    fixed.mix(a.toFloat());
                }
                fixed.mix(rhs.thermalEnergy.toFloat());
                Eos eos;
                if (evaluateEos(state, environment, eos) != Status::Success) {
                    std::printf("EOS failure at %g K mixture %d\n", temperature, mixture);
                    return 1;
                }
                inversion.mix(eos.temperature);
                inversion.mix(eos.heatCapacity);
                inversion.mix(eos.pressurePerHydrogen);
                inversion.mix(eos.adiabaticIndex);
                ++states;
            }
        }
    }
    std::printf("states %d fixed-temperature hash %016llx inversion hash %016llx\n", states,
        fixed.value, inversion.value);
    return 0;
}
