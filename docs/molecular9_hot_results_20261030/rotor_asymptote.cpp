// Converged binary64 rigid-rotor and harmonic-oscillator sums versus the runtime
// molecularModes, for H2 (theta_r = 85.4 K) and H2+ across the 6000 K ladder
// extension and the 1e4 K series switch. Compile with the same flags as
// bitwise_compare.cpp against the current tree.
#include <cmath>
#include <cosmictiger/chemistry/primordial_eos.hpp>
#include <cstdio>
#include <initializer_list>
int main() {
    for (double theta : {85.4, (30.21 - 0.8425) * 1.438776877}) {
        double const vib = theta == 85.4 ? 5987.0 : (2321.7 - 132.4) * 1.438776877;
        for (double T : {6000.0, 9000.0, 9999.0, 10000.0, 10001.0, 12000.0, 20000.0, 1.0e5, 1.0e6,
                 1.0e7, 1.0e8}) {
            double energy = 0.0, capacity = 0.0, zeven = 0.0, zodd = 0.0;
            for (int parity = 0; parity != 2; ++parity) {
                double partition = 0.0, first = 0.0, second = 0.0;
                for (int j = parity;; j += 2) {
                    double const level = theta * (j * (j + 1.0) - 2.0 * parity);
                    double const weight = (2 * j + 1) * std::exp(-level / T);
                    partition += weight;
                    first += weight * level;
                    second += weight * level * level;
                    if (level / T > 60.0)
                        break;
                }
                (parity == 0 ? zeven : zodd) = partition;
                double const mean = first / partition;
                energy += (parity == 0 ? 0.25 : 0.75) * mean;
                capacity +=
                    (parity == 0 ? 0.25 : 0.75) * (second / partition - mean * mean) / (T * T);
            }
            double const occ = std::exp(-vib / T);
            energy += vib * occ / (1.0 - occ);
            capacity += std::pow(vib / T, 2) * occ / std::pow(1.0 - occ, 2);
            auto const modes = primordial::detail::molecularModes(
                static_cast<float>(T), static_cast<float>(theta), static_cast<float>(vib));
            std::printf("theta %.4g T %.6g E ref %.10g run %.10g rel %.3e | cv ref %.10g run %.10g "
                        "rel %.3e | zodd/zeven-1 %.3e\n",
                theta, T, energy, modes.energyKelvin, modes.energyKelvin / energy - 1.0, capacity,
                modes.heatCapacity, modes.heatCapacity / capacity - 1.0,
                zodd * std::exp(-2.0 * theta / T) / zeven - 1.0);
        }
    }
    return 0;
}
