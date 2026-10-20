#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cosmictiger/cosmology/native_recombination.hpp>

int main(int argc, char** argv) {
    if (argc < 4 || argc > 5) {
        std::fputs(
            "Usage: recombination_reference NEFF MAX_LOG_STEP REDSHIFTS_FILE [RTOL]\n", stderr);
        return EXIT_FAILURE;
    }
    try {
        CosmicParameters parameters{};
        parameters.omegaB = 0.049389;
        parameters.omegaC = 0.26503;
        parameters.neff = std::stod(argv[1]);
        parameters.y = 0.2454006;
        parameters.theta = 2.7255 / constants::kTcmb;
        parameters.hubble = 0.6732;
        double const criticalDensity = 3.0 * std::pow(parameters.hubble * constants::kH0, 2) /
            (8.0 * std::acos(-1.0) * constants::kG);
        parameters.omegaGam = 4.0 * constants::kSigma * std::pow(2.7255, 4) /
            (std::pow(constants::kC, 3) * criticalDensity);
        parameters.omegaNu =
            parameters.omegaGam * parameters.neff * 7.0 / 8.0 * std::pow(4.0 / 11.0, 4.0 / 3.0);
        primordial::RecombinationOptions options;
        options.maxLogStep = std::stod(argv[2]);
        if (argc > 4) {
            options.relativeTolerance = std::stod(argv[4]);
        }
        primordial::RecombinationHistory const history(
            recombinationParameters(parameters), options);
        std::ifstream input(argv[3]);
        if (!input) {
            throw std::runtime_error("Unable to open redshift sample file");
        }
        std::vector<double> redshifts;
        double redshift;
        while (input >> redshift) {
            if (!std::isfinite(redshift) || redshift < 0.0) {
                throw std::invalid_argument("Redshift samples must be finite and nonnegative");
            }
            redshifts.push_back(redshift);
        }
        if (!input.eof() || redshifts.empty()) {
            throw std::invalid_argument("Invalid or empty redshift sample file");
        }
        std::puts("n_eff,z,a,x_e,T_m,x_H,x_He,x_He_double");
        for (double const z : redshifts) {
            double const scaleFactor = 1.0 / (1.0 + z);
            std::printf("%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n", parameters.neff, z,
                scaleFactor, history.electronFraction(scaleFactor),
                history.gasTemperature(scaleFactor), history.hydrogenIonizedFraction(scaleFactor),
                history.heliumSinglyIonizedFraction(scaleFactor),
                history.heliumDoublyIonizedFraction(scaleFactor));
        }
    } catch (std::exception const& error) {
        std::fprintf(stderr, "Native recombination reference failed: %s\n", error.what());
        return EXIT_FAILURE;
    }
}
