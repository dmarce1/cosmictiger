#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

#include <cosmictiger/cosmology/boltzmann.hpp>
#include <cosmictiger/cosmology/native_recombination.hpp>

int main(int argc, char** argv) {
    if (argc < 3 || argc > 7) {
        std::fputs("Usage: boltzmann_reference NEFF K_MPC "
                   "[MAX_L [RTOL [RECOMBINATION_LOG_STEP [RECOMBINATION_RTOL]]]]\n",
            stderr);
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
        double const k = std::stod(argv[2]);
        BoltzmannSolverOptions options;
        if (argc > 3) {
            options.maxMultipole = std::stoi(argv[3]);
        }
        if (argc > 4) {
            options.relativeTolerance = std::stod(argv[4]);
        }
        ZeroOrderUniverse universe;
        primordial::RecombinationOptions recombinationOptions;
        if (argc > 5) {
            recombinationOptions.maxLogStep = std::stod(argv[5]);
        }
        if (argc > 6) {
            recombinationOptions.relativeTolerance = std::stod(argv[6]);
        }
        createNativeZeroOrderUniverse(&universe, 1.1, parameters, recombinationOptions);
        auto const transfers = solveBoltzmannTransfers(universe, k, {0.02, 1.0}, options);
        std::puts("n_eff,z,k_mpc,delta_cdm,delta_baryon,delta_cb,delta_nu,h_prime,eta,eta_prime,"
                  "theta_b_over_k,constraint,steps,rejections");
        for (auto const& transfer : transfers) {
            double const combined =
                (parameters.omegaC * transfer.deltaCdm + parameters.omegaB * transfer.deltaBaryon) /
                (parameters.omegaC + parameters.omegaB);
            std::printf("%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
                        "%.17g,%zu,%zu\n",
                parameters.neff, 1.0 / transfer.scaleFactor - 1.0, k, transfer.deltaCdm,
                transfer.deltaBaryon, combined, transfer.deltaNeutrino, transfer.hPrime,
                transfer.eta, transfer.etaPrime, transfer.thetaBaryonOverK,
                transfer.constraintResidual, transfer.acceptedSteps, transfer.rejectedSteps);
        }
    } catch (std::exception const& error) {
        std::fprintf(stderr, "Boltzmann reference failed: %s\n", error.what());
        return EXIT_FAILURE;
    }
}
