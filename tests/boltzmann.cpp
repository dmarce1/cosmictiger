#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <cosmictiger/cosmology/boltzmann.hpp>
#include <cosmictiger/cosmology/native_recombination.hpp>

namespace detail {

void require(bool condition, char const* label) {
    if (!condition) {
        throw std::runtime_error(label);
    }
}

void close(double actual, double expected, double tolerance, char const* label) {
    if (!std::isfinite(actual) ||
        std::abs(actual - expected) > tolerance * std::max(std::abs(expected), 1.e-30)) {
        throw std::runtime_error(std::string(label) + ": actual=" + std::to_string(actual) +
            ", expected=" + std::to_string(expected));
    }
}

template <class Function>
void rejects(Function function, char const* label) {
    try {
        function();
    } catch (std::exception const&) {
        return;
    }
    throw std::runtime_error(label);
}

CosmicParameters parameters(double neff) {
    CosmicParameters params{};
    params.omegaB = 0.049389;
    params.omegaC = 0.26503;
    params.neff = neff;
    params.y = 0.2454006;
    params.theta = 2.7255 / constants::kTcmb;
    params.hubble = 0.6732;
    double const critical =
        3.0 * std::pow(params.hubble * constants::kH0, 2) / (8.0 * std::acos(-1.0) * constants::kG);
    params.omegaGam =
        4.0 * constants::kSigma * std::pow(2.7255, 4) / (std::pow(constants::kC, 3) * critical);
    params.omegaNu = params.omegaGam * neff * 7.0 / 8.0 * std::pow(4.0 / 11.0, 4.0 / 3.0);
    return params;
}

void testInitialConditions(ZeroOrderUniverse const& universe) {
    auto const initial = solveBoltzmannTransfers(universe, 0.1, {universe.amin}).front();
    double const rnu =
        universe.params.omegaNu / (universe.params.omegaGam + universe.params.omegaNu);
    close(initial.deltaBaryon / initial.deltaCdm, 1.0, 1.e-13, "Adiabatic baryon initial mode");
    close(initial.deltaNeutrino / initial.deltaCdm, 4.0 / 3.0, 1.e-13,
        "Adiabatic neutrino initial mode");
    close(initial.neutrinoShear / initial.deltaCdm, -8.0 / (3.0 * (15.0 + 4.0 * rnu)), 1.e-13,
        "MB (96) shear is half the quadrupole, not twice it");
    require(std::abs(initial.constraintResidual) < 1.e-14, "Initial Einstein constraint");
    require(initial.acceptedSteps == 0, "Initial output requires no evolution");
}

void testConvergence(ZeroOrderUniverse const& universe) {
    BoltzmannSolverOptions fine;
    fine.relativeTolerance = 1.e-8;
    fine.maxMultipole = 128;
    for (double const k : {1.e-6, 1.e-4, 0.01, 0.1, 1.0, 10.0}) {
        auto const ordinary = solveBoltzmannTransfers(universe, k, {0.02, 1.0});
        auto const refined = solveBoltzmannTransfers(universe, k, {0.02, 1.0}, fine);
        for (size_t epoch = 0; epoch < ordinary.size(); ++epoch) {
            close(ordinary[epoch].deltaCdm, refined[epoch].deltaCdm, 5.e-4,
                "CDM transfer convergence");
            close(ordinary[epoch].deltaBaryon, refined[epoch].deltaBaryon, 5.e-4,
                "Baryon transfer convergence");
            close(ordinary[epoch].eta, refined[epoch].eta, 5.e-4, "Metric convergence");
            require(std::abs(ordinary[epoch].constraintResidual) < 2.e-5,
                "Independently integrated Einstein trace agrees with the constraint");
            require(refined[epoch].acceptedSteps > ordinary[epoch].acceptedSteps,
                "Tighter tolerance refines the evolution");
            require(ordinary[epoch].acceptedSteps < 20000,
                "Scaled curvature avoids superhorizon ill-conditioning");
        }
        // Signed transfers share the MB eta=+1 convention and grow in magnitude.
        require(ordinary[0].deltaCdm < 0.0 && ordinary[1].deltaCdm < ordinary[0].deltaCdm &&
                ordinary[1].deltaBaryon < ordinary[0].deltaBaryon,
            "Growing matter mode");
    }
    // CAMB 1.6.5, massless Neff=3.046, h=.6732, Ob=.049389, Oc=.26503,
    // T=2.7255K, Y=.2454006, no reionization, eta convention reversed.
    // This broad regression catches sign/unit mistakes; the full independent
    // accuracy campaign uses native recombination and tests/camb_reference.py.
    auto const transfer = solveBoltzmannTransfers(universe, 0.1, {0.02, 1.0});
    close(transfer[0].deltaCdm, -428.45904541015625, 0.002, "CAMB early CDM transfer");
    close(transfer[0].deltaBaryon, -363.6638488769531, 0.002, "CAMB early baryon transfer");
    close(transfer[1].deltaCdm, -16327.255859375, 0.002, "CAMB present CDM transfer");
    close(transfer[1].deltaBaryon, -16252.814453125, 0.002, "CAMB present baryon transfer");
}

void testInvalidInput(ZeroOrderUniverse universe) {
    rejects([&] { solveBoltzmannTransfers(universe, 0.0, {1.0}); }, "Reject zero k");
    rejects(
        [&] { solveBoltzmannTransfers(universe, std::numeric_limits<double>::infinity(), {1.0}); },
        "Reject infinite k");
    rejects(
        [&] { solveBoltzmannTransfers(universe, std::numeric_limits<double>::quiet_NaN(), {1.0}); },
        "Reject NaN k");
    rejects([&] { solveBoltzmannTransfers(universe, 1.e-200, {1.0}); }, "Reject underflowing k");
    rejects([&] { solveBoltzmannTransfers(universe, 0.1, {1.0, 0.5}); }, "Reject unordered epochs");
    rejects([&] { solveBoltzmannTransfers(universe, 0.1, {2.0}); }, "Reject late extrapolation");
    rejects([&] { solveBoltzmannTransfers(universe, 0.1, {0.5 * universe.amin}); },
        "Reject early extrapolation");
    BoltzmannSolverOptions options;
    options.maxMultipole = 2;
    rejects(
        [&] { solveBoltzmannTransfers(universe, 0.1, {1.0}, options); }, "Reject short hierarchy");
    options = {};
    options.maxSteps = 1;
    rejects([&] { solveBoltzmannTransfers(universe, 0.1, {1.0}, options); }, "Honor step budget");
    auto broken = universe;
    broken.sigmaT.values.clear();
    rejects([&] { solveBoltzmannTransfers(broken, 0.1, {1.0}); }, "Reject empty opacity table");
    broken = universe;
    broken.cs2.values.front() = std::numeric_limits<double>::infinity();
    rejects([&] { solveBoltzmannTransfers(broken, 0.1, {1.0}); }, "Reject infinite sound speed");
    universe.params.hubble = std::numeric_limits<double>::infinity();
    rejects([&] { solveBoltzmannTransfers(universe, 0.1, {1.0}); }, "Reject infinite cosmology");
}

void testInitialTime(ZeroOrderUniverse const& universe) {
    auto later = universe;
    later.amin *= 2.0;
    BoltzmannSolverOptions options;
    options.relativeTolerance = 1.e-8;
    options.maxMultipole = 128;
    double maximum = 0.0;
    for (double const k : {1.0, 10.0}) {
        auto const early = solveBoltzmannTransfers(universe, k, {0.02, 1.0}, options);
        auto const late = solveBoltzmannTransfers(later, k, {0.02, 1.0}, options);
        for (size_t epoch = 0; epoch < early.size(); ++epoch) {
            close(late[epoch].deltaCdm, early[epoch].deltaCdm, 3.e-4,
                "Finite initial-time CDM convergence");
            close(late[epoch].deltaBaryon, early[epoch].deltaBaryon, 3.e-4,
                "Finite initial-time baryon convergence");
            maximum =
                std::max({maximum, std::abs(late[epoch].deltaCdm / early[epoch].deltaCdm - 1.0),
                    std::abs(late[epoch].deltaBaryon / early[epoch].deltaBaryon - 1.0)});
        }
    }
    std::printf("Initial scale factor doubled: maximum matter transfer change %.9g\n", maximum);
}

}    // namespace detail

TEST(Boltzmann, InitialConditions) {
    ZeroOrderUniverse universe;
    createNativeZeroOrderUniverse(&universe, 1.1, detail::parameters(3.046));
    detail::testInitialConditions(universe);
}

TEST(Boltzmann, Convergence) {
    ZeroOrderUniverse universe;
    createNativeZeroOrderUniverse(&universe, 1.1, detail::parameters(3.046));
    detail::testConvergence(universe);
}

TEST(Boltzmann, InitialTime) {
    ZeroOrderUniverse universe;
    createNativeZeroOrderUniverse(&universe, 1.1, detail::parameters(3.046));
    detail::testInitialTime(universe);
}

TEST(Boltzmann, InvalidInput) {
    ZeroOrderUniverse universe;
    createNativeZeroOrderUniverse(&universe, 1.1, detail::parameters(3.046));
    detail::testInvalidInput(universe);
}

TEST(Boltzmann, PhotonOnly) {
    ZeroOrderUniverse universe;
    createNativeZeroOrderUniverse(&universe, 1.1, detail::parameters(0.0));
    auto const photonOnly = solveBoltzmannTransfers(universe, 0.1, {1.0}).front();
    EXPECT_EQ(photonOnly.deltaNeutrino, 0.0);
    EXPECT_EQ(photonOnly.neutrinoShear, 0.0);
    EXPECT_LT(std::abs(photonOnly.constraintResidual), 2.e-5);
}
