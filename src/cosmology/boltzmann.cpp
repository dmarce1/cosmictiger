/*
CosmicTiger - A cosmological N-Body code
Copyright (C) 2021  Dominic C. Marcello

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
*/

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <cosmictiger/cosmology/boltzmann.hpp>

namespace detail {

// MB (1995), synchronous gauge, x=ln(a). The photon/baryon velocities are
// represented by their combined momentum and slip, avoiding subtraction of
// enormous Thomson drag coefficients in the tight-coupling limit.
enum Field {
    HPrime,
    DeltaCdm,
    DeltaBaryon,
    Velocity,
    DeltaPhoton,
    Slip,
    PhotonQuadrupole,
    Polarization0,
    Polarization1,
    Polarization2,
    DeltaNeutrino,
    VelocityNeutrino,
    NeutrinoQuadrupole,
    ScaledCurvature,
    CoreSize
};

using CoreMatrix = std::array<std::array<double, CoreSize>, CoreSize>;
constexpr std::array<int, 3> kQuadrupoles = {PhotonQuadrupole, Polarization2, NeutrinoQuadrupole};

struct LinearState {
    std::vector<double> values;
    double tau;
};

struct Background {
    double expansion;
    double eps;
    double cdm;
    double baryon;
    double photon;
    double neutrino;
    double soundSpeed2;
    double opacity;
    double baryonWeight;
    double photonWeight;
    double drag;
    double metricFriction;
};

class BoltzmannIntegrator {
    ZeroOrderUniverse const& universe_;
    double k_;
    BoltzmannSolverOptions options_;
    int tailSize_;
    int species_;
    size_t accepted_ = 0;
    size_t rejected_ = 0;

    size_t tailIndex(int species, int ell) const {
        return CoreSize + species * tailSize_ + ell - 3;
    }

    Background background(double logScale) const {
        double const a = std::exp(logScale);
        double const matter = universe_.params.omegaC + universe_.params.omegaB;
        double const radiation = universe_.params.omegaGam + universe_.params.omegaNu;
        Background bg{};
        bg.expansion = a * hubbleFunction(a, universe_.params.hubble, matter, radiation);
        bg.eps = k_ / bg.expansion;
        universe_.computeMatterFractions(bg.cdm, bg.baryon, a);
        universe_.computeRadiationFractions(bg.photon, bg.neutrino, a);
        bg.soundSpeed2 = universe_.cs2(a);
        bg.opacity = universe_.sigmaT(a);
        double const ratio = 4.0 * universe_.params.omegaGam / (3.0 * universe_.params.omegaB * a);
        bg.baryonWeight = 1.0 / (1.0 + ratio);
        bg.photonWeight = ratio / (1.0 + ratio);
        bg.drag = (1.0 + ratio) * bg.opacity;
        bg.metricFriction =
            0.5 * (bg.cdm + bg.baryon) + 2.0 * (1.0 - bg.cdm - bg.baryon - bg.photon - bg.neutrino);
        if (!(bg.opacity >= 0.0) || !(bg.soundSpeed2 >= 0.0) || !std::isfinite(bg.opacity) ||
            !std::isfinite(bg.soundSpeed2) || !std::isfinite(bg.expansion) ||
            !(bg.expansion > 0.0) || !std::isfinite(bg.eps) || !std::isfinite(bg.drag)) {
            throw std::runtime_error("Invalid Boltzmann background coefficient");
        }
        return bg;
    }

    double etaPrime(LinearState const& state, Background const& bg) const {
        return (1.5 * (bg.baryon + 4.0 * bg.photon / 3.0) * state.values[Velocity] +
                   2.0 * bg.neutrino * state.values[VelocityNeutrino]) /
            bg.eps;
    }

    CoreMatrix coreOperator(Background const& bg) const {
        CoreMatrix matrix{};
        matrix[HPrime][HPrime] = -bg.metricFriction;
        matrix[HPrime][DeltaCdm] = -3.0 * bg.cdm;
        matrix[HPrime][DeltaBaryon] = -3.0 * bg.baryon;
        matrix[HPrime][DeltaPhoton] = -6.0 * bg.photon;
        matrix[HPrime][DeltaNeutrino] = -6.0 * bg.neutrino;
        matrix[DeltaCdm][HPrime] = -0.5;
        matrix[DeltaBaryon][HPrime] = -0.5;
        matrix[DeltaBaryon][Velocity] = -bg.eps;
        matrix[DeltaBaryon][Slip] = -bg.eps * bg.photonWeight;
        matrix[DeltaPhoton][HPrime] = -2.0 / 3.0;
        matrix[DeltaPhoton][Velocity] = -4.0 * bg.eps / 3.0;
        matrix[DeltaPhoton][Slip] = 4.0 * bg.eps * bg.baryonWeight / 3.0;

        // v=(v_b+R v_gamma)/(1+R), s=v_b-v_gamma, R'= -R.
        // The R' contribution cancels the slip term in v', exactly.
        matrix[Velocity][Velocity] = -bg.baryonWeight;
        matrix[Velocity][DeltaBaryon] = bg.baryonWeight * bg.soundSpeed2 * bg.eps;
        matrix[Velocity][DeltaPhoton] = bg.photonWeight * bg.eps / 4.0;
        matrix[Velocity][PhotonQuadrupole] = -bg.photonWeight * bg.eps / 2.0;
        matrix[Slip][Velocity] = -1.0;
        matrix[Slip][DeltaBaryon] = bg.soundSpeed2 * bg.eps;
        matrix[Slip][DeltaPhoton] = -bg.eps / 4.0;
        matrix[Slip][PhotonQuadrupole] = bg.eps / 2.0;
        matrix[Slip][Slip] = -bg.photonWeight - bg.drag;

        double const etaVelocity = 1.5 * (bg.baryon + 4.0 * bg.photon / 3.0) / bg.eps;
        double const etaNeutrino = 2.0 * bg.neutrino / bg.eps;
        matrix[PhotonQuadrupole][HPrime] = 4.0 / 15.0;
        matrix[PhotonQuadrupole][Velocity] = 8.0 * bg.eps / 15.0 + 8.0 * etaVelocity / 5.0;
        matrix[PhotonQuadrupole][Slip] = -8.0 * bg.eps * bg.baryonWeight / 15.0;
        matrix[PhotonQuadrupole][VelocityNeutrino] = 8.0 * etaNeutrino / 5.0;
        matrix[PhotonQuadrupole][PhotonQuadrupole] = -0.9 * bg.opacity;
        matrix[PhotonQuadrupole][Polarization0] = bg.opacity / 10.0;
        matrix[PhotonQuadrupole][Polarization2] = bg.opacity / 10.0;
        matrix[Polarization0][Polarization1] = -bg.eps;
        matrix[Polarization0][PhotonQuadrupole] = bg.opacity / 2.0;
        matrix[Polarization0][Polarization0] = -bg.opacity / 2.0;
        matrix[Polarization0][Polarization2] = bg.opacity / 2.0;
        matrix[Polarization1][Polarization0] = bg.eps / 3.0;
        matrix[Polarization1][Polarization1] = -bg.opacity;
        matrix[Polarization1][Polarization2] = -2.0 * bg.eps / 3.0;
        matrix[Polarization2][Polarization1] = 2.0 * bg.eps / 5.0;
        matrix[Polarization2][PhotonQuadrupole] = bg.opacity / 10.0;
        matrix[Polarization2][Polarization0] = bg.opacity / 10.0;
        matrix[Polarization2][Polarization2] = -0.9 * bg.opacity;
        if (species_ == 3) {
            matrix[DeltaNeutrino][HPrime] = -2.0 / 3.0;
            matrix[DeltaNeutrino][VelocityNeutrino] = -4.0 * bg.eps / 3.0;
            matrix[VelocityNeutrino][DeltaNeutrino] = bg.eps / 4.0;
            matrix[VelocityNeutrino][NeutrinoQuadrupole] = -bg.eps / 2.0;
            matrix[NeutrinoQuadrupole][HPrime] = 4.0 / 15.0;
            matrix[NeutrinoQuadrupole][Velocity] = 8.0 * etaVelocity / 5.0;
            matrix[NeutrinoQuadrupole][VelocityNeutrino] =
                8.0 * bg.eps / 15.0 + 8.0 * etaNeutrino / 5.0;
        }
        // E=(k/Hc)^2 eta keeps curvature comparable to the fluid amplitudes
        // even far outside the horizon. Hc'/Hc=metricFriction-1.
        matrix[ScaledCurvature][ScaledCurvature] = 2.0 * (1.0 - bg.metricFriction);
        matrix[ScaledCurvature][Velocity] = bg.eps * bg.eps * etaVelocity;
        matrix[ScaledCurvature][VelocityNeutrino] = bg.eps * bg.eps * etaNeutrino;
        // Use the Hamiltonian constraint (MB 21a) for h' in the fluid
        // equations. Evolving eta with (21b) then feeds the metric back
        // into the fluid solution without subtracting an accumulated
        // O(1) curvature error from a tiny late-time eta at large k.
        // HPrime remains an independently evolved (21c+21a) trace check.
        for (int row = 1; row < CoreSize; ++row) {
            double const metric = matrix[row][HPrime];
            matrix[row][HPrime] = 0.0;
            matrix[row][ScaledCurvature] += metric * 2.0;
            matrix[row][DeltaCdm] += metric * 3.0 * bg.cdm;
            matrix[row][DeltaBaryon] += metric * 3.0 * bg.baryon;
            matrix[row][DeltaPhoton] += metric * 3.0 * bg.photon;
            matrix[row][DeltaNeutrino] += metric * 3.0 * bg.neutrino;
        }
        return matrix;
    }

    LinearState implicitStage(LinearState const& rhs, Background const& bg, double weight) const {
        LinearState result = rhs;
        result.tau += weight / bg.expansion;
        CoreMatrix matrix = coreOperator(bg);
        std::array<double, CoreSize> coreRhs{};
        for (int row = 0; row < CoreSize; ++row) {
            coreRhs[row] = rhs.values[row];
            for (int col = 0; col < CoreSize; ++col) {
                matrix[row][col] *= -weight;
            }
            matrix[row][row] += 1.0;
        }
        std::vector<double> recurrence(rhs.values.size(), 0.0);
        // Eliminate each tridiagonal tail back to its quadrupole. MB (51),(65)
        // give F_L'=eps F_(L-1)-(L+1)/(tau Hc)F_L, plus photon opacity.
        for (int species = 0; species < species_; ++species) {
            double const opacity = species < 2 ? bg.opacity : 0.0;
            for (int ell = options_.maxMultipole; ell >= 3; --ell) {
                size_t const index = tailIndex(species, ell);
                bool const last = ell == options_.maxMultipole;
                double const lower = -weight * bg.eps * (last ? 1.0 : double(ell) / (2 * ell + 1));
                double const upper = weight * bg.eps * (ell + 1) / (2 * ell + 1);
                double diagonal = 1.0 + weight * opacity;
                double value = rhs.values[index];
                if (last) {
                    diagonal += weight * (ell + 1) / (result.tau * bg.expansion);
                } else {
                    diagonal += upper * recurrence[index + 1];
                    value -= upper * result.values[index + 1];
                }
                recurrence[index] = -lower / diagonal;
                result.values[index] = value / diagonal;
            }
            int const quadrupole = kQuadrupoles[species];
            size_t const first = tailIndex(species, 3);
            double const upper = weight * 3.0 * bg.eps / 5.0;
            matrix[quadrupole][quadrupole] += upper * recurrence[first];
            coreRhs[quadrupole] -= upper * result.values[first];
        }

        // Partial-pivot elimination of the small coupled metric/fluid block.
        for (int col = 0; col < CoreSize; ++col) {
            int pivot = col;
            for (int row = col + 1; row < CoreSize; ++row) {
                if (std::abs(matrix[row][col]) > std::abs(matrix[pivot][col])) {
                    pivot = row;
                }
            }
            if (pivot != col) {
                std::swap(matrix[col], matrix[pivot]);
                std::swap(coreRhs[col], coreRhs[pivot]);
            }
            if (matrix[col][col] == 0.0) {
                throw std::runtime_error("Singular Boltzmann stage");
            }
            for (int row = col + 1; row < CoreSize; ++row) {
                double const factor = matrix[row][col] / matrix[col][col];
                for (int j = col + 1; j < CoreSize; ++j) {
                    matrix[row][j] -= factor * matrix[col][j];
                }
                coreRhs[row] -= factor * coreRhs[col];
            }
        }
        for (int row = CoreSize - 1; row >= 0; --row) {
            double value = coreRhs[row];
            for (int col = row + 1; col < CoreSize; ++col) {
                value -= matrix[row][col] * result.values[col];
            }
            result.values[row] = value / matrix[row][row];
        }
        for (int species = 0; species < species_; ++species) {
            double previous = result.values[kQuadrupoles[species]];
            for (int ell = 3; ell <= options_.maxMultipole; ++ell) {
                size_t const index = tailIndex(species, ell);
                result.values[index] += recurrence[index] * previous;
                previous = result.values[index];
            }
        }
        return result;
    }

    LinearState step(LinearState const& start, double logScale, double stepSize) const {
        // Alexander's three-stage, third-order, stiffly accurate SDIRK.
        // Its L-stability damps unresolved collision and hierarchy modes;
        // transport, metric and Thomson terms all share the same stages.
        double const gamma = 0.43586652150845899942;
        double const a21 = 0.5 * (1.0 - gamma);
        double const a31 = -1.5 * gamma * gamma + 4.0 * gamma - 0.25;
        double const a32 = 1.5 * gamma * gamma - 5.0 * gamma + 1.25;
        Background const firstBg = background(logScale + gamma * stepSize);
        Background const secondBg = background(logScale + (a21 + gamma) * stepSize);
        Background const thirdBg = background(logScale + stepSize);
        LinearState const first = implicitStage(start, firstBg, gamma * stepSize);
        LinearState secondRhs = start;
        for (size_t field = 0; field < start.values.size(); ++field) {
            secondRhs.values[field] += a21 / gamma * (first.values[field] - start.values[field]);
        }
        secondRhs.tau += a21 * stepSize / firstBg.expansion;
        LinearState const second = implicitStage(secondRhs, secondBg, gamma * stepSize);
        LinearState thirdRhs = start;
        for (size_t field = 0; field < start.values.size(); ++field) {
            thirdRhs.values[field] += a31 / gamma * (first.values[field] - start.values[field]) +
                a32 / gamma * (second.values[field] - secondRhs.values[field]);
        }
        thirdRhs.tau += stepSize * (a31 / firstBg.expansion + a32 / secondBg.expansion);
        LinearState result = implicitStage(thirdRhs, thirdBg, gamma * stepSize);
        return result;
    }

    double error(LinearState const& start, LinearState const& full, LinearState const& half) const {
        if (!std::isfinite(full.tau) || !std::isfinite(half.tau) || !(full.tau > 0.0) ||
            !(half.tau > 0.0)) {
            throw std::runtime_error("Invalid Boltzmann conformal time");
        }
        double norm = 0.0;
        // A common amplitude floor avoids resolving meaningless relative errors
        // at acoustic zeros, while resolving the growing mode from the start.
        double amplitude = 0.0;
        for (int field : {HPrime, DeltaCdm, DeltaBaryon, DeltaPhoton, DeltaNeutrino}) {
            amplitude = std::max(
                amplitude, std::max(std::abs(start.values[field]), std::abs(half.values[field])));
        }
        for (size_t field = 0; field < start.values.size(); ++field) {
            if (!std::isfinite(half.values[field]) || !std::isfinite(full.values[field])) {
                throw std::runtime_error("Nonfinite Boltzmann stage");
            }
            double const scale = options_.absoluteTolerance +
                options_.relativeTolerance *
                    std::max({std::abs(start.values[field]), std::abs(half.values[field]),
                        0.01 * amplitude});
            norm =
                std::max(norm, std::abs(half.values[field] - full.values[field]) / (7.0 * scale));
        }
        norm = std::max(
            norm, std::abs(half.tau - full.tau) / (7.0 * options_.relativeTolerance * half.tau));
        return norm;
    }

public:
    BoltzmannIntegrator(ZeroOrderUniverse const& universe, double k, BoltzmannSolverOptions options)
      : universe_(universe)
      , k_(k)
      , options_(options)
      , tailSize_(options.maxMultipole - 2)
      , species_(universe.params.omegaNu > 0.0 ? 3 : 2) {}

    std::vector<BoltzmannTransfer> solve(std::vector<double> const& scales) {
        double const radiation = universe_.params.omegaGam + universe_.params.omegaNu;
        double const matter = universe_.params.omegaB + universe_.params.omegaC;
        double const aInitial = universe_.amin;
        double logScale = std::log(aInitial);
        // Radiation+matter conformal age; Lambda is negligible at this start.
        double const tau = 2.0 * aInitial /
            (universe_.params.hubble * cosmic_constants::kH0 *
                (std::sqrt(radiation + matter * aInitial) + std::sqrt(radiation)));
        double const kt = k_ * tau;
        if (kt > 0.05 || matter * aInitial / radiation > 1.e-3) {
            throw std::invalid_argument(
                "Boltzmann background must start in the superhorizon radiation era");
        }
        double const rnu = universe_.params.omegaNu / radiation;
        // Rescale by (k tau_initial)^2 to keep initial fluid amplitudes O(1).
        // The returned transfers restore this factor and have MB C=1/2.
        double const normalization = kt * kt;
        if (!(normalization > std::numeric_limits<double>::min()) ||
            !std::isfinite(normalization)) {
            throw std::invalid_argument("Boltzmann wavenumber is outside the representable range");
        }
        double const initialEta =
            1.0 / normalization - (5.0 + 4.0 * rnu) / (12.0 * (15.0 + 4.0 * rnu));
        LinearState state{std::vector<double>(CoreSize + species_ * tailSize_, 0.0), tau};
        Background const initialBg = background(logScale);
        state.values[HPrime] = 1.0 / (tau * initialBg.expansion);
        state.values[DeltaCdm] = state.values[DeltaBaryon] = -0.25;
        state.values[DeltaPhoton] = -1.0 / 3.0;
        state.values[Velocity] = -kt / 36.0;
        if (species_ == 3) {
            state.values[DeltaNeutrino] = -1.0 / 3.0;
            state.values[VelocityNeutrino] =
                (23.0 + 4.0 * rnu) / (15.0 + 4.0 * rnu) * state.values[Velocity];
            state.values[NeutrinoQuadrupole] = 4.0 / (3.0 * (15.0 + 4.0 * rnu));
        }
        // MB (96) is a radiation-era series. At a finite start there is a
        // small matter fraction: enforce (21a) once, then normalize the
        // initial curvature back to the series value. Eta is subsequently
        // evolved independently using (21b), never projected onto (21a).
        double const initialDensity = initialBg.cdm * state.values[DeltaCdm] +
            initialBg.baryon * state.values[DeltaBaryon] +
            initialBg.photon * state.values[DeltaPhoton] +
            initialBg.neutrino * state.values[DeltaNeutrino];
        double const constrainedEta =
            (0.5 * state.values[HPrime] - 1.5 * initialDensity) / (initialBg.eps * initialBg.eps);
        double const initialAmplitude = initialEta / constrainedEta;
        for (double& value : state.values) {
            value *= initialAmplitude;
        }
        state.values[ScaledCurvature] = initialBg.eps * initialBg.eps * initialEta;
        std::vector<BoltzmannTransfer> transfers;
        double stepSize = std::min(0.01, options_.maxLogStep);
        for (double scaleFactor : scales) {
            double const target = std::log(scaleFactor);
            while (logScale < target) {
                if (accepted_ + rejected_ >= options_.maxSteps) {
                    throw std::runtime_error("Boltzmann solver exceeded its step limit");
                }
                double const stepValue = std::min(stepSize, target - logScale);
                if (logScale + stepValue == logScale) {
                    throw std::runtime_error("Boltzmann timestep underflow");
                }
                LinearState const full = step(state, logScale, stepValue);
                LinearState const firstHalf = step(state, logScale, 0.5 * stepValue);
                LinearState const half =
                    step(firstHalf, logScale + 0.5 * stepValue, 0.5 * stepValue);
                double const estimate = error(state, full, half);
                if (!std::isfinite(estimate)) {
                    throw std::runtime_error("Nonfinite Boltzmann integration error");
                }
                if (estimate <= 1.0) {
                    state = half;
                    logScale = std::min(logScale + stepValue, target);
                    ++accepted_;
                } else {
                    ++rejected_;
                }
                double const factor =
                    estimate == 0.0 ? 3.0 : std::clamp(0.85 * std::pow(estimate, -0.25), 0.2, 3.0);
                stepSize = std::min(options_.maxLogStep, stepValue * factor);
            }
            Background const bg = background(target);
            double const density = bg.cdm * state.values[DeltaCdm] +
                bg.baryon * state.values[DeltaBaryon] + bg.photon * state.values[DeltaPhoton] +
                bg.neutrino * state.values[DeltaNeutrino];
            double const constraint =
                state.values[ScaledCurvature] - 0.5 * state.values[HPrime] + 1.5 * density;
            double const denominator = std::abs(state.values[ScaledCurvature]) +
                0.5 * std::abs(state.values[HPrime]) + 1.5 * std::abs(density);
            transfers.push_back({scaleFactor, normalization * state.values[DeltaCdm],
                normalization * state.values[DeltaBaryon],
                normalization * state.values[DeltaNeutrino],
                normalization * (state.values[Velocity] + bg.photonWeight * state.values[Slip]),
                normalization * (2.0 * state.values[ScaledCurvature] + 3.0 * density),
                normalization * state.values[ScaledCurvature] / (bg.eps * bg.eps),
                normalization * etaPrime(state, bg),
                0.5 * normalization * state.values[PhotonQuadrupole],
                0.5 * normalization * state.values[NeutrinoQuadrupole], state.tau,
                constraint / std::max(denominator, std::numeric_limits<double>::min()), accepted_,
                rejected_});
        }
        return transfers;
    }
};

}    // namespace detail

std::vector<BoltzmannTransfer> solveBoltzmannTransfers(ZeroOrderUniverse const& universe, double k,
    std::vector<double> const& scaleFactors, BoltzmannSolverOptions const& options) {
    auto const& params = universe.params;
    if (!(k > 0.0) || !std::isfinite(k) || !(params.omegaB > 0.0) || !(params.omegaC >= 0.0) ||
        !(params.omegaGam > 0.0) || !(params.omegaNu >= 0.0) || !(params.hubble > 0.0) ||
        !(universe.amin > 0.0) || !(universe.amax >= universe.amin) ||
        !std::isfinite(params.omegaB) || !std::isfinite(params.omegaC) ||
        !std::isfinite(params.omegaGam) || !std::isfinite(params.omegaNu) ||
        !std::isfinite(params.hubble) || !std::isfinite(universe.amin) ||
        !std::isfinite(universe.amax)) {
        throw std::invalid_argument("Invalid Boltzmann cosmology or wavenumber");
    }
    auto const validTable = [&](InterpolationFunctor<double> const& table) {
        if (table.values.size() < 4 || table.values.size() != size_t(table.n) ||
            !std::isfinite(table.amin) || !std::isfinite(table.amax) ||
            !std::isfinite(table.minloga) || !std::isfinite(table.maxloga) ||
            !std::isfinite(table.dloga) || !(table.dloga > 0.0) || table.amin > universe.amin ||
            table.amax < universe.amax || table.minloga > std::log(universe.amin) ||
            table.maxloga < std::log(universe.amax)) {
            return false;
        }
        return std::all_of(table.values.begin(), table.values.end(),
            [](double value) { return std::isfinite(value) && value >= 0.0; });
    };
    if (!validTable(universe.sigmaT) || !validTable(universe.cs2)) {
        throw std::invalid_argument("Invalid Boltzmann background interpolation table");
    }
    if (!(options.relativeTolerance > 0.0) || !(options.absoluteTolerance > 0.0) ||
        !(options.maxLogStep > 0.0) || !std::isfinite(options.relativeTolerance) ||
        !std::isfinite(options.absoluteTolerance) || !std::isfinite(options.maxLogStep) ||
        options.maxMultipole < 3 || options.maxMultipole > 4096 || options.maxSteps == 0) {
        throw std::invalid_argument("Invalid Boltzmann integration options");
    }
    double previous = universe.amin;
    for (double scale : scaleFactors) {
        if (!std::isfinite(scale) || scale < previous || scale > universe.amax) {
            throw std::invalid_argument(
                "Boltzmann output scales must be ordered and inside the background");
        }
        previous = scale;
    }
    return detail::BoltzmannIntegrator(universe, k, options).solve(scaleFactors);
}
