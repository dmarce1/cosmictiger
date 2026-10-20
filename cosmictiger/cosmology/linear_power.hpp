/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello
 SPDX-License-Identifier: BSL-1.0
 */

#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace linear_power {

namespace detail {

    inline bool isFinite(double value) {
        return (std::bit_cast<std::uint64_t>(value) & UINT64_C(0x7ff0000000000000)) !=
            UINT64_C(0x7ff0000000000000);
    }

}    // namespace detail

// Interpolate in log(k), optionally also in log(value). Bounds are checked before
// accessing a stencil, including at either endpoint and on nonuniform input grids.
class LogarithmicTable {
    std::vector<double> logWavenumbers_;
    std::vector<double> values_;
    bool logarithmicValues_{};

public:
    LogarithmicTable() = default;

    LogarithmicTable(std::vector<double> const& wavenumbers, std::vector<double> values,
        bool logarithmicValues = false)
      : values_(std::move(values))
      , logarithmicValues_(logarithmicValues) {
        if (wavenumbers.size() < 2 || values_.size() != wavenumbers.size()) {
            throw std::invalid_argument("A spectrum requires at least two matching samples");
        }
        for (size_t i = 0; i < wavenumbers.size(); ++i) {
            if (!detail::isFinite(wavenumbers[i]) || !(wavenumbers[i] > 0.0) ||
                (i && !(wavenumbers[i] > wavenumbers[i - 1])) || !detail::isFinite(values_[i]) ||
                (logarithmicValues_ && !(values_[i] > 0.0))) {
                throw std::invalid_argument("Invalid or unordered spectrum samples");
            }
            logWavenumbers_.push_back(std::log(wavenumbers[i]));
            if (logarithmicValues_) {
                values_[i] = std::log(values_[i]);
            }
        }
    }

    bool empty() const {
        return values_.empty();
    }

    double minimumWavenumber() const {
        return std::exp(logWavenumbers_.front());
    }

    double maximumWavenumber() const {
        return std::exp(logWavenumbers_.back());
    }

    double operator()(double wavenumber) const {
        if (empty() || !detail::isFinite(wavenumber) || !(wavenumber > 0.0)) {
            throw std::invalid_argument("Invalid spectrum wavenumber or empty spectrum");
        }
        double coordinate = std::log(wavenumber);
        // Fourier wavenumbers are currently formed in float. Permit only their
        // rounding at a table boundary, never an unbounded extrapolation.
        constexpr double kBoundaryTolerance = 4.0 * std::numeric_limits<float>::epsilon();
        if (coordinate < logWavenumbers_.front() - kBoundaryTolerance ||
            coordinate > logWavenumbers_.back() + kBoundaryTolerance) {
            throw std::out_of_range("Power spectrum does not cover a requested Fourier mode");
        }
        coordinate = std::clamp(coordinate, logWavenumbers_.front(), logWavenumbers_.back());
        size_t const count = std::min(size_t(4), values_.size());
        auto const upper =
            std::upper_bound(logWavenumbers_.begin(), logWavenumbers_.end(), coordinate);
        size_t const interval =
            upper == logWavenumbers_.begin() ? 0 : size_t(upper - logWavenumbers_.begin() - 1);
        size_t const begin = std::min(interval ? interval - 1 : 0, values_.size() - count);
        double value = 0.0;
        for (size_t i = begin; i < begin + count; ++i) {
            double weight = 1.0;
            for (size_t j = begin; j < begin + count; ++j) {
                if (j != i) {
                    weight *= (coordinate - logWavenumbers_[j]) /
                        (logWavenumbers_[i] - logWavenumbers_[j]);
                }
            }
            value += weight * values_[i];
        }
        return logarithmicValues_ ? std::exp(value) : value;
    }
};

inline double topHatWindow(double argument) {
    double const squared = argument * argument;
    if (std::abs(argument) < 0.05) {
        return 1.0 +
            squared *
            (-1.0 / 10.0 +
                squared * (1.0 / 280.0 + squared * (-1.0 / 15120.0 + squared / 1330560.0)));
    }
    return 3.0 * (std::sin(argument) - argument * std::cos(argument)) / (argument * squared);
}

inline double sigmaSquared(LogarithmicTable const& power, double radius) {
    if (!detail::isFinite(radius) || !(radius > 0.0) || power.empty()) {
        throw std::invalid_argument("Invalid sigma radius or empty power spectrum");
    }
    constexpr int kIntervals = 16384;
    double const lower = std::log(power.minimumWavenumber());
    double const upper = std::log(power.maximumWavenumber());
    double const step = (upper - lower) / kIntervals;
    double integral = 0.0;
    for (int i = 0; i <= kIntervals; ++i) {
        double const wavenumber = std::exp(lower + i * step);
        double const window = topHatWindow(wavenumber * radius);
        double const weight = (i == 0 || i == kIntervals) ? 1.0 : (i % 2 ? 4.0 : 2.0);
        integral +=
            weight * wavenumber * wavenumber * wavenumber * power(wavenumber) * window * window;
    }
    double const pi = std::acos(-1.0);
    return integral * step / (6.0 * pi * pi);
}

inline double normalization(LogarithmicTable const& power, double sigma8, double hubble) {
    if (!detail::isFinite(sigma8) || !(sigma8 > 0.0) || !detail::isFinite(hubble) ||
        !(hubble > 0.0)) {
        throw std::invalid_argument("Invalid sigma8 or Hubble parameter");
    }
    double const variance = sigmaSquared(power, 8.0 / hubble);
    if (!detail::isFinite(variance) || !(variance > 0.0)) {
        throw std::runtime_error("Power spectrum has invalid sigma8 variance");
    }
    return sigma8 * sigma8 / variance;
}

inline double matterTransfer(double deltaCdm, double deltaBaryon, double baryonFraction) {
    return (1.0 - baryonFraction) * deltaCdm + baryonFraction * deltaBaryon;
}

// MB (29), (66), in x=ln(a): delta'_cb=-h'/2-f_b*k/(aH)*theta_b/k.
// This derivative determines the displacement velocity; synchronous theta_c=0
// is a coordinate condition, not a prescription for zero particle velocity.
inline double matterDerivative(double hPrime, double thetaBaryonOverK,
    double wavenumberOverConformalHubble, double baryonFraction) {
    return -0.5 * hPrime - baryonFraction * wavenumberOverConformalHubble * thetaBaryonOverK;
}

}    // namespace linear_power
