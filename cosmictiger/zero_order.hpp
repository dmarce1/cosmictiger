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

#pragma once

#include <functional>

#include <cosmictiger/constants.hpp>
#include <cosmictiger/interpolation.hpp>

struct CosmicParameters {
    double omegaB, omegaC, omegaGam, omegaNu;
    double neff, y, theta, hubble;
};

inline double hubbleFunction(double a, double littleh, double omegaM, double omegaR) {
    return littleh * cosmic_constants::kH0 *
        sqrt(omegaR / (a * a * a * a) + omegaM / (a * a * a) + (1 - omegaR - omegaM));
}

struct ZeroOrderUniverse {
    double amin;
    double amax;
    CosmicParameters params;
    InterpolationFunctor<double> sigmaT;
    InterpolationFunctor<double> cs2;
    InterpolationFunctor<double> k;
    std::function<double(double)> xe;

    void computeMatterFractions(double& oc, double& ob, double a) const;

    void computeRadiationFractions(double& ogam, double& onu, double a) const;

    double conformalTimeToScaleFactor(double taumax);

    double scaleFactorToConformalTime(double a);

    double redshiftToTime(double z) const;

    double redshiftToDensity(double z) const;
};

void createZeroOrderUniverse(
    ZeroOrderUniverse* uniPtr, std::function<double(double)> fxe, double amax, CosmicParameters);

class Cosmos {
    double a_;
    double t_;
    double tau_;
    double omegaM_;
    double omegaR_;
    double omegaLam_;
    double hValue_;

public:
    Cosmos(
        double omegaC, double omegaB, double omegaGam, double omegaNu, double hArg, double aArg) {
        hValue_ = constants::kH0 * hArg;
        omegaM_ = omegaC + omegaB;
        omegaR_ = omegaGam + omegaNu;
        omegaLam_ = 1.0 - omegaR_ - omegaM_;
        a_ = aArg;
        t_ = tau_ = 0.0;
    }
    double advance(double dtau0) {
        double const beta[3] = {1.0, 0.25, 2.0 / 3.0};

        double tau1 = tau_ + dtau0;
        while (tau_ < tau1) {
            double a0 = a_;
            double t0 = t_;
            double dtau = std::min(std::abs(1.0e-2 / a_ /
                                       (hValue_ *
                                           sqrt(omegaM_ / (a_ * a_ * a_) +
                                               omegaR_ / (a_ * a_ * a_ * a_) + omegaLam_))) *
                    0.01,
                tau1 - tau_);
            for (int rk = 0; rk < 3; rk++) {
                double da = (hValue_ *
                                sqrt(omegaM_ / (a_ * a_ * a_) + omegaR_ / (a_ * a_ * a_ * a_) +
                                    omegaLam_)) *
                    a_ * a_ * dtau;
                double dt = a_ * dtau;
                a_ = (1.0 - beta[rk]) * a0 + beta[rk] * (a_ + da);
                t_ = (1.0 - beta[rk]) * t0 + beta[rk] * (t_ + dt);
            }
            tau_ += dtau;
        }
        return t_;
    }
    double scale() const {
        return a_;
    }
    double time() const {
        return t_;
    }
};
