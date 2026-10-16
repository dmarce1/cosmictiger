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

#include <cosmictiger/analytic.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/particles.hpp>
#include <cosmictiger/timer.hpp>

using ReturnType = std::pair<Vector<double>, Array<Vector<double>, NDIM>>;

static ReturnType doAnalytic(
    Vector<Fixed32> const& sinkx, Vector<Fixed32> const& sinky, Vector<Fixed32> const& sinkz);

HPX_PLAIN_ACTION(doAnalytic, DoAnalyticAction);

using ReturnType = std::pair<Vector<double>, Array<Vector<double>, NDIM>>;

static ReturnType doAnalytic(
    Vector<Fixed32> const& sinkx, Vector<Fixed32> const& sinky, Vector<Fixed32> const& sinkz) {
#ifdef USE_CUDA
    Vector<hpx::future<ReturnType>> futs;
    for (auto c : hpxChildren()) {
        futs.push_back(hpx::async<DoAnalyticAction>(c, sinkx, sinky, sinkz));
    }

    auto results = gravityAnalyticCallKernel(sinkx, sinky, sinkz);

    for (auto& f : futs) {
        auto other = f.get();
        for (int i = 0; i < sinkx.size(); i++) {
            results.first[i] += other.first[i];
            for (int dim = 0; dim < NDIM; dim++) {
                results.second[dim][i] += other.second[dim][i];
            }
        }
    }
    return results;
#else
    return ReturnType();
#endif
}

Pair<double> analyticCompare(int numberSamples) {
#ifdef USE_CUDA
    Timer tm;
    tm.start();
    auto samples = particlesSample(numberSamples);
    tm.stop();
    PRINT("particles_sample: %e\n", tm.read());
    Vector<Fixed32> sinkx(numberSamples);
    Vector<Fixed32> sinky(numberSamples);
    Vector<Fixed32> sinkz(numberSamples);
    for (int i = 0; i < numberSamples; i++) {
        sinkx[i] = samples[i].x[XDIM];
        sinky[i] = samples[i].x[YDIM];
        sinkz[i] = samples[i].x[ZDIM];
    }
    double const gm = getOptions().gm;
    auto results = doAnalytic(sinkx, sinky, sinkz);
    Vector<double> gerrs, phierrs;
    for (int i = 0; i < numberSamples; i++) {
        double f1 = 0.0, f2 = 0.0;
        double g1 = 0.0, g2 = 0.0;
        for (int dim = 0; dim < NDIM; dim++) {
            g1 += sqr(samples[i].g[dim]);
            g2 += sqr(gm * results.second[dim][i]);
        }
        g1 = sqrt(g1);
        g2 = sqrt(g2);
        f1 = samples[i].p;
        f2 = gm * results.first[i];
        double const gerr = fabs(g1 - g2) / g2;
        double const phierr = fabs(f1 - f2) / fabs(f2);
        gerrs.push_back(gerr);
        phierrs.push_back(phierr);
        printf("%.10e %.10e %.10e | %.10e %.10e %.10e |%.10e %.10e %.10e \n", sinkx[i].toFloat(),
            sinky[i].toFloat(), sinkz[i].toFloat(), g1, g2, g2 / g1, f1, f2, f1 / f2);
    }
    int index = 99 * numberSamples / 100;
    std::sort(phierrs.begin(), phierrs.end());
    std::sort(gerrs.begin(), gerrs.end());
    double phi100 = phierrs[index];
    double force100 = gerrs[index];
    double phi50 = phierrs[numberSamples / 2] + phierrs[numberSamples / 2 + 1];
    double force50 = gerrs[numberSamples / 2] + gerrs[numberSamples / 2 + 1];
    phi50 *= 0.5;
    force50 *= 0.5;
    PRINT("Force 50  Error     = %e\n", force50);
    PRINT("Force 99  Error     = %e\n", force100);
    PRINT("Potential 50  Error = %e\n", phi50);
    PRINT("Potential 99  Error = %e\n", phi100);
    Pair<double> rc;
    rc.first = force50;
    rc.second = force100;
    return rc;
#else
    PRINT("analytic compare not available without CUDA\n");
#endif
}
