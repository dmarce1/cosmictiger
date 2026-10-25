/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2021  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/core/timer.hpp>
#include <tests/gravity/analytic.hpp>
#include <cosmictiger/math/math.hpp>
#include <cosmictiger/particles/particles.hpp>

using ReturnType = std::pair<Vector<double>, Array<Vector<double>, NDIM>>;

static ReturnType doAnalytic(
    Vector<Position> const& sinkx, Vector<Position> const& sinky, Vector<Position> const& sinkz);

HPX_PLAIN_ACTION(doAnalytic, DoAnalyticAction);

using ReturnType = std::pair<Vector<double>, Array<Vector<double>, NDIM>>;

static ReturnType doAnalytic(
    Vector<Position> const& sinkx, Vector<Position> const& sinky, Vector<Position> const& sinkz) {
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

Pair<double> analyticCompare(int numberOfSamples) {
#ifdef USE_CUDA
    Timer tm;
    tm.start();
    auto samples = particlesSample(numberOfSamples);
    tm.stop();
    PRINT("particles_sample: %e\n", tm.read());
    Vector<Position> sinkx(numberOfSamples);
    Vector<Position> sinky(numberOfSamples);
    Vector<Position> sinkz(numberOfSamples);
    for (int i = 0; i < numberOfSamples; i++) {
        sinkx[i] = samples[i].x[XDIM];
        sinky[i] = samples[i].x[YDIM];
        sinkz[i] = samples[i].x[ZDIM];
    }
    double const gm = getOptions().gravity.gm;
    auto results = doAnalytic(sinkx, sinky, sinkz);
    Vector<double> gerrs, phierrs;
    for (int i = 0; i < numberOfSamples; i++) {
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
    int index = 99 * numberOfSamples / 100;
    std::sort(phierrs.begin(), phierrs.end());
    std::sort(gerrs.begin(), gerrs.end());
    double phi100 = phierrs[index];
    double force100 = gerrs[index];
    double phi50 = phierrs[numberOfSamples / 2] + phierrs[numberOfSamples / 2 + 1];
    double force50 = gerrs[numberOfSamples / 2] + gerrs[numberOfSamples / 2 + 1];
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
