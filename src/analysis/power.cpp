/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2021  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <algorithm>
#include <cmath>
#include <complex>

#include <cosmictiger/analysis/power.hpp>
#include <cosmictiger/analysis/power_cpu.hpp>
#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/cosmology/fft.hpp>
#include <cosmictiger/math/math.hpp>
#include <cosmictiger/math/simulation_units.hpp>
#include <cosmictiger/particles/particles.hpp>

static void computeDensityFolded(int m);

HPX_PLAIN_ACTION(computeDensityFolded, ComputeDensityFoldedAction);

struct PowerSpectrumReturn {
    Vector<float> k;
    Vector<float> p;
    Vector<double> c;
    Vector<float> pdev;
};
static PowerSpectrumReturn powerSpectrumBin(Vector<float> const& kbins);
static PowerSpectrumReturn powerSpectrumBinDeviation(
    PowerSpectrumReturn, Vector<float> const& kbins);

HPX_PLAIN_ACTION(powerSpectrumBin, PowerSpectrumBinAction);
HPX_PLAIN_ACTION(powerSpectrumBinDeviation, PowerSpectrumBinDeviationAction);

static PowerSpectrumT powerSpectrumFromTransform() {
    Vector<float> kbins;
    double const boxSize =
        (units::scalesFromOptions(getOptions()).lengthScale() / units::kMegaparsec).value();
    int const nfour = getOptions().initial.fourierSize;
    double const ns = std::max(nfour / 8.0, 64.0);
    double const kmin = 0.999999;
    double const kmax = nfour / 2.0;
    double const logkmin = log(kmin);
    double const logkmax = log(kmax);
    double const dlogk = (logkmax - logkmin) / ns;
    for (int i = 0; i <= ns; i++) {
        double logk = logkmin + dlogk * i;
        kbins.push_back(exp(logk));
    }
    auto pr = powerSpectrumBin(kbins);
    auto const& count = pr.c;
    auto& power = pr.p;
    auto& ks = pr.k;
    PowerSpectrumT ret;
    ret.p.resize(kbins.size() - 1);
    ret.k.resize(kbins.size() - 1);
    ret.perr.resize(kbins.size() - 1, 0.0);
    for (int i = 0; i < power.size(); i++) {
        if (count[i] > 0) {
            power[i] /= count[i];
            ks[i] /= count[i];
        }
        ks[i] = sqrt(kbins[i] * kbins[i + 1]);
        ks[i] *= 2.0 * M_PI / boxSize;
        ret.p[i] = power[i];
        ret.k[i] = ks[i];
    }
    pr = powerSpectrumBinDeviation(pr, kbins);
    for (int i = 0; i < power.size(); i++) {
        if (pr.c[i] > 0) {
            if (count[i] > 1) {
                ret.perr[i] = sqrt(pr.pdev[i]) / sqrt(count[i] - 1) / sqrt(count[i]);
            } else {
                ret.p[i] = 0.0;
                ret.perr[i] = 0.0;
            }
        }
    }

    return ret;
}

static int searchKbins(Vector<float> kbins, double k, int binmin, int binmax) {
    if (k < 0.0 || k >= kbins.back()) {
        return -1;
    } else {
        if (binmax - binmin == 1) {
            ALWAYS_ASSERT(k >= kbins[binmin]);
            ALWAYS_ASSERT(k < kbins[binmin + 1]);
            return binmin;
        } else {
            int binmid = (binmax + binmin) / 2;
            if (k >= kbins[binmid]) {
                return searchKbins(kbins, k, binmid, binmax);
            } else {
                return searchKbins(kbins, k, binmin, binmid);
            }
        }
    }
    return -1;
};

static PowerSpectrumReturn powerSpectrumBin(Vector<float> const& kbins) {
    Vector<hpx::future<PowerSpectrumReturn>> futs;
    for (auto c : hpxChildren()) {
        futs.push_back(hpx::async<PowerSpectrumBinAction>(c, kbins));
    }
    int64_t const n = getOptions().initial.particlesPerDimension;
    float const boxSize =
        (units::scalesFromOptions(getOptions()).lengthScale() / units::kMegaparsec).value();
    auto const box = fft3dComplexRange();
    Vector<ComplexFloat>& yValue = fft3dComplexVector();
    Array<int64_t, NDIM> iValue;

    int const numberOfBins = kbins.size() - 1;
    Vector<float> power(numberOfBins, 0.0);
    Vector<double> count(numberOfBins, 0);
    Vector<float> ktot(numberOfBins, 0.0);
    for (iValue[0] = box.begin[0]; iValue[0] != box.end[0]; iValue[0]++) {
        futs.push_back(hpx::async(
            [n, numberOfBins, &kbins, box, &yValue](Array<int64_t, NDIM> iValue2) {
                int64_t const i = iValue2[0] < n / 2 ? iValue2[0] : iValue2[0] - n;
                double const sx = cloudFilter(2.0 * M_PI * i / n);
                PowerSpectrumReturn ret;
                Vector<float> power(numberOfBins, 0.0);
                Vector<double> count(numberOfBins, 0);
                Vector<float> ktot(numberOfBins, 0.0);
                for (iValue2[1] = box.begin[1]; iValue2[1] != box.end[1]; iValue2[1]++) {
                    int64_t const j = iValue2[1] < n / 2 ? iValue2[1] : iValue2[1] - n;
                    double const sy = cloudFilter(2.0 * M_PI * j / n);
                    for (iValue2[2] = box.begin[2]; iValue2[2] != box.end[2]; iValue2[2]++) {
                        int64_t const k = iValue2[2] < n / 2 ? iValue2[2] : iValue2[2] - n;
                        double const sz = cloudFilter(2.0 * M_PI * k / n);
                        double const c0 = sx * sy * sz;
                        yValue[box.index(iValue2)] *= c0;
                        double const iii = std::sqrt(i * i + j * j + k * k);
                        if (iii > 0) {
                            int const bin = searchKbins(kbins, iii, 0, numberOfBins);
                            if (bin >= 0) {
                                float const pwr = std::norm(yValue[box.index(iValue2)]);
                                count[bin] += 1;
                                power[bin] += pwr;
                                ktot[bin] += iii;
                            }
                        }
                    }
                }
                ret.p = std::move(power);
                ret.c = std::move(count);
                ret.k = std::move(ktot);
                return ret;
            },
            iValue));
    }
    for (auto& f : futs) {
        auto const pr = f.get();
        auto const& thisCount = pr.c;
        auto const& thisPower = pr.p;
        auto const& thisKtot = pr.k;
        for (int i = 0; i < thisCount.size(); i++) {
            count[i] += thisCount[i];
            power[i] += thisPower[i];
            ktot[i] += thisKtot[i];
        }
    }
    PowerSpectrumReturn rc;
    rc.p = std::move(power);
    rc.c = std::move(count);
    rc.k = std::move(ktot);
    return rc;
}

static PowerSpectrumReturn powerSpectrumBinDeviation(
    PowerSpectrumReturn rc, Vector<float> const& kbins) {
    Vector<hpx::future<PowerSpectrumReturn>> futs;
    for (auto c : hpxChildren()) {
        futs.push_back(hpx::async<PowerSpectrumBinDeviationAction>(c, rc, kbins));
    }
    int64_t const n = getOptions().initial.particlesPerDimension;
    float const boxSize =
        (units::scalesFromOptions(getOptions()).lengthScale() / units::kMegaparsec).value();
    auto const box = fft3dComplexRange();
    Vector<ComplexFloat>& yValue = fft3dComplexVector();
    Array<int64_t, NDIM> iValue;

    int const numberOfBins = kbins.size() - 1;
    Vector<float> devs(numberOfBins, 0.0);
    for (iValue[0] = box.begin[0]; iValue[0] != box.end[0]; iValue[0]++) {
        futs.push_back(hpx::async(
            [n, numberOfBins, &kbins, box, rc, &yValue](Array<int64_t, NDIM> iValue2) {
                int64_t const i = iValue2[0] < n / 2 ? iValue2[0] : iValue2[0] - n;
                double const sx = sinc(M_PI * i / n);
                PowerSpectrumReturn ret;
                Vector<float> pdev(numberOfBins, 0.0);
                for (iValue2[1] = box.begin[1]; iValue2[1] != box.end[1]; iValue2[1]++) {
                    int64_t const j = iValue2[1] < n / 2 ? iValue2[1] : iValue2[1] - n;
                    double const sy = sinc(M_PI * j / n);
                    for (iValue2[2] = box.begin[2]; iValue2[2] != box.end[2]; iValue2[2]++) {
                        int64_t const k = iValue2[2] < n / 2 ? iValue2[2] : iValue2[2] - n;
                        double const sz = sinc(M_PI * k / n);
                        double const c0 = pow(1.0 / (sx * sy * sz), 3);
                        yValue[box.index(iValue2)] *= c0;
                        double const iii = std::sqrt(i * i + j * j + k * k);
                        if (iii > 0) {
                            int const bin = searchKbins(kbins, iii, 0, numberOfBins);
                            if (bin >= 0) {
                                float const pwr = std::norm(yValue[box.index(iValue2)]);
                                pdev[bin] += sqr(pwr - rc.p[bin]);
                            }
                        }
                    }
                }
                ret.pdev = std::move(pdev);
                return ret;
            },
            iValue));
    }
    for (auto& f : futs) {
        auto const pr = f.get();
        auto const& thisPdev = pr.pdev;
        for (int i = 0; i < numberOfBins; i++) {
            devs[i] += thisPdev[i];
        }
    }
    rc.pdev = std::move(devs);
    return rc;
}

PowerSpectrumT powerSpectrumCompute(int m) {
    int const n = getOptions().initial.fourierSize;
    fft3dInit(n, -1.0f);
    computeDensityFolded(m);
    fft3dExecute();
    auto power = powerSpectrumFromTransform();
    for (int i = 0; i < power.k.size(); i++) {
        power.k[i] *= m;
    }
    fft3dDestroy();
    return power;
}

Vector<float> accumulateDensityCpu(int fold, int gridSize, Range<int64_t> box) {
    Vector<float> density(box.volume(), 0.0f);
    size_t const particleCount = particlesSize();
    if (particleCount == 0) {
        return density;
    }
    Vector<float> masses;
    if (getOptions().hydro.enabled) {
        masses.resize(particleCount);
        for (size_t index = 0; index < particleCount; index++) {
            masses[index] = particlesMass(index);
        }
    }
    accumulateDensityCpuKernel(&particlesPos(XDIM, 0), &particlesPos(YDIM, 0),
        &particlesPos(ZDIM, 0), masses.empty() ? nullptr : masses.data(), particleCount, fold,
        gridSize, box, density.data());
    return density;
}

static void computeDensityFolded(int m) {
    Vector<hpx::future<void>> futs1;
    for (auto c : hpxChildren()) {
        futs1.push_back(hpx::async<ComputeDensityFoldedAction>(c, m));
    }
    size_t const ndim = getOptions().initial.fourierSize;
    size_t const numberOfParticles = getOptions().initial.numberOfParticles;
    size_t const particlesPerRank = numberOfParticles / hpxSize();
    size_t nper = ndim;
    while (nper > 1 && sqr(nper) * nper > particlesPerRank) {
        nper /= 2;
    }
    for (size_t j = 0; j < ndim; j += nper) {
        for (size_t k = 0; k < ndim; k += nper) {
            for (size_t l = 0; l < ndim; l += nper) {
                Range<int64_t> intbox;
                intbox.begin[XDIM] = int64_t(j) + CLOUD_MIN;
                intbox.begin[YDIM] = int64_t(k) + CLOUD_MIN;
                intbox.begin[ZDIM] = int64_t(l) + CLOUD_MIN;
                intbox.end[XDIM] = std::min(j + nper, ndim) + CLOUD_MAX;
                intbox.end[YDIM] = std::min(k + nper, ndim) + CLOUD_MAX;
                intbox.end[ZDIM] = std::min(l + nper, ndim) + CLOUD_MAX;
                Vector<float> rho;
#ifdef USE_CUDA
                if (cudaEnabled()) {
                    rho = accumulateDensityCuda(m, ndim, intbox);
                } else
#endif
                {
                    rho = accumulateDensityCpu(m, ndim, intbox);
                }
                fft3dAccumulateReal(intbox, rho);
            }
        }
    }
    hpx::wait_all(futs1.begin(), futs1.end());
    for (auto& future : futs1) {
        future.get();
    }
}
