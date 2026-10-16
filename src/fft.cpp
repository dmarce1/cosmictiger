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

#include <fftw3.h>
#include <silo.h>

#include <cosmictiger/constants.hpp>
#include <cosmictiger/cuda.hpp>
#include <cosmictiger/fft.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/options.hpp>

#define MAX_BOX_SIZE (8 * 1024 * 1024)

static int64_t n;
static Vector<Range<int64_t>> realBoxes;
static Array<Vector<Range<int64_t>>, NDIM> cmplxBoxes;
static Range<int64_t> realMybox;
static Array<Range<int64_t>, NDIM> cmplxMybox;
static Vector<std::shared_ptr<SpinlockType>> mutexes;

static Vector<float> rValue;
static Vector<Cmplx> yValue;
static Vector<Cmplx> ym;
static Vector<Cmplx> y1Value;

static void findBoxes(Range<int64_t> box, Vector<Range<int64_t>>& boxes, int begin, int end);
static void splitBox(Range<int64_t> box, Vector<Range<int64_t>>& realBoxes);
static void transpose(int, int);
static void shift(bool);
static void update();
static void fft3dPhase1();
static void fft3dPhase2(int, bool);
static void fft3dPhase3();
static void finishForceReal();
static Vector<Cmplx> transposeRead(Range<int64_t> const&, int dim1, int dim2);
static Vector<Cmplx> shiftRead(Range<int64_t> const&, bool);
struct PspecRet {
    Vector<float> k;
    Vector<float> p;
    Vector<double> c;
    Vector<float> pdev;
};
static PspecRet powerSpectrumCompute(Vector<float> const& kbins);
static PspecRet powerSpectrumCompute2(PspecRet, Vector<float> const& kbins);

HPX_PLAIN_ACTION(fft3dAccumulateReal, Fft3dAccumulateRealAction);
HPX_PLAIN_ACTION(fft3dAccumulateComplex, Fft3dAccumulateComplexAction);
HPX_PLAIN_ACTION(fft3dInit, Fft3dInitAction);
HPX_PLAIN_ACTION(fft3dExecute, Fft3dExecuteAction);
HPX_PLAIN_ACTION(fft3dPhase1, Fft3dPhase1Action);
HPX_PLAIN_ACTION(fft3dPhase2, Fft3dPhase2Action);
HPX_PLAIN_ACTION(fft3dPhase3, Fft3dPhase3Action);
HPX_PLAIN_ACTION(fft3dReadReal, Fft3dReadRealAction);
HPX_PLAIN_ACTION(fft3dReadComplex, Fft3dReadComplexAction);
HPX_PLAIN_ACTION(fft3dForceReal, Fft3dForceRealAction);
HPX_PLAIN_ACTION(fft3dDestroy, Fft3dDestroyAction);
HPX_PLAIN_ACTION(transposeRead, TransposeReadAction);
HPX_PLAIN_ACTION(transpose, TransposeAction);
HPX_PLAIN_ACTION(shift, ShiftAction);
HPX_PLAIN_ACTION(shiftRead, ShiftReadAction);
HPX_PLAIN_ACTION(update, UpdateAction);
HPX_PLAIN_ACTION(finishForceReal, FinishForceRealAction);
HPX_PLAIN_ACTION(powerSpectrumCompute, PowerSpectrumComputeAction);
HPX_PLAIN_ACTION(powerSpectrumCompute2, PowerSpectrumCompute2Action);

void fft3d2silo(bool real) {
    if (real) {
        DBfile* db = DBCreateReal("real.silo", DB_CLOBBER, DB_LOCAL, "", DB_PDB);
        char const* coordnames[] = {"x", "y", "z"};
        Vector<float> x;
        Vector<float> y;
        Vector<float> z;
        auto box = realMybox;
        for (int i = 0; i < n; i++) {
            x.push_back(i);
        }
        for (int i = 0; i < n; i++) {
            y.push_back(i);
        }
        for (int i = 0; i < n; i++) {
            z.push_back(i);
        }
        float* coords[NDIM] = {x.data(), y.data(), z.data()};
        int dims[NDIM] = {(int) n, (int) n, (int) n};
        DBPutQuadmesh(
            db, "real_mesh", coordnames, coords, dims, NDIM, DB_FLOAT, DB_COLLINEAR, NULL);
        DBPutQuadvar1(db, "real_data", "real_mesh", rValue.data(), dims, NDIM, NULL, 0, DB_FLOAT,
            DB_NODECENT, NULL);
        DBClose(db);
    } else {
        DBfile* db = DBCreateReal("complex.silo", DB_CLOBBER, DB_LOCAL, "", DB_PDB);
        char const* coordnames[] = {"z", "y", "x"};
        Vector<float> x;
        Vector<float> y;
        Vector<float> z;
        auto box = cmplxMybox[ZDIM];
        for (int i = 0; i < n; i++) {
            x.push_back(i);
        }
        for (int i = 0; i < n; i++) {
            y.push_back(i);
        }
        for (int i = 0; i < n / 2 + 1; i++) {
            z.push_back(i);
        }
        float* coords[NDIM] = {z.data(), y.data(), x.data()};
        int dims[NDIM] = {(int) n / 2 + 1, (int) n, (int) n};
        DBPutQuadmesh(
            db, "complex_mesh", coordnames, coords, dims, NDIM, DB_FLOAT, DB_COLLINEAR, NULL);
        Vector<float> i;
        Vector<float> r;
        for (int j = 0; j < box.volume(); j++) {
            i.push_back(yValue[j].imag());
            r.push_back(yValue[j].real());
        }
        DBPutQuadvar1(db, "complex_real", "complex_mesh", r.data(), dims, NDIM, NULL, 0, DB_FLOAT,
            DB_NODECENT, NULL);
        DBPutQuadvar1(db, "complex_imag", "complex_mesh", i.data(), dims, NDIM, NULL, 0, DB_FLOAT,
            DB_NODECENT, NULL);
        DBClose(db);
    }
}

void fft3dExecute() {
    //	PRINT("FFT z\n");
    fft3dPhase1();
    //	PRINT("Transpose y-z\n");
    transpose(1, 2);
    //	PRINT("FFT y\n");
    fft3dPhase2(1, false);
    //	PRINT("Shifting\n");
    shift(false);
    //	PRINT("FFT x\n");
    fft3dPhase2(0, false);
    //	PRINT("Transpose z-x\n");
    transpose(2, 0);
    update();
    //	PRINT("done\n");
}

void fft3dInvExecute() {
    //	PRINT("Transpose z-x\n");
    transpose(0, 2);
    //	PRINT("inv FFT x\n");
    fft3dPhase2(0, true);
    //	PRINT("Shifting\n");
    shift(true);
    //	PRINT("inv FFT y\n");
    fft3dPhase2(1, true);
    //	PRINT("Transpose y-z\n");
    transpose(2, 1);
    //	PRINT("inv FFT z\n");
    fft3dPhase3();
    //	PRINT("done\n");
}

PowerSpectrumT fft3dPowerSpectrum() {
    Vector<float> kbins;
    double const boxSize = getOptions().codeToCm / constants::kMpcToCm;
    int const nfour = getOptions().nfour;
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
    auto pr = powerSpectrumCompute(kbins);
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
    pr = powerSpectrumCompute2(pr, kbins);
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

int searchKbins(Vector<float> kbins, double k, int binmin, int binmax) {
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

static PspecRet powerSpectrumCompute(Vector<float> const& kbins) {
    Vector<hpx::future<PspecRet>> futs;
    for (auto c : hpxChildren()) {
        futs.push_back(hpx::async<PowerSpectrumComputeAction>(c, kbins));
    }
    int64_t const n = getOptions().partsDim;
    float const boxSize = getOptions().codeToCm / constants::kMpcToCm;
    auto const& box = cmplxMybox[ZDIM];
    Array<int64_t, NDIM> iValue;

    int const nbins = kbins.size() - 1;
    Vector<float> power(nbins, 0.0);
    Vector<double> count(nbins, 0);
    Vector<float> ktot(nbins, 0.0);
    for (iValue[0] = box.begin[0]; iValue[0] != box.end[0]; iValue[0]++) {
        futs.push_back(hpx::async(
            [n, nbins, &kbins, box](Array<int64_t, NDIM> iValue2) {
                int64_t const i = iValue2[0] < n / 2 ? iValue2[0] : iValue2[0] - n;
                double const sx = cloudFilter(2.0 * M_PI * i / n);
                PspecRet ret;
                Vector<float> power(nbins, 0.0);
                Vector<double> count(nbins, 0);
                Vector<float> ktot(nbins, 0.0);
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
                            int const bin = searchKbins(kbins, iii, 0, nbins);
                            if (bin >= 0) {
                                float const pwr = yValue[box.index(iValue2)].norm();
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
    PspecRet rc;
    rc.p = std::move(power);
    rc.c = std::move(count);
    rc.k = std::move(ktot);
    return rc;
}

static PspecRet powerSpectrumCompute2(PspecRet rc, Vector<float> const& kbins) {
    Vector<hpx::future<PspecRet>> futs;
    for (auto c : hpxChildren()) {
        futs.push_back(hpx::async<PowerSpectrumCompute2Action>(c, rc, kbins));
    }
    int64_t const n = getOptions().partsDim;
    float const boxSize = getOptions().codeToCm / constants::kMpcToCm;
    auto const& box = cmplxMybox[ZDIM];
    Array<int64_t, NDIM> iValue;

    int const nbins = kbins.size() - 1;
    Vector<float> devs(nbins, 0.0);
    for (iValue[0] = box.begin[0]; iValue[0] != box.end[0]; iValue[0]++) {
        futs.push_back(hpx::async(
            [n, nbins, &kbins, box, rc](Array<int64_t, NDIM> iValue2) {
                int64_t const i = iValue2[0] < n / 2 ? iValue2[0] : iValue2[0] - n;
                double const sx = sinc(M_PI * i / n);
                PspecRet ret;
                Vector<float> pdev(nbins, 0.0);
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
                            int const bin = searchKbins(kbins, iii, 0, nbins);
                            if (bin >= 0) {
                                float const pwr = yValue[box.index(iValue2)].norm();
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
        for (int i = 0; i < nbins; i++) {
            devs[i] += thisPdev[i];
        }
    }
    rc.pdev = std::move(devs);
    return rc;
}

void fft3dForceReal() {
    Vector<hpx::future<void>> futs;
    for (auto c : hpxChildren()) {
        futs.push_back(hpx::async<Fft3dForceRealAction>(c));
    }
    auto const& box = cmplxMybox[ZDIM];
    Array<int64_t, NDIM> i;
    Range<int64_t> mirrorBox = box;
    Range<int64_t> slimBox = box;
    slimBox.end[ZDIM] = 1;
    ym.resize(slimBox.volume());
    mirrorBox.end[ZDIM] = 1;
    for (int dim = 0; dim < NDIM - 1; dim++) {
        mirrorBox.begin[dim] = n - box.end[dim] + 1;
        mirrorBox.end[dim] = n - box.begin[dim] + 1;
    }
    auto const data = fft3dReadComplex(mirrorBox);
    for (i[0] = mirrorBox.begin[0]; i[0] != mirrorBox.end[0]; i[0]++) {
        for (i[1] = mirrorBox.begin[1]; i[1] != mirrorBox.end[1]; i[1]++) {
            for (i[2] = mirrorBox.begin[2]; i[2] != mirrorBox.end[2]; i[2]++) {
                auto j = i;
                j[0] = (n - j[0]) % n;
                j[1] = (n - j[1]) % n;
                auto const y1 = yValue[box.index(j)];
                auto const y2 = data[mirrorBox.index(i)];
                auto const y = (y1 + y2.conj()) * 0.5;
                ym[slimBox.index(j)] = y;
            }
        }
    }

    hpx::wait_all(futs.begin(), futs.end());

    if (hpxRank() == 0) {
        finishForceReal();
    }
}

Range<int64_t> fft3dComplexRange() {
    return cmplxMybox[ZDIM];
}

Range<int64_t> fft3dRealRange() {
    return realMybox;
}

Vector<Cmplx>& fft3dComplexVector() {
    return yValue;
}

void finishForceReal() {
    Vector<hpx::future<void>> futs;
    for (auto c : hpxChildren()) {
        futs.push_back(hpx::async<FinishForceRealAction>(c));
    }
    auto const& box = cmplxMybox[ZDIM];
    Array<int64_t, NDIM> i;
    Range<int64_t> slimBox = box;
    slimBox.end[ZDIM] = 1;
    for (i[0] = slimBox.begin[0]; i[0] != slimBox.end[0]; i[0]++) {
        for (i[1] = slimBox.begin[1]; i[1] != slimBox.end[1]; i[1]++) {
            for (i[2] = slimBox.begin[2]; i[2] != slimBox.end[2]; i[2]++) {
                yValue[box.index(i)] = ym[slimBox.index(i)];
            }
        }
    }

    hpx::wait_all(futs.begin(), futs.end());
}

Vector<float> fft3dReadReal(Range<int64_t> const& thisBox) {
    Vector<hpx::future<void>> futs;
    Vector<float> data(thisBox.volume());
    if (realMybox.contains(thisBox)) {
        Array<int64_t, NDIM> i;
        for (i[0] = thisBox.begin[0]; i[0] != thisBox.end[0]; i[0]++) {
            for (i[1] = thisBox.begin[1]; i[1] != thisBox.end[1]; i[1]++) {
                for (i[2] = thisBox.begin[2]; i[2] != thisBox.end[2]; i[2]++) {
                    data[thisBox.index(i)] = rValue[realMybox.index(i)];
                }
            }
        }
    } else {
        for (int ri = 0; ri < hpxSize(); ri++) {
            Array<int64_t, NDIM> si;
            for (si[0] = -n; si[0] <= +n; si[0] += n) {
                for (si[1] = -n; si[1] <= +n; si[1] += n) {
                    for (si[2] = -n; si[2] <= +n; si[2] += n) {
                        auto const shiftedBox = thisBox.shift(si);
                        auto const inter = realBoxes[ri].intersection(shiftedBox);
                        if (inter.volume()) {
                            Vector<Range<int64_t>> inters;
                            splitBox(inter, inters);
                            for (auto thisInter : inters) {
                                auto fut =
                                    hpx::async<Fft3dReadRealAction>(hpxLocalities()[ri], thisInter);
                                futs.push_back(fut.then([si, thisBox, thisInter, &data](
                                                            hpx::future<Vector<float>> fut) {
                                    auto thisData = fut.get();
                                    Array<int64_t, NDIM> i;
                                    for (i[0] = thisInter.begin[0]; i[0] != thisInter.end[0];
                                        i[0]++) {
                                        for (i[1] = thisInter.begin[1]; i[1] != thisInter.end[1];
                                            i[1]++) {
                                            for (i[2] = thisInter.begin[2];
                                                i[2] != thisInter.end[2]; i[2]++) {
                                                auto j = i;
                                                for (int dim = 0; dim < NDIM; dim++) {
                                                    j[dim] -= si[dim];
                                                }
                                                data[thisBox.index(j)] =
                                                    thisData[thisInter.index(i)];
                                            }
                                        }
                                    }
                                }));
                            }
                        }
                    }
                }
            }
        }
    }
    hpx::wait_all(futs.begin(), futs.end());
    return std::move(data);
}

Vector<Cmplx> fft3dReadComplex(Range<int64_t> const& thisBox) {
    Vector<hpx::future<void>> futs;
    Vector<Cmplx> data(thisBox.volume());
    auto const mybox = cmplxMybox[ZDIM];
    if (mybox.contains(thisBox)) {
        Array<int64_t, NDIM> i;
        for (i[0] = thisBox.begin[0]; i[0] != thisBox.end[0]; i[0]++) {
            for (i[1] = thisBox.begin[1]; i[1] != thisBox.end[1]; i[1]++) {
                for (i[2] = thisBox.begin[2]; i[2] != thisBox.end[2]; i[2]++) {
                    data[thisBox.index(i)] = yValue[mybox.index(i)];
                }
            }
        }
    } else {
        for (int ri = 0; ri < hpxSize(); ri++) {
            Array<int64_t, NDIM> si;
            for (si[0] = -n; si[0] <= +n; si[0] += n) {
                for (si[1] = -n; si[1] <= +n; si[1] += n) {
                    for (si[2] = -n; si[2] <= +n; si[2] += n) {
                        auto const shiftedBox = thisBox.shift(si);
                        auto const inter = cmplxBoxes[ZDIM][ri].intersection(shiftedBox);
                        if (inter.volume()) {
                            Vector<Range<int64_t>> inters;
                            splitBox(inter, inters);
                            for (auto thisInter : inters) {
                                auto fut = hpx::async<Fft3dReadComplexAction>(
                                    hpxLocalities()[ri], thisInter);
                                futs.push_back(fut.then([si, thisBox, thisInter, &data](
                                                            hpx::future<Vector<Cmplx>> fut) {
                                    auto thisData = fut.get();
                                    Array<int64_t, NDIM> i;
                                    for (i[0] = thisInter.begin[0]; i[0] != thisInter.end[0];
                                        i[0]++) {
                                        for (i[1] = thisInter.begin[1]; i[1] != thisInter.end[1];
                                            i[1]++) {
                                            for (i[2] = thisInter.begin[2];
                                                i[2] != thisInter.end[2]; i[2]++) {
                                                auto j = i;
                                                for (int dim = 0; dim < NDIM; dim++) {
                                                    j[dim] -= si[dim];
                                                }
                                                data[thisBox.index(j)] =
                                                    thisData[thisInter.index(i)];
                                            }
                                        }
                                    }
                                }));
                            }
                        }
                    }
                }
            }
        }
    }
    hpx::wait_all(futs.begin(), futs.end());
    return std::move(data);
}

void fft3dAccumulateReal(Range<int64_t> const& thisBox, Vector<float> const& data) {
    Vector<hpx::future<void>> futs;
    if (realMybox.contains(thisBox)) {
        Array<int64_t, NDIM> i;
        for (i[0] = thisBox.begin[0]; i[0] != thisBox.end[0]; i[0]++) {
            for (i[1] = thisBox.begin[1]; i[1] != thisBox.end[1]; i[1]++) {
                auto const& box = realMybox;
                int64_t const mtxindex =
                    (i[0] - box.begin[0]) * (box.end[1] - box.begin[1]) + (i[1] - box.begin[1]);
                std::lock_guard<SpinlockType> lock(*mutexes[mtxindex]);
                for (i[2] = thisBox.begin[2]; i[2] != thisBox.end[2]; i[2]++) {
                    rValue[box.index(i)] += data[thisBox.index(i)];
                }
            }
        }
    } else {
        for (int bi = 0; bi < realBoxes.size(); bi++) {
            Array<int64_t, NDIM> si;
            for (si[0] = -n; si[0] <= +n; si[0] += n) {
                for (si[1] = -n; si[1] <= +n; si[1] += n) {
                    for (si[2] = -n; si[2] <= +n; si[2] += n) {
                        auto const shiftedBox = thisBox.shift(si);
                        auto const inter = realBoxes[bi].intersection(shiftedBox);
                        if (!inter.empty()) {
                            Vector<Range<int64_t>> inters;
                            splitBox(inter, inters);
                            for (auto thisInter : inters) {
                                Vector<float> thisData;
                                thisData.resize(thisInter.volume());
                                Array<int64_t, NDIM> i;
                                for (i[0] = thisInter.begin[0]; i[0] != thisInter.end[0]; i[0]++) {
                                    for (i[1] = thisInter.begin[1]; i[1] != thisInter.end[1];
                                        i[1]++) {
                                        for (i[2] = thisInter.begin[2]; i[2] != thisInter.end[2];
                                            i[2]++) {
                                            int64_t const k = thisInter.index(i);
                                            auto j = i;
                                            for (int dim = 0; dim < NDIM; dim++) {
                                                j[dim] -= si[dim];
                                            }
                                            int64_t const l = thisBox.index(j);
                                            ASSERT(k < thisData.size());
                                            ASSERT(l < data.size());
                                            thisData[k] = data[l];
                                        }
                                    }
                                }
                                auto fut = hpx::async<Fft3dAccumulateRealAction>(
                                    hpxLocalities()[bi], thisInter, std::move(thisData));
                                fut.get();
                                //								futs.push_back(std::move(fut));
                            }
                        }
                    }
                }
            }
        }
    }
    hpx::wait_all(futs.begin(), futs.end());
}

void fft3dAccumulateComplex(Range<int64_t> const& thisBox, Vector<Cmplx> const& data) {
    Vector<hpx::future<void>> futs;
    auto const& box = cmplxMybox[ZDIM];
    if (box.contains(thisBox)) {
        Array<int64_t, NDIM> i;
        for (i[0] = thisBox.begin[0]; i[0] != thisBox.end[0]; i[0]++) {
            for (i[1] = thisBox.begin[1]; i[1] != thisBox.end[1]; i[1]++) {
                int64_t const mtxindex =
                    (i[0] - box.begin[0]) * (box.end[1] - box.begin[1]) + (i[1] - box.begin[1]);
                std::lock_guard<SpinlockType> lock(*mutexes[mtxindex]);
                for (i[2] = thisBox.begin[2]; i[2] != thisBox.end[2]; i[2]++) {
                    yValue[box.index(i)] += data[thisBox.index(i)];
                }
            }
        }
    } else {
        for (int64_t bi = 0; bi < cmplxBoxes[ZDIM].size(); bi++) {
            Array<int64_t, NDIM> si;
            for (si[0] = -n; si[0] <= +n; si[0] += n) {
                for (si[1] = -n; si[1] <= +n; si[1] += n) {
                    for (si[2] = -n; si[2] <= +n; si[2] += n) {
                        auto const shiftedBox = thisBox.shift(si);
                        auto const inter = cmplxBoxes[ZDIM][bi].intersection(shiftedBox);
                        if (!inter.empty()) {
                            Vector<Range<int64_t>> inters;
                            splitBox(inter, inters);
                            for (auto thisInter : inters) {
                                Vector<Cmplx> thisData;
                                thisData.resize(thisInter.volume());
                                Array<int64_t, NDIM> i;
                                for (i[0] = thisInter.begin[0]; i[0] != thisInter.end[0]; i[0]++) {
                                    for (i[1] = thisInter.begin[1]; i[1] != thisInter.end[1];
                                        i[1]++) {
                                        for (i[2] = thisInter.begin[2]; i[2] != thisInter.end[2];
                                            i[2]++) {
                                            auto j = i;
                                            for (int dim = 0; dim < NDIM; dim++) {
                                                j[dim] -= si[dim];
                                            }
                                            thisData[thisInter.index(i)] = data[thisBox.index(j)];
                                        }
                                    }
                                }
                                auto fut = hpx::async<Fft3dAccumulateComplexAction>(
                                    hpxLocalities()[bi], thisInter, std::move(thisData));
                                futs.push_back(std::move(fut));
                            }
                        }
                    }
                }
            }
        }
    }

    hpx::wait_all(futs.begin(), futs.end());
}

void fft3dInit(int64_t gridSize, float initConst) {
    fft3dDestroy();
    Vector<hpx::future<void>> futs;
    for (auto c : hpxChildren()) {
        futs.push_back(hpx::async<Fft3dInitAction>(c, gridSize, initConst));
    }
    n = gridSize;
    Range<int64_t> box;
    for (int dim = 0; dim < NDIM; dim++) {
        box.begin[dim] = 0;
        box.end[dim] = n;
    }
    realBoxes.resize(hpxSize());
    findBoxes(box, realBoxes, 0, hpxSize());
    realMybox = realBoxes[hpxRank()];
    rValue.resize(realMybox.volume(), initConst);
    for (int dim = 0; dim < NDIM; dim++) {
        for (int dim1 = 0; dim1 < NDIM; dim1++) {
            box.begin[dim1] = 0;
            box.end[dim1] = n;
        }
        box.end[dim] = n / 2 + 1;
        cmplxBoxes[dim].resize(hpxSize());
        findBoxes(box, cmplxBoxes[dim], 0, hpxSize());
        cmplxMybox[dim] = cmplxBoxes[dim][hpxRank()];
    }
    yValue.resize(cmplxMybox[ZDIM].volume(), Cmplx(0.0, 0.0));
    mutexes.resize(n * n);
    for (int64_t i = 0; i < n * n; i++) {
        mutexes[i] = std::make_shared<SpinlockType>();
    }
    hpx::wait_all(futs.begin(), futs.end());
}

void fft3dDestroy() {
    Vector<hpx::future<void>> futs;
    for (auto c : hpxChildren()) {
        futs.push_back(hpx::async<Fft3dDestroyAction>(c));
    }
    rValue = decltype(rValue)();
    yValue = decltype(yValue)();
    y1Value = decltype(y1Value)();
    ym = decltype(ym)();

    hpx::wait_all(futs.begin(), futs.end());
}

static void update() {
    Vector<hpx::future<void>> futs;
    for (auto c : hpxChildren()) {
        futs.push_back(hpx::async<UpdateAction>(c));
    }
    yValue = std::move(y1Value);
    hpx::wait_all(futs.begin(), futs.end());
}

static void fft3dPhase1() {
    static MutexType mtx;
    Vector<hpx::future<void>> futs;
    for (auto c : hpxChildren()) {
        futs.push_back(hpx::async<Fft3dPhase1Action>(c));
    }
    Array<int64_t, NDIM> i;
    yValue.resize(cmplxMybox[ZDIM].volume());
    for (i[0] = realMybox.begin[0]; i[0] != realMybox.end[0]; i[0]++) {
        futs.push_back(hpx::async(
            [](Array<int64_t, NDIM> j) {
                fftwf_plan p;
                fftwf_complex out[n / 2 + 1];
                float in[n];
                std::unique_lock<MutexType> lock(mtx);
                p = fftwf_plan_dft_r2c_1d(n, in, out, FFTW_ESTIMATE);
                lock.unlock();
                auto i = j;
                for (i[1] = realMybox.begin[1]; i[1] != realMybox.end[1]; i[1]++) {
                    for (i[2] = 0; i[2] != n; i[2]++) {
                        in[i[2]] = rValue[realMybox.index(i)];
                    }
                    fftwf_execute(p);
                    for (i[2] = 0; i[2] < n / 2 + 1; i[2]++) {
                        int64_t const l = cmplxMybox[ZDIM].index(i);
                        yValue[l].real() = out[i[2]][0];
                        yValue[l].imag() = out[i[2]][1];
                    }
                }
                lock.lock();
                fftwf_destroy_plan(p);
                lock.unlock();
            },
            i));
    }
    hpx::wait_all(futs.begin(), futs.end());
    rValue = decltype(rValue)();
}

static void fft3dPhase2(int dim, bool inv) {
    static MutexType mtx;
    Vector<hpx::future<void>> futs;
    for (auto c : hpxChildren()) {
        futs.push_back(hpx::async<Fft3dPhase2Action>(c, dim, inv));
    }
    yValue = std::move(y1Value);
    Array<int64_t, NDIM> i;
    float const norm = inv ? 1.0f / n : 1.0;
    for (i[0] = cmplxMybox[dim].begin[0]; i[0] != cmplxMybox[dim].end[0]; i[0]++) {
        futs.push_back(hpx::async(
            [dim, inv, norm](Array<int64_t, NDIM> j) {
                fftwf_complex out[n];
                fftwf_complex in[n];
                fftwf_plan p;
                std::unique_lock<MutexType> lock(mtx);
                p = fftwf_plan_dft_1d(
                    n, in, out, inv ? FFTW_BACKWARD : FFTW_FORWARD, FFTW_ESTIMATE);
                lock.unlock();
                auto i = j;
                for (i[1] = cmplxMybox[dim].begin[1]; i[1] != cmplxMybox[dim].end[1]; i[1]++) {
                    ASSERT(cmplxMybox[dim].begin[2] == 0);
                    ASSERT(cmplxMybox[dim].end[2] == n);
                    for (i[2] = 0; i[2] < n; i[2]++) {
                        auto const l = cmplxMybox[dim].index(i);
                        ASSERT(l >= 0);
                        ASSERT(l < yValue.size());
                        in[i[2]][0] = yValue[l].real();
                        in[i[2]][1] = yValue[l].imag();
                    }
                    fftwf_execute(p);
                    for (i[2] = 0; i[2] < n; i[2]++) {
                        int64_t const l = cmplxMybox[dim].index(i);
                        yValue[l].real() = out[i[2]][0] * norm;
                        yValue[l].imag() = out[i[2]][1] * norm;
                    }
                }
                lock.lock();
                fftwf_destroy_plan(p);
                lock.unlock();
            },
            i));
    }
    hpx::wait_all(futs.begin(), futs.end());
}

static void fft3dPhase3() {
    static MutexType mtx;
    Vector<hpx::future<void>> futs;
    for (auto c : hpxChildren()) {
        futs.push_back(hpx::async<Fft3dPhase3Action>(c));
    }
    Array<int64_t, NDIM> i;
    yValue = std::move(y1Value);
    rValue.resize(realMybox.volume());
    float const ninv = 1.0 / n;
    for (i[0] = cmplxMybox[ZDIM].begin[0]; i[0] != cmplxMybox[ZDIM].end[0]; i[0]++) {
        futs.push_back(hpx::async(
            [ninv](Array<int64_t, NDIM> j) {
                fftwf_plan p;
                fftwf_complex in[n / 2 + 1];
                float out[n];
                std::unique_lock<MutexType> lock(mtx);
                p = fftwf_plan_dft_c2r_1d(n, in, out, FFTW_ESTIMATE);
                lock.unlock();
                auto i = j;
                for (i[1] = cmplxMybox[ZDIM].begin[1]; i[1] != cmplxMybox[ZDIM].end[1]; i[1]++) {
                    for (i[2] = 0; i[2] < n / 2 + 1; i[2]++) {
                        int64_t const l = cmplxMybox[ZDIM].index(i);
                        in[i[2]][0] = yValue[l].real();
                        in[i[2]][1] = yValue[l].imag();
                        /*					if( i[2] == 0 || i[2] == N / 2) {
                         const float value = Y[l].imag() / (Y[l].abs()+1e-20);
                         if(std::abs(value) > 5.0e-3) {
                         PRINT( "!!!! %i %i %i %e %e\n", i[0], i[1], i[2], Y[l].real(),
                         Y[l].imag());
                         }
                         }*/
                    }
                    fftwf_execute(p);
                    for (i[2] = 0; i[2] != n; i[2]++) {
                        rValue[realMybox.index(i)] = out[i[2]] * ninv;
                    }
                }
                lock.lock();
                fftwf_destroy_plan(p);
                lock.unlock();
            },
            i));
    }
    hpx::wait_all(futs.begin(), futs.end());
    yValue = decltype(yValue)();
}

static void findBoxes(Range<int64_t> box, Vector<Range<int64_t>>& boxes, int begin, int end) {
    if (end - begin == 1) {
        boxes[begin] = box;
    } else {
        int const xdim = (box.end[0] - box.begin[0] > 1) ? 0 : 1;
        auto left = box;
        auto right = box;
        int const mid = (begin + end) / 2;
        float const w = float(mid - begin) / float(end - begin);
        left.end[xdim] = right.begin[xdim] =
            (((1.0 - w) * box.begin[xdim] + w * box.end[xdim]) + 0.5);
        findBoxes(left, boxes, begin, mid);
        findBoxes(right, boxes, mid, end);
    }
}

static void transpose(int dim1, int dim2) {
    Vector<hpx::future<void>> futs;
    for (auto c : hpxChildren()) {
        futs.push_back(hpx::async<TransposeAction>(c, dim1, dim2));
    }
    Range<int64_t> tbox = cmplxMybox[dim1].transpose(dim1, dim2);
    y1Value.resize(cmplxMybox[dim1].volume());
    for (int64_t bi = 0; bi < cmplxBoxes[dim2].size(); bi++) {
        auto const tinter = cmplxBoxes[dim2][bi].intersection(tbox);
        Vector<float> data;
        if (!tinter.empty()) {
            Vector<Range<int64_t>> tinters;
            splitBox(tinter, tinters);
            for (auto thisTinter : tinters) {
                auto inter = thisTinter.transpose(dim1, dim2);
                auto fut =
                    hpx::async<TransposeReadAction>(hpxLocalities()[bi], thisTinter, dim1, dim2);
                futs.push_back(fut.then([inter, dim1](hpx::future<Vector<Cmplx>> fut) {
                    auto data = fut.get();
                    Array<int64_t, NDIM> i;
                    for (i[0] = inter.begin[0]; i[0] != inter.end[0]; i[0]++) {
                        for (i[1] = inter.begin[1]; i[1] != inter.end[1]; i[1]++) {
                            for (i[2] = inter.begin[2]; i[2] != inter.end[2]; i[2]++) {
                                int const k = inter.index(i);
                                int64_t const l = cmplxMybox[dim1].index(i);
                                ASSERT(k < data.size());
                                ASSERT(l < y1Value.size());
                                y1Value[l] = data[k];
                            }
                        }
                    }
                }));
            }
        }
    }

    hpx::wait_all(futs.begin(), futs.end());
}

static Vector<Cmplx> transposeRead(Range<int64_t> const& thisBox, int dim1, int dim2) {
    Vector<Cmplx> data(thisBox.volume());
    ASSERT(cmplxMybox[dim2].contains(thisBox));
    auto tbox = thisBox.transpose(dim1, dim2);
    Array<int64_t, NDIM> i;
    for (i[0] = thisBox.begin[0]; i[0] != thisBox.end[0]; i[0]++) {
        for (i[1] = thisBox.begin[1]; i[1] != thisBox.end[1]; i[1]++) {
            for (i[2] = thisBox.begin[2]; i[2] != thisBox.end[2]; i[2]++) {
                auto j = i;
                std::swap(j[dim1], j[dim2]);
                data[tbox.index(j)] = yValue[cmplxMybox[dim2].index(i)];
            }
        }
    }
    return std::move(data);
}

static void shift(bool inv) {
    Vector<hpx::future<void>> futs;
    for (auto c : hpxChildren()) {
        futs.push_back(hpx::async<ShiftAction>(c, inv));
    }
    int const dim2 = inv ? YDIM : XDIM;
    int const dim1 = inv ? XDIM : YDIM;
    Range<int64_t> tbox = inv ? cmplxMybox[dim2].shiftUp() : cmplxMybox[dim2].shiftDown();
    y1Value.resize(cmplxMybox[dim2].volume());
    for (int64_t bi = 0; bi < cmplxBoxes[dim1].size(); bi++) {
        auto const tinter = cmplxBoxes[dim1][bi].intersection(tbox);
        Vector<float> data;
        if (!tinter.empty()) {
            Vector<Range<int64_t>> tinters;
            splitBox(tinter, tinters);
            for (auto thisTinter : tinters) {
                auto inter = inv ? thisTinter.shiftDown() : thisTinter.shiftUp();
                auto fut = hpx::async<ShiftReadAction>(hpxLocalities()[bi], thisTinter, inv);
                futs.push_back(fut.then([inter, dim2](hpx::future<Vector<Cmplx>> fut) {
                    auto data = fut.get();
                    Array<int64_t, NDIM> i;
                    for (i[0] = inter.begin[0]; i[0] != inter.end[0]; i[0]++) {
                        for (i[1] = inter.begin[1]; i[1] != inter.end[1]; i[1]++) {
                            for (i[2] = inter.begin[2]; i[2] != inter.end[2]; i[2]++) {
                                int64_t const k = inter.index(i);
                                int64_t const l = cmplxMybox[dim2].index(i);
                                ASSERT(k < data.size());
                                ASSERT(l < y1Value.size());
                                y1Value[l] = data[k];
                            }
                        }
                    }
                }));
            }
        }
    }

    hpx::wait_all(futs.begin(), futs.end());
}

static Vector<Cmplx> shiftRead(Range<int64_t> const& thisBox, bool inv) {
    Vector<Cmplx> data(thisBox.volume());
    auto tbox = inv ? thisBox.shiftDown() : thisBox.shiftUp();
    Array<int64_t, NDIM> i;
    int const dim = inv ? XDIM : YDIM;
    for (i[0] = thisBox.begin[0]; i[0] != thisBox.end[0]; i[0]++) {
        for (i[1] = thisBox.begin[1]; i[1] != thisBox.end[1]; i[1]++) {
            for (i[2] = thisBox.begin[2]; i[2] != thisBox.end[2]; i[2]++) {
                auto j = inv ? shiftDown(i) : shiftUp(i);
                data[tbox.index(j)] = yValue[cmplxMybox[dim].index(i)];
            }
        }
    }
    return std::move(data);
}

static void splitBox(Range<int64_t> box, Vector<Range<int64_t>>& realBoxes) {
    if (box.volume() < MAX_BOX_SIZE) {
        realBoxes.push_back(box);
    } else {
        auto children = box.split();
        splitBox(children.first, realBoxes);
        splitBox(children.second, realBoxes);
    }
}
