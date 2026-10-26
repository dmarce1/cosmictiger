/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2021  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <complex>
#include <mutex>

#include <fftw3.h>
#include <hpx/serialization/complex.hpp>
#include <silo.h>

#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/cosmology/fft.hpp>
#include <cosmictiger/cosmology/fftw_lock.hpp>
#include <cosmictiger/cuda/cuda.hpp>
#include <cosmictiger/io/output.hpp>

constexpr int kMaxBoxSize = (8 * 1024 * 1024);

static int64_t n;
static Vector<Range<int64_t>> realBoxes;
static Array<Vector<Range<int64_t>>, kDimensions> cmplxBoxes;
static Range<int64_t> realMybox;
static Array<Range<int64_t>, kDimensions> cmplxMybox;
static Vector<std::shared_ptr<SpinlockType>> mutexes;

static Vector<float> rValue;
static Vector<ComplexFloat> yValue;
static Vector<ComplexFloat> ym;
static Vector<ComplexFloat> y1Value;

static void findBoxes(Range<int64_t> box, Vector<Range<int64_t>>& boxes, int begin, int end);
static void splitBox(Range<int64_t> box, Vector<Range<int64_t>>& realBoxes);
static void transpose(int, int);
static void shift(bool);
static void update();
static void fft3dPhase1();
static void fft3dPhase2(int, bool);
static void fft3dPhase3();
static void finishForceReal();
static Vector<ComplexFloat> transposeRead(Range<int64_t> const&, int dim1, int dim2);
static Vector<ComplexFloat> shiftRead(Range<int64_t> const&, bool);

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

void fft3d2silo(bool real) {
    if (real) {
        DBfile* db =
            DBCreateReal(outputPath("real.silo").c_str(), DB_CLOBBER, DB_LOCAL, "", DB_PDB);
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
        float* coords[kDimensions] = {x.data(), y.data(), z.data()};
        int dims[kDimensions] = {(int) n, (int) n, (int) n};
        DBPutQuadmesh(
            db, "real_mesh", coordnames, coords, dims, kDimensions, DB_FLOAT, DB_COLLINEAR, NULL);
        DBPutQuadvar1(db, "real_data", "real_mesh", rValue.data(), dims, kDimensions, NULL, 0,
            DB_FLOAT, DB_NODECENT, NULL);
        DBClose(db);
    } else {
        DBfile* db =
            DBCreateReal(outputPath("complex.silo").c_str(), DB_CLOBBER, DB_LOCAL, "", DB_PDB);
        char const* coordnames[] = {"z", "y", "x"};
        Vector<float> x;
        Vector<float> y;
        Vector<float> z;
        auto box = cmplxMybox[kZDimension];
        for (int i = 0; i < n; i++) {
            x.push_back(i);
        }
        for (int i = 0; i < n; i++) {
            y.push_back(i);
        }
        for (int i = 0; i < n / 2 + 1; i++) {
            z.push_back(i);
        }
        float* coords[kDimensions] = {z.data(), y.data(), x.data()};
        int dims[kDimensions] = {(int) n / 2 + 1, (int) n, (int) n};
        DBPutQuadmesh(db, "complex_mesh", coordnames, coords, dims, kDimensions, DB_FLOAT,
            DB_COLLINEAR, NULL);
        Vector<float> i;
        Vector<float> r;
        for (int j = 0; j < box.volume(); j++) {
            i.push_back(yValue[j].imag());
            r.push_back(yValue[j].real());
        }
        DBPutQuadvar1(db, "complex_real", "complex_mesh", r.data(), dims, kDimensions, NULL, 0,
            DB_FLOAT, DB_NODECENT, NULL);
        DBPutQuadvar1(db, "complex_imag", "complex_mesh", i.data(), dims, kDimensions, NULL, 0,
            DB_FLOAT, DB_NODECENT, NULL);
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

void fft3dForceReal() {
    Vector<hpx::future<void>> futs;
    for (auto c : hpxChildren()) {
        futs.push_back(hpx::async<Fft3dForceRealAction>(c));
    }
    auto const& box = cmplxMybox[kZDimension];
    Array<int64_t, kDimensions> i;
    Range<int64_t> mirrorBox = box;
    Range<int64_t> slimBox = box;
    slimBox.end[kZDimension] = 1;
    ym.resize(slimBox.volume());
    mirrorBox.end[kZDimension] = 1;
    for (int dim = 0; dim < kDimensions - 1; dim++) {
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
                auto const y = (y1 + std::conj(y2)) * 0.5f;
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
    return cmplxMybox[kZDimension];
}

Range<int64_t> fft3dRealRange() {
    return realMybox;
}

Vector<ComplexFloat>& fft3dComplexVector() {
    return yValue;
}

void finishForceReal() {
    Vector<hpx::future<void>> futs;
    for (auto c : hpxChildren()) {
        futs.push_back(hpx::async<FinishForceRealAction>(c));
    }
    auto const& box = cmplxMybox[kZDimension];
    Array<int64_t, kDimensions> i;
    Range<int64_t> slimBox = box;
    slimBox.end[kZDimension] = 1;
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
        Array<int64_t, kDimensions> i;
        for (i[0] = thisBox.begin[0]; i[0] != thisBox.end[0]; i[0]++) {
            for (i[1] = thisBox.begin[1]; i[1] != thisBox.end[1]; i[1]++) {
                for (i[2] = thisBox.begin[2]; i[2] != thisBox.end[2]; i[2]++) {
                    data[thisBox.index(i)] = rValue[realMybox.index(i)];
                }
            }
        }
    } else {
        for (int ri = 0; ri < hpxSize(); ri++) {
            Array<int64_t, kDimensions> si;
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
                                    Array<int64_t, kDimensions> i;
                                    for (i[0] = thisInter.begin[0]; i[0] != thisInter.end[0];
                                        i[0]++) {
                                        for (i[1] = thisInter.begin[1]; i[1] != thisInter.end[1];
                                            i[1]++) {
                                            for (i[2] = thisInter.begin[2];
                                                i[2] != thisInter.end[2]; i[2]++) {
                                                auto j = i;
                                                for (int dim = 0; dim < kDimensions; dim++) {
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

Vector<ComplexFloat> fft3dReadComplex(Range<int64_t> const& thisBox) {
    Vector<hpx::future<void>> futs;
    Vector<ComplexFloat> data(thisBox.volume());
    auto const mybox = cmplxMybox[kZDimension];
    if (mybox.contains(thisBox)) {
        Array<int64_t, kDimensions> i;
        for (i[0] = thisBox.begin[0]; i[0] != thisBox.end[0]; i[0]++) {
            for (i[1] = thisBox.begin[1]; i[1] != thisBox.end[1]; i[1]++) {
                for (i[2] = thisBox.begin[2]; i[2] != thisBox.end[2]; i[2]++) {
                    data[thisBox.index(i)] = yValue[mybox.index(i)];
                }
            }
        }
    } else {
        for (int ri = 0; ri < hpxSize(); ri++) {
            Array<int64_t, kDimensions> si;
            for (si[0] = -n; si[0] <= +n; si[0] += n) {
                for (si[1] = -n; si[1] <= +n; si[1] += n) {
                    for (si[2] = -n; si[2] <= +n; si[2] += n) {
                        auto const shiftedBox = thisBox.shift(si);
                        auto const inter = cmplxBoxes[kZDimension][ri].intersection(shiftedBox);
                        if (inter.volume()) {
                            Vector<Range<int64_t>> inters;
                            splitBox(inter, inters);
                            for (auto thisInter : inters) {
                                auto fut = hpx::async<Fft3dReadComplexAction>(
                                    hpxLocalities()[ri], thisInter);
                                futs.push_back(fut.then([si, thisBox, thisInter, &data](
                                                            hpx::future<Vector<ComplexFloat>> fut) {
                                    auto thisData = fut.get();
                                    Array<int64_t, kDimensions> i;
                                    for (i[0] = thisInter.begin[0]; i[0] != thisInter.end[0];
                                        i[0]++) {
                                        for (i[1] = thisInter.begin[1]; i[1] != thisInter.end[1];
                                            i[1]++) {
                                            for (i[2] = thisInter.begin[2];
                                                i[2] != thisInter.end[2]; i[2]++) {
                                                auto j = i;
                                                for (int dim = 0; dim < kDimensions; dim++) {
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
        Array<int64_t, kDimensions> i;
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
            Array<int64_t, kDimensions> si;
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
                                Array<int64_t, kDimensions> i;
                                for (i[0] = thisInter.begin[0]; i[0] != thisInter.end[0]; i[0]++) {
                                    for (i[1] = thisInter.begin[1]; i[1] != thisInter.end[1];
                                        i[1]++) {
                                        for (i[2] = thisInter.begin[2]; i[2] != thisInter.end[2];
                                            i[2]++) {
                                            int64_t const k = thisInter.index(i);
                                            auto j = i;
                                            for (int dim = 0; dim < kDimensions; dim++) {
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

void fft3dAccumulateComplex(Range<int64_t> const& thisBox, Vector<ComplexFloat> const& data) {
    Vector<hpx::future<void>> futs;
    auto const& box = cmplxMybox[kZDimension];
    if (box.contains(thisBox)) {
        Array<int64_t, kDimensions> i;
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
        for (int64_t bi = 0; bi < cmplxBoxes[kZDimension].size(); bi++) {
            Array<int64_t, kDimensions> si;
            for (si[0] = -n; si[0] <= +n; si[0] += n) {
                for (si[1] = -n; si[1] <= +n; si[1] += n) {
                    for (si[2] = -n; si[2] <= +n; si[2] += n) {
                        auto const shiftedBox = thisBox.shift(si);
                        auto const inter = cmplxBoxes[kZDimension][bi].intersection(shiftedBox);
                        if (!inter.empty()) {
                            Vector<Range<int64_t>> inters;
                            splitBox(inter, inters);
                            for (auto thisInter : inters) {
                                Vector<ComplexFloat> thisData;
                                thisData.resize(thisInter.volume());
                                Array<int64_t, kDimensions> i;
                                for (i[0] = thisInter.begin[0]; i[0] != thisInter.end[0]; i[0]++) {
                                    for (i[1] = thisInter.begin[1]; i[1] != thisInter.end[1];
                                        i[1]++) {
                                        for (i[2] = thisInter.begin[2]; i[2] != thisInter.end[2];
                                            i[2]++) {
                                            auto j = i;
                                            for (int dim = 0; dim < kDimensions; dim++) {
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
    for (int dim = 0; dim < kDimensions; dim++) {
        box.begin[dim] = 0;
        box.end[dim] = n;
    }
    realBoxes.resize(hpxSize());
    findBoxes(box, realBoxes, 0, hpxSize());
    realMybox = realBoxes[hpxRank()];
    rValue.resize(realMybox.volume(), initConst);
    for (int dim = 0; dim < kDimensions; dim++) {
        for (int dim1 = 0; dim1 < kDimensions; dim1++) {
            box.begin[dim1] = 0;
            box.end[dim1] = n;
        }
        box.end[dim] = n / 2 + 1;
        cmplxBoxes[dim].resize(hpxSize());
        findBoxes(box, cmplxBoxes[dim], 0, hpxSize());
        cmplxMybox[dim] = cmplxBoxes[dim][hpxRank()];
    }
    yValue.resize(cmplxMybox[kZDimension].volume(), ComplexFloat(0.0, 0.0));
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
    Vector<hpx::future<void>> futs;
    for (auto c : hpxChildren()) {
        futs.push_back(hpx::async<Fft3dPhase1Action>(c));
    }
    Array<int64_t, kDimensions> i;
    yValue.resize(cmplxMybox[kZDimension].volume());
    for (i[0] = realMybox.begin[0]; i[0] != realMybox.end[0]; i[0]++) {
        futs.push_back(hpx::async(
            [](Array<int64_t, kDimensions> j) {
                fftwf_plan p;
                fftwf_complex out[n / 2 + 1];
                float in[n];
                std::unique_lock<std::mutex> lock(fftwPlannerMutex());
                p = fftwf_plan_dft_r2c_1d(n, in, out, FFTW_ESTIMATE);
                lock.unlock();
                auto i = j;
                for (i[1] = realMybox.begin[1]; i[1] != realMybox.end[1]; i[1]++) {
                    for (i[2] = 0; i[2] != n; i[2]++) {
                        in[i[2]] = rValue[realMybox.index(i)];
                    }
                    fftwf_execute(p);
                    for (i[2] = 0; i[2] < n / 2 + 1; i[2]++) {
                        int64_t const l = cmplxMybox[kZDimension].index(i);
                        yValue[l].real(out[i[2]][0]);
                        yValue[l].imag(out[i[2]][1]);
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
    Vector<hpx::future<void>> futs;
    for (auto c : hpxChildren()) {
        futs.push_back(hpx::async<Fft3dPhase2Action>(c, dim, inv));
    }
    yValue = std::move(y1Value);
    Array<int64_t, kDimensions> i;
    float const norm = inv ? 1.0f / n : 1.0;
    for (i[0] = cmplxMybox[dim].begin[0]; i[0] != cmplxMybox[dim].end[0]; i[0]++) {
        futs.push_back(hpx::async(
            [dim, inv, norm](Array<int64_t, kDimensions> j) {
                fftwf_complex out[n];
                fftwf_complex in[n];
                fftwf_plan p;
                std::unique_lock<std::mutex> lock(fftwPlannerMutex());
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
                        yValue[l].real(out[i[2]][0] * norm);
                        yValue[l].imag(out[i[2]][1] * norm);
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
    Vector<hpx::future<void>> futs;
    for (auto c : hpxChildren()) {
        futs.push_back(hpx::async<Fft3dPhase3Action>(c));
    }
    Array<int64_t, kDimensions> i;
    yValue = std::move(y1Value);
    rValue.resize(realMybox.volume());
    float const ninv = 1.0 / n;
    for (i[0] = cmplxMybox[kZDimension].begin[0]; i[0] != cmplxMybox[kZDimension].end[0]; i[0]++) {
        futs.push_back(hpx::async(
            [ninv](Array<int64_t, kDimensions> j) {
                fftwf_plan p;
                fftwf_complex in[n / 2 + 1];
                float out[n];
                std::unique_lock<std::mutex> lock(fftwPlannerMutex());
                p = fftwf_plan_dft_c2r_1d(n, in, out, FFTW_ESTIMATE);
                lock.unlock();
                auto i = j;
                for (i[1] = cmplxMybox[kZDimension].begin[1];
                    i[1] != cmplxMybox[kZDimension].end[1]; i[1]++) {
                    for (i[2] = 0; i[2] < n / 2 + 1; i[2]++) {
                        int64_t const l = cmplxMybox[kZDimension].index(i);
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
                futs.push_back(fut.then([inter, dim1](hpx::future<Vector<ComplexFloat>> fut) {
                    auto data = fut.get();
                    Array<int64_t, kDimensions> i;
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

static Vector<ComplexFloat> transposeRead(Range<int64_t> const& thisBox, int dim1, int dim2) {
    Vector<ComplexFloat> data(thisBox.volume());
    ASSERT(cmplxMybox[dim2].contains(thisBox));
    auto tbox = thisBox.transpose(dim1, dim2);
    Array<int64_t, kDimensions> i;
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
    int const dim2 = inv ? kYDimension : kXDimension;
    int const dim1 = inv ? kXDimension : kYDimension;
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
                futs.push_back(fut.then([inter, dim2](hpx::future<Vector<ComplexFloat>> fut) {
                    auto data = fut.get();
                    Array<int64_t, kDimensions> i;
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

static Vector<ComplexFloat> shiftRead(Range<int64_t> const& thisBox, bool inv) {
    Vector<ComplexFloat> data(thisBox.volume());
    auto tbox = inv ? thisBox.shiftDown() : thisBox.shiftUp();
    Array<int64_t, kDimensions> i;
    int const dim = inv ? kXDimension : kYDimension;
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
    if (box.volume() < kMaxBoxSize) {
        realBoxes.push_back(box);
    } else {
        auto children = box.split();
        splitBox(children.first, realBoxes);
        splitBox(children.second, realBoxes);
    }
}
