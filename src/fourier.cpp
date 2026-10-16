#include <cosmictiger/fourier.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/vector.hpp>

static std::vector<Vector<Cmplx>> y;
static Vector<std::shared_ptr<SpinlockType>> mutexes;
static int n;
static int begin;
static std::vector<hpx::id_type> localities;
static int end;
static int span;
static int rank;
static int nranks;

Vector<Cmplx> fourier3dTranspose(int xbegin, int xend, int zi, Vector<Cmplx>);
void fourier3dTransposeXz();
void fft1d(Cmplx* y, int n);
std::pair<std::vector<float>, std::vector<size_t>> fourier3dGetPowerSpectrum();

HPX_PLAIN_ACTION(fourier3dGetPowerSpectrum, Fourier3dGetPowerSpectrumAction);
HPX_PLAIN_ACTION(fourier3dInitialize, Fourier3dInitializeAction);
HPX_PLAIN_ACTION(fourier3dDestroy, Fourier3dDestroyAction);
HPX_PLAIN_ACTION(fourier3dTranspose, Fourier3dTransposeAction);
HPX_PLAIN_ACTION(fourier3dAccumulate, Fourier3dAccumulateAction);
HPX_PLAIN_ACTION(fourier3dAccumulateReal, Fourier3dAccumulateRealAction);
HPX_PLAIN_ACTION(fourier3dTransposeXz, Fourier3dTransposeXzAction);
HPX_PLAIN_ACTION(fourier3dRead, Fourier3dReadAction);
HPX_PLAIN_ACTION(fourier3dReadReal, Fourier3dReadRealAction);
HPX_PLAIN_ACTION(fourier3dMirror, Fourier3dMirrorAction);
HPX_PLAIN_ACTION(fourier3dInvExecute, Fourier3dInvExecuteAction);

int slabToRank(int xi) {
    return std::min((int) ((xi + int(n % nranks != 0)) * nranks / n), nranks - 1);
}

void fourier3dMirror() {
    PRINT("Mirroring\n");
    std::vector<hpx::future<void>> futs;
    if (rank == 0) {
        for (int i = 1; i < nranks; i++) {
            futs.push_back(hpx::async<Fourier3dMirrorAction>(localities[i]));
        }
    }
    for (int xi = begin; xi < end; xi++) {
        if (xi <= n / 2) {
            int xi0 = (n - xi) % n;
            auto future = hpx::async<Fourier3dReadAction>(
                localities[slabToRank(xi0)], xi0, xi0 + 1, 0, n, 0, n);
            futs.push_back(future.then([xi](hpx::future<Vector<Cmplx>> fut) {
                auto data = fut.get();
                for (int yi = 0; yi < n; yi++) {
                    for (int zi = 0; zi < n; zi++) {
                        y[xi - begin][yi * n + zi] =
                            data[((n - yi) % n) * n + ((n - zi) % n)].conj();
                    }
                }
            }));
        }
    }
    hpx::wait_all(futs.begin(), futs.end());
    PRINT("Done Mirroring\n");
}

Vector<Cmplx> fourier3dRead(int xb, int xe, int yb, int ye, int zb, int ze) {
    Vector<Cmplx> data;
    int yspan = ye - yb;
    int zspan = ze - zb;
    data.resize(zspan * yspan * (xe - xb));
    std::vector<hpx::future<void>> futs;
    if (slabToRank(xb) != rank || slabToRank(xe - 1) != rank) {
        int const rankb = slabToRank(xb);
        int const ranke = slabToRank(xe - 1);
        for (int otherRank = rankb; otherRank <= ranke; otherRank++) {
            int const thisXb = std::max(xb, otherRank * n / nranks);
            int const thisXe = std::min(xe, (otherRank + 1) * n / nranks);
            int const thisXspan = thisXe - thisXb;
            auto fut = hpx::async<Fourier3dReadAction>(
                localities[otherRank], thisXb, thisXe, yb, ye, zb, ze);
            futs.push_back(fut.then([&data, thisXb, thisXe, yspan, zspan, xb, yb, ye, zb, ze](
                                        hpx::future<Vector<Cmplx>> fut) {
                auto rankData = fut.get();
                std::vector<hpx::future<void>> theseFuts;
                int nthreads = hardware_concurrency();
                for (int proc = 0; proc < nthreads; proc++) {
                    auto const func = [xb, thisXb, proc, thisXe, nthreads, yb, ye, zb, ze, &data,
                                          &rankData, zspan, yspan]() {
                        for (int xi = thisXb + proc; xi < thisXe; xi += nthreads) {
                            for (int yi = yb; yi < ye; yi++) {
                                for (int zi = zb; zi < ze; zi++) {
                                    data[zspan * yspan * (xi - xb) + zspan * (yi - yb) +
                                        (zi - zb)] = rankData[zspan * yspan * (xi - thisXb) +
                                        zspan * (yi - yb) + (zi - zb)];
                                }
                            }
                        }
                    };
                    theseFuts.push_back(hpx::async(func));
                }
                hpx::wait_all(theseFuts.begin(), theseFuts.end());
            }));
        }
    } else {
        int nthreads = hardware_concurrency();
        for (int proc = 0; proc < nthreads; proc++) {
            auto const func = [xb, proc, xe, nthreads, yb, ye, zb, ze, yspan, zspan, &data]() {
                for (int xi = xb + proc; xi < xe; xi += nthreads) {
                    for (int yi = yb; yi < ye; yi++) {
                        for (int zi = zb; zi < ze; zi++) {
                            data[zspan * yspan * (xi - xb) + zspan * (yi - yb) + (zi - zb)] =
                                y[xi - begin][zspan * (yi - yb) + (zi - zb)];
                        }
                    }
                }
            };
            futs.push_back(hpx::async(func));
        }
    }
    hpx::wait_all(futs.begin(), futs.end());
    return std::move(data);
}

std::vector<float> fourier3dReadReal(int xb, int xe, int yb, int ye, int zb, int ze) {
    std::vector<float> data;
    int yspan = ye - yb;
    int zspan = ze - zb;
    data.resize(zspan * yspan * (xe - xb));
    std::vector<hpx::future<void>> futs;
    if (slabToRank(xb) != rank || slabToRank(xe - 1) != rank) {
        int const rankb = slabToRank(xb);
        int const ranke = slabToRank(xe - 1);
        for (int otherRank = rankb; otherRank <= ranke; otherRank++) {
            int const thisXb = std::max(xb, otherRank * n / nranks);
            int const thisXe = std::min(xe, (otherRank + 1) * n / nranks);
            int const thisXspan = thisXe - thisXb;
            auto fut = hpx::async<Fourier3dReadRealAction>(
                localities[otherRank], thisXb, thisXe, yb, ye, zb, ze);
            futs.push_back(fut.then([&data, thisXb, thisXe, yspan, zspan, xb, yb, ye, zb, ze](
                                        hpx::future<std::vector<float>> fut) {
                auto rankData = fut.get();
                std::vector<hpx::future<void>> theseFuts;
                int nthreads = hardware_concurrency();
                for (int proc = 0; proc < nthreads; proc++) {
                    auto const func = [xb, thisXb, proc, thisXe, nthreads, yb, ye, zb, ze, &data,
                                          &rankData, zspan, yspan]() {
                        for (int xi = thisXb + proc; xi < thisXe; xi += nthreads) {
                            for (int yi = yb; yi < ye; yi++) {
                                for (int zi = zb; zi < ze; zi++) {
                                    data[zspan * yspan * (xi - xb) + zspan * (yi - yb) +
                                        (zi - zb)] = rankData[zspan * yspan * (xi - thisXb) +
                                        zspan * (yi - yb) + (zi - zb)];
                                }
                            }
                        }
                    };
                    theseFuts.push_back(hpx::async(func));
                }
                hpx::wait_all(theseFuts.begin(), theseFuts.end());
            }));
        }
    } else {
        int nthreads = hardware_concurrency();
        for (int proc = 0; proc < nthreads; proc++) {
            auto const func = [xb, proc, xe, nthreads, yb, ye, zb, ze, yspan, zspan, &data]() {
                for (int xi = xb + proc; xi < xe; xi += nthreads) {
                    for (int yi = yb; yi < ye; yi++) {
                        for (int zi = zb; zi < ze; zi++) {
                            data[zspan * yspan * (xi - xb) + zspan * (yi - yb) + (zi - zb)] =
                                y[xi - begin][zspan * (yi - yb) + (zi - zb)].real();
                        }
                    }
                }
            };
            futs.push_back(hpx::async(func));
        }
    }
    hpx::wait_all(futs.begin(), futs.end());
    return std::move(data);
}

void fourier3dAccumulate(int xb, int xe, int yb, int ye, int zb, int ze, Vector<Cmplx> data) {
    int const xspan = xe - xb;
    int const yspan = ye - yb;
    int const zspan = ze - zb;
    std::vector<hpx::future<void>> futs;
    int maxNthreads = hardware_concurrency();
    int nthreads;
    if (xb < 0) {
        int xspan1 = xe - 0;
        int xspan2 = 0 - xb;
        Vector<Cmplx> data1(xspan1 * yspan * zspan);
        Vector<Cmplx> data2(xspan2 * yspan * zspan);
        nthreads = std::min(maxNthreads, yspan);
        for (int proc = 0; proc < nthreads; proc++) {
            futs.push_back(hpx::async([xb, xe, yb, ye, zb, ze, xspan1, xspan2, yspan, zspan, &data,
                                          &data1, &data2, proc, nthreads]() {
                for (int i = 0; i < xe; i++) {
                    for (int j = yb + proc; j < ye; j += nthreads) {
                        for (int k = zb; k < ze; k++) {
                            data1[i * yspan * zspan + (j - yb) * zspan + (k - zb)] =
                                data[(i - xb) * yspan * zspan + (j - yb) * zspan + (k - zb)];
                        }
                    }
                }
                for (int i = xb; i < 0; i++) {
                    for (int j = yb + proc; j < ye; j += nthreads) {
                        for (int k = zb; k < ze; k++) {
                            data2[(i - xb) * yspan * zspan + (j - yb) * zspan + (k - zb)] =
                                data[(i - xb) * yspan * zspan + (j - yb) * zspan + (k - zb)];
                        }
                    }
                }
            }));
        }
        hpx::wait_all(futs.begin(), futs.end());
        data = decltype(data)();
        fourier3dAccumulate(0, xe, yb, ye, zb, ze, std::move(data1));
        fourier3dAccumulate(xb + n, n, yb, ye, zb, ze, std::move(data2));
    } else if (xe > n) {
        int xspan1 = n - xb;
        int xspan2 = xe - n;
        Vector<Cmplx> data1(xspan1 * yspan * zspan);
        Vector<Cmplx> data2(xspan2 * yspan * zspan);
        nthreads = std::min(maxNthreads, yspan);
        for (int proc = 0; proc < nthreads; proc++) {
            futs.push_back(hpx::async([xb, xe, yb, ye, zb, ze, xspan1, xspan2, yspan, zspan, &data,
                                          &data1, &data2, proc, nthreads]() {
                for (int i = xb; i < n; i++) {
                    for (int j = yb + proc; j < ye; j += nthreads) {
                        for (int k = zb; k < ze; k++) {
                            data1[(i - xb) * yspan * zspan + (j - yb) * zspan + (k - zb)] =
                                data[(i - xb) * yspan * zspan + (j - yb) * zspan + (k - zb)];
                        }
                    }
                }
                for (int i = n; i < xe; i++) {
                    for (int j = yb + proc; j < ye; j += nthreads) {
                        for (int k = zb; k < ze; k++) {
                            data2[(i - n) * yspan * zspan + (j - yb) * zspan + (k - zb)] =
                                data[(i - xb) * yspan * zspan + (j - yb) * zspan + (k - zb)];
                        }
                    }
                }
            }));
        }
        hpx::wait_all(futs.begin(), futs.end());
        data = decltype(data)();
        fourier3dAccumulate(xb, n, yb, ye, zb, ze, std::move(data1));
        fourier3dAccumulate(0, xe - n, yb, ye, zb, ze, std::move(data2));
    } else if (yb < 0) {
        int yspan1 = ye - 0;
        int yspan2 = 0 - yb;
        Vector<Cmplx> data1(xspan * yspan1 * zspan);
        Vector<Cmplx> data2(xspan * yspan2 * zspan);
        nthreads = std::min(maxNthreads, xspan);
        for (int proc = 0; proc < nthreads; proc++) {
            futs.push_back(hpx::async([xb, xe, yb, ye, zb, ze, yspan1, yspan2, xspan, yspan, zspan,
                                          &data, &data1, &data2, proc, nthreads]() {
                for (int i = xb + proc; i < xe; i += nthreads) {
                    for (int j = 0; j < ye; j++) {
                        for (int k = zb; k < ze; k++) {
                            data1[(i - xb) * yspan1 * zspan + j * zspan + (k - zb)] =
                                data[(i - xb) * yspan * zspan + (j - yb) * zspan + (k - zb)];
                        }
                    }
                    for (int j = yb; j < 0; j++) {
                        for (int k = zb; k < ze; k++) {
                            data2[(i - xb) * yspan2 * zspan + (j - yb) * zspan + (k - zb)] =
                                data[(i - xb) * yspan * zspan + (j - yb) * zspan + (k - zb)];
                        }
                    }
                }
            }));
        }
        hpx::wait_all(futs.begin(), futs.end());
        data = decltype(data)();
        fourier3dAccumulate(xb, xe, 0, ye, zb, ze, std::move(data1));
        fourier3dAccumulate(xb, xe, yb + n, n, zb, ze, std::move(data2));
    } else if (ye > n) {
        int yspan1 = n - yb;
        int yspan2 = ye - n;
        Vector<Cmplx> data1(xspan * yspan1 * zspan);
        Vector<Cmplx> data2(xspan * yspan2 * zspan);
        nthreads = std::min(maxNthreads, xspan);
        for (int proc = 0; proc < nthreads; proc++) {
            futs.push_back(hpx::async([xb, xe, yb, ye, zb, ze, yspan1, yspan2, xspan, yspan, zspan,
                                          &data, &data1, &data2, proc, nthreads]() {
                for (int i = xb + proc; i < xe; i += nthreads) {
                    for (int j = yb; j < n; j++) {
                        for (int k = zb; k < ze; k++) {
                            data1[(i - xb) * yspan1 * zspan + (j - yb) * zspan + (k - zb)] =
                                data[(i - xb) * yspan * zspan + (j - yb) * zspan + (k - zb)];
                        }
                    }
                    for (int j = n; j < ye; j++) {
                        for (int k = zb; k < ze; k++) {
                            data2[(i - xb) * yspan2 * zspan + (j - n) * zspan + (k - zb)] =
                                data[(i - xb) * yspan * zspan + (j - yb) * zspan + (k - zb)];
                        }
                    }
                }
            }));
        }
        hpx::wait_all(futs.begin(), futs.end());
        data = decltype(data)();
        fourier3dAccumulate(xb, xe, yb, n, zb, ze, std::move(data1));
        fourier3dAccumulate(xb, xe, 0, ye - n, zb, ze, std::move(data2));
    } else if (zb < 0) {
        int zspan1 = ze - 0;
        int zspan2 = 0 - zb;
        Vector<Cmplx> data1(xspan * yspan * zspan1);
        Vector<Cmplx> data2(xspan * yspan * zspan2);
        nthreads = std::min(maxNthreads, xspan);
        for (int proc = 0; proc < nthreads; proc++) {
            futs.push_back(hpx::async([xb, xe, yb, ye, zb, ze, xspan, zspan1, zspan2, yspan, zspan,
                                          &data, &data1, &data2, proc, nthreads]() {
                for (int i = xb + proc; i < xe; i += nthreads) {
                    for (int j = yb; j < ye; j++) {
                        for (int k = 0; k < ze; k++) {
                            data1[(i - xb) * yspan * zspan1 + j * zspan1 + k] =
                                data[(i - xb) * yspan * zspan + (j - yb) * zspan + (k - zb)];
                        }
                        for (int k = 0; k < ze; k++) {
                            data2[(i - xb) * yspan * zspan2 + (j - yb) * zspan2 + (k - zb)] =
                                data[(i - xb) * yspan * zspan + (j - yb) * zspan + (k - zb)];
                        }
                    }
                }
            }));
        }
        hpx::wait_all(futs.begin(), futs.end());
        data = decltype(data)();
        fourier3dAccumulate(xb, xe, yb, ye, 0, ze, std::move(data1));
        fourier3dAccumulate(xb, xe, yb, ye, zb + n, n, std::move(data2));
    } else if (ze > n) {
        int zspan1 = n - zb;
        int zspan2 = ze - n;
        Vector<Cmplx> data1(xspan * yspan * zspan1);
        Vector<Cmplx> data2(xspan * yspan * zspan2);
        nthreads = std::min(maxNthreads, xspan);
        for (int proc = 0; proc < nthreads; proc++) {
            futs.push_back(hpx::async([xb, xe, yb, ye, zb, ze, xspan, yspan, zspan1, zspan2, zspan,
                                          &data, &data1, &data2, proc, nthreads]() {
                for (int i = xb + proc; i < xe; i += nthreads) {
                    for (int j = yb; j < ye; j++) {
                        for (int k = zb; k < n; k++) {
                            data1[(i - xb) * yspan * zspan1 + j * zspan1 + (k - zb)] =
                                data[(i - xb) * yspan * zspan + (j - yb) * zspan + (k - zb)];
                        }
                        for (int k = n; k < ze; k++) {
                            data2[(i - xb) * yspan * zspan2 + (j - yb) * zspan2 + (k - n)] =
                                data[(i - xb) * yspan * zspan + (j - yb) * zspan + (k - zb)];
                        }
                    }
                }
            }));
        }
        hpx::wait_all(futs.begin(), futs.end());
        data = decltype(data)();
        fourier3dAccumulate(xb, xe, yb, ye, zb, n, std::move(data1));
        fourier3dAccumulate(xb, xe, yb, ye, 0, ze - n, std::move(data2));
    } else if (slabToRank(xb) != rank || slabToRank(xe - 1) != rank) {
        int const rankb = slabToRank(xb);
        int const ranke = slabToRank(xe - 1);
        for (int otherRank = rankb; otherRank <= ranke; otherRank++) {
            int const thisXb = std::max(xb, otherRank * n / nranks);
            int const thisXe = std::min(xe, (otherRank + 1) * n / nranks);
            int const thisXspan = thisXe - thisXb;
            Vector<Cmplx> rankData(thisXspan * yspan * zspan);
            int nthreads = hardware_concurrency();
            std::vector<hpx::future<void>> theseFuts;
            for (int proc = 0; proc < nthreads; proc++) {
                auto const func = [thisXb, thisXe, proc, nthreads, xb, yb, ye, zb, ze, zspan, yspan,
                                      &data, &rankData]() {
                    for (int xi = thisXb + proc; xi < thisXe; xi += nthreads) {
                        for (int yi = yb; yi < ye; yi++) {
                            for (int zi = zb; zi < ze; zi++) {
                                rankData[zspan * yspan * (xi - thisXb) + zspan * (yi - yb) +
                                    (zi - zb)] =
                                    data[zspan * yspan * (xi - xb) + yspan * (yi - yb) + (zi - zb)];
                            }
                        }
                    }
                };
                theseFuts.push_back(hpx::async(func));
            }
            hpx::wait_all(theseFuts.begin(), theseFuts.end());
            futs.push_back(hpx::async<Fourier3dAccumulateAction>(
                localities[otherRank], thisXb, thisXe, yb, ye, zb, ze, std::move(rankData)));
        }
    } else {
        int nthreads = hardware_concurrency();
        for (int proc = 0; proc < nthreads; proc++) {
            auto const func = [proc, nthreads, xb, xe, yb, ye, zb, ze, &data, zspan, yspan]() {
                for (int xi = xb + proc; xi < xe; xi += nthreads) {
                    std::lock_guard<SpinlockType> lock(*mutexes[xi - xb]);
                    for (int yi = yb; yi < ye; yi++) {
                        for (int zi = zb; zi < ze; zi++) {
                            y[xi - xb][n * (yi - yb) + (zi - zb)] +=
                                data[zspan * yspan * (xi - xb) + zspan * (yi - yb) + (zi - zb)];
                        }
                    }
                }
            };
            futs.push_back(hpx::async(func));
        }
    }
    hpx::wait_all(futs.begin(), futs.end());
}

void fourier3dAccumulateReal(int xb, int xe, int yb, int ye, int zb, int ze, Vector<float> data) {
    int const xspan = xe - xb;
    int const yspan = ye - yb;
    int const zspan = ze - zb;
    std::vector<hpx::future<void>> futs;
    int maxNthreads = hardware_concurrency();
    int nthreads;
    if (xb < 0) {
        int xspan1 = xe - 0;
        int xspan2 = 0 - xb;
        Vector<float> data1(xspan1 * yspan * zspan);
        Vector<float> data2(xspan2 * yspan * zspan);
        nthreads = std::min(maxNthreads, yspan);
        for (int proc = 0; proc < nthreads; proc++) {
            futs.push_back(hpx::async([xb, xe, yb, ye, zb, ze, xspan1, xspan2, yspan, zspan, &data,
                                          &data1, &data2, proc, nthreads]() {
                for (int i = 0; i < xe; i++) {
                    for (int j = yb + proc; j < ye; j += nthreads) {
                        for (int k = zb; k < ze; k++) {
                            data1[i * yspan * zspan + (j - yb) * zspan + (k - zb)] =
                                data[(i - xb) * yspan * zspan + (j - yb) * zspan + (k - zb)];
                        }
                    }
                }
                for (int i = xb; i < 0; i++) {
                    for (int j = yb + proc; j < ye; j += nthreads) {
                        for (int k = zb; k < ze; k++) {
                            data2[(i - xb) * yspan * zspan + (j - yb) * zspan + (k - zb)] =
                                data[(i - xb) * yspan * zspan + (j - yb) * zspan + (k - zb)];
                        }
                    }
                }
            }));
        }
        hpx::wait_all(futs.begin(), futs.end());
        data = decltype(data)();
        fourier3dAccumulateReal(0, xe, yb, ye, zb, ze, std::move(data1));
        fourier3dAccumulateReal(xb + n, n, yb, ye, zb, ze, std::move(data2));
    } else if (xe > n) {
        int xspan1 = n - xb;
        int xspan2 = xe - n;
        Vector<float> data1(xspan1 * yspan * zspan);
        Vector<float> data2(xspan2 * yspan * zspan);
        nthreads = std::min(maxNthreads, yspan);
        for (int proc = 0; proc < nthreads; proc++) {
            futs.push_back(hpx::async([xb, xe, yb, ye, zb, ze, xspan1, xspan2, yspan, zspan, &data,
                                          &data1, &data2, proc, nthreads]() {
                for (int i = xb; i < n; i++) {
                    for (int j = yb + proc; j < ye; j += nthreads) {
                        for (int k = zb; k < ze; k++) {
                            data1[(i - xb) * yspan * zspan + (j - yb) * zspan + (k - zb)] =
                                data[(i - xb) * yspan * zspan + (j - yb) * zspan + (k - zb)];
                        }
                    }
                }
                for (int i = n; i < xe; i++) {
                    for (int j = yb + proc; j < ye; j += nthreads) {
                        for (int k = zb; k < ze; k++) {
                            data2[(i - n) * yspan * zspan + (j - yb) * zspan + (k - zb)] =
                                data[(i - xb) * yspan * zspan + (j - yb) * zspan + (k - zb)];
                        }
                    }
                }
            }));
        }
        hpx::wait_all(futs.begin(), futs.end());
        data = decltype(data)();
        fourier3dAccumulateReal(xb, n, yb, ye, zb, ze, std::move(data1));
        fourier3dAccumulateReal(0, xe - n, yb, ye, zb, ze, std::move(data2));
    } else if (yb < 0) {
        int yspan1 = ye - 0;
        int yspan2 = 0 - yb;
        Vector<float> data1(xspan * yspan1 * zspan);
        Vector<float> data2(xspan * yspan2 * zspan);
        nthreads = std::min(maxNthreads, xspan);
        for (int proc = 0; proc < nthreads; proc++) {
            futs.push_back(hpx::async([xb, xe, yb, ye, zb, ze, yspan1, yspan2, xspan, yspan, zspan,
                                          &data, &data1, &data2, proc, nthreads]() {
                for (int i = xb + proc; i < xe; i += nthreads) {
                    for (int j = 0; j < ye; j++) {
                        for (int k = zb; k < ze; k++) {
                            data1[(i - xb) * yspan1 * zspan + j * zspan + (k - zb)] =
                                data[(i - xb) * yspan * zspan + (j - yb) * zspan + (k - zb)];
                        }
                    }
                    for (int j = yb; j < 0; j++) {
                        for (int k = zb; k < ze; k++) {
                            data2[(i - xb) * yspan2 * zspan + (j - yb) * zspan + (k - zb)] =
                                data[(i - xb) * yspan * zspan + (j - yb) * zspan + (k - zb)];
                        }
                    }
                }
            }));
        }
        hpx::wait_all(futs.begin(), futs.end());
        data = decltype(data)();
        fourier3dAccumulateReal(xb, xe, 0, ye, zb, ze, std::move(data1));
        fourier3dAccumulateReal(xb, xe, yb + n, n, zb, ze, std::move(data2));
    } else if (ye > n) {
        int yspan1 = n - yb;
        int yspan2 = ye - n;
        Vector<float> data1(xspan * yspan1 * zspan);
        Vector<float> data2(xspan * yspan2 * zspan);
        nthreads = std::min(maxNthreads, xspan);
        for (int proc = 0; proc < nthreads; proc++) {
            futs.push_back(hpx::async([xb, xe, yb, ye, zb, ze, yspan1, yspan2, xspan, yspan, zspan,
                                          &data, &data1, &data2, proc, nthreads]() {
                for (int i = xb + proc; i < xe; i += nthreads) {
                    for (int j = yb; j < n; j++) {
                        for (int k = zb; k < ze; k++) {
                            data1[(i - xb) * yspan1 * zspan + (j - yb) * zspan + (k - zb)] =
                                data[(i - xb) * yspan * zspan + (j - yb) * zspan + (k - zb)];
                        }
                    }
                    for (int j = n; j < ye; j++) {
                        for (int k = zb; k < ze; k++) {
                            data2[(i - xb) * yspan2 * zspan + (j - n) * zspan + (k - zb)] =
                                data[(i - xb) * yspan * zspan + (j - yb) * zspan + (k - zb)];
                        }
                    }
                }
            }));
        }
        hpx::wait_all(futs.begin(), futs.end());
        data = decltype(data)();
        fourier3dAccumulateReal(xb, xe, yb, n, zb, ze, std::move(data1));
        fourier3dAccumulateReal(xb, xe, 0, ye - n, zb, ze, std::move(data2));
    } else if (zb < 0) {
        int zspan1 = ze - 0;
        int zspan2 = 0 - zb;
        Vector<float> data1(xspan * yspan * zspan1);
        Vector<float> data2(xspan * yspan * zspan2);
        nthreads = std::min(maxNthreads, xspan);
        for (int proc = 0; proc < nthreads; proc++) {
            futs.push_back(hpx::async([xb, xe, yb, ye, zb, ze, xspan, zspan1, zspan2, yspan, zspan,
                                          &data, &data1, &data2, proc, nthreads]() {
                for (int i = xb + proc; i < xe; i += nthreads) {
                    for (int j = yb; j < ye; j++) {
                        for (int k = 0; k < ze; k++) {
                            data1[(i - xb) * yspan * zspan1 + j * zspan1 + k] =
                                data[(i - xb) * yspan * zspan + (j - yb) * zspan + (k - zb)];
                        }
                        for (int k = 0; k < ze; k++) {
                            data2[(i - xb) * yspan * zspan2 + (j - yb) * zspan2 + (k - zb)] =
                                data[(i - xb) * yspan * zspan + (j - yb) * zspan + (k - zb)];
                        }
                    }
                }
            }));
        }
        hpx::wait_all(futs.begin(), futs.end());
        data = decltype(data)();
        fourier3dAccumulateReal(xb, xe, yb, ye, 0, ze, std::move(data1));
        fourier3dAccumulateReal(xb, xe, yb, ye, zb + n, n, std::move(data2));
    } else if (ze > n) {
        int zspan1 = n - zb;
        int zspan2 = ze - n;
        Vector<float> data1(xspan * yspan * zspan1);
        Vector<float> data2(xspan * yspan * zspan2);
        nthreads = std::min(maxNthreads, xspan);
        for (int proc = 0; proc < nthreads; proc++) {
            futs.push_back(hpx::async([xb, xe, yb, ye, zb, ze, xspan, yspan, zspan1, zspan2, zspan,
                                          &data, &data1, &data2, proc, nthreads]() {
                for (int i = xb + proc; i < xe; i += nthreads) {
                    for (int j = yb; j < ye; j++) {
                        for (int k = zb; k < n; k++) {
                            int i0 = (i - xb) * yspan * zspan1 + j * zspan1 + (k - zb);
                            int i1 = (i - xb) * yspan * zspan + (j - yb) * zspan + (k - zb);
                            data1[i0] = data[i1];
                        }
                        for (int k = n; k < ze; k++) {
                            int i0 = (i - xb) * yspan * zspan2 + (j - yb) * zspan2 + (k - n);
                            int i1 = (i - xb) * yspan * zspan + (j - yb) * zspan + (k - zb);
                            data2[i0] = data[i1];
                        }
                    }
                }
            }));
        }
        hpx::wait_all(futs.begin(), futs.end());
        data = decltype(data)();
        fourier3dAccumulateReal(xb, xe, yb, ye, zb, n, std::move(data1));
        fourier3dAccumulateReal(xb, xe, yb, ye, 0, ze - n, std::move(data2));
    } else if (slabToRank(xb) != rank || slabToRank(xe - 1) != rank) {
        int const rankb = slabToRank(xb);
        int const ranke = slabToRank(xe - 1);
        for (int otherRank = rankb; otherRank <= ranke; otherRank++) {
            int const thisXb = std::max(xb, otherRank * n / nranks);
            int const thisXe = std::min(xe, (otherRank + 1) * n / nranks);
            int const thisXspan = thisXe - thisXb;
            Vector<float> rankData(thisXspan * yspan * zspan);
            int nthreads = hardware_concurrency();
            std::vector<hpx::future<void>> theseFuts;
            for (int proc = 0; proc < nthreads; proc++) {
                auto const func = [thisXb, thisXe, proc, nthreads, xb, yb, ye, zb, ze, zspan, yspan,
                                      &data, &rankData]() {
                    for (int xi = thisXb + proc; xi < thisXe; xi += nthreads) {
                        for (int yi = yb; yi < ye; yi++) {
                            for (int zi = zb; zi < ze; zi++) {
                                int i0 =
                                    zspan * yspan * (xi - thisXb) + zspan * (yi - yb) + (zi - zb);
                                int i1 = zspan * yspan * (xi - xb) + zspan * (yi - yb) + (zi - zb);
                                assert(i0 >= 0);
                                assert(i0 < rankData.size());
                                assert(i1 >= 0);
                                assert(i1 < data.size());
                                rankData[i0] = data[i1];
                            }
                        }
                    }
                };
                theseFuts.push_back(hpx::async(func));
            }
            hpx::wait_all(theseFuts.begin(), theseFuts.end());
            futs.push_back(hpx::async<Fourier3dAccumulateRealAction>(
                localities[otherRank], thisXb, thisXe, yb, ye, zb, ze, std::move(rankData)));
        }
    } else {
        int nthreads = hardware_concurrency();
        for (int proc = 0; proc < nthreads; proc++) {
            auto const func = [proc, nthreads, xb, xe, yb, ye, zb, ze, &data, zspan, yspan]() {
                for (int xi = xb + proc; xi < xe; xi += nthreads) {
                    std::lock_guard<SpinlockType> lock(*mutexes[xi - xb]);
                    for (int yi = yb; yi < ye; yi++) {
                        for (int zi = zb; zi < ze; zi++) {
                            y[xi - xb][n * (yi - yb) + (zi - zb)].real() +=
                                data[zspan * yspan * (xi - xb) + zspan * (yi - yb) + (zi - zb)];
                        }
                    }
                }
            };
            futs.push_back(hpx::async(func));
        }
    }
    hpx::wait_all(futs.begin(), futs.end());
}

void fourier3dDo2dpart() {
    for (int i = 0; i < span; i++) {
        fft2d(y[i].data(), n);
    }
}

void fourier3dDo1dpart() {
    for (int i = 0; i < span; i++) {
        fft1d(y[i].data(), n);
    }
}

HPX_PLAIN_ACTION(fourier3dDo1dpart, Fourier3dDo1dpartAction);
HPX_PLAIN_ACTION(fourier3dDo2dpart, Fourier3dDo2dpartAction);

std::vector<float> fourier3dPowerSpectrum() {
    std::vector<hpx::future<std::pair<std::vector<float>, std::vector<size_t>>>> futs;
    for (int i = 0; i < nranks; i++) {
        futs.push_back(hpx::async<Fourier3dGetPowerSpectrumAction>(localities[i]));
    }
    std::vector<float> power;
    std::vector<size_t> counts;
    int maxN = n / 2;
    power.resize(maxN, 0.0f);
    counts.resize(maxN, 0);
    for (auto& f : futs) {
        auto data = f.get();
        auto thisPower = std::move(data.first);
        auto thisCounts = std::move(data.second);
        for (int i = 0; i < maxN; i++) {
            power[i] += thisPower[i];
            counts[i] += thisCounts[i];
        }
    }
    for (int i = 0; i < maxN; i++) {
        if (counts[i]) {
            power[i] /= counts[i];
        }
    }
    return power;
}

std::pair<std::vector<float>, std::vector<size_t>> fourier3dGetPowerSpectrum() {
    std::vector<float> power;
    std::vector<size_t> counts;
    int maxN = n / 2;
    power.resize(maxN, 0.0f);
    counts.resize(maxN, 0);
    int const nthreads = hardware_concurrency();
    static MutexType mutex;
    std::vector<hpx::future<void>> futs;
    for (int proc = 0; proc < nthreads; proc++) {
        futs.push_back(hpx::async([maxN, proc, nthreads, &counts, &power]() {
            std::vector<float> thisPower;
            std::vector<size_t> thisCounts;
            thisPower.resize(maxN, 0.0f);
            thisCounts.resize(maxN, 0);
            for (int yi = proc; yi < n; yi += nthreads) {
                int const yi0 = yi < n / 2 ? yi : yi - n;
                for (int xi = begin; xi < end; xi++) {
                    int const xi0 = xi < n / 2 ? xi : xi - n;
                    for (int zi = 0; zi < n; zi++) {
                        int const zi0 = zi < n / 2 ? zi : zi - n;
                        int const i2 = xi0 * xi0 + yi0 * yi0 + zi0 * zi0;
                        int const i = int(std::sqrt(i2));
                        if (i < maxN) {
                            assert(y.size() > xi - begin);
                            float const mag = y[xi - begin][n * yi + zi].norm();
                            thisPower[i] += mag;
                            thisCounts[i]++;
                        }
                    }
                }
            }
            std::lock_guard<MutexType> lock(mutex);
            for (int i = 0; i < maxN; i++) {
                power[i] += thisPower[i];
                counts[i] += thisCounts[i];
            }
        }));
    }
    hpx::wait_all(futs.begin(), futs.end());
    return std::make_pair(std::move(power), std::move(counts));
}

void fourier3dExecute() {
    PRINT("Executing fourier\n");
    std::vector<hpx::future<void>> futs;
    PRINT("doing 2d\n");
    for (int i = 0; i < nranks; i++) {
        futs.push_back(hpx::async<Fourier3dDo2dpartAction>(localities[i]));
    }
    hpx::wait_all(futs.begin(), futs.end());
    PRINT("transposing\n");
    fourier3dTransposeXz();
    futs.resize(0);
    for (int i = 0; i < nranks; i++) {
        futs.push_back(hpx::async<Fourier3dDo1dpartAction>(localities[i]));
    }
    hpx::wait_all(futs.begin(), futs.end());
    PRINT("transposing\n");
    fourier3dTransposeXz();
    PRINT("Done executing fourier\n");
}

void fourier3dInvExecute() {
    std::vector<hpx::future<void>> futs;
    if (rank == 0) {
        for (int i = 1; i < nranks; i++) {
            futs.push_back(hpx::async<Fourier3dInvExecuteAction>(localities[i]));
        }
    }
    float const factor = 1.0f / (n * n * n);
    for (int i = 0; i < span; i++) {
        for (int j = 0; j < n * n; j++) {
            y[i][j].real() *= factor;
            y[i][j].imag() *= factor;
        }
    }
    hpx::wait_all(futs.begin(), futs.end());
    if (rank == 0) {
        fourier3dExecute();
    }
}

void fourier3dInitialize(int gridSize) {
    std::vector<hpx::future<void>> futs;
    rank = hpxRank();
    nranks = hpxSize();
    localities = hpxLocalities();
    if (rank == 0) {
        fourier3dDestroy();
        for (int i = 1; i < nranks; i++) {
            futs.push_back(hpx::async<Fourier3dInitializeAction>(localities[i], gridSize));
        }
    }
    n = gridSize;
    begin = rank * n / nranks;
    end = (rank + 1) * n / nranks;
    span = end - begin;
    y.resize(span);
    mutexes.resize(span);
    for (int i = 0; i < span; i++) {
        mutexes[i] = std::make_shared<SpinlockType>();
        y[i].resize(n * n);
        for (int j = 0; j < n * n; j++) {
            y[i][j].real() = y[i][j].imag() = 0.0;
        }
    }
    hpx::wait_all(futs.begin(), futs.end());
}

void fourier3dDestroy() {
    std::vector<hpx::future<void>> futs;
    if (rank == 0) {
        for (int i = 1; i < nranks; i++) {
            futs.push_back(hpx::async<Fourier3dDestroyAction>(localities[i]));
        }
    }
    y = decltype(y)();
    mutexes = decltype(mutexes)();
    hpx::wait_all(futs.begin(), futs.end());
}

void fourier3dTransposeXz() {
    std::vector<hpx::future<void>> futs;
    rank = hpxRank();
    nranks = hpxSize();
    localities = hpxLocalities();
    if (rank == 0) {
        for (int i = 1; i < nranks; i++) {
            futs.push_back(hpx::async<Fourier3dTransposeXzAction>(localities[i]));
        }
    }
    for (int other = rank; other < nranks; other++) {
        for (int xi = begin; xi < end; xi++) {
            Vector<Cmplx> data;
            int const zend = (other + 1) * n / nranks;
            int const zbegin = other * n / nranks + ((other == rank) ? (xi - begin) : 0);
            int const zspan = zend - zbegin;
            data.resize(n * zspan);
            for (int j = 0; j < n; j++) {
                for (int k = zbegin; k < zend; k++) {
                    data[j * zspan + k - zbegin] = y[xi - begin][j * n + k];
                }
            }
            auto future = hpx::async<Fourier3dTransposeAction>(
                localities[other], zbegin, zend, xi, std::move(data));
            futs.push_back(future.then([zspan, zbegin, zend, xi](hpx::future<Vector<Cmplx>> fut) {
                auto data = fut.get();
                for (int j = 0; j < n; j++) {
                    for (int k = zbegin; k < zend; k++) {
                        y[xi - begin][j * n + k] = data[j * zspan + k - zbegin];
                    }
                }
            }));
        }
    }

    hpx::wait_all(futs.begin(), futs.end());
}

Vector<Cmplx> fourier3dTranspose(int xbegin, int xend, int zi, Vector<Cmplx> data) {
    int const xspan = xend - xbegin;
    for (int xi = 0; xi < xspan; xi++) {
        for (int yi = 0; yi < n; yi++) {
            auto& a = data[yi * xspan + xi];
            auto& b = y[xi + xbegin - begin][yi * n + zi];
            swap(a, b);
        }
    }
    return std::move(data);
}
