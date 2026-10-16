#include <cosmictiger/hpx.hpp>
#include <cosmictiger/particle_server.hpp>
#include <cosmictiger/zeldovich.hpp>

HPX_PLAIN_ACTION(twolpt, TwolptAction);
HPX_PLAIN_ACTION(twolptCorrection1, TwolptCorrection1Action);
HPX_PLAIN_ACTION(twolptCorrection2, TwolptCorrection2Action);
HPX_PLAIN_ACTION(twolptInit, TwolptInitAction);
HPX_PLAIN_ACTION(twolptDestroy, TwolptDestroyAction);
HPX_PLAIN_ACTION(twolptPhase, TwolptPhaseAction);
HPX_PLAIN_ACTION(phi1ToParticles, Phi1ToParticlesAction);
HPX_PLAIN_ACTION(phi2ToParticles, Phi2ToParticlesAction);

static Vector<Cmplx> delta2;
static std::vector<float> delta2Part;

void twolptInit(int n) {
    std::vector<hpx::future<void>> futs;
    if (hpxRank() == 0) {
        for (int i = 1; i < hpxSize(); i++) {
            futs.push_back(hpx::async<TwolptInitAction>(hpxLocalities()[i], n));
        }
    }
    int xbegin = hpxRank() * n / hpxSize();
    int xend = (hpxRank() + 1) * n / hpxSize();
    int xspan = xend - xbegin;
    delta2.resize(xspan * n * n);
    for (int i = 0; i < xspan * n * n; i++) {
        delta2[i] = Cmplx(0.0, 0.0);
    }
    delta2Part.resize(xspan * n * n, 0.0);
    hpx::wait_all(futs.begin(), futs.end());
}

void twolptDestroy() {
    std::vector<hpx::future<void>> futs;
    if (hpxRank() == 0) {
        for (int i = 1; i < hpxSize(); i++) {
            futs.push_back(hpx::async<TwolptDestroyAction>(hpxLocalities()[i]));
        }
    }
    delta2Part = decltype(delta2Part)();
    delta2 = decltype(delta2)();
    hpx::wait_all(futs.begin(), futs.end());
}

void twolptPhase(int n, int phase) {
    std::vector<hpx::future<void>> futs;
    if (hpxRank() == 0) {
        for (int i = 1; i < hpxSize(); i++) {
            futs.push_back(hpx::async<TwolptPhaseAction>(hpxLocalities()[i], n, phase));
        }
    }
    int xbegin = hpxRank() * n / hpxSize();
    int xend = (hpxRank() + 1) * n / hpxSize();
    int xspan = xend - xbegin;
    if (phase > 2 * NDIM) {
        auto thisDelta2Part = fourier3dReadReal(xbegin, xend, 0, n, 0, n);
        for (int i = 0; i < xspan * n * n; i++) {
            delta2[i].real() -= thisDelta2Part[i] * thisDelta2Part[i];
        }
    } else {
        if (phase % 2 == 0) {
            delta2Part = fourier3dReadReal(xbegin, xend, 0, n, 0, n);
        } else {
            auto delta2ThisPart = fourier3dReadReal(xbegin, xend, 0, n, 0, n);
            for (int i = 0; i < xspan * n * n; i++) {
                delta2[i].real() += delta2Part[i] * delta2ThisPart[i];
            }
        }
    }
    hpx::wait_all(futs.begin(), futs.end());
}

static double constrainRange(double& r) {
    while (r >= 1.0) {
        r -= 1.;
    }
    while (r < 0.) {
        r += 1.;
    }
    return r;
}

float phi1ToParticles(int n, float boxSize, float d1, float prefactor, int dim) {
    std::vector<hpx::future<float>> futs;
    if (hpxRank() == 0) {
        for (int i = 1; i < hpxSize(); i++) {
            futs.push_back(hpx::async<Phi1ToParticlesAction>(
                hpxLocalities()[i], n, boxSize, d1, prefactor, dim));
        }
    }
    double dmax = 0.0f;
    particle_server pserv;
    particle_set& parts = pserv.get_particle_set();
    int xb = hpxRank() * n / hpxSize();
    int xe = (hpxRank() + 1) * n / hpxSize();
    int xspan = xe - xb;
    PartInt count = xspan * n * n;
    if (dim == 0) {
        parts.resize(count);
        for (int xi = xb; xi < xe; xi++) {
            float const x = (float(xi) + 0.5f) / float(n);
            for (int yi = 0; yi < n; yi++) {
                float const y = (float(yi) + 0.5f) / float(n);
                for (int zi = 0; zi < n; zi++) {
                    float const z = (float(zi) + 0.5f) / float(n);
                    PartInt index = n * n * (xi - xb) + n * yi + zi;
                    parts.pos(0, index) = x;
                    parts.pos(1, index) = y;
                    parts.pos(2, index) = z;
                }
            }
        }
    }
    auto phi1 = fourier3dReadReal(xb, xe, 0, n, 0, n);
    float const factor = -d1 / boxSize;
    for (int xi = xb; xi < xe; xi++) {
        float const x = (float(xi) + 0.5f) / float(n);
        for (int yi = 0; yi < n; yi++) {
            float const y = (float(yi) + 0.5f) / float(n);
            for (int zi = 0; zi < n; zi++) {
                float const z = (float(zi) + 0.5f) / float(n);
                PartInt i0 = n * n * (xi - xb) + n * yi + zi;
                double const dx = phi1[i0] * factor;
                double x = parts.pos(dim, i0).toDouble() + dx;
                constrainRange(x);
                parts.pos(dim, i0) = x;
                parts.vel(dim, i0) = prefactor * dx;
                dmax = std::max(dmax, dx);
            }
        }
    }
    dmax *= n;
    for (auto& f : futs) {
        dmax = std::max(f.get(), (float) dmax);
    }
    return dmax;
}

float phi2ToParticles(int n, float boxSize, float d2, float prefactor, int dim) {
    std::vector<hpx::future<float>> futs;
    if (hpxRank() == 0) {
        for (int i = 1; i < hpxSize(); i++) {
            futs.push_back(hpx::async<Phi2ToParticlesAction>(
                hpxLocalities()[i], n, boxSize, d2, prefactor, dim));
        }
    }
    double dmax = 0.0f;
    particle_server pserv;
    particle_set& parts = pserv.get_particle_set();
    int xb = hpxRank() * n / hpxSize();
    int xe = (hpxRank() + 1) * n / hpxSize();
    int xspan = xe - xb;
    PartInt count = xspan * n * n;
    auto phi2 = fourier3dReadReal(xb, xe, 0, n, 0, n);
    float const factor = d2 / boxSize;
    for (int xi = xb; xi < xe; xi++) {
        float const x = (float(xi) + 0.5f) / float(n);
        for (int yi = 0; yi < n; yi++) {
            float const y = (float(yi) + 0.5f) / float(n);
            for (int zi = 0; zi < n; zi++) {
                float const z = (float(zi) + 0.5f) / float(n);
                PartInt i0 = n * n * (xi - xb) + n * yi + zi;
                double const dx = phi2[i0] * factor;
                double x = parts.pos(dim, i0).toDouble() + dx;
                constrainRange(x);
                parts.pos(dim, i0) = x;
                parts.vel(dim, i0) += prefactor * dx;
                dmax = std::max(dmax, dx);
            }
        }
    }
    dmax *= n;
    for (auto& f : futs) {
        dmax = std::max(f.get(), (float) dmax);
    }
    return dmax;
}

void twolpt(InterpFunctor<float> const denK, int n, float boxSize, int dim1, int dim2, int seed) {
    if (hpxRank() == 0) {
        fourier3dInitialize(n);
    }
    std::vector<hpx::future<void>> futs;
    if (hpxRank() == 0) {
        for (int i = 1; i < hpxSize(); i++) {
            futs.push_back(
                hpx::async<TwolptAction>(hpxLocalities()[i], denK, n, boxSize, dim1, dim2, seed));
        }
    }
    Vector<Cmplx> yValue;
    int xbegin = hpxRank() * n / hpxSize();
    int xend = (hpxRank() + 1) * n / hpxSize();
    int xspan = xend - xbegin;
    yValue.resize(xspan * n * n);
    float const factor = std::pow(boxSize, -1.5);
    execute_2lpt_kernel(yValue.data(), xbegin, xend, denK, n, boxSize, dim1, dim2);
    fourier3dAccumulate(xbegin, xend, 0, n, 0, n, std::move(yValue));
    hpx::wait_all(futs.begin(), futs.end());
    if (hpxRank() == 0) {
        fourier3dMirror();
        fourier3dExecute();
    }
}
void twolptCorrectionF2delta2(int n);
HPX_PLAIN_ACTION(twolptCorrectionF2delta2, TwolptCorrectionF2delta2Action);

void twolptCorrectionF2delta2(int n) {
    std::vector<hpx::future<void>> futs;
    if (hpxRank() == 0) {
        for (int i = 1; i < hpxSize(); i++) {
            futs.push_back(hpx::async<TwolptCorrectionF2delta2Action>(hpxLocalities()[i], n));
        }
    }
    int xbegin = hpxRank() * n / hpxSize();
    int xend = (hpxRank() + 1) * n / hpxSize();
    delta2 = fourier3dRead(xbegin, xend, 0, n, 0, n);
    hpx::wait_all(futs.begin(), futs.end());
}

void twolptCorrection1(int n, float boxSize) {
    if (hpxRank() == 0) {
        fourier3dInitialize(n);
    }
    std::vector<hpx::future<void>> futs;
    if (hpxRank() == 0) {
        for (int i = 1; i < hpxSize(); i++) {
            futs.push_back(hpx::async<TwolptCorrection1Action>(hpxLocalities()[i], n, boxSize));
        }
    }
    int xbegin = hpxRank() * n / hpxSize();
    int xend = (hpxRank() + 1) * n / hpxSize();
    fourier3dAccumulate(xbegin, xend, 0, n, 0, n, std::move(delta2));
    hpx::wait_all(futs.begin(), futs.end());
    if (hpxRank() == 0) {
        fourier3dInvExecute();
    }
}

void twolptCorrection2(int n, float boxSize, int dim) {
    int xbegin = hpxRank() * n / hpxSize();
    int xend = (hpxRank() + 1) * n / hpxSize();
    if (hpxRank() == 0) {
        if (dim == 0) {
            twolptCorrectionF2delta2(n);
        }
        fourier3dInitialize(n);
    }
    std::vector<hpx::future<void>> futs;
    if (hpxRank() == 0) {
        for (int i = 1; i < hpxSize(); i++) {
            futs.push_back(
                hpx::async<TwolptCorrection2Action>(hpxLocalities()[i], n, boxSize, dim));
        }
    }
    auto yValue = delta2;
    execute_2lpt_correction_kernel(yValue.data(), xbegin, xend, n, boxSize, dim);
    fourier3dAccumulate(xbegin, xend, 0, n, 0, n, std::move(yValue));
    hpx::wait_all(futs.begin(), futs.end());
    if (hpxRank() == 0) {
        fourier3dExecute();
    }
}
