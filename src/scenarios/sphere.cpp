/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <algorithm>
#include <cmath>
#include <numbers>
#include <vector>

#include <hpx/future.hpp>

#include <cosmictiger/core/safe_io.hpp>
#include <cosmictiger/cuda/cuda.hpp>
#include <cosmictiger/scenarios/sphere.hpp>

namespace detail {

// Rows of the pairwise force pass handed to one HPX task.
constexpr int kSphereSurfaceRowsPerTask = 256;

struct SphereSurfaceShell {
    float* x;
    float* y;
    float* z;
    int count;
    float h;
    float mass;
};

struct SphereSurfaceForceSums {
    float gmax;
    double phi;
};

// Accumulates the softened repulsion on rows [begin, end) into gx, gy, gz.
static SphereSurfaceForceSums sphereSurfaceForces(SphereSurfaceShell const& shell, float* gx,
    float* gy, float* gz, int const begin, int const end) {
    float const* const x = shell.x;
    float const* const y = shell.y;
    float const* const z = shell.z;
    float const h2 = shell.h * shell.h;
    float const hinv = 1.f / shell.h;
    float const h2inv = hinv * hinv;
    float const h3inv = h2inv * hinv;
    float const m = shell.mass;
    SphereSurfaceForceSums sums{0.f, 0.0};
    for (int i = begin; i < end; i++) {
        float const xi = x[i];
        float const yi = y[i];
        float const zi = z[i];
        float fx = 0.f;
        float fy = 0.f;
        float fz = 0.f;
        float phi = 0.f;
        auto const accumulate = [&](int const jBegin, int const jEnd) {
            for (int j = jBegin; j < jEnd; j++) {
                float const dx = xi - x[j];
                float const dy = yi - y[j];
                float const dz = zi - z[j];
                float const r2 = dx * dx + dy * dy + dz * dz;
                bool const far = r2 > h2;
                float const farRinv = 1.f / std::sqrt(far ? r2 : h2);
                float const rinv = far ? farRinv : (1.5f - 0.5f * r2 * h2inv) * hinv;
                float const r3inv = far ? farRinv * farRinv * farRinv : h3inv;
                fx += m * r3inv * dx;
                fy += m * r3inv * dy;
                fz += m * r3inv * dz;
                phi += m * rinv;
            }
        };
        accumulate(0, i);
        accumulate(i + 1, shell.count);
        gx[i] = fx;
        gy[i] = fy;
        gz[i] = fz;
        sums.gmax = std::max(sums.gmax, std::sqrt(fx * fx + fy * fy + fz * fz));
        sums.phi += phi;
    }
    return sums;
}

// Mirrors sphereSurfaceKernel in sphere.cu for one shell.
static void relaxSphereSurfaceShell(SphereSurfaceShell shell) {
    int const n = shell.count;
    float* const x = shell.x;
    float* const y = shell.y;
    float* const z = shell.z;
    for (int stream = 0; stream < std::min(n, kSphereSurfaceStreams); stream++) {
        unsigned state = stream;
        for (int i = stream; i < n; i += kSphereSurfaceStreams) {
            x[i] = sphereSurfaceRandom(state);
            y[i] = sphereSurfaceRandom(state);
            z[i] = sphereSurfaceRandom(state);
        }
    }
    auto const project = [x, y, z](int const i) {
        float const inv = 1.f / std::sqrt(x[i] * x[i] + y[i] * y[i] + z[i] * z[i]);
        x[i] *= inv;
        y[i] *= inv;
        z[i] *= inv;
    };
    // One point feels no force, and its zero potential would make the
    // convergence ratio 0/0, which -ffast-math does not order.
    if (n < 2) {
        for (int i = 0; i < n; i++) {
            project(i);
        }
        return;
    }
    shell.h = kSphereSurfaceSoftening * std::sqrt(4.f * std::numbers::pi_v<float> / n);
    shell.mass = 1.0f / n;
    std::vector<float> vx(n, 0.f);
    std::vector<float> vy(n, 0.f);
    std::vector<float> vz(n, 0.f);
    std::vector<float> gx(n);
    std::vector<float> gy(n);
    std::vector<float> gz(n);
    int const tasks = (n + kSphereSurfaceRowsPerTask - 1) / kSphereSurfaceRowsPerTask;
    std::vector<hpx::future<SphereSurfaceForceSums>> futs;
    futs.reserve(tasks);
    int iter = 0;
    float error;
    do {
        futs.clear();
        for (int task = 1; task < tasks; task++) {
            int const begin = task * kSphereSurfaceRowsPerTask;
            int const end = std::min(n, begin + kSphereSurfaceRowsPerTask);
            futs.push_back(hpx::async([&shell, &gx, &gy, &gz, begin, end]() {
                return sphereSurfaceForces(shell, gx.data(), gy.data(), gz.data(), begin, end);
            }));
        }
        SphereSurfaceForceSums sums = sphereSurfaceForces(
            shell, gx.data(), gy.data(), gz.data(), 0, std::min(n, kSphereSurfaceRowsPerTask));
        for (auto& fut : futs) {
            auto const part = fut.get();
            sums.gmax = std::max(sums.gmax, part.gmax);
            sums.phi += part.phi;
        }
        float const dt = kSphereSurfaceEta *
            std::min(std::sqrt(shell.h / sums.gmax), 1.f / kSphereSurfaceDamping);
        float const halfdt = 0.5f * dt;
        float const deninv = 1.f / (1.f + halfdt * kSphereSurfaceDamping);
        double kin = 0.0;
        for (int i = 0; i < n; i++) {
            vx[i] = (vx[i] + halfdt * gx[i]) * deninv;
            vy[i] = (vy[i] + halfdt * gy[i]) * deninv;
            vz[i] = (vz[i] + halfdt * gz[i]) * deninv;
            x[i] += vx[i] * dt;
            y[i] += vy[i] * dt;
            z[i] += vz[i] * dt;
            vx[i] = (vx[i] + halfdt * gx[i]) * deninv;
            vy[i] = (vy[i] + halfdt * gy[i]) * deninv;
            vz[i] = (vz[i] + halfdt * gz[i]) * deninv;
            project(i);
            float const vr = x[i] * vx[i] + y[i] * vy[i] + z[i] * vz[i];
            vx[i] -= x[i] * vr;
            vy[i] -= y[i] * vr;
            vz[i] -= z[i] * vr;
            kin += 0.5f * shell.mass * (vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i]);
        }
        error = kin / sums.phi;
        iter++;
    } while (error > kSphereSurfaceTolerance || iter < kSphereSurfaceMinIterations);
    PRINT("Solved for %i in %i iterations\n", n, iter);
}

}    // namespace detail

void solveSphereSurfaceProblemCpu(Vector<Array<Vector<float>, kDimensions>>& x) {
    std::vector<hpx::future<void>> futs;
    futs.reserve(x.size());
    for (auto& shell : x) {
        detail::SphereSurfaceShell const points{shell[kXDimension].data(),
            shell[kYDimension].data(), shell[kZDimension].data(),
            static_cast<int>(shell[kXDimension].size()), 0.f, 0.f};
        futs.push_back(hpx::async(detail::relaxSphereSurfaceShell, points));
    }
    hpx::wait_all(futs);
    for (auto& fut : futs) {
        fut.get();
    }
}

void solveSphereSurfaceProblem(Vector<Array<Vector<float>, kDimensions>>& x) {
#ifdef USE_CUDA
    if (cudaEnabled()) {
        solveSphereSurfaceProblemCuda(x);
        return;
    }
#endif
    solveSphereSurfaceProblemCpu(x);
}
