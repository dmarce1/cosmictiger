/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

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

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include <cosmictiger/sph_cuda.hpp>

// Only the options-constructor and host assertion services are replaced here.
// Density, smoothing, forces, heating, device assertions, and allocation use the
// production translation units. The test does not start an HPX runtime.
Options const& getOptions() {
    static Options const options{};
    return options;
}

void cosmictigerAssert(char const* condition, bool value, char const* file, int line) {
    if (!value) {
        throw std::runtime_error(std::string(file) + ":" + std::to_string(line) + ": " + condition);
    }
}

namespace detail {

constexpr double kPi = 3.141592653589793238462643383279502884;

void checkCuda(cudaError_t status, char const* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

void require(bool condition, char const* label) {
    if (!condition) {
        throw std::runtime_error(label);
    }
}

void requireClose(char const* label, double actual, double expected, double tolerance) {
    if (!std::isfinite(actual) ||
        std::abs(actual - expected) > tolerance * std::max(1.0, std::abs(expected))) {
        throw std::runtime_error(std::string(label) + ": got " + std::to_string(actual) +
            ", expected " + std::to_string(expected));
    }
}

class Fixture {
    std::vector<void*> allocations_;

    template <typename Value>
    Value* allocate(size_t count) {
        Value* pointer = nullptr;
        checkCuda(cudaMallocManaged(&pointer, count * sizeof(Value)), "managed test allocation");
        allocations_.push_back(pointer);
        for (size_t i = 0; i < count; i++) {
            new (pointer + i) Value{};
        }
        return pointer;
    }

    void resetReduction() {
        *reduction_ = {};
        reduction_->hmin = std::numeric_limits<float>::max();
    }

public:
    int count_;
    SphRunParameters params_;
    SphRunCudaData data_{};
    SphReduction* reduction_;

    explicit Fixture(int count, int treeCount = 1)
      : count_(count) {
        data_.x = allocate<Fixed32>(count);
        data_.y = allocate<Fixed32>(count);
        data_.z = allocate<Fixed32>(count);
        data_.vx = allocate<float>(count);
        data_.vy = allocate<float>(count);
        data_.vz = allocate<float>(count);
        data_.eint = allocate<float>(count);
        data_.h = allocate<float>(count);
        data_.rho = data_.rhoSnk = allocate<float>(count);
        data_.preSnk = allocate<float>(count);
        data_.omegaSnk = allocate<float>(count);
        data_.omegaPSnk = allocate<float>(count);
        data_.shearSnk = allocate<float>(count);
        data_.divvSnk = allocate<float>(count);
        data_.deint1Snk = allocate<float>(count);
        data_.deint2Snk = allocate<float>(count);
        data_.rec1Snk = allocate<SphRecord1>(count);
        data_.rec2Snk = allocate<SphRecord2>(count);
        data_.rec6Snk = allocate<SphRecord6>(count);
        data_.rungs = data_.rungsSnk = allocate<char>(count);
        data_.convergedSnk = allocate<char>(count);
        data_.saSnk = allocate<char>(count);
        data_.dmIndexSnk = allocate<ParticleInteger>(count);
        data_.trees = allocate<SphTreeNode>(treeCount);
        data_.selfs = allocate<int>(treeCount);
        data_.neighbors = allocate<int>(treeCount);
        reduction_ = allocate<SphReduction>(1);
        data_.numberOfTargetNodes = treeCount;
        data_.m = 1.0f / count;
        data_.n = 32.0f;
        data_.defGamma = 5.0f / 3.0f;
        data_.eta = 0.25f;
        data_.gsoft = 0.05f;
        params_.a = 1.0f;
        params_.adot = 0.0f;
        params_.minRung = 1;
        params_.maxRung = 1;
        params_.t0 = 1.0f;
        params_.maxDt = 1.0f;
        params_.phase = 0;
        params_.cfl = 0.25f;
        params_.alpha0 = 1.0f;
        params_.beta = 2.0f;
        params_.viscosityEpsilon = 0.01f;
        params_.hmin = 0.1f;
        params_.hmax = 0.49f;
        for (int i = 0; i < count; i++) {
            data_.dmIndexSnk[i] = i;
            data_.rungs[i] = 1;
            data_.eint[i] = 1.0f;
            data_.h[i] = data_.rec2Snk[i].h = 0.2f;
        }
        for (int tree = 0; tree < treeCount; tree++) {
            data_.selfs[tree] = data_.neighbors[tree] = tree;
            auto& node = data_.trees[tree];
            node.particleRange = {tree * count / treeCount, (tree + 1) * count / treeCount};
            node.sinkParticleRange = node.particleRange;
            node.neighborRange = {0, treeCount};
            for (int dim = 0; dim < NDIM; dim++) {
                node.innerBox.begin[dim] = 0.0;
                node.innerBox.end[dim] = 1.0;
                node.outerBox.begin[dim] = -1.0;
                node.outerBox.end[dim] = 2.0;
            }
        }
    }

    Fixture(Fixture const&) = delete;
    Fixture& operator=(Fixture const&) = delete;

    ~Fixture() {
        for (void* allocation : allocations_) {
            cudaFree(allocation);
        }
    }

    void solveSmoothing(float minimum, float maximum, float initial) {
        params_.hmin = minimum;
        params_.hmax = maximum;
        for (int i = 0; i < count_; i++) {
            data_.convergedSnk[i] = false;
            data_.h[i] = data_.rec2Snk[i].h = initial;
        }
        resetReduction();
        sphCudaPrehydro1<<<1, PREHYDRO1_BLOCK_SIZE>>>(params_, data_, reduction_);
        checkCuda(cudaGetLastError(), "prehydro1 launch");
        checkCuda(cudaDeviceSynchronize(), "prehydro1 execution");
        require(reduction_->flag == 0, "complete neighbor fixture must converge without refresh");
        for (int i = 0; i < count_; i++) {
            require(data_.convergedSnk[i], "smoothing solve did not mark convergence");
            data_.h[i] = data_.rec2Snk[i].h;
            require(data_.h[i] >= minimum && data_.h[i] <= maximum,
                "smoothing solve exceeded configured bounds");
        }
    }

    void density() {
        resetReduction();
        sphCudaPrehydro2<<<1, PREHYDRO2_BLOCK_SIZE>>>(params_, data_, reduction_);
        checkCuda(cudaGetLastError(), "prehydro2 launch");
        checkCuda(cudaDeviceSynchronize(), "prehydro2 execution");
        require(reduction_->flag == 0, "density fixture unexpectedly needs neighbor refresh");
        for (int i = 0; i < count_; i++) {
            require(data_.rho[i] > 0.0f && std::isfinite(data_.rho[i]), "positive finite density");
            requireClose("ideal-gas pressure", data_.preSnk[i],
                (data_.defGamma - 1.0f) * data_.rho[i] * data_.eint[i], 2.0e-6);
            require(data_.omegaSnk[i] == 1.0f && data_.omegaPSnk[i] == 1.0f,
                "classical SPH must not apply grad-h correction");
        }
    }

    void hydro(int phase = 0) {
        params_.phase = phase;
        resetReduction();
        sphCudaHydro<<<1, HYDRO_BLOCK_SIZE>>>(params_, data_, reduction_);
        checkCuda(cudaGetLastError(), "hydro launch");
        checkCuda(cudaDeviceSynchronize(), "hydro execution");
        for (int i = 0; i < count_; i++) {
            require(std::isfinite(data_.deint2Snk[i]), "finite active-kernel thermal rate");
            for (int dim = 0; dim < NDIM; dim++) {
                require(
                    std::isfinite(data_.rec6Snk[i].dvel[dim]), "finite active-kernel acceleration");
            }
            if (phase == 1) {
                require(data_.rungs[i] >= 1 && data_.rungs[i] < MAX_RUNG,
                    "valid active-kernel timestep rung");
            }
        }
    }

    void checkConservation() const {
        std::array<double, NDIM> momentum{};
        double energy = 0.0;
        double absoluteEnergy = 0.0;
        double absoluteForce = 0.0;
        for (int i = 0; i < count_; i++) {
            auto const& acceleration = data_.rec6Snk[i].dvel;
            double const kinetic = data_.m *
                (data_.vx[i] * acceleration[XDIM] + data_.vy[i] * acceleration[YDIM] +
                    data_.vz[i] * acceleration[ZDIM]);
            double const thermal = data_.m * data_.deint2Snk[i];
            energy += kinetic + thermal;
            absoluteEnergy += std::abs(kinetic) + std::abs(thermal);
            for (int dim = 0; dim < NDIM; dim++) {
                momentum[dim] += data_.m * acceleration[dim];
                absoluteForce += std::abs(data_.m * acceleration[dim]);
            }
        }
        for (int dim = 0; dim < NDIM; dim++) {
            requireClose("active-kernel total momentum rate", momentum[dim], 0.0,
                3.0e-5 * std::max(1.0, absoluteForce));
        }
        requireClose(
            "active-kernel total energy rate", energy, 0.0, 3.0e-5 * std::max(1.0, absoluteEnergy));
    }
};

void initializeLattice(Fixture& fixture, int side) {
    int i = 0;
    for (int ix = 0; ix < side; ix++) {
        for (int iy = 0; iy < side; iy++) {
            for (int iz = 0; iz < side; iz++) {
                fixture.data_.x[i] = (ix + 0.5) / side;
                fixture.data_.y[i] = (iy + 0.5) / side;
                fixture.data_.z[i] = (iz + 0.5) / side;
                i++;
            }
        }
    }
}

void testPeriodicSupport() {
    Fixed32Range box;
    Array<Fixed32, NDIM> source;
    for (int dim = 0; dim < NDIM; dim++) {
        box.begin[dim] = 0.49;
        box.end[dim] = 0.51;
        source[dim] = 0.5;
    }
    box.begin[XDIM] = 0.90;
    box.end[XDIM] = 0.95;
    source[XDIM] = 0.40;
    require(sphSupportIntersects(box, source, 0.49f),
        "periodic support reaches the far end of a sink box");
    require(!sphSupportIntersects(box, source, 0.40f),
        "periodic support excludes a genuinely separated box");
    source[XDIM] = 0.02;
    require(sphSupportIntersects(box, source, 0.10f), "periodic support crosses the box boundary");
}

void testLattice() {
    constexpr int kSide = 5;
    Fixture fixture(kSide * kSide * kSide);
    initializeLattice(fixture, kSide);
    fixture.solveSmoothing(0.1f, 0.24f, 0.2f);
    for (int i = 0; i < fixture.count_; i++) {
        requireClose("upper support bound", fixture.data_.h[i], 0.24f, 1.0e-7);
    }
    fixture.solveSmoothing(0.43f, 0.49f, 0.48f);
    for (int i = 0; i < fixture.count_; i++) {
        requireClose("lower support bound", fixture.data_.h[i], 0.43f, 1.0e-7);
    }
    fixture.solveSmoothing(0.3f, 0.3f, 0.17f);
    for (int i = 0; i < fixture.count_; i++) {
        requireClose("fixed support bound", fixture.data_.h[i], 0.3f, 1.0e-7);
    }
    fixture.solveSmoothing(0.1f, 0.49f, 0.2f);
    fixture.density();
    for (int i = 0; i < fixture.count_; i++) {
        double const support = fixture.data_.h[i];
        double const count =
            4.0 * kPi / 3.0 * fixture.data_.rho[i] * support * support * support / fixture.data_.m;
        requireClose("weighted neighbor target", count, fixture.data_.n, 2.0e-4);
        requireClose("uniform lattice density", fixture.data_.rho[i], fixture.data_.rho[0], 2.0e-5);
        requireClose("uniform lattice continuum density", fixture.data_.rho[i], 1.0, 0.03);
    }
    fixture.hydro(1);
    for (int i = 0; i < fixture.count_; i++) {
        requireClose("uniform gas thermal rate", fixture.data_.deint2Snk[i], 0.0, 2.0e-5);
        for (int dim = 0; dim < NDIM; dim++) {
            requireClose("uniform gas force", fixture.data_.rec6Snk[i].dvel[dim], 0.0, 2.0e-4);
        }
    }
    require(fixture.reduction_->dtinvCfl > 0.0f && fixture.reduction_->dtinvVisc > 0.0f,
        "active timestep computes sound and viscosity restrictions");

    // A smooth periodic perturbation exercises nonzero work and force for every target.
    for (int i = 0; i < fixture.count_; i++) {
        double const angle = 2.0 * kPi * fixture.data_.x[i].toDouble();
        fixture.data_.vx[i] = 0.02 * std::sin(angle);
        fixture.data_.eint[i] = 1.0 + 0.03 * std::cos(angle);
    }
    fixture.hydro();
    fixture.checkConservation();
    double forceNorm = 0.0;
    for (int i = 0; i < fixture.count_; i++) {
        forceNorm += std::abs(fixture.data_.rec6Snk[i].dvel[XDIM]);
    }
    require(forceNorm > 0.1, "perturbed gas must exercise nonzero acceleration");

    // The physical expansion term is tested at nonunit a and nonmonatomic gamma.
    fixture.data_.defGamma = 1.4f;
    fixture.params_.a = 0.6f;
    fixture.params_.adot = 0.09f;
    for (int i = 0; i < fixture.count_; i++) {
        fixture.data_.vx[i] = 0.0f;
        fixture.data_.eint[i] = 2.0f;
    }
    fixture.density();
    fixture.hydro();
    double const background =
        (5.0f - 3.0f * fixture.data_.defGamma) * fixture.params_.adot / fixture.params_.a * 2.0f;
    for (int i = 0; i < fixture.count_; i++) {
        requireClose("homogeneous cosmological energy source", fixture.data_.deint2Snk[i],
            background, 2.0e-6);
    }
}

void testUnequalSupports() {
    Fixture fixture(2, 2);
    auto& data = fixture.data_;
    data.x[0] = 0.35;
    data.x[1] = 0.65;
    for (int i = 0; i < 2; i++) {
        data.y[i] = data.z[i] = 0.5;
        data.convergedSnk[i] = true;
        data.eint[i] = i == 0 ? 0.7f : 1.2f;
        data.h[i] = data.rec2Snk[i].h = i == 0 ? 0.2f : 0.45f;
        auto& node = data.trees[i];
        for (int dim = 0; dim < NDIM; dim++) {
            double const position = dim == XDIM ? data.x[i].toDouble() : 0.5;
            node.innerBox.begin[dim] = node.innerBox.end[dim] = position;
            node.outerBox.begin[dim] = position - data.h[i];
            node.outerBox.end[dim] = position + data.h[i];
        }
    }
    fixture.density();
    require(data.rho[0] != data.rho[1], "unequal supports must produce unequal densities");
    data.vx[0] = 0.1f;
    data.vx[1] = -0.1f;
    fixture.hydro();
    fixture.checkConservation();
    require(data.rec6Snk[0].dvel[XDIM] < 0.0f && data.rec6Snk[1].dvel[XDIM] > 0.0f,
        "union support must include both directions of repulsive force");
    std::array<float, 2> const viscousEnergy = {data.deint2Snk[0], data.deint2Snk[1]};
    fixture.params_.alpha0 = fixture.params_.beta = 0.0f;
    fixture.hydro();
    for (int i = 0; i < 2; i++) {
        require(viscousEnergy[i] > data.deint2Snk[i], "approach must add viscosity heating");
    }
    data.vx[0] = -0.1f;
    data.vx[1] = 0.1f;
    fixture.hydro();
    std::array<float, 2> const inviscidEnergy = {data.deint2Snk[0], data.deint2Snk[1]};
    fixture.params_.alpha0 = 1.0f;
    fixture.params_.beta = 2.0f;
    fixture.hydro();
    fixture.checkConservation();
    for (int i = 0; i < 2; i++) {
        requireClose("separating pair has no viscosity heating", data.deint2Snk[i],
            inviscidEnergy[i], 2.0e-6);
    }
}

}    // namespace detail

int main() {
    try {
        int deviceCount = 0;
        detail::checkCuda(cudaGetDeviceCount(&deviceCount), "CUDA device discovery");
        detail::require(deviceCount > 0, "CUDA regression requires a device");
        detail::checkCuda(cudaSetDevice(0), "select CUDA device");
        cudaMemoryInitialize(4 * 1024 * 1024);
        detail::checkCuda(cudaGetLastError(), "production workspace allocator initialization");
        detail::testPeriodicSupport();
        detail::testLattice();
        detail::testUnequalSupports();
        std::puts("Monaghan SPH active CUDA kernel regressions passed (one block per launch)");
    } catch (std::exception const& error) {
        std::fprintf(stderr, "Monaghan SPH CUDA regression failed: %s\n", error.what());
        return 1;
    }
    return 0;
}
