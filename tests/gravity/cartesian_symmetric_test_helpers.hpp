/* CosmicTiger - full Cartesian operator test fixtures; BSL-1.0. */

#pragma once

#include <algorithm>
#include <cmath>
#include <type_traits>

#include <cosmictiger/gravity/kernels/cartesian_operators.hpp>
#include <cosmictiger/math/simd_math.hpp>

namespace detail {

namespace symmetric = cosmictiger::gravity::cartesian;

template <class Real, int order>
struct SymmetricResult {
    symmetric::Multipole<Real, order> moments;
    symmetric::Multipole<Real, order> shiftedMoments;
    symmetric::Green<Real, order> green;
    symmetric::Expansion<Real, order> pointLocal;
    symmetric::Expansion<Real, order> local;
    symmetric::Expansion<Real, order> centeredLocal;
    symmetric::Expansion<Real, order> shiftedLocal;
    cosmictiger::gravity::Field<Real> field;
    cosmictiger::gravity::Field<Real> pointField;
    cosmictiger::gravity::Field<Real> centeredField;
    cosmictiger::gravity::Field<Real> noPotentialField;
    cosmictiger::gravity::Field<Real> quadraticField;
};

// Different samples (and different SIMD lanes) exercise different geometry.
// Nonzero destination seeds also cover every operator's accumulation contract.
template <class Real, int order>
CUDA_EXPORT void evaluateSymmetricOperators(Real sample, SymmetricResult<Real, order>& result) {
    auto const value = [](double x) { return simdBroadcast<Real>(x); };
    Vector3<Real> const position{value(0.037) + sample * value(0.001), value(-0.029), value(0.021)};
    Vector3<Real> const shift{value(0.024), value(-0.018) + sample * value(0.0007), value(0.031)};
    Vector3<Real> const separation{
        value(0.8) + sample * value(0.011), value(-0.4), value(0.9) - sample * value(0.003)};
    Real const mass = value(1.37) + sample * value(0.007);
    result.moments = value(0.017);
    result.shiftedMoments = value(0.019);
    result.pointLocal = value(-0.023);
    result.local = value(0.027);
    result.centeredLocal = value(-0.031);
    result.shiftedLocal = value(0.037);
    for (int component = 0; component < 4; ++component) {
        result.field[component] = value(0.041);
        result.pointField[component] = value(-0.043);
        result.centeredField[component] = value(0.047);
        result.noPotentialField[component] = value(0.053);
        result.quadraticField[component] = value(0);
    }
    P2M(result.moments, mass, position);
    M2M(result.shiftedMoments, result.moments, shift);
    symmetric::evaluateGreen(result.green, separation, value(0.3));
    P2L(result.pointLocal, mass, result.green);
    M2L(result.local, result.shiftedMoments, result.green);
    M2L<true>(result.centeredLocal, result.shiftedMoments, result.green);
    L2L(result.shiftedLocal, result.local, shift);
    L2P(result.field, result.shiftedLocal, position);
    M2P(result.pointField, result.shiftedMoments, result.green);
    M2P<true>(result.centeredField, result.shiftedMoments, result.green);
    M2P(result.noPotentialField, result.shiftedMoments, result.green, false);
    L2P(result.noPotentialField, result.shiftedLocal, position, false);
    symmetric::Multipole<Real, order> pointMoment{};
    symmetric::Green<Real, order> quadratic{};
    pointMoment = value(0);
    quadratic = value(0);
    P2M(pointMoment, mass, position);
    quadratic[0] = separation[0] * separation[0] + separation[1] * separation[1] +
        separation[2] * separation[2];
    quadratic(1, 0, 0) = value(2) * separation[0];
    quadratic(0, 1, 0) = value(2) * separation[1];
    quadratic(0, 0, 1) = value(2) * separation[2];
    quadratic(2, 0, 0) = quadratic(0, 2, 0) = quadratic(0, 0, 2) = value(2);
    M2P(result.quadraticField, pointMoment, quadratic);
}

// Host-only traversal keeps CUDA results strongly typed without relying on
// object padding or reinterpreting a complete object as an array of scalars.
template <class Real, class Reference, int order, class Compare>
void compareSymmetricResults(SymmetricResult<Real, order> const& actual,
    SymmetricResult<Reference, order> const& expected, Compare&& compare) {
    auto tensor = [&](auto const& left, auto const& right) {
        using Tensor = std::remove_cvref_t<decltype(right)>;
        for (int rank = 0; rank <= Tensor::kDegree_; ++rank) {
            double scale = 1.0;
            for (int x = 0; x <= rank; ++x) {
                for (int y = 0; y <= rank - x; ++y) {
                    scale = std::max(scale, std::abs(double(right(x, y, rank - x - y))));
                }
            }
            for (int x = 0; x <= rank; ++x) {
                for (int y = 0; y <= rank - x; ++y) {
                    compare(left(x, y, rank - x - y), right(x, y, rank - x - y), scale);
                }
            }
        }
    };
    tensor(actual.moments, expected.moments);
    tensor(actual.shiftedMoments, expected.shiftedMoments);
    tensor(actual.green, expected.green);
    tensor(actual.pointLocal, expected.pointLocal);
    tensor(actual.local, expected.local);
    tensor(actual.centeredLocal, expected.centeredLocal);
    tensor(actual.shiftedLocal, expected.shiftedLocal);
    for (int component = 0; component < 4; ++component) {
        compare(actual.field[component], expected.field[component], 1.0);
        compare(actual.pointField[component], expected.pointField[component], 1.0);
        compare(actual.centeredField[component], expected.centeredField[component], 1.0);
        compare(actual.noPotentialField[component], expected.noPotentialField[component], 1.0);
        compare(actual.quadraticField[component], expected.quadraticField[component], 1.0);
    }
}

}    // namespace detail
