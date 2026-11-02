#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <limits>

#include <gtest/gtest.h>

#include <cosmictiger/math/matrix_eigen.hpp>

namespace detail {

template <class Real, int dimensions>
void checkEigenpairs(math::Matrix<Real, dimensions> const& input,
    math::SymmetricEigenpairs<Real, dimensions> const& eigenpairs, Real tolerance) {
    ASSERT_TRUE(eigenpairs.converged);
    Real scale = Real(0);
    for (int row = 0; row < dimensions; ++row) {
        for (int column = 0; column < dimensions; ++column) {
            scale = std::max(scale, std::abs(input(row, column)));
        }
    }
    for (int axis = 0; axis < dimensions; ++axis) {
        if (axis > 0) {
            EXPECT_GE(eigenpairs.values[axis - 1], eigenpairs.values[axis]);
        }
        for (int row = 0; row < dimensions; ++row) {
            Real product = Real(0);
            for (int column = 0; column < dimensions; ++column) {
                product += input(row, column) * eigenpairs.vectors(column, axis);
            }
            EXPECT_NEAR(product, eigenpairs.values[axis] * eigenpairs.vectors(row, axis),
                tolerance * scale);
        }
        for (int other = 0; other < dimensions; ++other) {
            Real product = Real(0);
            for (int component = 0; component < dimensions; ++component) {
                product +=
                    eigenpairs.vectors(component, axis) * eigenpairs.vectors(component, other);
            }
            EXPECT_NEAR(product, axis == other ? Real(1) : Real(0), tolerance);
        }
    }
}

template <class Real>
void checkScaledEigenpairs() {
    // This rotated indefinite tensor exercises the general dimension, sign,
    // descending order, and scale handling independently of the 3D callers.
    math::Matrix<Real, 4> const rotation{{Real(0.5), Real(0.5), Real(0.5), Real(0.5)},
        {Real(0.5), Real(-0.5), Real(0.5), Real(-0.5)},
        {Real(0.5), Real(0.5), Real(-0.5), Real(-0.5)},
        {Real(0.5), Real(-0.5), Real(-0.5), Real(0.5)}};
    math::DiagonalMatrix<Real, 4> const diagonal{Real(9), Real(4), Real(0), Real(-2)};
    for (Real const scale : {Real(1.0e-20), Real(1), Real(1.0e20)}) {
        auto const tensor = (rotation * diagonal * math::transpose(rotation)) * scale;
        math::SymmetricEigenpairs<Real, 4> eigenpairs;
        ASSERT_EQ(math::trySymmetricEigenpairs(tensor, eigenpairs), math::MatrixStatus::Success);
        Real const tolerance = Real(128) * std::numeric_limits<Real>::epsilon();
        checkEigenpairs(tensor, eigenpairs, tolerance);
        for (int axis = 0; axis < 4; ++axis) {
            EXPECT_NEAR(eigenpairs.values[axis] / scale, diagonal(axis, axis), tolerance * Real(9));
        }
    }
}

}    // namespace detail

TEST(MatrixEigen, FloatScaledIndefiniteFourDimensional) {
    detail::checkScaledEigenpairs<float>();
}

TEST(MatrixEigen, DoubleScaledIndefiniteFourDimensional) {
    detail::checkScaledEigenpairs<double>();
}

TEST(MatrixEigen, OneDimensionalAndZero) {
    math::Matrix<double, 1> tensor;
    tensor(0, 0) = -3.0;
    math::SymmetricEigenpairs<double, 1> eigenpairs;
    ASSERT_EQ(math::trySymmetricEigenpairs(tensor, eigenpairs), math::MatrixStatus::Success);
    EXPECT_EQ(eigenpairs.values[0], -3.0);
    EXPECT_EQ(eigenpairs.vectors(0, 0), 1.0);
    EXPECT_TRUE(eigenpairs.converged);
    tensor(0, 0) = 0.0;
    ASSERT_EQ(math::trySymmetricEigenpairs(tensor, eigenpairs), math::MatrixStatus::Success);
    EXPECT_EQ(eigenpairs.values[0], 0.0);
    EXPECT_EQ(eigenpairs.vectors(0, 0), 1.0);
    EXPECT_TRUE(eigenpairs.converged);
}

TEST(MatrixEigen, NonFiniteLeavesOutputUnchanged) {
    math::Matrix<double, 2> tensor{{2.0, 0.0}, {0.0, 1.0}};
    math::SymmetricEigenpairs<double, 2> eigenpairs;
    ASSERT_EQ(math::trySymmetricEigenpairs(tensor, eigenpairs), math::MatrixStatus::Success);
    for (double const invalid :
        {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
        tensor(0, 1) = invalid;
        EXPECT_EQ(math::trySymmetricEigenpairs(tensor, eigenpairs), math::MatrixStatus::NonFinite);
        EXPECT_EQ(eigenpairs.values[0], 2.0);
        EXPECT_EQ(eigenpairs.values[1], 1.0);
        EXPECT_EQ(eigenpairs.vectors(0, 0), 1.0);
        EXPECT_EQ(eigenpairs.vectors(1, 1), 1.0);
        EXPECT_TRUE(eigenpairs.converged);
    }
}

TEST(MatrixEigen, IterationBudgetAndSymmetrization) {
    math::Matrix<double, 2> const asymmetric{{2.0, 3.0}, {-1.0, 2.0}};
    math::Matrix<double, 2> const symmetric{{2.0, 1.0}, {1.0, 2.0}};
    math::SymmetricEigenpairs<double, 2> eigenpairs;
    math::SymmetricEigenOptions<double> options;
    options.maxIterations = 0;
    ASSERT_EQ(
        math::trySymmetricEigenpairs(asymmetric, eigenpairs, options), math::MatrixStatus::Success);
    EXPECT_FALSE(eigenpairs.converged);
    options.maxIterations = 8;
    options.pivot = math::JacobiPivot::Cyclic;
    options.useHypot = false;
    ASSERT_EQ(
        math::trySymmetricEigenpairs(asymmetric, eigenpairs, options), math::MatrixStatus::Success);
    detail::checkEigenpairs(symmetric, eigenpairs, 1.0e-14);
    EXPECT_NEAR(eigenpairs.values[0], 3.0, 1.0e-14);
    EXPECT_NEAR(eigenpairs.values[1], 1.0, 1.0e-14);
}
