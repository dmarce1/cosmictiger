/* CosmicTiger - fixed small matrix tests; BSL-1.0. */

#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <type_traits>

#include <gtest/gtest.h>

#include <cosmictiger/math/matrix.hpp>
#include <cosmictiger/math/vector3.hpp>

namespace detail {
using math::Matrix;
using math::MatrixStatus;
using math::SymmetricMatrix;
using math::Vector;
static_assert(sizeof(Matrix<double, 3>) == 9 * sizeof(double));
static_assert(sizeof(SymmetricMatrix<double, 3>) == 6 * sizeof(double));
static_assert(sizeof(math::DiagonalMatrix<double, 3>) == 3 * sizeof(double));
static_assert(
    std::is_same_v<decltype(Matrix<double, 2, 3>{} * Vector<double, 3>{}), Vector<double, 2>>);
static_assert(
    std::is_same_v<decltype(Vector<double, 2>{} * Matrix<double, 2, 3>{}), Vector<double, 3>>);
static_assert(std::is_same_v<decltype(SymmetricMatrix<double, 3>{} * SymmetricMatrix<double, 3>{}),
    Matrix<double, 3>>);

template <class A, class B>
void expectIdentityProduct(A const& a, B const& inverse, long double tolerance) {
    for (int row = 0; row < A::rows(); ++row) {
        for (int column = 0; column < A::rows(); ++column) {
            long double value = 0;
            for (int k = 0; k < A::rows(); ++k) {
                value += static_cast<long double>(a(row, k)) * inverse(k, column);
            }
            EXPECT_NEAR(value, row == column ? 1 : 0, tolerance);
        }
    }
}

template <class T, int dimensions, bool symmetric>
void checkRandomSystems() {
    using A = std::conditional_t<symmetric, SymmetricMatrix<T, dimensions>, Matrix<T, dimensions>>;
    std::mt19937 generator(137 + dimensions);
    std::uniform_real_distribution<double> random(-1, 1);
    long double const tolerance = 128 * dimensions * std::numeric_limits<T>::epsilon();
    for (int trial = 0; trial < 32; ++trial) {
        A a{};
        for (int i = 0; i < dimensions; ++i) {
            for (int j = 0; j < (symmetric ? i + 1 : dimensions); ++j) {
                // Include symmetric indefinite matrices as well as SPD ones.
                a(i, j) = T(
                    random(generator) + (i == j ? (trial % 2 && i % 2 ? -2 : 2) * dimensions : 0));
            }
        }
        Vector<T, dimensions> expected{}, rhs{};
        for (int i = 0; i < dimensions; ++i) {
            expected[i] = T(random(generator));
        }
        for (int i = 0; i < dimensions; ++i) {
            long double value = 0;
            for (int j = 0; j < dimensions; ++j) {
                value += static_cast<long double>(a(i, j)) * expected[j];
            }
            rhs[i] = T(value);
        }
        for (int algorithm = 0; algorithm < 3; ++algorithm) {
            A inverse{};
            Vector<T, dimensions> solution{};
            MatrixStatus inverseStatus{}, solveStatus{};
            if (algorithm == 0) {
                if constexpr (dimensions <= 6) {
                    inverseStatus = math::tryInverseExplicit(a, inverse);
                    solveStatus = math::trySolveExplicit(a, rhs, solution);
                } else {
                    continue;
                }
            } else if (algorithm == 1) {
                inverseStatus = math::tryInversePivoted(a, inverse);
                solveStatus = math::trySolvePivoted(a, rhs, solution);
            } else {
                inverseStatus = math::tryInverse(a, inverse);
                solveStatus = math::trySolve(a, rhs, solution);
            }
            ASSERT_EQ(inverseStatus, MatrixStatus::Success) << dimensions << ' ' << algorithm;
            ASSERT_EQ(solveStatus, MatrixStatus::Success) << dimensions << ' ' << algorithm;
            expectIdentityProduct(a, inverse, tolerance);
            for (int i = 0; i < dimensions; ++i) {
                EXPECT_NEAR(solution[i], expected[i], tolerance);
            }
        }
    }
}
template <class T, int dimensions>
void checkSize() {
    checkRandomSystems<T, dimensions, false>();
    checkRandomSystems<T, dimensions, true>();
}
template <class T>
void checkSizes() {
    checkSize<T, 1>();
    checkSize<T, 2>();
    checkSize<T, 3>();
    checkSize<T, 4>();
    checkSize<T, 5>();
    checkSize<T, 6>();
    checkSize<T, 16>();
}
struct PivotPolicy : math::DefaultMatrixPolicy {
    static constexpr int kGeneralInverseThreshold = 0;
    static constexpr int kSymmetricInverseThreshold = 0;
};
struct ExplicitSolvePolicy : math::DefaultMatrixPolicy {
    static constexpr int kGeneralSolveThreshold = 3;
};
}    // namespace detail

TEST(Matrix, RectangularProductsAndVectorOrientation) {
    math::Matrix<double, 2, 3> const a{{1, 2, 3}, {4, 5, 6}};
    math::Matrix<double, 3, 2> const b{{7, 8}, {9, 10}, {11, 12}};
    auto const c = a * b;
    EXPECT_EQ(c(0, 0), 58);
    EXPECT_EQ(c(0, 1), 64);
    EXPECT_EQ(c(1, 0), 139);
    EXPECT_EQ(c(1, 1), 154);
    auto const column = a * Vector3<double>{1, 2, 3};
    EXPECT_EQ(column[0], 14);
    EXPECT_EQ(column[1], 32);
    auto const row = math::Vector<double, 2>{2, 3} * a;
    EXPECT_EQ(row[0], 14);
    EXPECT_EQ(row[1], 19);
    EXPECT_EQ(row[2], 24);
    auto const t = math::transpose(a);
    EXPECT_EQ(t(2, 1), 6);
    auto const outer = math::outerProduct(math::Vector<double, 2>{2, 3}, Vector3<double>{4, 5, 6});
    EXPECT_EQ(outer(1, 2), 18);
}

TEST(Matrix, PackedSymmetryAndMixedProducts) {
    math::SymmetricMatrix<double, 3> a{};
    a(0, 0) = 2;
    a(1, 1) = 3;
    a(2, 2) = 4;
    a(0, 1) = 5;
    EXPECT_EQ(&a(0, 1), &a(1, 0));
    EXPECT_EQ(math::trace(a), 9);
    EXPECT_EQ(math::squaredNorm(a), 79);
    auto const product = a * math::symmetricOuterProduct(Vector3<double>{1, 2, 3});
    EXPECT_EQ(product(0, 1), 24);
    EXPECT_EQ(product(1, 0), 11);
    math::DiagonalMatrix<double, 3> d{};
    d.diagonal(0) = 2;
    d.diagonal(1) = 3;
    d.diagonal(2) = 4;
    EXPECT_EQ((d * a)(0, 1), 10);
    EXPECT_EQ((a * d)(0, 1), 15);
    EXPECT_EQ((d * Vector3<double>{1, 2, 3})[2], 12);
}

TEST(Matrix, RandomFloatSystems) {
    detail::checkSizes<float>();
}
TEST(Matrix, RandomDoubleSystems) {
    detail::checkSizes<double>();
}

TEST(Matrix, PivotingMultipleRightSidesAndAliasing) {
    math::Matrix<double, 3> a{{0, 2, 1}, {1, 0, 3}, {4, 1, 0}};
    math::Matrix<double, 3, 2> const expected{{1, 4}, {2, 5}, {3, 6}};
    auto rhs = a * expected;
    ASSERT_EQ(math::trySolvePivoted(a, rhs, rhs), math::MatrixStatus::Success);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 2; ++j) {
            EXPECT_NEAR(rhs(i, j), expected(i, j), 1e-14);
        }
    }
    auto const original = a;
    ASSERT_EQ(math::tryInverse(a, a), math::MatrixStatus::Success);
    detail::expectIdentityProduct(original, a, 1e-14);
    double determinant{};
    ASSERT_EQ(math::tryDeterminant(original, determinant), math::MatrixStatus::Success);
    EXPECT_NEAR(determinant, 25, 1e-13);
}

TEST(Matrix, SingularAndNonFiniteLeaveOutputsUnchanged) {
    math::Matrix<double, 3> a{{1, 2, 3}, {1, 2, 3}, {4, 5, 6}};
    math::Matrix<double, 3> out(17);
    math::Vector<double, 3> rhs{1, 2, 3}, solution{17, 18, 19};
    double determinant = 17;
    EXPECT_EQ(math::tryInverse(a, out), math::MatrixStatus::Singular);
    EXPECT_EQ(math::trySolve(a, rhs, solution), math::MatrixStatus::Singular);
    EXPECT_EQ(math::tryDeterminant(a, determinant), math::MatrixStatus::Singular);
    EXPECT_EQ(determinant, 17);
    EXPECT_EQ(solution[2], 19);
    EXPECT_EQ(out(2, 2), 17);
    for (double bad :
        {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
        a(0, 0) = bad;
        EXPECT_EQ(math::tryInverse(a, out), math::MatrixStatus::NonFinite);
        EXPECT_EQ(out(0, 0), 17);
    }
    a = math::Matrix<double, 3>(math::identityMatrix<double, 3>());
    rhs[1] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_EQ(math::trySolve(a, rhs, solution), math::MatrixStatus::NonFinite);
    EXPECT_EQ(solution[1], 18);
}

TEST(Matrix, ScaleRangeAndAbsolutePivotFloor) {
    math::Matrix<double, 3> const base{{2, 1, 0}, {1, 3, 1}, {0, 1, 2}};
    for (double scale : {1e-200, 1.0, 1e200}) {
        auto const a = base * scale;
        math::Matrix<double, 3> inverse{};
        ASSERT_EQ(math::tryInverse(a, inverse), math::MatrixStatus::Success);
        detail::expectIdentityProduct(a, inverse, 1e-14);
        auto rhs = a * Vector3<double>{1, 2, 3};
        math::Vector<double, 3> solution{};
        ASSERT_EQ(math::trySolve(a, rhs, solution), math::MatrixStatus::Success);
        EXPECT_NEAR(solution[1], 2, 1e-14);
    }
    math::Matrix<double, 3> a{{1, 0, 0}, {0, 1, 0}, {0, 0, 1e-10}};
    double determinant = 0;
    ASSERT_EQ(math::tryDeterminant(a, determinant), math::MatrixStatus::Success);
    EXPECT_NEAR(determinant, 1e-10, 1e-25);
    math::Matrix<double, 3> inverse{};
    EXPECT_EQ(math::tryInversePivoted(a, inverse, {0, 1e-9}), math::MatrixStatus::Singular);
    EXPECT_EQ(math::tryInversePivoted(a, inverse, {0, 1e-11}), math::MatrixStatus::Success);
}

TEST(Matrix, DiagonalAlwaysExplicitAtAnySize) {
    math::DiagonalMatrix<double, 16> a{};
    math::Vector<double, 16> rhs{}, solution{};
    for (int i = 0; i < 16; ++i) {
        a.diagonal(i) = i < 8 ? 1e200 : 1e-200;
        rhs[i] = a.diagonal(i);
    }
    math::DiagonalMatrix<double, 16> inverse{};
    ASSERT_EQ(math::tryInverse(a, inverse), math::MatrixStatus::Success);
    ASSERT_EQ(math::trySolve(a, rhs, solution), math::MatrixStatus::Success);
    double determinant{};
    ASSERT_EQ(math::tryDeterminant(a, determinant), math::MatrixStatus::Success);
    EXPECT_NEAR(determinant, 1, 1e-14);
    for (int i = 0; i < 16; ++i) {
        EXPECT_NEAR(a.diagonal(i) * inverse.diagonal(i), 1, 1e-14);
        EXPECT_EQ(solution[i], 1);
    }
}

TEST(Matrix, CompileTimePolicyOverrides) {
    math::Matrix<double, 3> const a{{0, 1, 0}, {1, 0, 0}, {0, 0, 2}};
    math::Matrix<double, 3> inverse{};
    EXPECT_EQ(math::tryInverse<detail::PivotPolicy>(a, inverse), math::MatrixStatus::Success);
    detail::expectIdentityProduct(a, inverse, 0);
}

TEST(Matrix, AutomaticSolveRetriesOverflowingCofactorIntermediate) {
    // adj(A / scale)(2,2) is 2: adj*rhs overflows in the explicit path,
    // while the original-units pivoted solve returns the finite rhs/scale.
    double volatile matrixScale = 1e100;
    double volatile rhsScale = 1e308;
    double const scale = matrixScale;
    math::Matrix<double, 3> const a{{scale, scale, 0}, {-scale, scale, 0}, {0, 0, scale}};
    math::Vector<double, 3> const rhs{0, 0, rhsScale};
    math::Vector<double, 3> solution{17, 18, 19};
    ASSERT_EQ(
        math::trySolve<detail::ExplicitSolvePolicy>(a, rhs, solution), math::MatrixStatus::Success);
    EXPECT_EQ(solution[0], 0);
    EXPECT_EQ(solution[1], 0);
    EXPECT_NEAR(solution[2] / 1e208, 1, 1e-14);
}

TEST(Matrix, DivisionPreservesIntegersAndAliasedScalar) {
    math::Matrix<int, 2> dense{{2, 4}, {6, 8}};
    dense /= dense(0, 0);
    EXPECT_EQ(dense(0, 0), 1);
    EXPECT_EQ(dense(1, 1), 4);
    math::SymmetricMatrix<int, 2> symmetric{};
    symmetric(0, 0) = 2;
    symmetric(1, 0) = 4;
    symmetric(1, 1) = 6;
    symmetric /= symmetric(0, 0);
    EXPECT_EQ(symmetric(1, 0), 2);
    EXPECT_EQ(symmetric(1, 1), 3);
    math::DiagonalMatrix<int, 2> diagonal{};
    diagonal.diagonal(0) = 2;
    diagonal.diagonal(1) = 8;
    diagonal /= diagonal.diagonal(0);
    EXPECT_EQ(diagonal.diagonal(1), 4);
#ifndef __FAST_MATH__
    // Fast-math environments may flush subnormal inputs to zero globally.
    math::Matrix<double, 1> tiny{{1e-310}};
    tiny /= 1e-310;
    EXPECT_EQ(tiny(0, 0), 1);
#endif
}

TEST(Matrix, SimdArithmeticIncludesEveryLane) {
    using Pack = SimdVector<float, 4>;
    math::SymmetricMatrix<Pack, 3> a{};
    for (int i = 0; i < 3; ++i) {
        a(i, i) = Pack([i](auto lane) { return float(i + 1 + int(lane)); });
    }
    Vector3<Pack> const x{Pack(2), Pack(3), Pack(4)};
    auto const y = a * x;
    auto const product = a * a;
    for (int lane = 0; lane < 4; ++lane) {
        for (int i = 0; i < 3; ++i) {
            EXPECT_EQ(y[i][lane], float(i + 1 + lane) * float(i + 2));
            EXPECT_EQ(product(i, i)[lane], float((i + 1 + lane) * (i + 1 + lane)));
            EXPECT_EQ(product(i, (i + 1) % 3)[lane], 0);
        }
    }
}

TEST(Matrix, IllConditionedCofactorAttemptUsesPivotFallback) {
    math::SymmetricMatrix<double, 3> a{};
    long double const eigenvalues[3] = {1, 1e-4L, 1e-8L};
    long double q[3][3]{};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            q[i][j] = (i == j ? 1.L : 0.L) - 2.L * (i + 1) * (j + 1) / 14.L;
        }
    }
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j <= i; ++j) {
            long double value = 0;
            for (int k = 0; k < 3; ++k) {
                value += q[i][k] * eigenvalues[k] * q[j][k];
            }
            a(i, j) = double(value);
        }
    }
    math::SymmetricMatrix<double, 3> inverse{}, pivoted{};
    ASSERT_EQ(math::tryInverse(a, inverse), math::MatrixStatus::Success);
    ASSERT_EQ(math::tryInversePivoted(a, pivoted), math::MatrixStatus::Success);
    detail::expectIdentityProduct(a, inverse, 1e-8);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j <= i; ++j) {
            EXPECT_EQ(inverse(i, j), pivoted(i, j));
        }
    }
}

TEST(Matrix, PivotDeterminantAvoidsIntermediateOverflow) {
    math::Matrix<double, 4> matrix{};
    matrix(0, 0) = 1e200;
    matrix(1, 1) = -1e200;
    matrix(2, 2) = 1e-200;
    matrix(3, 3) = 1e-200;
    double determinant{};
    ASSERT_EQ(math::tryDeterminant(matrix, determinant, {0, 0}), math::MatrixStatus::Success);
    EXPECT_NEAR(determinant, -1, 1e-14);
}

TEST(Matrix, NearlyRankOneSymmetricSystems) {
    math::Vector<double, 3> const vector{1, 2, 3};
    for (double const delta : {1e-4, 1e-6, 1e-8}) {
        auto a = math::symmetricOuterProduct(vector);
        for (int i = 0; i < 3; ++i) {
            a(i, i) += delta;
        }
        long double const condition = (14.L + delta) / delta;
        long double const tolerance = 16 * std::numeric_limits<double>::epsilon() * condition;
        math::SymmetricMatrix<double, 3> automatic{}, pivoted{};
        math::Matrix<double, 3> dense{};
        ASSERT_EQ(math::tryInverse(a, automatic), math::MatrixStatus::Success);
        ASSERT_EQ(math::tryInversePivoted(a, pivoted), math::MatrixStatus::Success);
        ASSERT_EQ(math::tryInversePivoted(a, dense), math::MatrixStatus::Success);
        detail::expectIdentityProduct(a, automatic, tolerance);
        detail::expectIdentityProduct(a, pivoted, tolerance);
        detail::expectIdentityProduct(a, dense, tolerance);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j <= i; ++j) {
                EXPECT_EQ(automatic(i, j), pivoted(i, j));
            }
        }
        double determinant = 0;
        ASSERT_EQ(math::tryDeterminant(a, determinant), math::MatrixStatus::Success);
        long double const expected = static_cast<long double>(delta) * delta * (14.L + delta);
        EXPECT_NEAR(
            determinant / expected, 1.L, 64 * std::numeric_limits<double>::epsilon() * condition);
    }
}

TEST(Matrix, FrobeniusNormAvoidsSquaringOverflowAndUnderflow) {
    for (double const scale : {1e-200, 1.0, 1e200}) {
        math::Matrix<double, 1> const single{{scale}};
        EXPECT_NEAR(math::frobeniusNorm(single) / scale, 1, 1e-15);
        math::Matrix<double, 2> const matrix{{scale, 2 * scale}, {2 * scale, 0}};
        EXPECT_NEAR(math::frobeniusNorm(matrix) / scale, 3, 1e-14);
    }
}
