#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#include <gtest/gtest.h>

#include <cosmictiger/math/math.hpp>

void cosmictigerAssert(char const* expression, bool condition, char const* file, int line) {
    if (!condition) {
        std::fprintf(stderr, "Assertion failed: %s at %s:%d\n", expression, file, line);
        throw std::runtime_error("Regression check failed; see diagnostic above");
    }
}

TEST(RockstarEigenpair, PrincipalAxes) {
    Array<Array<double, NDIM>, NDIM> tensor{};
    tensor[0][0] = 9.e-6;
    tensor[1][1] = 4.e-6;
    tensor[2][2] = 1.e-6;
    double const targets[] = {0.01, 4.e-6, 0.0};
    double const expected[] = {9.e-6, 4.e-6, 1.e-6};
    for (int axis = 0; axis < NDIM; ++axis) {
        auto const result = findEigenpair(tensor, targets[axis]);
        ASSERT_TRUE(std::isfinite(result.first)) << "axis " << axis;
        EXPECT_NEAR(result.first / expected[axis], 1.0, 1.e-12) << "axis " << axis;
        EXPECT_NEAR(std::abs(result.second[axis]), 1.0, 1.e-12) << "axis " << axis;
    }
}
