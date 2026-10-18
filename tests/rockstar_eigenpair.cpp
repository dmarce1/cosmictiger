#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <cosmictiger/math.hpp>

void cosmictigerAssert(char const* expression, bool condition, char const* file, int line) {
    if (!condition) {
        std::fprintf(stderr, "Assertion failed: %s at %s:%d\n", expression, file, line);
        std::abort();
    }
}

int main() {
    Array<Array<double, NDIM>, NDIM> tensor{};
    tensor[0][0] = 9.e-6;
    tensor[1][1] = 4.e-6;
    tensor[2][2] = 1.e-6;
    double const targets[] = {0.01, 4.e-6, 0.0};
    double const expected[] = {9.e-6, 4.e-6, 1.e-6};
    for (int axis = 0; axis < NDIM; ++axis) {
        auto const result = findEigenpair(tensor, targets[axis]);
        if (!std::isfinite(result.first) ||
            std::abs(result.first / expected[axis] - 1.0) > 1.e-12 ||
            std::abs(std::abs(result.second[axis]) - 1.0) > 1.e-12) {
            std::fprintf(stderr, "findEigenpair failed axis %d: %.12g\n", axis, result.first);
            return 1;
        }
    }
    std::puts("ROCKSTAR eigenpair wrapper regression passed");
    return 0;
}
