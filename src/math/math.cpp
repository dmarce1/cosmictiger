#include <cmath>

#include <cosmictiger/analysis/rockstar_properties.hpp>
#include <cosmictiger/math/math.hpp>

Pair<double, Array<double, NDIM>> findEigenpair(
    Array<Array<double, NDIM>, NDIM> const& matrix, double target) {
    static_assert(NDIM == 3);
    rockstar::Matrix3 tensor;
    for (int row = 0; row < NDIM; ++row) {
        for (int column = 0; column < NDIM; ++column) {
            tensor[row][column] = matrix[row][column];
        }
    }
    auto const eigenpairs = rockstar::symmetricEigenpairs(tensor);
    int closest = 0;
    for (int axis = 1; axis < NDIM; ++axis) {
        if (std::abs(eigenpairs.values[axis] - target) <
            std::abs(eigenpairs.values[closest] - target)) {
            closest = axis;
        }
    }
    Pair<double, Array<double, NDIM>> result;
    result.first = eigenpairs.values[closest];
    for (int component = 0; component < NDIM; ++component) {
        result.second[component] = eigenpairs.vectors[closest][component];
    }
    return result;
}
