#include <cmath>
#include <stdexcept>

#include <cosmictiger/math/math.hpp>
#include <cosmictiger/math/matrix_eigen.hpp>

Pair<double, Vector3<double>> findEigenpair(
    Array<Array<double, kDimensions>, kDimensions> const& matrix, double target) {
    static_assert(kDimensions == 3);
    math::Matrix<double, 3> tensor;
    for (int row = 0; row < kDimensions; ++row) {
        for (int column = 0; column < kDimensions; ++column) {
            tensor[row][column] = matrix[row][column];
        }
    }
    math::SymmetricEigenpairs<double, 3> eigenpairs;
    if (math::trySymmetricEigenpairs(tensor, eigenpairs) != math::MatrixStatus::Success) {
        throw std::invalid_argument("Matrix must have finite eigenpairs");
    }
    int closest = 0;
    for (int axis = 1; axis < kDimensions; ++axis) {
        if (std::abs(eigenpairs.values[axis] - target) <
            std::abs(eigenpairs.values[closest] - target)) {
            closest = axis;
        }
    }
    Pair<double, Vector3<double>> result;
    result.first = eigenpairs.values[closest];
    for (int component = 0; component < kDimensions; ++component) {
        result.second[component] = eigenpairs.vectors[component][closest];
    }
    return result;
}
