/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <cosmictiger/gravity/fmm_kernels/spherical_ewald_kernels.hpp>
#include <cosmictiger/gravity/fmm_order.hpp>
#include <cosmictiger/gravity/nitadori/kernels.hpp>

namespace pmmm {
namespace detail {

    void checkOrder(int order) {
        if (order < kFmmMinOrder || order > kFmmMaxOrder) {
            throw std::invalid_argument("PMMM order is outside the compiled FMM order range");
        }
    }

    int index(int degree, int mode) {
        return degree * (degree + 1) + mode;
    }

    // C(n,-m)=(-1)^m*conj(C(n,m)); negative storage slots contain the
    // imaginary part of the positive-mode coefficient, not C(n,-m).
    struct HarmonicParts {
        int realSlot;
        int imaginarySlot;
        int realSign;
        int imaginarySign;
    };

    HarmonicParts harmonicParts(int degree, int mode) {
        int const magnitude = std::abs(mode);
        int const parity = mode < 0 && magnitude % 2 != 0 ? -1 : 1;
        return {index(degree, magnitude), index(degree, -magnitude), parity,
            mode == 0 ? 0 : (mode < 0 ? -parity : parity)};
    }

}    // namespace detail

std::vector<ContractionTerm> contractionTerms(int order) {
    detail::checkOrder(order);
    int const localDegree = fmmLocalDegree(order);
    int const sourceDegree = fmmMultipoleDegree(order);
    using TermKey = std::tuple<int, int, int>;
    std::map<TermKey, int> weights;
    for (int localRank = 0; localRank <= localDegree; ++localRank) {
        for (int localMode = 0; localMode <= localRank; ++localMode) {
            int const localReal = detail::index(localRank, localMode);
            int const localImaginary = detail::index(localRank, -localMode);
            for (int sourceRank = 0; sourceRank <= std::min(sourceDegree, localDegree - localRank);
                ++sourceRank) {
                for (int sourceMode = -sourceRank; sourceMode <= sourceRank; ++sourceMode) {
                    auto source = detail::harmonicParts(sourceRank, sourceMode);
                    auto const kernel =
                        detail::harmonicParts(localRank + sourceRank, localMode + sourceMode);
                    source.imaginarySign = -source.imaginarySign;
                    weights[{localReal, source.realSlot, kernel.realSlot}] +=
                        source.realSign * kernel.realSign;
                    weights[{localReal, source.imaginarySlot, kernel.imaginarySlot}] -=
                        source.imaginarySign * kernel.imaginarySign;
                    if (localMode != 0) {
                        weights[{localImaginary, source.realSlot, kernel.imaginarySlot}] +=
                            source.realSign * kernel.imaginarySign;
                        weights[{localImaginary, source.imaginarySlot, kernel.realSlot}] +=
                            source.imaginarySign * kernel.realSign;
                    }
                }
            }
        }
    }
    std::vector<ContractionTerm> terms;
    terms.reserve(weights.size());
    for (auto const& [indices, weight] : weights) {
        if (weight != 0) {
            auto const [local, source, kernel] = indices;
            terms.push_back({local, source, kernel, double(weight)});
        }
    }
    return terms;
}

FlopCount addMoments(int order, double* moments, double mass, std::array<double, 3> const& offset) {
    detail::checkOrder(order);
    FlopCount count = 0;
    fmmDispatch(order, [&](auto selectedOrder) {
        constexpr int kDegree = decltype(selectedOrder)::value - 2;
        cosmictiger::gravity::spherical::detail::P2M<kDegree>::template add<double>(
            moments, mass, offset[0], offset[1], offset[2]);
        count = cosmictiger::gravity::spherical::detail::P2M<kDegree>::kConventionalFlops_;
    });
    return count;
}

FlopCount green(int order, double* coefficients, std::array<double, 3> const& boxOffset,
    double cellWidth, bool near) {
    detail::checkOrder(order);
    if (!(cellWidth > 0.0) || !std::isfinite(cellWidth)) {
        throw std::invalid_argument("PMMM cell width must be finite and positive");
    }
    for (double const coordinate : boxOffset) {
        if (!std::isfinite(coordinate) || std::abs(coordinate) > 0.5) {
            throw std::invalid_argument("PMMM Green displacement is outside the minimum-image box");
        }
    }
    if (!near && boxOffset[0] == 0.0 && boxOffset[1] == 0.0 && boxOffset[2] == 0.0) {
        throw std::invalid_argument("PMMM primary origin must be excluded by the near-field mask");
    }
    FlopCount count = 0;
    fmmDispatch(order, [&](auto selectedOrder) {
        constexpr int kDegree = decltype(selectedOrder)::value - 1;
        double trace = 0.0;
        cosmictiger::gravity::spherical::EwaldGreen<kDegree>::template evaluate<double>(
            coefficients, trace, boxOffset[0], boxOffset[1], boxOffset[2]);
        count = cosmictiger::gravity::spherical::EwaldGreen<kDegree>::kConventionalFlops_;
        if (!near) {
            std::array<double, (kDegree + 1) * (kDegree + 1)> primary;
            cosmictiger::gravity::spherical::detail::Green<kDegree>::template evaluate<double>(
                primary, boxOffset[0], boxOffset[1], boxOffset[2]);
            for (size_t slot = 0; slot < primary.size(); ++slot) {
                coefficients[slot] += primary[slot];
            }
            count += cosmictiger::gravity::spherical::detail::Green<kDegree>::kConventionalFlops_ +
                primary.size();
        }
        double scale = 1.0;
        for (int rank = 0; rank <= kDegree; ++rank) {
            for (int slot = rank * rank; slot < (rank + 1) * (rank + 1); ++slot) {
                coefficients[slot] *= scale;
            }
            scale *= cellWidth;
        }
        count += (kDegree + 1) * (kDegree + 1) + (kDegree + 1);
    });
    return count;
}

FlopCount evaluate(int order, std::array<double, 4>& field, double const* coefficients,
    double localTrace, std::array<double, 3> const& cellOffset, double cellWidth) {
    detail::checkOrder(order);
    if (!(cellWidth > 0.0) || !std::isfinite(cellWidth)) {
        throw std::invalid_argument("PMMM cell width must be finite and positive");
    }
    field = {};
    FlopCount count = 0;
    fmmDispatch(order, [&](auto selectedOrder) {
        constexpr int kDegree = decltype(selectedOrder)::value - 1;
        cosmictiger::gravity::spherical::EwaldL2P<kDegree>::template add<double>(
            field, coefficients, localTrace, cellOffset[0], cellOffset[1], cellOffset[2]);
        count = cosmictiger::gravity::spherical::EwaldL2P<kDegree>::kConventionalFlops_;
    });
    for (int component = 1; component < 4; ++component) {
        field[component] /= cellWidth;
    }
    return count + 3;
}

}    // namespace pmmm
