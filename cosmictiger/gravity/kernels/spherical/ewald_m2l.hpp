/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once
#include <cosmictiger/gravity/kernels/spherical/ewald_common.hpp>
#include <cosmictiger/gravity/kernels/spherical/m2l_derivatives.hpp>

namespace cosmictiger::gravity::spherical {
#if (!defined(FMM_MIN_ORDER) || FMM_MIN_ORDER <= 3) && \
    (!defined(FMM_MAX_ORDER) || FMM_MAX_ORDER >= 3)
template <>
class EwaldM2L<2> {
public:
    static constexpr int kFlops_ = 35;
    static constexpr int kConventionalFlops_ = 58;
    template <typename Real, typename Local, typename Moments, typename Greens>
    CUDA_EXPORT static void add(Local& local, Real& localTrace, Moments const& moments, Real,
        Greens const& greens, Real greenTrace) {
        using namespace math;
        Fma const fma{};
        detail::M2LDerivatives<2>::template add<Real>(local, moments, greens);
        local[3] = fma(-broadcast<Real>(2) * greenTrace, moments[3], local[3]);
        local[1] = fma(-broadcast<Real>(2) * greenTrace, moments[1], local[1]);
        local[2] = fma(-greenTrace, moments[2], local[2]);
        localTrace = fma(greenTrace, moments[0], localTrace);
    }
};
#endif

#if (!defined(FMM_MIN_ORDER) || FMM_MIN_ORDER <= 4) && \
    (!defined(FMM_MAX_ORDER) || FMM_MAX_ORDER >= 4)
template <>
class EwaldM2L<3> {
public:
    static constexpr int kFlops_ = 100;
    static constexpr int kConventionalFlops_ = 173;
    template <typename Real, typename Local, typename Moments, typename Greens>
    CUDA_EXPORT static void add(Local& local, Real& localTrace, Moments const& moments,
        Real sourceTrace, Greens const& greens, Real greenTrace) {
        using namespace math;
        Fma const fma{};
        detail::M2LDerivatives<3>::template add<Real>(local, moments, greens);
        local[0] = fma(-broadcast<Real>(0.5) * greenTrace, sourceTrace, local[0]);
        local[3] = fma(-broadcast<Real>(2) * greenTrace, moments[3], local[3]);
        local[1] = fma(-broadcast<Real>(2) * greenTrace, moments[1], local[1]);
        local[2] = fma(-greenTrace, moments[2], local[2]);
        localTrace = fma(greenTrace, moments[0], localTrace);
    }
};
#endif

#if (!defined(FMM_MIN_ORDER) || FMM_MIN_ORDER <= 5) && \
    (!defined(FMM_MAX_ORDER) || FMM_MAX_ORDER >= 5)
template <>
class EwaldM2L<4> {
public:
    static constexpr int kFlops_ = 228;
    static constexpr int kConventionalFlops_ = 409;
    template <typename Real, typename Local, typename Moments, typename Greens>
    CUDA_EXPORT static void add(Local& local, Real& localTrace, Moments const& moments,
        Real sourceTrace, Greens const& greens, Real greenTrace) {
        using namespace math;
        Fma const fma{};
        detail::M2LDerivatives<4>::template add<Real>(local, moments, greens);
        local[0] = fma(-broadcast<Real>(0.5) * greenTrace, sourceTrace, local[0]);
        local[3] = fma(-broadcast<Real>(2) * greenTrace, moments[3], local[3]);
        local[1] = fma(-broadcast<Real>(2) * greenTrace, moments[1], local[1]);
        local[2] = fma(-greenTrace, moments[2], local[2]);
        localTrace = fma(greenTrace, moments[0], localTrace);
    }
};
#endif

#if (!defined(FMM_MIN_ORDER) || FMM_MIN_ORDER <= 6) && \
    (!defined(FMM_MAX_ORDER) || FMM_MAX_ORDER >= 6)
template <>
class EwaldM2L<5> {
public:
    static constexpr int kFlops_ = 457;
    static constexpr int kConventionalFlops_ = 841;
    template <typename Real, typename Local, typename Moments, typename Greens>
    CUDA_EXPORT static void add(Local& local, Real& localTrace, Moments const& moments,
        Real sourceTrace, Greens const& greens, Real greenTrace) {
        using namespace math;
        Fma const fma{};
        detail::M2LDerivatives<5>::template add<Real>(local, moments, greens);
        local[0] = fma(-broadcast<Real>(0.5) * greenTrace, sourceTrace, local[0]);
        local[3] = fma(-broadcast<Real>(2) * greenTrace, moments[3], local[3]);
        local[1] = fma(-broadcast<Real>(2) * greenTrace, moments[1], local[1]);
        local[2] = fma(-greenTrace, moments[2], local[2]);
        localTrace = fma(greenTrace, moments[0], localTrace);
    }
};
#endif

#if (!defined(FMM_MIN_ORDER) || FMM_MIN_ORDER <= 7) && \
    (!defined(FMM_MAX_ORDER) || FMM_MAX_ORDER >= 7)
template <>
class EwaldM2L<6> {
public:
    static constexpr int kFlops_ = 829;
    static constexpr int kConventionalFlops_ = 1553;
    template <typename Real, typename Local, typename Moments, typename Greens>
    CUDA_EXPORT static void add(Local& local, Real& localTrace, Moments const& moments,
        Real sourceTrace, Greens const& greens, Real greenTrace) {
        using namespace math;
        Fma const fma{};
        detail::M2LDerivatives<6>::template add<Real>(local, moments, greens);
        local[0] = fma(-broadcast<Real>(0.5) * greenTrace, sourceTrace, local[0]);
        local[3] = fma(-broadcast<Real>(2) * greenTrace, moments[3], local[3]);
        local[1] = fma(-broadcast<Real>(2) * greenTrace, moments[1], local[1]);
        local[2] = fma(-greenTrace, moments[2], local[2]);
        localTrace = fma(greenTrace, moments[0], localTrace);
    }
};
#endif

#if (!defined(FMM_MIN_ORDER) || FMM_MIN_ORDER <= 8) && \
    (!defined(FMM_MAX_ORDER) || FMM_MAX_ORDER >= 8)
template <>
class EwaldM2L<7> {
public:
    static constexpr int kFlops_ = 1396;
    static constexpr int kConventionalFlops_ = 2649;
    template <typename Real, typename Local, typename Moments, typename Greens>
    CUDA_EXPORT static void add(Local& local, Real& localTrace, Moments const& moments,
        Real sourceTrace, Greens const& greens, Real greenTrace) {
        using namespace math;
        Fma const fma{};
        detail::M2LDerivatives<7>::template add<Real>(local, moments, greens);
        local[0] = fma(-broadcast<Real>(0.5) * greenTrace, sourceTrace, local[0]);
        local[3] = fma(-broadcast<Real>(2) * greenTrace, moments[3], local[3]);
        local[1] = fma(-broadcast<Real>(2) * greenTrace, moments[1], local[1]);
        local[2] = fma(-greenTrace, moments[2], local[2]);
        localTrace = fma(greenTrace, moments[0], localTrace);
    }
};
#endif

#if (!defined(FMM_MIN_ORDER) || FMM_MIN_ORDER <= 9) && \
    (!defined(FMM_MAX_ORDER) || FMM_MAX_ORDER >= 9)
template <>
class EwaldM2L<8> {
public:
    static constexpr int kFlops_ = 2216;
    static constexpr int kConventionalFlops_ = 4245;
    template <typename Real, typename Local, typename Moments, typename Greens>
    CUDA_EXPORT static void add(Local& local, Real& localTrace, Moments const& moments,
        Real sourceTrace, Greens const& greens, Real greenTrace) {
        using namespace math;
        Fma const fma{};
        detail::M2LDerivatives<8>::template add<Real>(local, moments, greens);
        local[0] = fma(-broadcast<Real>(0.5) * greenTrace, sourceTrace, local[0]);
        local[3] = fma(-broadcast<Real>(2) * greenTrace, moments[3], local[3]);
        local[1] = fma(-broadcast<Real>(2) * greenTrace, moments[1], local[1]);
        local[2] = fma(-greenTrace, moments[2], local[2]);
        localTrace = fma(greenTrace, moments[0], localTrace);
    }
};
#endif
}    // namespace cosmictiger::gravity::spherical
