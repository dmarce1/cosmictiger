/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

/*
 * Variational SPH pressure force and compatible thermal work, after
 *
 *   P. F. Hopkins, "A general class of Lagrangian smoothed particle hydrodynamics
 *   methods and implications for fluid mixing problems", MNRAS 428 (2013), 2840-2856,
 *   doi:10.1093/mnras/sts210, https://arxiv.org/abs/1206.5006,
 *   equations 12-13 with the support-constraint weight x~_j = 1; and
 *   D. J. Price, "Smoothed particle hydrodynamics and magnetohydrodynamics",
 *   J. Comput. Phys. 231 (2012), 759-794, doi:10.1016/j.jcp.2010.12.011,
 *   https://arxiv.org/abs/1012.1885, equations 30 and 35 for the density formulation.
 *
 * Particle i has mass m_i, fixed volume weight x_i, weighted density y_i and volume
 * V_i = x_i/y_i. cosmictiger/sph/volume.hpp evaluates y_i and the grad-H correction
 * ζ_i for the support constraint H_i^D n_i = constant. Its spatial derivative is
 *
 *   dy_i = Σ_j (x_j - ζ_i) g_i . dr_ij,   g_i = grad_i W_ij(H_i).
 *
 * The adiabatic first law d(m_i u_i) = -P_i dV_i = (P_i x_i/y_i^2) dy_i and
 * L = Σ_i m_i (v_i^2/2 - u_i) give the directed coefficient and inviscid rates
 *
 *   Q_ij = (P_i x_i/y_i^2)(x_j - ζ_i),
 *   m_i dv_i/dt = -Σ_j [Q_ij g_i + Q_ji g_j],
 *   m_i du_i/dt =  Σ_j Q_ij v_ij . g_i,
 *
 * where r_ij = r_i-r_j, v_ij = v_i-v_j and g_j = grad_i W_ij(H_j).
 * Hopkins' f_ij = 1 - ζ_i/x_j is absorbed into Q_ij. For equal masses with
 * x_i = m_i, y_i = ρ_i and x_j-ζ_i = m_i/Ω_i, these become Price's
 * equations 30 and 35. A held support uses ζ_i = 0.
 *
 * The same directed coefficient supplies force and work. Callers divide by m_i
 * and apply the cosmological scale factors derived in docs/variational_sph.md;
 * artificial viscosity is supplied separately by the Monaghan operator.
 */
#pragma once

#include <cosmictiger/cuda/cuda_export.hpp>

namespace variational {

// P_i x_i/y_i^2 = -P_i dV_i/dy_i from V_i = x_i/y_i and the adiabatic first law.
// clang-format off
/*
 * Derived form of Hopkins (2013), equations (12)–(13), and Price (2012), (30), (35).
 * Full citations and derivation above; https://arxiv.org/abs/1206.5006,
 * https://arxiv.org/abs/1012.1885.
 *   P_i x_i/y_i² = −P_i ∂V_i/∂y_i.
 * Symbol directory:
 *   P_i → pressure; x_i → weight; y_i → weighted density value; expression → return value.
 */
// clang-format on
template <typename Real>
CUDA_EXPORT inline Real pressureVolumeFactor(Real P_i, Real x_i, Real y_i) {
    return P_i * x_i / (y_i * y_i);
}

// Hopkins (2013), equations 12-13: Q_ij = (P_i x_i/y_i^2)(x_j - ζ_i).
// clang-format off
/*
 * Derived form of Hopkins (2013), equations (12)–(13), and Price (2012), (30), (35).
 * Full citations and derivation above; https://arxiv.org/abs/1206.5006,
 * https://arxiv.org/abs/1012.1885.
 *   Q_ij = (P_i x_i/y_i²)(x_j − ζ_i).
 * Symbol directory:
 *   P_i x_i/y_i² → sink pressure volume factor; x_j → neighbor volume weight; ζ_i → sink grad-H correction; Q_ij → return value.
 *   Code names: P_i_x_i_over_y2_i = P_i x_i/y_i².
 */
// clang-format on
template <typename Real>
CUDA_EXPORT inline Real directedCoefficient(Real P_i_x_i_over_y2_i, Real x_j, Real ζ_i) {
    return P_i_x_i_over_y2_i * (x_j - ζ_i);
}

// Hopkins (2013), equation 12, or Price (2012), equation 30 in the equal-mass limit.
// With g_i = F_i r_ij and g_j = F_j r_ij, m_i dv_i/dt gains -(this) r_ij from pair ij.
// clang-format off
/*
 * Derived form of Hopkins (2013), equations (12)–(13), and Price (2012), (30), (35).
 * Full citations and derivation above; https://arxiv.org/abs/1206.5006,
 * https://arxiv.org/abs/1012.1885.
 *   m_i dv_i/dt|_j = −(Q_ij F_i + Q_ji F_j) r_ij.
 * Symbol directory:
 *   Q_ij, Q_ji → ij coefficient, ji coefficient; F_i, F_j → i gradient factor, j gradient factor
 *   positive parenthesis → return value.
 */
// clang-format on
template <typename Real>
CUDA_EXPORT inline Real pairForceFactor(Real Q_ij, Real Q_ji, Real F_i, Real F_j) {
    return Q_ij * F_i + Q_ji * F_j;
}

// The adiabatic first law above; Price (2012), equation 35 in the equal-mass limit.
// m_i du_i/dt gains Q_ij v_ij . g_i = Q_ij F_i (v_ij . r_ij) from pair ij.
// clang-format off
/*
 * Derived form of Hopkins (2013), equations (12)–(13), and Price (2012), (30), (35).
 * Full citations and derivation above; https://arxiv.org/abs/1206.5006,
 * https://arxiv.org/abs/1012.1885.
 *   m_i du_i/dt|_j = Q_ij F_i (v_ij · r_ij).
 * Symbol directory:
 *   Q_ij → ij coefficient; F_i → i gradient factor; v_ij · r_ij → velocity dot separation
 *   pair work → return value.
 *   Code names: v_ij_dot_r_ij = v_ij · r_ij.
 */
// clang-format on
template <typename Real>
CUDA_EXPORT inline Real pairWork(Real Q_ij, Real F_i, Real v_ij_dot_r_ij) {
    return Q_ij * F_i * v_ij_dot_r_ij;
}

}    // namespace variational
