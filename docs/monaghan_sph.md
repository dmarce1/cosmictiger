# Monaghan (1992) SPH baseline

This implementation follows the nonrelativistic SPH equations in J. J. Monaghan,
*Smoothed Particle Hydrodynamics*, **Annual Review of Astronomy and Astrophysics**
30 (1992), 543–574,
[doi:10.1146/annurev.aa.30.090192.002551](https://doi.org/10.1146/annurev.aa.30.090192.002551),
ADS bibcode `1992ARA&A..30..543M`
([open-access scan](https://articles.adsabs.harvard.edu/pdf/1992ARA%26A..30..543M)).
Equation and printed-page references below were checked against that scan.

The active fluid model uses density and specific internal energy with an ideal-gas
equation of state. Stellar particles, star formation, chemistry, thermal conduction,
and species diffusion have been removed, along with their particle storage and
solver branches. Their former options are no longer accepted. Ordinary particle
advection is used; the optional XSPH variant in equation 3.19 is not enabled.

## Notation in the code

The computation code transcribes the paper. `cosmictiger/sph/monaghan_sph.hpp` holds the
per-pair formulae, and the pair, density, and support-radius loops in
`src/sph/sph_hydro_cpu.cpp`, `src/sph/sph_hydro.cu`, `src/sph/sph_prehydro_cpu.cpp`,
`src/sph/sph_prehydro.cu`, and `cosmictiger/sph/sph_cuda.hpp` apply them to the staged particle
arrays. Inside those regions the local variables carry the paper's symbols and each
block names the equation it implements; function parameters and data fields keep their
descriptive names, and the regions are fenced with
`NOLINTBEGIN/NOLINTEND(readability-identifier-naming)`. Relative positions and
velocities use the `Vector3` class of `cosmictiger/math/vector3.hpp`, which supplies
`dot`, `norm`, and the arithmetic operators for scalar, SIMD-pack, and device element
types.

| Paper | Code | Meaning |
| --- | --- | --- |
| \(a\), \(b\) | `_a`, `_b`, `_ab` suffixes | the sink and a neighbor |
| \(m_b\) | `m_b`, `m_b_over_a` | the common gas mass, and divided by the scale factor |
| \(\rho_a, P_a, u_a, c_a, \gamma_a\) | `ρ_a`, `P_a`, `u_a`, `c_a`, `γ_a` | density, pressure, specific energy, sound speed, adiabatic index |
| \(h_a\) | `H_a` | stored support radius, \(2h_a\) for the cubic spline |
| \(h\) in equation 4.2 | `h_ab` | pair viscosity length \(\tfrac12(h_a+h_b)\) |
| \(\mathbf r_{ab}, \mathbf v_{ab}\) | `r_ab`, `v_ab` | \(\mathbf r_a-\mathbf r_b\), \(\mathbf v_a-\mathbf v_b\) |
| \(r_{ab}^2\), \(\eta^2\) | `r_ab2`, `η2` | squares; superscript digits are not identifier characters |
| \(\nabla_a\widetilde W_{ab}\) | `gradW̃_ab`, `F_ab` | symmetric kernel gradient, equal to `F_ab * r_ab` |
| \(\Pi_{ab}, \mu_{ab}\) | `Π_ab`, `μ_ab` | artificial viscosity and its velocity measure |
| \(\alpha, \beta, \eta^2/h^2\) | `α`, `β`, `ε` | viscosity constants |
| \(\bar c_{ab}, \bar\rho_{ab}\) | `c̄_ab`, `ρ̄_ab` | pair averages |
| \(P_b/\rho_b^2+P_a/\rho_a^2\) | `P_over_ρ2_ab` | symmetric pressure term |
| \(\max_b\mu_{ab}\) | `μ_max` | compression magnitude \(\max_b(-\mu_{ab})\) |
| \(\delta t_{\rm cv}, \delta t_{\rm f}\) | `one_over_δt_cv`, `one_over_δt_f` | reciprocals of the section 10.3 limits |
| \(\sum_b W_{ab}\) | `ΣW`, `dΣW_dH_a` | support-radius constraint sum and its derivative |

Quantities summed over SIMD lanes carry a `_pack` suffix until they are reduced.
GCC 13 and CUDA 12.8 accept Greek letters, combining marks, and subscript letters in
identifiers but reject `∇`, `∂`, and superscript digits, hence `gradW̃_ab` and `r_ab2`.

## Kernel and density

The default spline in section 7, pp.554–555, is written here with the paper's
length denoted by \(h_M\):

\[
 W(r,h_M)=\frac{1}{\pi h_M^3}
 \begin{cases}
  1-\frac32q^2+\frac34q^3,&0\le q\le1,\\
  \frac14(2-q)^3,&1<q\le2,\\
  0,&q>2,
 \end{cases}
 \qquad q=r/h_M.
\]

The code stores the **support radius** \(H\), equal to \(2h_M\) for this default
cubic. Its dimensionless kernel has normalization \(8/\pi\), changes polynomial
at \(r/H=1/2\), and vanishes at \(r/H=1\). The common effective radius is
\(h=2\sigma\), where \(\sigma\) is the one-coordinate standard deviation of
the normalized kernel. It retains the existing `2 sigma` value without a
factor-of-two conversion. For the cubic, \(H=\sqrt{10/3}\,h\); thus the common
\(h\) and the paper's \(h_M\) are distinct.

The default `hydro.neighborNumber` is 42. The name cubic B-spline refers to this
radial M4 kernel; it is not a tensor-product bicubic interpolation.
`hydro.kernelFamily` and `hydro.kernelDegree` can instead select cardinal splines
of degrees 3–9, Wendland C0/C2/C4, or Ferrers F1–F4. Gravity has an independent
selection. See [kernel choices and length conventions](kernels.md) for the valid
degrees, conversion factors, and migration of explicit old settings. Production
gravity and SPH use three-dimensional tables; a separate arbitrary-dimensional
density-table generator is available as described there. Select SIMD CPU SPH with
`--compute.cuda=false`, or CUDA SPH with `--compute.cuda=true`. Both backends
implement support-radius solving, density, forces/heating, and neighbor timestep
propagation using the same kernel selections and length conventions. CPU workers
process separate tree leaves, with eight-lane SIMD packs over their neighbors.
All source fields are staged before a phase updates local sinks, including sources
fetched from other MPI ranks.

Particle drift also uses the CPU when `compute.cuda=false`. The complete executable
still requires CUDA for other initialization and analysis paths.
The standalone `sph_cpu_test` target can be built with
`COSMICTIGER_WITH_CUDA=OFF` and does not call a CUDA device. The installed HPX
library may itself link the CUDA runtime.

Write the selected density kernel as \(W_H(r)=H^{-3}w(r/H)\). Density includes
the self contribution and uses the gather sum

\[
 \rho_i=\sum_j m_j W_{H_i}(r_{ij}).
\]

This gather choice is a deliberate practical variation: equation 6.3 on p.553
writes the adaptive density interpolant with the **source** length \(h_j\).
Section 6 permits summation instead of integrating the continuity equation, but
does not prescribe this gather solve or its numerical parameters.

Gas particles share one constant class mass. Other particle classes may have
different constant masses; there is no per-particle SPH mass field. The gas
smoothing solve holds

\[
 \frac{4\pi}{3}\frac{\rho_i H_i^3}{m}=N_{\rm target},
\]

so \(H_i\propto\rho_i^{-1/3}\), consistent with equation 6.2. The target is a
kernel-weighted count, including self, rather than an integer number of neighbors.
The self contribution imposes \(N_{\rm target}>(4\pi/3)w(0)\), which is
\(32/3\) for the default cubic. Startup checks use the selected kernel's floor,
including its single-precision representation. The solve respects the configured
radius bounds, supplied as `hydro.minEffectiveRadius`/`hydro.maxEffectiveRadius`
or `hydro.minSupportRadius`/`hydro.maxSupportRadius` and converted with the selected
hydro kernel. Explicit effective-radius settings take precedence with a warning
naming each ignored support-radius setting. It refreshes neighbors if the search
box is too small, and caps each local solve at 128 iterations. Its relative residual tolerance
is \(10^{-4}\); a bound-limited solution need not attain the requested target.
The periodic search uses one nearest image per pair, so the effective maximum
support is also capped at half the box width; a requested minimum above that limit
is rejected. The effective limits are printed at startup in initial interparticle
spacings. These numerical choices are implementation details, not prescriptions
in the paper.

## Pair force, viscosity, and thermal work

Let \(\mathbf r_{ij}=\mathbf r_i-\mathbf r_j\),
\(\mathbf v_{ij}=\mathbf v_i-\mathbf v_j\). The symmetric gradient is

\[
 \nabla_i\widetilde W_{ij}
 =\tfrac12[\nabla_i W_{H_i}(r_{ij})+\nabla_i W_{H_j}(r_{ij})].
\]

This is one of the two choices explicitly offered on p.554. A pair participates
when it lies inside either support radius. The zero-separation gradient is zero.

Equations 6.4–6.5 give the physical-coordinate pair rates

\[
 \frac{d\mathbf v_i}{dt}
 =-\sum_j m_j C_{ij}\nabla_i\widetilde W_{ij},\qquad
 \frac{du_i}{dt}
 =\frac12\sum_jm_j C_{ij}\mathbf v_{ij}\cdot\nabla_i\widetilde W_{ij},
\]

\[
 C_{ij}=\frac{P_i}{\rho_i^2}+\frac{P_j}{\rho_j^2}+\Pi_{ij},\qquad
 P_i=(\gamma-1)\rho_i u_i,\qquad c_i^2=\gamma(\gamma-1)u_i.
\]

The artificial viscosity follows equation 4.1, the unnumbered definition of
\(\Pi_{ij}\) that follows it, and equation 4.2, pp.550–551:

\[
 \mu_{ij}=\frac{\bar\ell_{ij}\,\mathbf v_{ij}\cdot\mathbf r_{ij}}
 {r_{ij}^2+\epsilon\bar\ell_{ij}^2},\quad
 \Pi_{ij}=\frac{-\alpha\bar c_{ij}\mu_{ij}+\beta\mu_{ij}^2}{\bar\rho_{ij}}
 \quad\hbox{for }\mathbf v_{ij}\cdot\mathbf r_{ij}<0,
\]

and \(\Pi_{ij}=0\) otherwise. Arithmetic means are used for density and sound
speed. The viscosity length scale is
\(\bar\ell_{ij}=\sqrt{5/6}\,(h_i+h_j)/2\), using the effective radii
\(h_i=H_i/\texttt{supportRadiusPerEffectiveRadius}\). The fixed factor makes
\(\bar\ell_{ij}=(H_i+H_j)/4\) for the cubic and preserves the previous
interpretation of the viscosity parameters. The same factor applies to every
selected family. Defaults remain the paper's \(\alpha=1\), \(\beta=2\), and
\(\epsilon=0.01\).

The active formulation has no pressure-entropy weighting, grad-h factors, evolving
viscosity coefficient, or shear limiter. The same symmetric pressure and viscosity
factors enter force and thermal work. Their simultaneous pair rates conserve linear
and angular momentum and balance kinetic with internal energy. This algebraic
identity does not imply exact conservation by a finite-step asynchronous integrator.

## Time integration and cosmological variables

The existing individual-rung predictor/corrector integration and particle drift
are retained. Section 10.3, p.563, permits leapfrog and predictor/corrector methods.
Neighbor rung limiting uses the union of the two supports and immutable source-rung
snapshots during each pass; it does not wake inactive particles. The support radius
is held fixed during drift; active particles and selected inactive neighbors solve
the neighbor constraint before hydro;
there is no separate radius predictor or divergence timestep bound.

The Courant-viscous limit uses

\[
 \Delta t_{\rm cv}=C_{\rm CFL}\min_i
 \frac{h_i}{c_i+0.6[\alpha c_i+\beta\max_j(-\mu_{ij},0)]}.
\]

The code interprets two apparent issues in the printed p.563 formulas:

- The printed force limit \(h/|\mathbf f|\) lacks a square root and has dimensions
  of time squared. The implemented acceleration limit scales as
  \(\sqrt{h/|\mathbf a|}\).
- The printed viscous limit uses \(\max\mu\), although equation 4.2 makes \(\mu\)
  negative for approaching particles. The implementation uses the nonnegative
  compression magnitude \(\max(-\mu,0)\).

The numerator uses the effective radius \(h_i=2\sigma_i\). The Courant coefficient
defaults to `0.22821773229381923`, converted from `0.25` by \(\sqrt{5/6}\)
to preserve the default cubic Courant-viscous limit. The hydro acceleration
coefficient is `gravity.timestepEta` times the fixed reference calibration
`0.7115965735877551`; the new default gravity eta is `0.20401599335954483`.
This preserves the former default cubic/Ferrers F4 hydro acceleration limit.
The additional fractional thermal-energy bound applies the fixed
\(\sqrt{5/6}\) reference conversion to preserve its previous physical limit. The explicit integrator retains its cosmological expansion limits.
The gravity acceleration criterion uses its own effective radius, independently of
the selected hydro family. These reference conversions do not establish equal
timestep accuracy for every kernel shape.

The `star` test relaxes its initial polytrope with a velocity damping term outside
the paper, \(d\mathbf w/d\tau \mathrel{-}= \lambda_{\rm d}\mathbf w\), where
\(\lambda_{\rm d}\) is the internal `hydro.damping` rate per code time unit. The
timestep criteria above do not include it, so the explicit kick is stable only while
\(\lambda_{\rm d}\Delta t\) stays below order one. The test therefore sets
\(\lambda_{\rm d}=1/t_{\rm dyn}\) with \(t_{\rm dyn}=(G\rho_0)^{-1/2}\); with the
cosmological code units \(t_{\rm dyn}\approx2.7\times10^5\) code times, so the former
\(\lambda_{\rm d}=1\) gave \(\lambda_{\rm d}\Delta t\approx3\times10^3\) and the rungs
diverged on the second step.

CosmicTiger evolves comoving positions \(\mathbf x\), conformal time
\(d\tau=dt/a\), stored velocity \(\mathbf w=a\mathbf v_{\rm pec}\), and stored
specific internal energy \(E=a^2u\). Density is comoving. The driver passes
\(a\,da/dt\) to the SPH `expansionRate` field; this differs from the driver's own physical
\(da/dt\) field. The scaled physical relative velocity used for viscosity is

\[
 a\,\Delta\mathbf v_{\rm physical}
 =\Delta\mathbf w+a\frac{da}{dt}\Delta\mathbf x.
\]

Peculiar velocity differences supply the pair pressure work. Physical relative
velocities supply viscosity and its heating, so homogeneous expansion does not
cause artificial viscosity. The homogeneous adiabatic source is

\[
 \frac{dE}{d\tau}\bigg|_{\rm background}
 =(5-3\gamma)\frac{da}{dt}E,
\]

which vanishes for \(\gamma=5/3\). The pair-force and thermal-work prefactors contain
\(m/a\) after converting to these stored variables. These transformations are
repository requirements; the cited equations in the review are noncosmological.

## Parameters and compatibility

Application options use `groupName.optionName` with camelCase names. The principal
hydrodynamic controls are:

| Option | Default | Meaning |
| --- | ---: | --- |
| `hydro.enabled` | `false` | Enable the Monaghan gas solver |
| `hydro.kernelFamily` | `spline` | Density family: `spline`, `wendland`, or `ferrers` |
| `hydro.kernelDegree` | `3` | Density polynomial degree; valid values depend on family |
| `hydro.gamma` | `5/3` | Ideal-gas adiabatic index |
| `hydro.viscosityAlpha` | `1` | Linear viscosity coefficient |
| `hydro.viscosityBeta` | `2` | Quadratic viscosity coefficient |
| `hydro.viscosityEpsilon` | `0.01` | Denominator coefficient multiplying \(\bar\ell^2\) |
| `hydro.neighborNumber` | `42` | Kernel-weighted support-sphere count |
| `hydro.minEffectiveRadius` | unset | Alternative minimum effective radius \(h=2\sigma\), in initial interparticle spacings |
| `hydro.minSupportRadius` | `1e-6` | Minimum \(H\), in initial interparticle spacings |
| `hydro.maxEffectiveRadius` | unset | Alternative maximum effective radius \(h=2\sigma\), in initial interparticle spacings |
| `hydro.maxSupportRadius` | `1000` | Requested maximum \(H\), in initial interparticle spacings; capped at half the box |
| `hydro.centralEffectiveRadius` | `0.0054772255750516604` | Imposed central-force effective radius in box units; uses the gravity kernel |
| `hydro.centralSupportRadius` | derived | Alternative central-force support radius in box units; uses the gravity kernel |
| `hydro.centralSoftening` | legacy alias | Alias for `hydro.centralEffectiveRadius` with the same value and units |
| `hydro.cfl` | `0.22821773229381923` | Courant-viscous coefficient using the effective radius \(h=2\sigma\); also controls the calibrated fractional-energy bound |

For each radius pair, an explicit effective-radius key wins over the
support-radius key, including across CLI and INI inputs, with a warning naming
the ignored key. For the central force, precedence is `centralEffectiveRadius`,
then the `centralSoftening` alias, then `centralSupportRadius`. The command line
still overrides INI values for the same key. See [kernel radii](kernels.md) for
the equivalent gravity options and conversion factors.

Gas and dark matter share one fixed comoving gravity kernel radius.
SPH density, pressure, viscosity, and thermal work retain the gas-only hydro
kernel and variable smoothing radii. See
[gravitational softening](kernels.md#fixed-comoving-gravitational-softening) for
the gravity radius options and their independence from SPH smoothing.

Checkpoints use version 3, and the low byte of `reserved` identifies the Monaghan
discretization. The next two bytes encode independent gravity and hydro kernel
selections. Bit 24 identifies the compact gas layout after removal of stellar,
chemistry, and obsolete solver fields. Older SPH checkpoints lack this bit and
are rejected before particle data are loaded; the gas payload also has its own
version 2 header. Compatible dark-matter checkpoint layouts remain unchanged.
Historical kernel marker values retain their original meanings; the Ferrers F2
default changes the selected gravity marker. A change of kernel selection on
restart is rejected.
A separate parameter block also rejects changes to viscosity coefficients,
smoothing constraints, timestep
controls, gravity enablement, and external accelerations on SPH restart. These
checks still apply when migrating an old configuration to the new length and
timestep conventions; old SPH checkpoints are not automatically converted.
Remove `hydro.chemistry`, `hydro.conduction`, `hydro.diffusion`, `hydro.yReflect`,
and `gravity.sphSmoothing` from older INI files, including entries set to false.
The `run.scenario=polytropicStar` scenario is a self-gravitating gas sphere; it does
not use stellar particles or star formation.

## Validation scope

The independent `monaghan_sph` CTest target retains the reference cubic checks:
float and double kernel normalization, spline joins, analytic versus
finite-difference derivatives, the paper's \(H=2h_M\) convention, viscosity signs
and coefficients, regularization, and invariance under length-unit changes and
velocity boosts. Pair tests use unequal pressures,
densities, masses, and supports to check net force, torque, and total energy rate,
including a pair inside only one support. They also check separation, uniform
translation, and the distinction between peculiar and physical thermal work.

The `monaghan_sph_cuda` target invokes the production density, smoothing, and
hydrodynamics kernels. It checks lower, upper, and fixed support bounds; the
weighted-neighbor residual; periodic support intersection; a uniform lattice;
nonzero perturbations; timestep selection; unequal-support pair conservation;
and homogeneous expansion with a nonmonatomic gamma. Selected-kernel checks
exercise all 14 family/degree choices through the production SPH path.
The same target compares all four SIMD CPU passes directly with CUDA, including
both hydro phases, permuted source/sink mappings, periodic positions, inactive and
semiactive particles, support-search retries, and immutable source snapshots.

The `sph_cpu` regression exercises the production SIMD code with ordinary host
allocations. It checks independent analytic density/force/timestep references,
pair conservation, expansion, every kernel selection, SIMD tails, periodic
geometry, activity, support bounds/retries, and iterative neighbor rung changes.
On Linux it enables floating-point traps to catch invalid arithmetic in masked
lanes. `smoke_sod_cpu`, `smoke_sph_cpu`, and `smoke_sph_cpu_mpi` cover shock-tube
evolution, cosmological evolution with restart, and two-rank MPI evolution through
the application driver.

The separate `radial_kernel` and `radial_kernel_cuda` targets check the generated
CPU/SIMD/CUDA tables, moments, gravitational force/potential consistency, and
float behavior near zero and the support boundary; see [kernels.md](kernels.md).

Historical validation on 2026-10-16, before selectable kernels and the effective
radius conversion: the complete CUDA build and all 23 CTest checks passed,
including grouped CLI/INI checks, CPU/CUDA mixed-mass gravity, SPH checkpoint
compatibility, MPI migration and evolution, the coarse Sod problem, and a short
cosmological SPH evolution with restart. At the smoke test's coarse resolution,
Sod density relative L1 error was 0.11484; the largest sampled relative total-energy
drift was 8.24e-8. These are regression results, not a resolution-convergence study
or evidence of long-time accuracy. No performance claim follows from them.
