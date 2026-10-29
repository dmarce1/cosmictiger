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

The computation code transcribes the paper. `cosmictiger/sph/monaghan/equations.hpp` and `viscosity.hpp` hold the
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
| \(f_a^{\rm AV}, (f_a^{\rm AV}+f_b^{\rm AV})/2\) | `f_a_AV`, `f_ab_AV` | GADGET-4 equation (72) shear factors |
| \(\rho_a\nabla\cdot\mathbf v_a, \rho_a\nabla\times\mathbf v_a\) | `ρ_a_div_v_a`, `ρ_a_curl_v_a` | unnormalized derivative sums; `∇` cannot appear in identifiers |
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
of degrees 2–9, Wendland C0/C2/C4, or Ferrers F1–F4. Gravity has an independent
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
selected family. The regularization remains \(\epsilon=0.01\). With variable
viscosity disabled the default \(\alpha=1\) and \(\beta=2\) recover the constant
coefficients of the paper. The default evolving coefficients are described below.

The Balsara (1995) shear limiter is enabled by default, using equation (72) of
Springel, Pakmor, Zier & Reinecke (2021), *Simulating cosmic structure formation
with the GADGET-4 code*, MNRAS 506, 2871–2949,
[doi:10.1093/mnras/stab1855](https://doi.org/10.1093/mnras/stab1855)
([author PDF, section 5.3](https://wwwmpa.mpa-garching.mpg.de/gadget4/gadget4-code-paper.pdf)).
Equation (72) belongs to GADGET-4, not to the original Balsara paper.
During the density pass,
the code accumulates the gather estimates
\(\rho_i\,\nabla\!\cdot\mathbf v_i=-\sum_j m_j\mathbf v_{ij}\cdot\nabla_i W_{ij}(H_i)\)
and
\(\rho_i\,\nabla\!\times\mathbf v_i=\sum_j m_j\mathbf v_{ij}\times\nabla_i W_{ij}(H_i)\).
It computes \(f_i=|\nabla\!\cdot\mathbf v_i|/(|\nabla\!\cdot\mathbf v_i|+
|\nabla\!\times\mathbf v_i|)\), with \(f_i=0\) when both derivatives vanish,
and multiplies \(\Pi_{ij}\) by \((f_i+f_j)/2\) in both force and heating.
The density normalization cancels in the ratio. The derivatives use the physical
relative velocity, including Hubble flow, consistently with the viscosity.
For this shear correction, only \(f_i\), one float per gas particle, is retained between density and force
passes; the divergence and curl are discarded. This derived factor is recomputed
after restart and migration and is not stored in checkpoints. Checkpoint metadata
records whether the limiter is enabled and rejects a restart with a different setting.
Use `--hydro.balsaraSwitch=false` to disable this shear correction.

The active formulation has no pressure-entropy weighting or grad-h factors.
The same symmetric pressure and viscosity factors enter force and thermal work.
Their simultaneous pair rates conserve linear
and angular momentum and balance kinetic with internal energy. This algebraic
identity does not imply exact conservation by a finite-step asynchronous integrator.

## Morris–Monaghan variable viscosity

`hydro.viscosityEvolution=morrisAndMonaghan` (the default) enables the original additive switch of
Morris & Monaghan (1997), *A Switch to Reduce SPH Viscosity*, JCP 136, 41–50,
[doi:10.1006/jcph.1997.5690](https://doi.org/10.1006/jcph.1997.5690):

\[
 \frac{d\alpha_i}{dt}=-\frac{\alpha_i-\alpha^\ast}{\tau_i}+S_i
 \quad\text{(5)},\qquad
 \tau_i=\frac{h_{M,i}}{\mathcal C_1c_i}\quad\text{(6)},\qquad
 S_i=\max(-\nabla\!\cdot\mathbf v_i,0)\quad\text{(13)}.
\]

The defaults are \(\alpha^\ast=0.1\) and \(\mathcal C_1=0.2\), as used in their
calculations. There is no hard upper cap or \((\alpha_{\max}-\alpha)\) source
multiplier. The source uses equation (13) directly, without an additional
adiabatic-index scaling. The decay length is the same calibrated viscosity length
\(h_{M,i}=\sqrt{5/6}\,H_i/\texttt{supportRadiusPerEffectiveRadius}\), equal to
\(H_i/2\) for the cubic spline. New particles start at \(\alpha^\ast\); this
smooth-flow initialization is an implementation choice, since the paper uses
problem-dependent initial values, including elevated α near the initial Sod interface.

Pair forces and heating use \(\bar\alpha_{ij}=(\alpha_i+\alpha_j)/2\), preserving
pair symmetry. The single quadratic coefficient `hydro.viscosityBeta` (default 2)
has a model-dependent meaning:

- With `morrisAndMonaghan` and `cullenAndDehnen`, \(\beta_{ij}=\texttt{viscosityBeta}\,
  \bar\alpha_{ij}\), the convention of Morris & Monaghan's equation (3), so β decays
  with α.
- With `none` and `priceEtAl`, \(\beta_{ij}=\texttt{viscosityBeta}\) is constant: the
  fixed Monaghan coefficients, and Phantom's equation (41), in which β does not
  multiply α.

With `--hydro.viscosityEvolution=none` every pair uses the constant
`hydro.viscosityAlpha` and `hydro.viscosityBeta`; the disabled mode allocates no
per-particle variable-viscosity state.

The density pass computes the instantaneous velocity divergence and the new α
rate for active particles. Divergence and curl are then discarded; neither their
history nor \(\nabla\cdot(\rho\mathbf v)\) is needed. Momentum divergence contains
an advected-density term and is not a Galilean-invariant compression indicator.
The derivative-of-divergence indicator described after equation (72) in GADGET-4
belongs to a later switch, not to Morris–Monaghan equation (13).

α and its previous rate survive sorting, domain migration, and checkpoints. The
existing kick predictor/corrector also retains a predicted base while closing a
step. Density retries do not advance α, and semiactive neighbors keep the old
rate needed when their own staggered step closes. Silo writes a dimensionless
`alpha` gas field when viscosity evolution is enabled. Morris–Monaghan retains
and writes no divergence field; Cullen–Dehnen has additional state described below.

In the cosmological variables below, the normalized density derivative sum is
\(D_i=a^2\nabla_{\rm physical}\cdot\mathbf v_{\rm physical}\). Thus

\[
 \frac{d\alpha_i}{d\tau}
 =-(\alpha_i-\alpha^\ast)\frac{\mathcal C_1c_{\rm stored,i}}{a h_{M,i}}
 +\frac{\max(-D_i,0)}{a}.
\]

This includes homogeneous expansion in the same physical relative velocity used
by artificial viscosity. It yields no compression source in expanding flow.

## Cullen–Dehnen variable viscosity

`--hydro.viscosityEvolution=cullenAndDehnen` selects Cullen & Dehnen (2010),
*Inviscid smoothed particle hydrodynamics*, MNRAS 408, 669–683,
[doi:10.1111/j.1365-2966.2010.17158.x](https://doi.org/10.1111/j.1365-2966.2010.17158.x)
([author paper](https://arxiv.org/pdf/1006.1524)). Their equations (13)–(16) define

\[
 A_i=\xi_i\max\!\left[-\frac{D(\nabla\cdot\mathbf v)_i}{Dt},0\right],\qquad
 \alpha_{{\rm loc},i}=\alpha_{\max}
 \frac{H_i^2 A_i}{(v_i^{\rm sig})^2+H_i^2 A_i},
\]
\[
 v_i^{\rm sig}=\max_{j:r_{ij}\le H_i}
 \left[\frac{c_i+c_j}{2}-\min(0,\mathbf v_{ij}\cdot\widehat{\mathbf r}_{ij})\right],
 \qquad \tau_i=\frac{H_i}{2\ell v_i^{\rm sig}}.
\]

Here the paper's smoothing length is the **full support radius** \(H_i\), unlike
the Morris–Monaghan cubic length \(H_i/2\). The CD floor and initial α are zero;
\(\ell=0.05\) follows the paper. `hydro.viscosityAlphaMax=2` is an implementation
choice. Equation (A15) raises α immediately to the target when needed and otherwise
decays it exponentially toward that target.

The limiter is equation (18), with the neighbor compression-sign measure from
(17):
\[
 R_i=\frac{\sum_jm_j\operatorname{sign}[(\nabla\cdot\mathbf v)_j]W_{ij}}{\rho_i},
 \qquad
 \xi_i=\frac{|2(1-R_i)^4\nabla\cdot\mathbf v_i|^2}
 {|2(1-R_i)^4\nabla\cdot\mathbf v_i|^2+\operatorname{tr}(\mathbf S_i\mathbf S_i^T)},
\]
where \(\mathbf S_i=\tfrac12(\mathbf V_i+\mathbf V_i^T)
-\tfrac13\operatorname{tr}(\mathbf V_i)\mathbf I\) and
\(\mathbf V_i=\nabla\mathbf v_i\). The matrix-corrected gradient estimates follow
(B2), (B4), and (B7), using mass/kernel-gradient weights so they do not read
densities being updated concurrently on other ranks. A symmetric pseudoinverse
handles planar or collinear neighborhoods on their resolved subspace.
Equation (B12) computes the material divergence derivative as
\(\operatorname{tr}(\nabla\mathbf a-\mathbf V^2)\), using the previous accelerations;
it does not finite-difference stored divergence values.

CD nevertheless retains the previous divergence for the immutable neighbor-sign
snapshot in equation (17), following the sweep order in Appendix A4. CD and the
Price et al. switch below allocate this extra state and write the physical velocity
divergence as `divv` in Silo, with inverse-time units. Every evolution scheme writes
dimensionless `alpha`; selecting `none` allocates neither α evolution state nor
divergence state. Under CD, β follows α as described above.

## Price et al. (Phantom) variable viscosity

`--hydro.viscosityEvolution=priceEtAl` selects the switch of Price et al. (2018),
*Phantom: A smoothed particle hydrodynamics and magnetohydrodynamics code for
astrophysics*, PASA 35, e031,
[doi:10.1017/pasa.2018.25](https://doi.org/10.1017/pasa.2018.25)
([arXiv:1702.03930](https://arxiv.org/abs/1702.03930)), section 2.2.9. It is their
variation on Cullen & Dehnen, implemented in `sph/price_etal/`:

\[
 A_i=\xi_i\max\!\left[-\frac{d}{dt}(\nabla\cdot\mathbf v_i),0\right]\ \text{(46)},\qquad
 \xi_i=\frac{|\nabla\cdot\mathbf v_i|^2}{|\nabla\cdot\mathbf v_i|^2+|\nabla\times\mathbf v_i|^2}
 \ \text{(47)},
\]
\[
 \alpha_{{\rm loc},i}=\min\!\left(\frac{10\,h_i^2A_i}{c_{s,i}^2},\alpha_{\max}\right)
 \ \text{(48)},\qquad
 \tau_i=\frac{h_i}{\sigma_{\rm decay}c_{s,i}}.
\]

The differences from CD are: ξ is the Balsara-type ratio of equation (47) rather than
the sign-weighted \(\operatorname{tr}(\mathbf S\mathbf S^T)\) limiter, so no neighbor
divergence snapshot is gathered; the target uses the sound speed and the fixed factor
10, capped at `hydro.viscosityAlphaMaxPrice` (default 1, the paper's \(\alpha_{\max}\));
the decay time uses the sound speed with `hydro.viscosityDecayPrice`
(\(\sigma_{\rm decay}=0.1\)); and when \(\alpha_{{\rm loc}}\le\alpha\) the decay of
equation (49) is integrated with the backward Euler step of equation (50),
\(\alpha^{n+1}=(\alpha^n+\alpha_{\rm loc}\Delta t/\tau)/(1+\Delta t/\tau)\), otherwise α
is set to \(\alpha_{\rm loc}\) immediately. The paper's \(h_i\) is the smoothing length
with support \(2h_i\) for the cubic spline, so the code uses the same calibrated
viscosity length \(h_{M,i}\) as Morris–Monaghan. The velocity gradient, its curl, and
\(d(\nabla\cdot\mathbf v)/dt=\operatorname{tr}(\nabla\mathbf a-\mathbf V^2)\)
(their equations (51)–(54)) come from the same CD appendix-B matrix reconstruction,
with the previous accelerations and the cosmological terms described above. The
initial α is zero, and the paper's quadratic coefficient is the constant
`hydro.viscosityBeta` of their equation (41); the pair viscosity itself remains
Monaghan's \(\Pi_{ij}\) above rather than Phantom's \(q_{ab}\) form. α and the
previous divergence share the CD storage, migration, checkpoint, and Silo fields.

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

The `star` test's constant-α schedule (zero during relaxation, 0.05 afterward)
applies only with `hydro.viscosityEvolution=none`; evolving viscosity follows the
selected switch. It relaxes its initial polytrope with a velocity damping term outside
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
| `hydro.viscosityAlpha` | `1` | Constant linear coefficient when variable viscosity is disabled |
| `hydro.viscosityBeta` | `2` | Quadratic coefficient: multiplies \(\bar\alpha_{ij}\) for `morrisAndMonaghan` and `cullenAndDehnen`, constant for `none` and `priceEtAl` |
| `hydro.viscosityEvolution` | `morrisAndMonaghan` | α evolution: `none`, `morrisAndMonaghan`, `cullenAndDehnen`, or `priceEtAl` |
| `hydro.viscosityAlphaMin` | `0.1` | Morris–Monaghan positive α floor and initial value |
| `hydro.viscosityDecay` | `0.2` | Morris–Monaghan decay coefficient \(\mathcal C_1\) |
| `hydro.viscosityAlphaMax` | `2` | Cullen–Dehnen maximum α; the MM scheme has no upper cap |
| `hydro.viscosityDecayCD` | `0.05` | Cullen–Dehnen decay coefficient \(\ell\) |
| `hydro.viscosityAlphaMaxPrice` | `1` | Price et al. maximum α in equation (48) |
| `hydro.viscosityDecayPrice` | `0.1` | Price et al. decay coefficient \(\sigma_{\rm decay}\) |
| `hydro.viscosityEpsilon` | `0.01` | Denominator coefficient multiplying \(\bar\ell^2\) |
| `hydro.balsaraSwitch` | `true` | Apply the Balsara shear limiter to artificial viscosity |
| `hydro.neighborNumber` | `42` | Kernel-weighted support-sphere count |
| `compute.sphBucketSize` | `0` (automatic) | SPH tree leaf limit; automatic sizing uses `floor(8/pi * hydro.neighborNumber)`, giving `106` at the default neighbor target |
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

SPH and gravity use independent tree bucket sizes. `compute.bucketSize` controls
the gravity tree, while a positive `compute.sphBucketSize` fixes the SPH leaf
limit. Zero selects the historical SPH sizing heuristic above after CLI and INI
neighbor options have been resolved. Startup output and snapshot metadata report
the resolved positive size. The gravity compile-time bucket cap does not apply
to SPH, whose kernels iterate over each leaf's particle range.

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

The independent `sph_kernels` and `monaghan_sph` CTest targets retain the reference cubic checks:
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
exercise all 15 family/degree choices through the production SPH path.
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

## Source layout

Common particle storage, source staging, neighbor traversal, CPU/CUDA execution,
and timestep scheduling stay directly in `cosmictiger/sph/` and `src/sph/`.
The numerical components live in author-named directories:

- `monaghan/`: pair pressure work and the baseline artificial-viscosity formula.
- `morris_and_monaghan/`: compression/decay rate and predictor/corrector updates of alpha.
- `cullen_and_dehnen/`: gradient reconstruction, shock detection, alpha updates, and model diagnostics.
- `price_etal/`: the Phantom shock detector and backward Euler alpha update, reusing the CD gradient reconstruction and state.
- `balsara/`: the shear limiter, composable with every evolving-alpha model.

The common `viscosity_state.hpp` defines alpha storage used by all evolution
models; the Cullen–Dehnen diagnostic state is shared with Price et al. Allocation,

reordering, migration, and checkpoint traversal remain in common particle code.
The shared density and hydro passes call these policies without duplicating
neighbor walks. Standalone policy tests use the same author-directory organization.
