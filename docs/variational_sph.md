# Variational density-energy SPH with grad-H terms

`hydro.pressureOperator=variational` replaces the Monaghan (1992) pressure force and
reversible thermal work of [the Monaghan baseline](monaghan_sph.md) with the
compatible density-energy operator derived from the discrete SPH Lagrangian with
variable supports ("grad-h" or "grad-H" terms). The default,
`hydro.pressureOperator=monaghan`, is unchanged and remains the regression baseline.
Artificial viscosity, its switches, the support solve, the timestep criteria,
neighbor propagation and the external forces are shared by both operators.

References:

- P. F. Hopkins, *A general class of Lagrangian smoothed particle hydrodynamics
  methods and implications for fluid mixing problems*, MNRAS 428 (2013) 2840,
  [arXiv:1206.5006](https://arxiv.org/abs/1206.5006): equation 8 (weighted
  volume sum), 9–11 (its derivatives), 12 (equation of motion), 13 (\(f_{ij}\)) and
  14 (the density special case).
- D. J. Price, *Smoothed particle hydrodynamics and magnetohydrodynamics*,
  J. Comput. Phys. 231 (2012) 759, [arXiv:1012.1885](https://arxiv.org/abs/1012.1885):
  equations 27 (\(\Omega_a\)), 30 (equation of motion) and 35 (thermal energy).

## Selection and supported physics

| Option | Values | Default |
| --- | --- | --- |
| `hydro.pressureOperator` | `monaghan`, `variational` | `monaghan` |

The variational operator is implemented for the constant-\(\gamma\) ideal gas,
\(P=(\gamma-1)\rho u\), on CPU and CUDA, with every viscosity model
(`hydro.viscosityEvolution`) and the Balsara limiter, with or without self-gravity,
in isolated scenarios and in comoving cosmology, with the traditional and global
time integrators. It is rejected, with a diagnostic, together with
`chemistry.enabled=true` or `starFormation.enabled=true`; those keep the Monaghan
operator until a composition-dependent equation of state is carried through the
variational volume. No entropy variable is stored: the operator evolves the stored
internal energy, as the Monaghan operator does.

## Notation

The stored length is the compact support radius \(H\) and the selected kernel is
\(W(r,H)=H^{-3}w(q)\), \(q=r/H\). It is not the papers' smoothing length \(h\), whose
normalization differs between kernels; only \(H\,\partial W/\partial H\) ratios enter,
so the formulas below are unchanged. Particle \(i\) carries a mass \(m_i\) and a
volume weight \(x_i\); \(r_{ij}=r_i-r_j\) and \(v_{ij}=v_i-v_j\).

| Symbol | Meaning | Code |
| --- | --- | --- |
| \(n_i=\sum_j W_{ij}(H_i)\) | unweighted number density (constraint) | `sph_volume::numberDensity` |
| \(y_i=\sum_j x_jW_{ij}(H_i)\) | weighted density | `sph_volume::weightedDensity` |
| \(V_i=x_i/y_i\) | particle volume | `sph_volume::volume` |
| \(\rho^*_i=m_iy_i/x_i\) | density implied by \(V_i\) | `sph_volume::volumeDensity` |
| \(\Omega_i\) | grad-H factor of the constraint | `sph_volume::omega`, `Ω_a` |
| \(\zeta_i\) | weighted grad-H correction, in units of \(x\) | `sph_volume::zeta`, `ζ_a` |
| \(Q_{ij}\) | directed pressure coefficient | `sph_volume::directedCoefficient` |
| \(g_i=\nabla_iW_{ij}(H_i)\), \(g_j=\nabla_iW_{ij}(H_j)\) | separate kernel gradients | `F_a r_ab`, `F_b r_ab` |

\(\partial W/\partial H=H^{-4}[-3w(q)-qw'(q)]\) (`sph_volume::kernelSupportDerivative`).
This stage uses \(x_i=m_i\), so \(y_i\) is the ordinary mass-summed density, and all
production gas particles have one mass. The shared algebra in
`cosmictiger/sph/volume.hpp` keeps \(n\) and \(y\) distinct and accepts unequal
weights and masses; the algebra tests use both.

## Constraint and equations

The support solve holds \(H_i^3 n_i\) constant: \(\sum_j w(q_{ij})=3N_{\rm target}/(4\pi)\)
to a relative residual \(10^{-4}\) (`kSphSupportRadiusTolerance`), within the bounds
\(H_{\min}\le H_i\le H_{\max}\). For a free support,
\[
\Omega_i=1+\frac{H_i}{3n_i}\frac{\partial n_i}{\partial H_i}
=-\frac{\sum_j q_{ij}w'(q_{ij})}{3\sum_j w(q_{ij})},\qquad
\zeta_i=\frac{H_i}{3n_i\Omega_i}\frac{\partial y_i}{\partial H_i}
=\frac{3\sum_jx_jw(q_{ij})+\sum_jx_jq_{ij}w'(q_{ij})}{\sum_jq_{ij}w'(q_{ij})}.
\]
The second forms are what the code evaluates: they avoid the cancellation
\(1+(-1-\epsilon)\) of the defining form and contain no power of \(H\). For the
nonincreasing kernels of `radial_kernel`, \(\Omega_i\ge 0\). Differentiating the
constraint gives \(dH_i=-H_i/(3n_i\Omega_i)\,dn_i\) and
\[
dy_i=\sum_j\,(x_j-\zeta_i)\;g_i\cdot dr_{ij}.
\]
With the adiabatic \(d(m_iu_i)=-P_i\,dV_i=P_ix_i/y_i^2\,dy_i\), the Lagrangian
\(L=\sum_i m_i(v_i^2/2-u_i)\) gives Hopkins (2013) equations 12–13 with the constraint
scalar \(\tilde x=1\), written with the directed coefficient
\[
Q_{ij}=\frac{P_ix_i}{y_i^2}\,(x_j-\zeta_i),\qquad
m_i\frac{dv_i}{dt}=-\sum_j\bigl[Q_{ij}g_i+Q_{ji}g_j\bigr],\qquad
m_i\frac{du_i}{dt}=\sum_jQ_{ij}\,v_{ij}\cdot g_i .
\]
Hopkins' \(f_{ij}=1-\zeta_i/x_j\) multiplies \(x_j\) inside \(Q_{ij}\); the code forms
\(x_j-\zeta_i\) directly instead of dividing by \(x_j\) and multiplying back. Force and
work use the same \(Q_{ij}\). For equal-mass density SPH, \(\zeta_i=m(1-1/\Omega_i)\),
\(f_{ij}=1/\Omega_i\) and the equations are Price (2012) equations 30 and 35:
\[
\frac{dv_i}{dt}=-\sum_jm_j\Bigl[\frac{P_i}{\Omega_i\rho_i^2}g_i+\frac{P_j}{\Omega_j\rho_j^2}g_j\Bigr],\qquad
\frac{du_i}{dt}=\frac{P_i}{\Omega_i\rho_i^2}\sum_jm_jv_{ij}\cdot g_i .
\]
Unlike the Monaghan operator, which averages the two gradients and gives each pair
member half of the symmetric pressure work, each particle's work here is its own
adiabatic rate \(P_ix_i/y_i^2\,dy_i/dt\): the discrete entropy of each particle is
unchanged by the reversible terms. The pair rates conserve linear and angular
momentum (each pair force is along \(r_{ij}\)) and exchange kinetic and internal
energy exactly, for any values of \(\zeta\); the accuracy of \(\zeta\) decides only how
closely the force is the gradient of the discrete thermal energy.

In code, with \(x=m\) for every gas particle,
\(Q_{ab}/m_a=(P_a/\rho_a^2)(m-\zeta_a)\) and
\(Q_{ba}/m_a=(P_b/\rho_b^2)(m-\zeta_b)\); see `src/sph/sph_hydro_cpu.cpp` and
`src/sph/sph_hydro.cu`.

## Supports at bounds

The support solve accepts \(H_i=H_{\min}\) with more than the target sum and
\(H_i=H_{\max}\) with less; these supports do not meet the constraint and do not
move with the particles, \(dH_i/dr=0\). The periodic search caps \(H_{\max}\) at half
the box, and a requested \(H_{\min}=H_{\max}\) fixes every support. For such a held
support the derivation above gives \(\zeta_i=0\) (a density grad-H factor of one) and
\(m_i\,du_i/dt=\sum_j x_jP_ix_i/y_i^2\,v_{ij}\cdot g_i\).
`sph_volume::classifySupport` decides from the support and its relative residual at
the stored \(H\):

| Classification | Condition | \(\zeta_i\) |
| --- | --- | --- |
| `Fixed` | \(H_{\min}=H_{\max}\) | 0 |
| `LowerBound` | \(H_i\le H_{\min}\) and residual \(>10^{-4}\) | 0 |
| `UpperBound` | \(H_i\ge H_{\max}\) and residual \(<-10^{-4}\) | 0 |
| `Free` | otherwise, including a bound met within tolerance | formula above |

At \(H_{\min}=H_{\max}\) with equal supports, \(g_i=g_j\) and the variational force equals
the Monaghan pressure force; the work is split between the pair members differently,
with the same total (`SphCpuTest.VariationalHeldSupports`).

## Conditioning and diagnostics

A free support whose \(\Omega_i\) or \(\zeta_i\) is non-finite, or whose \(\Omega_i\le0\),
stops the run with a diagnostic naming its support, density, sums and residual; the
code never clamps a correction to unity. Finiteness is tested on the exponent bits
because the hydro sources are compiled with `-ffast-math`. A free support with
\(0<\Omega_i<0.1\) (`sph_volume::kMinimumWellConditionedOmega`) is counted as
ill-conditioned: its force amplifies the support-solve residual and the summation
error by \(1/\Omega_i\). Each density preparation prints

```text
SPH grad-H: Omega_min = ... residual_max = ... held = ... ill-conditioned (Omega < 0.1) = ...
```

and a warning when any support is ill-conditioned. `Omega_min` and `residual_max`
are taken over free supports; counts include supports that a later pass of the
same preparation evaluates again.

The production solve stops at a relative residual of \(10^{-4}\). The float rates at
the converged supports match a double-precision evaluation at the same supports to
\(5\times10^{-7}\) of the largest acceleration, and differ from rates at exactly
solved supports by \(8\times10^{-5}\) (`SphCpuTest.VariationalAgainstReference`). The
double-precision tests in `tests/sph/volume.cpp` solve supports to roundoff.

## Viscosity

The artificial viscosity is the Monaghan \(\Pi_{ij}\) with the symmetric gradient
\(\tfrac12(g_i+g_j)\), its pair-symmetric force and its heating
\(\tfrac12\sum_j m_j\Pi_{ij}v^{\rm phys}_{ij}\cdot\tfrac12(g_i+g_j)\), as in the
Monaghan operator; `hydro.viscosityEvolution` and `hydro.balsaraSwitch` act on it
unchanged. Only the reversible pressure terms change.

## Cosmological variables

Positions and density are comoving, conformal time obeys \(d\tau=dt/a\), the stored
velocity is \(w=av_{\rm pec}\) and the stored energy \(E=a^2u\)
([Monaghan baseline](monaghan_sph.md#time-integration-and-cosmological-variables)).
\(\zeta\) is a ratio of sums in \(q\) and does not depend on \(a\). With
\(P_c=(\gamma-1)\rho_cE=a^5P\), \(y_c=a^3y\) and \(g_c=a^4g\), the physical
\(Q_{ij}=aQ^c_{ij}\) and
\[
\frac{dw_i}{d\tau}=-\frac1{a\,m_i}\sum_j\bigl[Q^c_{ij}g^c_i+Q^c_{ji}g^c_j\bigr],\qquad
\frac{dE_i}{d\tau}=\frac1{a\,m_i}\sum_jQ^c_{ij}\,w_{ij}\cdot g^c_i
+(5-3\gamma)\frac{da/d\tau}{a}E_i ,
\]
the same \(m/a\) prefactor as the Monaghan pair terms. The pair work uses the stored
(peculiar) velocity difference. This is exact, not a continuum approximation: in
comoving coordinates \(V_i=a^3V^c_i\) with \(V^c_i\) a function of comoving positions
and comoving supports, so
\(d(m_iu_i)/dt=P_iV_i/y^c_i\,dy^c_i/dt-3(\dot a/a)P_iV_i\). The first term is the
peculiar pair work, the second the analytic homogeneous term, for free supports and
for supports held at comoving bounds alike. For a free support the Hubble flow's own
pair work is also exactly \(-3(\dot a/a)P_iV_i\), because
\(\sum_j(x_j-\zeta_i)\,r_{ij}\cdot g_i=-3y_i\) identically; adding the Hubble flow to
the pair work would therefore count it twice. A held support has no such identity,
which is why the analytic term, not physical pair velocities, is used.
For \(\gamma=5/3\) the expansion term vanishes and these variables evolve as a static
gas in \(ds=d\tau/a\): the stored \(\sum_im_i(w_i^2/2+E_i)\) is conserved by the pair rates.

## Implementation and data lifecycle

- **Density pass** (`kSphRunPrehydro2`, CPU `src/sph/sph_prehydro_cpu.cpp`, CUDA
  `src/sph/sph_prehydro.cu`): for every semiactive particle, after its support has
  converged, the loop also accumulates \(\sum_jw\) and \(\sum_jqw'\);
  `sphGradHEvaluation` (`cosmictiger/sph/sph_kernel.hpp`) classifies the support and
  forms \(\zeta\), which is written beside the density.
- **Storage**: `sphParticlesGradHCorrections`, one float per gas particle, exists
  only when the operator is variational. It is swapped with the particle arrays,
  initialized to NaN for new slots and set to NaN for every particle by
  `sphParticlesResetConverged`, which starts each density preparation. It is neither
  checkpointed nor migrated.
- **Staging**: the force pass stages \(\zeta\) with the density through
  `sphParticlesGlobalReadDensity`; remote sources share the density cache line, which
  every `sphRun` clears. Source snapshots stay immutable during each pass.
- **Freshness**: every source that interacts with an active sink in the force pass
  is semiactive, and every semiactive particle's density and \(\zeta\) are rebuilt in
  the same step's density pass. A correction left from an earlier step, a migrated
  slot or a restart is NaN, is masked in non-interacting SIMD lanes, and stops the
  force pass if it reaches an interacting pair (the rates are tested for finiteness).
  The star-formation conversion path rebuilds densities before its next force use;
  the chemistry and star-formation reruns of the force pass are excluded with this
  operator. Restart and migration therefore need no stored correction; an exact
  restart reproduces the uninterrupted run bit for bit.
- **CUDA** stages \(\zeta_b\) in each `HydroNeighbor`; the CPU and CUDA paths share
  `sphGradHEvaluation` and the `sph_volume` helpers.

## Checkpoint compatibility

The low byte of the checkpoint physics code identifies the pressure operator:
1 for Monaghan (the historical value) and 2 for variational. A restart with the
other operator is rejected with
`Checkpoint SPH pressure operator (...) differs from hydro.pressureOperator=...`;
a checkpoint is never reinterpreted with another operator, and every older SPH
checkpoint remains a Monaghan checkpoint. The gas payload and its version are
unchanged because \(\zeta\) is not stored.

## Validation

Recorded October 30, 2026 (CPU Release build with chemistry and stars compiled in;
CUDA 13.1 build for sm_86, executed on the GPU). Spatial (operator) and temporal
(integrator) errors are reported separately: exact pair-rate identities do not make
a finite-step integrator conserve exactly, least of all with individual rungs.

**Algebra, double precision** (`sph_volume` CTest, `tests/sph/volume.cpp`, supports
solved to roundoff in a 48-particle isolated cloud):

- \(\partial W/\partial H\) and \(\nabla W\) agree with central differences for all 15
  selectable kernels (spline 2–9, Wendland C0/C2/C4, Ferrers F1–F4).
- \(n\), \(y\), their \(H\) derivatives, \(V\), \(\rho^*\), \(\Omega\) and \(\zeta\) agree with
  direct sums and finite differences for unequal weights and masses; the equal-mass
  reduction \(\zeta=m(1-1/\Omega)\), \(f_{ij}=1/\Omega\) and Price's equation 30 hold to
  \(10^{-12}\).
- \(dy_i/dr_k\) and \(dH_i/dr_k\) with the support re-solved match the constrained
  derivatives.
- The force equals minus the finite-difference gradient of
  \(\sum_im_iu_i(\rho^*_i)\) at fixed entropy, with every support re-solved, to
  \(10^{-6}\) of the largest force, for free supports (cubic, Wendland C4, Ferrers F3,
  spline 5), unequal masses, unequal weights, fixed supports and both bounds active,
  at \(\gamma=5/3\) and \(7/5\).
- Pair rates conserve momentum, angular momentum and kinetic plus internal energy to
  \(10^{-13}\); each particle's work equals its own adiabatic rate (zero entropy rate)
  for free and held supports; the free-support Hubble identity holds to \(10^{-12}\).
- A copy of `volume.hpp` that drops \(\zeta\) fails four of these tests.

**Production kernels, float** (`sph_cpu` CTest, `SphCpuTest.Variational*`):

| Check | Result |
| --- | --- |
| Rates vs double reference at the production supports (perturbed 6³ lattice, \(a=0.7\), \(\gamma=1.4\)) | \(4.7\times10^{-7}\) acceleration, \(2.3\times10^{-7}\) energy rate |
| Rates vs exactly solved supports | \(7.7\times10^{-5}\), \(5.0\times10^{-5}\) (solve tolerance \(10^{-4}\)) |
| Isolated cluster with supports held at \(H_{\max}\) | \(\zeta\) matches, held count matches; momentum, angular momentum and energy rates vanish to \(3\times10^{-5}\) with viscosity |
| Hubble flow through a free support | pair work \(=-3hP/\rho\) to \(2\times10^{-5}\); held supports differ (no identity) |
| All supports held (\(H_{\min}\), \(H_{\max}\), fixed) | \(\zeta=0\); with fixed equal supports the force equals the Monaghan force |
| Uniform lattice | zero force and work; homogeneous source exact at \(\gamma=5/3\), \(7/5\) |
| NaN correction on a non-interacting source | masked under FP traps; on an interacting source the pass stops |

**Synchronous (global-step) evolution** with the production kernels and the production
predictor-corrector kick-drift-kick (8³ lattices, inviscid unless stated):

| Test | Steps | Energy error | Entropy drift, variational | Entropy drift, Monaghan |
| --- | --- | --- | --- | --- |
| Sound wave, \(\delta=0.05\), half a period | 12 / 24 / 48 | \(3.5\times10^{-2}\) / \(4.3\times10^{-3}\) / \(5.5\times10^{-4}\) of the wave energy (both operators) | \(1.1\times10^{-3}\) / \(2.8\times10^{-4}\) / \(7.0\times10^{-5}\) | \(2.2\times10^{-3}\) / \(1.3\times10^{-3}\) / \(1.1\times10^{-3}\) |
| Compression to \(\rho_{\max}=1.28\), \(\gamma=1.4\) | 20 / 40 | \(1.3\times10^{-3}\) / \(3.0\times10^{-4}\) | \(1.2\times10^{-4}\) / \(3.0\times10^{-5}\) | \(1.7\times10^{-3}\) / \(1.7\times10^{-3}\) |

The energy error is temporal and second order for both operators. The variational
entropy drift is also temporal and converges; the Monaghan drift stops at a
resolution-limited floor because its pair work is split symmetrically. The wave
profile error after half a period, a spatial error at eight particles per
wavelength, is 0.085 (variational) and 0.10 (Monaghan).

Comoving sound waves in an Einstein–de Sitter background, \(a=(\tau/\tau_0)^2\) from 1 to
4: the residual of the stored energy budget (pair exchange plus the integrated
\((5-3\gamma)\) term) is \(4.5\times10^{-3}\), \(1.4\times10^{-3}\), \(3.9\times10^{-4}\) of the
wave energy for \(\gamma=5/3\) and \(1.4\times10^{-4}\), \(3.6\times10^{-5}\),
\(8.9\times10^{-6}\) of the expansion work for \(\gamma=7/5\) at 40, 80, 160 steps; a uniform
lattice follows \(E\propto a^{5-3\gamma}\) to \(1.2\times10^{-5}\) at 160 steps.

**Application** (`variational_sph_application` CTest runs `tests/run_variational_sph.py
--quick`; the full driver ran 45 cases):

| Case | Monaghan | Variational |
| --- | --- | --- |
| Sod 16³, traditional: density L1 vs exact; final energy error | 0.053; \(1.5\times10^{-6}\) | 0.050; \(1.8\times10^{-6}\) |
| Sod 16³, global | 0.055; \(1.2\times10^{-5}\) | 0.053; \(1.4\times10^{-5}\) |
| Sod 16³, global, CFL halved (energy error) | \(1.0\times10^{-4}\to1.5\times10^{-5}\) | \(1.1\times10^{-4}\to1.7\times10^{-5}\) |
| hydroWave 16³, traditional / global (max energy error) | \(8.4\times10^{-5}\) / \(6.5\times10^{-4}\) | \(1.3\times10^{-4}\) / \(9.5\times10^{-4}\) |
| hydroWave 16³, global, CFL halved | ratio 0.12 | ratio 0.12 |
| Sedov blast 8³ and 16³ | aborts (negative internal energy) in all four runs | 8³ both integrators and 16³ global complete, energy error \(\le6\times10^{-5}\); 16³ traditional stops at the rung limit |
| SPH cosmology 8³, 5 steps, \(\gamma=5/3\) and \(7/5\), gravity on | completes | completes, same energy diagnostics |
| hydroWave 8³ on two localities vs one | — | final kinetic energy differs by \(3.8\times10^{-6}\) |
| `sph_migration_mpi` regression with the variational operator | — | passes |
| CUDA application runs (Sod 16³, hydroWave 8³ and 16³) vs CPU | matches | matches; final kinetic energy within \(5.5\times10^{-5}\) |
| Chemistry with the variational operator | — | rejected with the documented diagnostic |

Sod L1 values use the 1000 sampled particles of `sod.txt`; repeated runs vary by about
7 %. The Sedov blast aborts with the Monaghan operator also with a binary built before
this operator existed; its cold background (\(u=10^{-20}\)) is driven negative by the
neighbor-pressure half of the Monaghan work. The variational 16³ traditional failure is
the existing fractional-energy timestep bound on viscously heated cold particles
(\(\delta t\) falls by \(4\times10^4\), beyond `kMaxRung`), not an energy error: the same
case with the global integrator recovers and conserves energy to \(6\times10^{-7}\).

**Restart, checkpoints and parity**: exact restarts of each operator from a
staggered-rung checkpoint reproduce the uninterrupted gas payload bit for bit, and
a restart with the other operator is rejected (`checkpoint_integrity`); the
checkpoint self-test checks both formulation codes. CPU/CUDA kernel parity
(`monaghan_sph_cuda`, `VariationalPressureCpuParity`) holds for densities,
corrections, rates, rungs and diagnostics with three viscosity models, including a
NaN correction on an excluded source; it ran on the GPU.

**Not validated**: production-scale or long cosmological runs; individual-rung energy
behavior beyond the restart, hydroWave, Sod and blast fixtures; multi-rank CUDA;
ill-conditioned supports (\(\Omega<0.1\)) in an application run, which none of the tests
produced; performance.
