# Hopkins meshless hydrodynamics

Select `hydro.method=sph|mfm|mfv` in an INI file or on the command line.
`sph` remains the default and retains its existing numerical algorithms.
The MFM and MFV paths share the gas particle infrastructure and kernel options.
They are an optional build capability, disabled by default. Enable them with
`-DCOSMICTIGER_ENABLE_HOPKINS=ON`, `-DCOSMICTIGER_ENABLE_SPH=ON`, and
`-DCOSMICTIGER_POSITION_TYPE=fixed64`. CMake rejects Hopkins without SPH support
or Fixed64 positions. With Hopkins disabled, its driver and sound-wave scenario
are excluded from the build, and active MFM/MFV selections are rejected at startup.

The formulation follows Philip F. Hopkins, *A new class of accurate, mesh-free
hydrodynamic simulation methods*, MNRAS **450**, 53–110 (2015),
[doi:10.1093/mnras/stv195](https://doi.org/10.1093/mnras/stv195),
[arXiv:1409.7395](https://arxiv.org/html/1409.7395v2).
Equation numbers below refer to the arXiv version. This is CosmicTiger's
implementation of that formulation, not vendored GIZMO code or a complete
implementation of GIZMO's physics and numerical options.

## Discretization

For displacement `r_ij = x_j - x_i`, the local number density and volume are
`omega_i = sum_j W(r_ij, H_i)` and `V_i = 1 / omega_i`, including the self term.
The density is `rho_i = m_i / V_i`. Unlike SPH density summation, the volume
partition is independent of neighboring particle masses (equations 18–22).

The weighted moment matrix is
`E_i = sum_j (r_ij tensor r_ij) W(r_ij,H_i) / omega_i`.
Its inverse `B_i` gives the least-squares gradient
`grad f_i = B_i sum_j (f_j - f_i) r_ij W(r_ij,H_i) / omega_i`
(equations 24–26). With nonsingular geometry this reproduces linear fields.

The effective oriented face is
`A_ij = V_i B_i r_ij W(r_ij,H_i)/omega_i + V_j B_j r_ij W(r_ij,H_j)/omega_j`.
Pair fluxes update integrated mass, momentum, and total energy with opposite
signs, using `dQ_i/dt = -sum_j F_ij dot A_ij` (equations 30–35).

Slope-limited primitive states are reconstructed to the kernel-weighted
quadrature point and predicted by half a timestep before the Riemann solve.
The midpoint flux also predicts the effective face vector from the moving
volume/moment/support geometry and advances the face velocity with hydrodynamic
and external acceleration. If the predicted face is nonfinite or reverses
orientation, as can occur when a pair leaves the compact kernel support, that
pair uses a conservative first-order flux for the interval.
The MFV face follows the interpolated particle velocity; MFM additionally
moves the face at the Riemann contact speed, making its mass flux zero
(sections 2.4 and 2.6). The common Riemann flux supplies shock dissipation;
the SPH artificial-viscosity controls apply to the SPH method.

The slope-limiter parameters are fixed in the implementation, with no command-line
or INI controls: the global limiter uses beta bounds 1 and 2 and condition-number
scale 100, and the pair limiter uses `psi_1=0.5`, `psi_2=0.25` (Appendix B).
The current geometry acceptance bound is also 100, so accepted gradients use
the upper beta bound of 2.

The numerical primitives are in `cosmictiger/hydro/hopkins/hopkins.hpp`, with the shared
scalar/SIMD reconstruction and HLLC arithmetic in
`cosmictiger/hydro/hopkins/hopkins_simd.hpp`. HLLC uses the Davis extrema of the two
one-sided acoustic speeds. For face vector \(\boldsymbol A\) and input frame
velocity \(\boldsymbol w\), define \(A=|\boldsymbol A|\),
\(\boldsymbol n=\boldsymbol A/A\),
\(\boldsymbol v_K=\boldsymbol v_K^{\rm lab}-\boldsymbol w\), and
\(u_K=\boldsymbol v_K\cdot\boldsymbol n\), for \(K=L,R\).
A zero-area face returns zero flux. All following states and fluxes are in
this moving frame until the final boost:

\[
 E_K=\frac{P_K}{\gamma-1}+\frac12\rho_K|\boldsymbol v_K|^2,\qquad
 U_K=(\rho_K,\rho_K\boldsymbol v_K,E_K),
\]
\[
 F_K=(\rho_Ku_K,\rho_Ku_K\boldsymbol v_K+P_K\boldsymbol n,(E_K+P_K)u_K).
\]

The acoustic speeds and Davis wave bounds are

\[
 S_L=\min(u_L-c_L,u_R-c_R),\qquad
 S_R=\max(u_L+c_L,u_R+c_R),\qquad
 c_K=\sqrt{\gamma P_K/\rho_K}.
\]

These are equation (48) in [Toro's HLLC lecture](https://www.prague-sum.com/download/2012/Toro_2-HLLC-RiemannSolver.pdf).
They require only arithmetic, two square roots, and lane-wise
minimum/maximum operations. There is no Roe average or iterative pressure solve.
The contact speed and star states use these unmodified signal speeds. Define
\(D_K=\rho_K(S_K-u_K)\) and \(D=D_L-D_R\). Then

\[
 S_* = \frac{P_R-P_L+D_Lu_L-D_Ru_R}{D},\qquad
 P_K^*=P_K+D_K(S_*-u_K),\qquad
 P_* = \frac12(P_L^*+P_R^*).
\]

The two star pressures agree algebraically; their average treats both sides
symmetrically under floating-point rounding. For each side,

\[
 \rho_K^*=\frac{D_K}{S_K-S_*},\qquad
 \boldsymbol m_K^*=\rho_K^*
     [\boldsymbol v_K+(S_*-u_K)\boldsymbol n],
\]
\[
 E_K^*=\frac{(S_K-u_K)E_K-P_Ku_K+P_*S_*}{S_K-S_*},\qquad
 U_K^*=(\rho_K^*,\boldsymbol m_K^*,E_K^*).
\]

For MFV, the flux uses the clipped *correction* speeds:

\[
 F_{\mathrm{HLLC}}=
 \begin{cases}
 F_L+\min(0,S_L)(U_L^*-U_L),&S_*\geq0,\\
 F_R+\max(0,S_R)(U_R^*-U_R),&S_*<0.
 \end{cases}
\]

Clipping the corrections combines the usual four wave regions into two contact
sides while retaining the physical upwind flux in a supersonic flow. The packed
kernel first checks whether all active lanes choose the same contact side; a
uniform pack takes that side directly, and a mixed pack selects its state with
a lane mask. Both then evaluate one physical flux and one correction. MFM
instead follows the contact-moving face and uses
\(F_{\rm MFM}=(0,P_*\boldsymbol n,P_*S_*)\), so its mass flux is exactly zero.
For either method, writing the selected frame flux as
\((f_\rho,\boldsymbol f_m,f_E)\), the returned area-integrated laboratory flux is

\[
 \mathcal F_\rho=A f_\rho,\qquad
 \boldsymbol{\mathcal F}_m=A(\boldsymbol f_m+\boldsymbol w f_\rho),\qquad
 \mathcal F_E=A\left(f_E+\boldsymbol w\cdot\boldsymbol f_m
                     +\tfrac12|\boldsymbol w|^2 f_\rho\right).
\]

Thus MFM returns momentum flux \(AP_*\boldsymbol n\) and energy flux
\(AP_*(S_*+\boldsymbol w\cdot\boldsymbol n)\).
The reported signal speed is \(\max(|S_L|,|S_R|)\); the contact speed is \(S_*\).

Inputs must be finite with \(\rho_K,P_K>0\) and \(\gamma>1\). The solver requires
finite ordered wave bounds, \(D<0\), \(S_L<S_*<S_R\), and \(P_*>0\), together
with finite star states, \(\rho_K^*>0\), and
\(E_K^*-|\boldsymbol m_K^*|^2/(2\rho_K^*)>0\) on both sides. Nonfinite area
norms or output fluxes also fail. Any failure terminates the solve with an
exception; SIMD diagnostics identify the failing lane. There are no wave-speed
retries, HLLE fallback, or exact-contact fallback.
Ill-conditioned gradient matrices cause the actual neighbor search
to expand, up to the configured support bound, rather than introducing an SPH
pressure-force fallback. An invalid face-area predictor uses a conservative
first-order face flux.
A nonpositive mass or thermal energy at an own-step closure stops with a diagnostic
requesting a lower `hydro.cfl`. Already published transfers cannot be retried on
only one endpoint: there is no one-sided first-order retry or nonconservative
mass/energy floor.
Thermal energy is recovered by subtracting kinetic energy from the conserved
total energy. Hopkins's Appendix D dual-energy/entropy switch is not implemented,
so extremely high-Mach, cold flows can lose thermal accuracy through cancellation
or fail the positive-energy check. This backend does not reproduce the full
high-Mach robustness machinery of GIZMO.

## Implementation scope

The restored implementation uses double-precision hydrodynamic state and
arithmetic, and requires a build with `COSMICTIGER_POSITION_TYPE=fixed64`.
This requirement applies to MFM/MFV; the existing SPH and gravity-only methods
retain their supported position choices. The CPU meshless backend supports `run.timeIntegrator=traditional` and `global`. Meshless hydro
does not support `hierarchical` integration. The gravity solver is an independent
choice: either `gravity.method.far=fmm` or `pmmm` can accompany traditional or
global meshless hydro. PMMM also supports hierarchical integration for gravity
without hydrodynamics. A uniform global timestep is optional; choosing PMMM does
not select it. Both chemistry and star formation must be disabled for MFM/MFV;
incompatible combinations are rejected during option validation.

`src/hydro/hopkins.cpp` is a CPU SIMD backend, including in a
CUDA-enabled build. It gathers gas state and tree-derived neighbor ranges from
all HPX localities onto the coordinating locality, runs CPU SIMD loops there, and
scatters updated states to the particle owners. MPI domain ownership and
migration are supported, but the meshless calculation is not distributed across
the locality CPUs: its global gas state and neighbor lists must fit in the
coordinator's memory. Communication and coordinator memory limit scalability.
`gravity.cuda` and `drift.cuda` independently select their CUDA-capable
components. Set `hydro.cuda=false` for this CPU meshless backend; selecting CUDA
for gravity or drift does not move the meshless flux calculation to the GPU.

The SIMD backend is `NativeSimdVector<double>` from
`cosmictiger/math/simd_backend.hpp`, exposed to Hopkins as `hopkins::SimdReal`.
It selects the compiler target's native double-precision width through standard
`<simd>` when available, or `<experimental/simd>` otherwise. The repository also
has `SimdDouble` (a fixed two-lane pack used by the tree) and `SimdDouble8` (eight
lanes); Hopkins uses the native-width type instead of fixing its width to either.
Hydrodynamic storage and arithmetic remain double precision.

Neighbor distance/kernel evaluations, support-radius residuals and derivatives,
geometry moments and their time derivatives, gradient contributions and extrema,
and timestep signal estimates are computed in packs. Pair reconstruction and
Riemann fluxes are also evaluated in packs. Fixed64 position differences are
formed before conversion to double. Reductions retain their scalar particle-ID
order, and the pair scatter retains its conservative equal-and-opposite updates.
Final partial packs use safe inactive inputs and store only active lanes.

The tree search, selected kernel, and gas particle infrastructure are reused
from SPH. The meshless support-radius solve uses the same kernel-number-density
equation with its own ordered scalar reduction, so sorting and MPI ownership do
not change the accumulation order. The meshless path computes its own volume,
density, gradients, Riemann fluxes, and conservative update. It does not call
the SPH support, density/viscosity, pressure/thermal-force, or predictor/corrector
operators.

With `run.timeIntegrator=traditional`, gas particles select individual
power-of-two timestep rungs from their local signal-speed, acceleration, and
expansion limits. A particle commits its conserved state and reconstructs its
primitive state and gradients only at its own endpoint or an explicit wake-up.
The persisted state consists of the conserved anchor `U0`, pending integrated
transfers `deltaU`, the anchor volume, gradients, source and trajectory anchors,
and explicit physical and integer beginning/end clocks. Primitive anchors are
recovered from `U0` and its own volume, never from `U0 + deltaU` while inactive.

Each pair opens a MUSCL–Hancock interval ending at the earlier of its endpoints'
scheduled boundaries. The beginning and predicted midpoint fluxes are retained.
At the pair endpoint, one integrated transfer is deposited with opposite signs
into both particles' pending ledgers, including an inactive coarse endpoint.
Coarse–coarse records remain untouched at intermediate fine events. They are
not discarded merely because the pair leaves the current neighbor graph.
Stable particle identifiers retain pair ownership through sorting and migration.

Inactive primitives, face velocities, geometry rates, and timestep signals use
one prediction from their retained own-step anchor. Positions drift to each
global event using the separately anchored mesh velocity and acceleration; a
fine drift neither commits `U0 + deltaU` nor rebuilds the coarse trajectory.
The coordinator still gathers gas and reconstructs support/geometry at every
global event. Individual rungs save primitive reconstructions, pair Riemann
solves, and conservative/source commits, not that global preparation work.

A newly restrictive limit or a nonphysical primitive prediction wakes an inactive
particle. Neighbor propagation bounds interacting timestep ratios by four.
Before a wake commits, each incident live pair is integrated exactly once over
its elapsed portion, using the retained linear flux representation; both signed
transfers are deposited and the unused future tail is retired. New intervals
open from the wake boundary. All gas accelerations are refreshed by the gravity
pass so newly woken gas has a fresh source endpoint; source commits still occur
only at each gas particle's own event. This correctness choice costs more gravity
force evaluations than selecting only the current active gas sinks.

With `run.timeIntegrator=global`, all gas shares the raw global duration selected
from its signal-speed, acceleration, expansion, gravity, and endpoint bounds.
It is not rounded to a power of two. The same pair-open, pair-settle, and own-step
closure sequence uses physical times; open pair intervals persist across a
checkpoint until the next event. Both modes support a terminal closure that
settles every elapsed pair, commits all gas, and opens no new interval.

The validation results recorded later in this document predate this own-step
scheduler unless explicitly identified otherwise; they do not validate the new
wake, source, or overlap ordering.

Conserved accounting includes `U0 + deltaU`. Between coarse endpoints this is an
asynchronous ledger, not a claim that every cell's physical source terms are
synchronized to the fine clock. Before any gravity source construction, MFV mass
is projected to that force time from the anchor, completed pending transfers,
and the elapsed prefixes of all live pair records. This read-only projection
updates only the public mass mirror and never commits a hydro anchor. Terminal
diagnostics first close every remaining interval.

Mass, momentum, total energy, source histories, primitive/gradient anchors,
pair fluxes, and support radii persist in double precision. Coordinates reside
only in the shared Fixed64 particle array. Pair separations subtract the
periodic fixed-point integers before converting the local displacement to
double, preserving low coordinate bits that would be lost by first converting
both absolute positions. Drift adds a Fixed64-encoded double displacement to
the existing Fixed64 position; there is no double absolute-coordinate sidecar.
Ordinary gas fields and the tree's support-radius mirror remain float
projections. The support solve retains its double radius through neighbor-search
refreshes and targets a relative kernel-count residual of `1e-13`. These changes
do not alter SPH's storage or numerical algorithms. Meshless checkpoint sidecars
use particle schema 5 and pair schema 2. Earlier asynchronous state layouts are
rejected rather than reinterpreted under the new event semantics.
The current CPU and CUDA drift interfaces retain double timestep and scale
arguments. CPU and CUDA coupled regression checks are described below. Ordinary particle output
fields remain limited by their float storage; conserved diagnostic reductions
and the sound-wave CSV diagnostic read the double meshless state.

The implemented CFL condition is
`Delta tau <= hydro.cfl * a * R_i / signal_i`, with
`R_i = (3 V_i / (4 pi))^(1/3)`, stored sound speed
`sqrt(gamma (gamma-1) a^2 u)`, and the maximum pair signal speed of equation 37.
This uses the volume-equivalent cell radius, not Hopkins's kernel-length
normalization in equation 36; the same numerical `hydro.cfl` is more conservative
here. The existing SPH CFL convention remains unchanged.
The dedicated one-dimensional `soundWave` scenario instead uses the paper's
`2 * supportRadius` Courant length with `hydro.cfl=0.2`.

The force stage calls `hopkinsKick` after fresh gravity is available. It settles
completed or interrupted pair intervals, closes each due particle's source and
conservative interval, and opens new intervals after active reconstruction.
Source-independent support/geometry preparation has a separate entry point so
it can overlap gravity evaluation after both tree-sorting phases have completed.
FMM gravity runs in an HPX future while the coordinator prepares hydro geometry;
the future is joined before importing fresh gas accelerations, closing source
terms, choosing timesteps, or reconstructing the next interval's fluxes. The
particle's stored `sourceAcceleration` remains its old force sample while the
gravity solver writes the new sample to the gas acceleration arrays. Neither
sample depends on enabling the optional analysis acceleration output, and no
stored potential is needed by this source integration.

For MFV, the gravity source mass is projected from the conserved anchor, pending
mass, and elapsed portions of live pair intervals before gravity builds its tree.
This projection does not commit a particle or alter its reconstruction anchor.
PMMM remains sequential because its particle ordering differs from hydro's tree.
Distributed CUDA FMM also remains sequential: remote GPU source staging currently
resizes shared particle storage. Single-locality CUDA FMM and CPU FMM can overlap.

Gravity and external forces use an opening source half-step with the old mass
and old acceleration, the accumulated conservative hydro exchange, and a closing
half-step with the new mass and fresh acceleration. These operations execute
once at the particle's own closure. Damping uses its analytic exponential on
each half-step; expansion evolves thermal energy with the corresponding endpoint
coefficient. No later fine-event source correction modifies an inactive anchor.

The CPU/CUDA drift stage only moves positions along the anchored trajectory; it
performs no Riemann solve or conservative update. The predictor, pending ledger,
source moments, stable identifiers, and unexpired pair intervals persist so a
restart may occur between coarse-particle commits.

The existing tree, gas storage, domain distribution, gravity, cosmological
units, scenario initialization, and output infrastructure are reused. MFV mass
is carried with each gas particle through redistribution and checkpointing;
restarts require the same hydrodynamic method. Cosmological conversions use
the existing stored `a v` velocity and `a^2 u` internal energy, with explicit
expansion source evolution. This implementation has not been validated for
large cosmological production simulations.

MFV also includes the gravitational work associated with mass moving between
cells, the effect discussed in Appendix H1. The implemented normalization is
defined explicitly here: for mass rate `F > 0` from cell `i` to cell `j` and
`r = x_j - x_i`, set `q = F r / 2`. The energy source rates at the endpoints
are `q dot g_i` and `q dot g_j`, with `g = -grad Phi`. Their sum is
`F r dot (g_i + g_j) / 2`, exactly `-F (Phi_j - Phi_i)` for a spatially linear
acceleration field. This cancels the potential-energy change of the transported
mass. These two work terms have the same orientation; they are not an
antisymmetric hydrodynamic energy exchange. MFM has zero mass flux and zero
such work.

The beginning and midpoint flux/separation predictions define a linear `q(t)`
on each pair interval. Settlement accumulates `I = integral q(t) dt` and
`J = integral (t - sourceTime) q(t) dt` in each endpoint's own pending source
history. At that endpoint's closure of duration `T`, the energy source includes
`I dot g_old + J dot (g_new - g_old) / T`. This integrates the endpoint-linear
force interpolation over the same completed mass transport, including intervals
shortened by a wake. Sorting, migration, and checkpoints preserve both moments.

With no damping, the own-interval momentum source is exactly
`T (m_old g_old + m_new g_new) / 2`, the endpoint-mass trapezoid in equation 100.
The source half-steps update conserved momentum and energy directly; without
sources, closure reduces to the single conserved addition `U0 + deltaU`.
Flux and mass-work predictions retain their own midpoint time centering.

This is the chosen work normalization, rather than a claim of
literal coefficient agreement with printed equation 101. That equation prints
`T/2` for each temporal mass-work endpoint term, while the spatially symmetric
work defined above uses `q = F r / 2`. The same printed factors occur in the
[published PDF, Appendix H1](https://academic.oup.com/mnras/article-pdf/450/1/53/18507840/stv195.pdf).
The implementation's normalization is checked against the independent potential
change, and its production integration helper is checked against analytic
time-dependent potential transport. SPH retains its original particle-state
interpretation and update.
The Appendix H1 mass-flux gravitational work is distinct from Appendix H2's
adaptive gravitational-softening correction. This integration retains the
existing fixed gravitational softening and does not implement the Appendix H2
adaptive-softening terms or claim full GIZMO gravity equivalence.

The ideal-gas equation of state uses `hydro.gamma`. Coupling chemistry and
star formation to the conserved meshless state requires additional
implementation. The cosmology INI enables chemistry, so explicitly use
`--chemistry.enabled=false --starFormation.enabled=false` when selecting a
meshless method with that configuration. These solvers do not add GIZMO's
MHD, feedback, particle splitting, or other optional physics.

For example, from the repository root run the existing Sod setup with either
meshless solver, substituting your build directory. Replace `traditional` with
`global` to select a raw global timestep:

```sh
path/to/build/cosmictiger --configuration=configurations/sod.ini --hydro.method=mfm \
  --gravity.method.far=fmm --gravity.expansionDegree=6 --run.timeIntegrator=traditional \
  --gravity.cuda=false --drift.cuda=false --hydro.cuda=false --hpx:threads=4
path/to/build/cosmictiger --configuration=configurations/sod.ini --hydro.method=mfv \
  --gravity.method.far=fmm --gravity.expansionDegree=6 --run.timeIntegrator=traditional \
  --gravity.cuda=false --drift.cuda=false --hydro.cuda=false --hpx:threads=4
```

## Validation

The numerical results dated 2026-10-21 and 2026-10-22 below are historical
measurements copied with the saved implementation. They are not fresh validation
of this integration with current main. The historical tables, plots, audit
scripts, and logs under `docs/hopkins_sound_wave_results_20261022/` were not
restored here; their saved source is the third parent of stash
`cef72fbeea6e33168c17002172f5918cb42a7df6`.

### SIMD unit validation on 2026-11-02

The 49 tests in `tests/hopkins.cpp` pass in standalone GCC 15 C++20 builds at
`-O2`, with fast math and contraction disabled, on both the native four-double
target and the baseline SSE2 two-double target. A C++26 native build also passes;
the installed library still selects `<experimental/simd>` in that language
mode, so the standard `<simd>` branch is not validated here. The tests cover an independent
four-region HLLC reference, uniform and mixed contact masks, scalar/packed
agreement, moving-frame covariance, pair reversal, exact zero MFM mass flux,
every partial-pack length, zero-area faces, poisoned inactive inputs without
invalid/divide-by-zero exceptions, active-lane fatal diagnostics, packed
reconstruction and physical recovery, and FLOP accounting. The existing
independent Sod evolution remains positive, conservative, and convergent.
These unit checks do not establish application, MPI, GPU, or performance results
for the SIMD revision.

### SIMD application validation on 2026-11-02

A fresh `build/hopkins_simd` CPU build uses GCC 15, C++20, `-O2 -march=native`,
Fixed64 positions, FMM expansion degree 6, and traditional/global integrators.
Hopkins retains `-fno-fast-math -ffp-contract=off`. The 49 unit CTests and
`hopkins_application`, `hopkins_global_sources`,
`hopkins_fmm_traditional_cpu`, `hopkins_fmm_global_cpu`, and
`hopkins_flops_application` pass. These cover MFM/MFV evolution, conservation,
individual-rung cache/wake-up behavior, exact physical restarts, source laws,
SPH default dispatch, and live FMM coupling. The repeated FLOP fixture initially
refused to overwrite its existing checkpoints; it passed after moving that
prior output aside and rerunning with a fresh output directory.

Both methods also completed the sound-wave campaign at `N=16,32,34,64,128`.
The non-multiple-of-four `N=34` supplements partial-pack unit coverage.
Density L1 errors decrease from `6.77544e-8` to `1.64421e-9` for MFM and
from `9.33296e-8` to `2.37154e-9` for MFV between `N=16` and `128`.
Artifacts are in `build/hopkins_simd/sound_wave_final/summary.json` and
`build/hopkins_simd/validation/` (logs, build/source hashes, and benchmark source).
No MPI, CUDA, PMMM, or long production run was performed for this revision.

A separate one-core GCC 15 `-O3 -march=native` kernel microbenchmark uses
16,384 varied physical faces, repeated 96 times, with uniform and mixed contact
masks for MFM/MFV. Five-run median paired speedups over the shared scalar solver
are 2.11–2.51 times including input gathering, and 2.31–2.84 times for prepacked
inputs. Checksums agree to the 12 printed significant digits. This measures
Riemann-kernel throughput, excluding production reconstruction, geometry,
scatter and HPX overhead; concurrent compilation and unpinned execution limit
timing precision. No end-to-end application speedup is established.

### Integration validation on 2026-11-01

Release builds with GCC 15, HPX, Fixed64 positions, potential expansion degree 6,
and all three gravity integrators compiled successfully with CUDA disabled and
with CUDA 13.1 targeting an RTX 3090. The following checks passed:

- Hopkins numerical primitives, Gresho initialization, and drift dispatch,
  including preservation of low Fixed64 bits and double drift intervals.
- Both MFM and MFV with FMM, PMMM-PP, and PMMM-FMM gravity under traditional and
  global stepping, on CPU and with CUDA gravity/drift. These short fixtures
  check live gravity, mass conservation, exact restarts, and clipped global
  endpoints for every combination.
- The serial wave, Sod, Gresho, and blast regression suite, including individual
  rungs, inactive predictor retention, pair-flux reuse, conservative updates,
  cross-method restart rejection, and default/explicit SPH parity.
- Two MPI localities for individual-rung MFM/MFV conservation and exact restart,
  and CPU gravity checks for imported MFV source masses and gas cache handling.
- Analytic uniform-force laws with unequal raw global timesteps; MFV restart
  with nonzero pending gravitational work; and independent analytic integration
  of cached mass-flux work in a time-dependent potential. The latter tests the
  production work-integration helper against potential-energy transport.
- CPU analysis pipelines with meshless mass and velocity overrides.

The run logs are under the untracked local directory
`build/hopkins_merge_20261101/`. These checks establish the tested integration
paths, not full-duration hydrodynamic or self-gravitating convergence, CUDA/MPI
scaling, or cosmological production accuracy. The stopped sound-wave amplitude
campaign was not resumed. Those historical runs executed coordinating-CPU hydro
sequentially with gravity; they did not exercise the newer preparation overlap.

For a Fixed64 build with `COSMICTIGER_ENABLE_HOPKINS=ON`, FMM and PMMM,
potential expansion degree 6, traditional
and global integration, SPH infrastructure, and testing enabled, build the application, `hopkins_test`,
and `gresho_test`, then run the primitive and end-to-end checks:

```sh
ctest --test-dir path/to/build --output-on-failure -L '^hopkins$' --parallel 2
```

The end-to-end script can also be run directly:

```sh
python3 tests/run_hopkins.py path/to/build/cosmictiger /tmp/hopkins_integration
```

For the individual-rung fixtures on two MPI localities, use an MPI-enabled build:

```sh
python3 tests/run_hopkins.py path/to/build/cosmictiger /tmp/hopkins_mpi \
  --mpi-exec=/path/to/mpiexec --localities=2 --async-only
```

`tests/hopkins.cpp` exercises geometric consistency, reconstruction, moving-face
fluxes, conservation, and shock/contact behavior. `tests/run_hopkins.py` runs
small three-dimensional wave, Sod, Gresho, and blast fixtures for both methods;
checks finite positive states and mass/momentum/energy conservation; verifies
MFV mass exchange and exact restart state; rejects cross-method restarts; and
compares explicit `sph` selection with the default. Uniform-force fixtures
check synchronized momentum and energy against analytic solutions; forced
restarts include nonzero pending MFV gravitational work. An additional blast
fixture requires distinct gas rungs and recorded reuse of pair fluxes while gas
is inactive, checks conservation at every fine-step checkpoint, and restarts
inside an unexpired coarse-pair interval. Restart comparison includes the
conservative/source state, inactive prediction anchors, and cached pair fluxes.
It checks that anchor ages stay within the gas-rung interval and that inactive
primitive/gradient anchors remain unchanged across intervening fine steps.
The optional MPI run reduces
all rank payloads and requires cached pairs crossing rank ownership. These short regression
fixtures do not establish convergence, full-duration vortex fidelity, or
production-scale MPI performance.

### Historical published sound-wave test on 2026-10-22

`soundWave` is a dedicated one-dimensional scenario for
[Hopkins (2015)](https://arxiv.org/html/1409.7395v2), section 4.2.1 and Figure 2.
It uses a periodic unit interval, mean
density and sound speed one, `gamma=5/3`, background pressure `3/5`, and
amplitude `A=1e-6`. The right-going linear reference is

```text
delta(x,t) = A sin(2 pi (x-t))
rho(x,t)   = 1 + delta(x,t)
vx(x,t)    = delta(x,t)
P(x,t)     = 3/5 + delta(x,t).
```

Each run ends after one crossing, at `t=1`. The sweep uses
`N=16,32,64,128,256,512,1024,2048`, CFL 0.2, cubic splines, HLLC and the ordinary
limiters. Every particle, including extrema, enters the unweighted density
`L1 = sum(abs(rho_i-rho_exact(x_i,t)))/N`. The Python runner recomputes the
analytic reference independently from the reported double coordinates and
checks the C++ reference columns. This is combined spatial/CFL refinement;
it does not isolate temporal order at a fixed particle count.

We initialize equal masses `m=1/N` and invert
`q=x+A*(1-cos(2*pi*x))/(2*pi)` at `q=(i+1/2)/N`. The historical mass/position
recipe is not specified in the paper, so this is a reconstruction of its
continuum initial condition, not a claim to have recovered the original IC
file. The later downloadable author IC has a different sound speed and crossing
time and was not substituted for the published setup.

The main configuration uses effective neighbor number four with the inspected
[author code's one-dimensional normalization](https://github.com/pfhopkins/gizmo-public/blob/78b66ffcfad6362f8aa0798f8a23a6c9ba5dd2ca/declarations/constants.h#L255-L257),
`N_ngb=2*h*sum(W)`, giving `h` approximately `2/N`. Equation 38 of the paper
prints the coefficient as one instead. A separate sensitivity configuration
uses `hydro.neighborNumber=8` in our convention, implementing the printed
equation with paper neighbor number four and `h` approximately `4/N`. The
historical discrepancy remains unresolved.

To run a new full main sequence with the restored implementation:

```sh
python3 tests/run_hopkins_sound_wave.py path/to/cosmictiger /tmp/sound_wave
```

The runner saves commands, resolved options, binary/configuration/runner hashes,
initial and final norms, adjacent-resolution orders, and conservation changes.
It refuses to mix changed executables or configurations within a campaign.
`--reprocess /tmp/sound_wave` recalculates diagnostics without running a solver.

The historical archive includes the failed original-storage campaigns. An
independent initial-only audit reproduced their density floor using Fixed32
coordinates and float separations. With ideal coordinates and double separations,
initial density error instead falls from `1.6342e-8` at N16 to `9.9887e-13` at N2048, following
`N^-2`. This establishes an initialization/representation error; it does not
by itself explain the later evolved error or validate the integrator. That
audit launched no simulation; its script is part of the historical archive
identified above, not this restored source tree.

The printed-equation run also exposed support iterations alternating between
adjacent float radii. A storage-bracket termination fix allowed that old-storage
sequence to complete; it still failed convergence. The subsequent precision
revision retains coordinates, drift increments, separations and support radii
in double, without changing the reconstruction, limiter or flux integrator.
For the current-convention N2048 run, `dt=1/8192`; the initial velocity amplitude
`1e-6` gives a displacement of at most `1.22e-10` per step. The Fixed32 spacing
is `2^-32=2.33e-10`. Computing a drift in double and then discarding its
sub-Fixed32 part at every step therefore loses relevant motion. The historical
persistent double coordinate preserved those increments. The restored backend
now uses authoritative Fixed64 coordinates, whose spacing is `2^-64`, instead;
the historical double-coordinate convergence results below do not validate
that changed representation.

Both historical double-precision sequences recovered approximately second-order
density L1 convergence. For the main normalization, the final errors are:

| N | MFM density L1 | MFV density L1 |
| ---: | ---: | ---: |
| 256 | 4.19896e-10 | 6.10911e-10 |
| 512 | 1.05712e-10 | 1.54517e-10 |
| 1024 | 2.61380e-11 | 3.82724e-11 |
| 2048 | 6.63130e-12 | 9.47034e-12 |

An ordinary least-squares fit of `log(L1)` against `log(N)` over these four
resolutions gives orders **1.99697 for MFM** and **2.00476 for MFV**. The
printed-equation normalization gives **1.97869** and **1.97608**, respectively.
The main-normalization density L2 orders are lower, 1.86119 and 1.87228;
the printed-equation sequence gives 1.98431 and 1.98089. The second-order claim
here refers specifically to the paper's density L1 resolution diagnostic,
with every particle included. No extrema were removed and no limiter or time
integration formula was changed to obtain these results.
An independent read of all 64 archived initial/final particle tables reproduced
every reported norm. Across the 32 runs, maximum absolute relative changes in
total mass and total energy were `2.22e-16` and `1.23e-16`, respectively.

On the final 2026-10-22 executable, all 48 focused tests passed (34 Hopkins,
five CPU drift and nine drift-dispatch tests). The complete CPU integration
suite and the individual-rung conservation/restart suite with two MPI localities
and one worker per locality also passed. Those historical checks included
default-versus-explicit-SPH equivalence and preservation of the SPH drift
rounding then in use. CUDA drift compiled successfully; GPU runtime accuracy
was not tested. The historical archive contains the commands, hashes, logs,
fit values, and initial/final particle tables. These results do not establish
that the restored implementation passes the current build or test suite.

Relevant papers citing Hopkins include:

- [Hopkins & Raives (2016)](https://academic.oup.com/mnras/article/455/1/51/983632),
  section 3.1/Figure 1, and
  [Hubber, Rosotti & Booth (2018), GANDALF](https://arxiv.org/html/1709.04488),
  section 6.1/Figure 4, report second-order wave convergence. These are
  resolution sequences, not isolated fixed-resolution time-refinement tests.
- [Alonso Asensio et al. (2023), PKDGRAV3](https://academic.oup.com/mnras/article/519/1/300/6847218),
  section 3.1.1/Figure 2, explicitly identify single-precision particle masses
  as preventing a `1e-6`-amplitude convergence test in their implementation.
  Their plot instead uses amplitude `1e-3`, gamma 1.4, 32 neighbors and disabled
  gradient limiting. Its fitted MFM/MFV orders are 1.84/1.89. Their mass-storage
  limitation differs from the coordinate/separation error demonstrated here.
- [Groth et al. (2023), OpenGadget3](https://arxiv.org/html/2301.03612v2),
  section 4.2.1/Figure 3, find between first- and second-order convergence for
  their different three-dimensional wave setup and attribute the reduction
  to limiting; they also discuss support-iteration-related scatter.
- [Pakmor et al. (2016), AREPO](https://arxiv.org/html/1503.00562v2), Figure 4,
  require both improved gradients and time integration to recover second-order
  vortex convergence. Their implemented time update combines primitive
  prediction and endpoint flux averaging. This is not evidence that GIZMO's
  published sound-wave test is first order.
- [Zier & Springel (2022)](https://arxiv.org/html/2205.07916v2), section 5.1/Figure 8,
  provide a fixed-resolution temporal-refinement test of their improved
  moving-mesh scheme, with approximately second-order errors.

### Historical individual-rung checks before the precision revision

These measurements were made on 2026-10-22 before the double-coordinate
revision. The fixed-resolution temporal probe below has not been repeated
with the final precision changes; the published sound-wave sequence above
does not independently establish its temporal convergence order.

The complete CPU integration suite passed with two worker threads, including
wave, Sod, Gresho, blast, forced evolution, method-mismatch rejection, and
explicit/default-SPH equivalence. The individual-rung suite also passed with
two MPI localities and two worker threads per locality. Restarts inside a
coarse-pair interval reproduced every physical particle field and persisted
solver field exactly; MPI comparison uses stable particle identifiers to allow
storage reordering and excludes native padding.

The 432-particle blast fixtures took eight fine steps to `t=0.0005`:

| Method | Minimum active gas | Maximum reused pair fluxes | Retained prediction anchors across steps |
| --- | ---: | ---: | ---: |
| MFM | 1 / 432 | 10750 / 10800 | 2867 |
| MFV | 15 / 432 | 10148 / 10800 | 1668 |

The tests verified that unchanged anchor clocks retain exactly unchanged
primitive/gradient values, and that inferred ages remain within the owning
gas-rung interval. Across the unforced CPU fixtures and MPI blast fixtures,
maximum relative conserved-energy drift was `3.27e-16`, momentum drift divided
by `sqrt(initial mass * initial energy)` was `2.01e-18`, and total mass was
unchanged. Public float fields passed an independent storage-precision check.
These conservation measurements compare each fine-step checkpoint with the
first checkpoint.

An independent smooth-wave probe used a regular periodic `N^3` lattice with
`m=1/N^3`, zero velocity, and
`u=(1+0.001*sin(2*pi*x))/(gamma-1)`, with `gamma=5/3`. These were separately
seeded checkpoints, rather than the built-in displaced `hydroWave` initial
condition. At `t=0.04`, uniform timesteps `0.005`, `0.0025`, and `0.00125`
for `N=8,16,32` gave the following velocity L2 errors relative to the analytic
linearized acoustic-plus-entropy solution, normalized by its velocity-mode RMS:

| Method | N8 | N16 | N32 |
| --- | ---: | ---: | ---: |
| MFM | 0.1806 | 0.04710 | 0.005752 |
| MFV | 0.1888 | 0.05172 | 0.008563 |

The analytic reference neglects terms quadratic in the perturbation amplitude.
These measurements show combined spatial and temporal refinement. A separate
fixed-N8 test with a 1% perturbation and successively halved timesteps showed
approximately first-order temporal self-convergence against the finest run.
Independent second-order time convergence is therefore not established by
these tests.

For the 432-cell blast at `t=0.0005`, halving the default CFL twice reduced
velocity RMS differences from a uniform-rung-7 reference from
`0.00823` to `0.00383` to `0.00169` for MFM and from
`0.01333` to `0.00624` to `0.00280` for MFV. All states remained positive and
MFM particle masses stayed fixed. The final robustness build repeated the
N8/N16 wave cases and the default-CFL/reference blast cases with exactly
unchanged decoded positions, masses, velocities, and internal energies. N32
and the two intermediate CFL runs were measured on the preceding build before
its support-edge safeguards; they were not repeated on the final build.

### Historical shared-rung checks on 2026-10-21

The measurements below used the earlier shared-rung integrator. They record
its validation history and do not establish accuracy of the individual-rung
implementation described above.

The CPU executable completed eight additional runs using two HPX threads:
`hydroWave` to `t=0.1` and `sod` to `t=0.03`, each with MFM/MFV and
`initial.particlesPerDimension=8,16`. All used 32 coarse intervals, the default
`gamma=5/3`, 42 neighbors, and `hydro.cfl=0.22821773229381923`, with gravity,
chemistry, and Silo disabled. All final masses, densities, and internal energies
were positive and finite. Across these cases, maximum relative mass and total
energy drift were `6.19e-8` and `6.91e-8`; maximum momentum drift divided by
`sqrt(initial mass * initial energy)` was `5.04e-9`. Initial energies were
computed from the scenario's stored float values, avoiding text-output rounding.

The existing Sod diagnostic compares 1000 sampled particles with its exact
solution. These are sampled relative L1 errors, not volume-integrated norms:

| Method | Density, N8 | Density, N16 | Pressure, N8 | Pressure, N16 |
| --- | ---: | ---: | ---: | ---: |
| MFM | 0.09225 | 0.03405 | 0.14028 | 0.05661 |
| MFV | 0.10682 | 0.04096 | 0.15392 | 0.05922 |

N8 and N16 initialize 576 and 4608 Sod gas particles, respectively. The errors
decrease with refinement, but these two sampled measurements do not establish
a convergence order. `hydroWave` has no exact-solution diagnostic and its
displacement scales with particle spacing; its runs establish stability and
conservation over this interval only.

To run the same Sod scenario with the current individual-rung implementation,
use the following and repeat with `mfv` and/or `particlesPerDimension=16`, using
a separate output directory for each case. This does not reproduce the
historical shared-rung integrator exactly:

```sh
path/to/build/cosmictiger --run.scenario=sod --hydro.method=mfm \
  --gravity.method.far=fmm --gravity.expansionDegree=6 --run.timeIntegrator=traditional \
  --initial.particlesPerDimension=8 --run.endTime=0.03 --run.numberOfSteps=32 \
  --gravity.cuda=false --drift.cuda=false --hydro.cuda=false --gravity.enabled=false --chemistry.enabled=false \
  --output.silo=false --output.directory=output/sod_mfm_n8 --hpx:threads=2
```

The printed `L1` is the sampled density error. `sod.txt` contains `x`, density,
exact density, pressure, and exact pressure; the pressure metric above is
`sum(abs(P - P_exact)) / sum(P_exact)`. Change `run.scenario` to `hydroWave`
and `run.endTime` to `0.1` for the wave checks.

Two-locality CPU checks also completed four wave steps with 512 gas particles
for each meshless method. Global checkpoint reductions conserved mass,
momentum, and energy to storage roundoff, with maximum relative energy drift
`1.06e-8`; MFM masses stayed fixed and MFV masses changed. The SPH and MFV
gravity/migration fixtures and a default-SPH distributed wave run passed.

Small cosmology runs with 512 gas and 512 dark-matter particles exercised both
methods with gravity and expansion enabled and chemistry disabled. With
`run.maxIterations=2`, the existing loop logged three steps and reached
`a` approximately `0.02032`, saving finite velocities and positive finite gas
mass, density, thermal energy, and support radii. These are execution checks,
not cosmological convergence or energy-conservation measurements. Affected
CUDA translation units were compile-checked; GPU execution was not tested.

### Own-step integration and independent backends: November 2, 2026

The current revision uses conserved anchors plus pending conservative increments,
retained gradients, anchored drift trajectories, and elapsed pair-flux intervals.
Particle-state checkpoints use version 5; live pair records use version 2. Old
particle-state layouts are rejected. Fresh force samples and the own-step source
anchor remain separate, including when a particle wakes before its scheduled end.

Fresh validation used Fixed64 positions and FMM potential degree 6 (order 7):

- CPU-only and CUDA 13.1 application builds succeeded.
- The 101-case CPU selection covering Hopkins, SPH kernels, drift dispatch, and
  PMMM-PP/PMMM-FMM traditional/global coupling passed; all 19 CUDA SPH kernel
  tests passed, including deferred versus unsplit acceleration-source assembly.
- The full Hopkins application and global-source regressions passed, including
  conservative mass/momentum/energy accounting and exact restarts.
- Natural MFM/MFV blast fixtures completed two coarse steps with actual early
  wakes. MFM closed 381 particles early and truncated 10,222 pair intervals;
  MFV closed 8 early and truncated 248 pairs. The tests independently integrate
  every retired pair prefix and compare both endpoints' deposits. Three occupied
  bins span rungs 0 through 3; four simultaneously populated bins were not tested.
- Two-locality CPU individual-step/wake runs and gas migration passed, including
  exact restart immediately before wake-up.
- CPU Hopkins with CUDA FMM gravity and drift passed traditional/global coupling
  and exact-restart checks. CPU SPH with CUDA FMM gravity passed traditional and
  global smoke/restart checks. These include evidence of actual CUDA force launches.
- One-period MFM/MFV sound waves at 16, 32, and 64 particles completed. Density
  L1 error decreased from 6.67e-8 to 6.51e-9 for MFM and from 9.27e-8 to 9.05e-9
  for MFV. This is combined space/time refinement, not a temporal-order proof.

The module-specific CLI/INI migration passed 508 grouped-options cases and the
startup checks. CPU analysis in a CUDA-enabled executable also passed with a
runtime-call guard after fixing host allocation release before CUDA initialization.
Coordinator profiles and CUDA elapsed-work records do not establish a measured
overlap speedup; no such speedup is claimed.

Intermediate energy and output mass fields remain views of the asynchronous
conserved ledger; unfinished live-pair/source intervals are not reconstructed at
arbitrary observation times. In particular, an MFV lightcone mass is not an
independent reconstruction at its crossing time. Terminal synchronized diagnostics
are covered by the application tests. Large cosmological production evolution,
CUDA MPI overlap, and long-time gravitational-energy convergence remain unvalidated.
