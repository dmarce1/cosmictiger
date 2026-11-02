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

The numerical primitives are in `cosmictiger/hydro/hopkins.hpp`. HLLC uses
Roe wave-speed estimates and broader-wave retries. The final fallback is HLLE
for MFV and a bracketed ideal-gas contact solution for MFM, preserving zero MFM
mass flux. Ill-conditioned gradient matrices cause the actual neighbor search
to expand, up to the configured support bound, rather than introducing an SPH
pressure-force fallback. If a second-order update produces nonpositive mass or
thermal energy, the current elapsed interval is retried at first order, refreshing
all its pair fluxes. A still-invalid
update stops with a diagnostic requesting a lower `hydro.cfl`; there is no
nonconservative mass or energy floor applied to accept that update.
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

`src/hydro/hopkins_driver.cpp` is a host reference backend, including in a
CUDA-enabled build. It gathers gas state and tree-derived neighbor ranges from
all HPX localities onto the coordinating locality, runs CPU loops there, and
scatters updated states to the particle owners. MPI domain ownership and
migration are supported, but the meshless calculation is not distributed across
the locality CPUs: its global gas state and neighbor lists must fit in the
coordinator's memory. Communication and coordinator memory limit scalability.
`compute.cuda` still selects existing CUDA-capable components such as gravity
and particle drift; it does not move the meshless flux calculation to the GPU.

The tree search, selected kernel, and gas particle infrastructure are reused
from SPH. The meshless support-radius solve uses the same kernel-number-density
equation with its own ordered scalar reduction, so sorting and MPI ownership do
not change the accumulation order. The meshless path computes its own volume,
density, gradients, Riemann fluxes, and conservative update. It does not call
the SPH support, density/viscosity, pressure/thermal-force, or predictor/corrector
operators.

With `run.timeIntegrator=traditional`, gas particles select individual
power-of-two timestep rungs from their local
signal-speed, acceleration, and expansion limits. Active particles may coarsen
by one rung at an allowed clock boundary; a newly restrictive limit wakes an
inactive particle. Neighbor propagation prevents interacting timesteps from
differing by more than a factor of four. `run.minRung` remains a lower bound on
every rung and can force a uniformly fine reference calculation.

A pair's flux interval uses the shorter endpoint timestep. The solver stores
its beginning and midpoint fluxes and integrates the resulting linear function
over each actual elapsed global interval. Each increment is added to one
endpoint and subtracted from the other, including when one or both endpoints
are inactive. Future exchanges are not booked in advance. A pair is solved again
when an endpoint is active, its interval expires, or it enters the neighbor
graph; other pairs reuse the cached flux. Stable particle identifiers retain
pair ownership through sorting and MPI migration. Active particles recover
their primitive state from the conservative state and reconstruct gradients
using neighbors predicted to the same time. Inactive particles retain their
primitive and gradient anchors and predict forward from the inferred anchor
time, following Appendix G's active/inactive distinction. The coordinator still
gathers gas state and reconstructs geometry at each global event; individual
rungs save gradient reconstructions and pair Riemann solves, while retaining
that global preparation work.

The existing gas rung sets each cell's refresh interval. A 16-bit `lastKickTick`
stores the low bits of the global finest-clock tick at its last active
primitive/gradient reconstruction (32768 ticks per base interval). Anchor age
is computed lazily by modular subtraction from the global clock, bounded by
the gas-rung interval. The phase is written only when that cell is active; an
inactive cell has no age counter to increment. Conservative quantities advance
at every global fine step,
including for inactive cells, so they share the global physical time and do not
need a separate per-cell conservative clock.

With `run.timeIntegrator=global`, every gas particle is active at every force
event and has rung zero. The coordinator reduces the raw signal-speed,
acceleration, and expansion timestep limits over all gathered gas and applies
the driver's gravity and endpoint bounds. The resulting duration is not rounded
to a power of two. Each pair is reconstructed for that duration, evaluated at
its midpoint, and updated conservatively over the complete interval. There are
no inactive predictor ages or unfinished pair intervals to retain; the global
pair cache is empty at checkpoints. Source correction uses the previous raw
duration from the driver, independently of the newly chosen next duration.
The driver preserves that duration for an exact restart. Both integration modes
support a terminal source correction without opening another hydro interval.

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
use schema 4 and reject the historical schema 3 double-coordinate layout.
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

The force stage calls `hopkinsKick` to advance conservative hydrodynamics and
source terms over the elapsed interval and store a time-centered mesh velocity.
The ordinary CPU/CUDA drift stage only moves positions using that velocity;
it performs no Riemann solve or hydrodynamic state update. The physical fluid
velocity remains separate from the mesh velocity. The terminal diagnostic force
stage corrects pending source evolution and reports synchronized energies
without starting another flux interval or drifting particles.

Gravity and external forces use source half steps around the conservative
exchange. Between fresh force evaluations, source evolution uses the particle's
stored acceleration; its next active force evaluation corrects that prediction
using the new acceleration and the elapsed source interval. Checkpointed physical
and conservative states are at the global step end. The source anchor, pending
work history, primitive/gradient prediction anchors, stable identifiers, and
unexpired pair intervals are saved so that a restart can occur between
coarse-particle updates.

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
on each pair interval. For each endpoint, the advance accumulates
`I = integral q(t) dt` and `J = integral (t - sourceTime) q(t) dt` over only the
elapsed portion. It applies `I dot g_old` immediately, then adds
`J dot (g_new - g_old) / T` when a fresh force closes that endpoint's source
interval of duration `T`. Thus the combined work integrates the linear force
interpolation over the same elapsed mass transport, including unequal cached
subintervals. The factor `1/2` in `q` is spatial; the temporal moment supplies
the separate time centering. Sorting, migration, and checkpoints preserve `J`,
the source acceleration, and the source clock.

With no damping, the momentum source sequence on one interval is an opening
half kick using the old mass, a conservative exchange, and a closing half kick
using the updated mass. Both initially use `g_old`; the subsequent correction
adds `m_new T (g_new - g_old) / 2`. Their sum is precisely
`T (m_old g_old + m_new g_new) / 2`, the endpoint-mass trapezoid in equation 100.
The flux calculation and mass-flux work still require their own midpoint
centering; the kick ordering alone does not provide it.

This is the discretization implemented and tested here, rather than a claim of
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
  --compute.cuda=false --hpx:threads=4
path/to/build/cosmictiger --configuration=configurations/sod.ini --hydro.method=mfv \
  --gravity.method.far=fmm --gravity.expansionDegree=6 --run.timeIntegrator=traditional \
  --compute.cuda=false --hpx:threads=4
```

## Validation

The numerical results dated 2026-10-21 and 2026-10-22 below are historical
measurements copied with the saved implementation. They are not fresh validation
of this integration with current main. The historical tables, plots, audit
scripts, and logs under `docs/hopkins_sound_wave_results_20261022/` were not
restored here; their saved source is the third parent of stash
`cef72fbeea6e33168c17002172f5918cb42a7df6`.

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
campaign was not resumed. Hydro remains on the coordinating CPU and executes
sequentially with gravity; CPU/GPU overlap has not been implemented.

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
  --compute.cuda=false --gravity.enabled=false --chemistry.enabled=false \
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
