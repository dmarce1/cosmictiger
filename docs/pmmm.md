# Periodic particle mesh multipoles

Select **PMMM-PP** or **PMMM-FMM** and the expansion order in `./build.sh`.
The default build is global FMM at order seven. To configure PMMM directly with
CMake, use:

```sh
cmake -S . -B build_pmmm -DCOSMICTIGER_GRAVITY_METHOD=pmmm \
  -DCOSMICTIGER_PMMM_NEAR_METHOD=pp -DCOSMICTIGER_WITH_ORDER=8
cmake --build build_pmmm --parallel 12
```

Set `COSMICTIGER_PMMM_NEAR_METHOD=fmm` for an adaptive FMM near field. Each build
uses one solver variant and one order from 3 through 9. PMMM uses a periodic
uniform mesh, spherical multipoles about cell centers and FFT convolution for
distant interactions. Neighboring cells use the compiled direct-pair or adaptive
FMM near field.

```sh
./build_pmmm/cosmictiger --gravity.meshSize=32 --gravity.pmmmNearWidth=2 \
  --initial.particlesPerDimension=32 --hydro.enabled=false \
  --compute.cuda=false --hpx:threads=12
```

The runtime settings can also be supplied in an INI file:

```ini
[gravity]
meshSize = 32
pmmmNearWidth = 2

[hydro]
enabled = false
```

The former `gravity.method`, `gravity.pmmmNearMethod`, and `gravity.fmmOrder`
inputs are rejected on both the CLI and in INI files. `--help` and the `build.*`
startup fields report the selected build configuration.

`gravity.meshSize=n` sets `n` cells on each axis of the normalized unit box,
independently of the initial-condition particle count. The default value zero
is unset and is rejected for PMMM. `gravity.pmmmNearWidth=w` sets the ghost
width and accepts one or two, defaulting to two. The near region contains
`(2*w+1)^3` cells (27 or 125), including the sink cell. The mesh must satisfy
`n >= 2*w+1`, so periodic near cells are distinct. Odd mesh sizes are supported.

`COSMICTIGER_PMMM_NEAR_METHOD=pp|fmm` selects how those near cells are evaluated
at build time. The default `pp` sums all primary particle pairs. The `fmm` choice
uses the existing Cartesian FMM operators and adaptive tree acceptance within
the same near-cell region. It first exchanges complete remote source-cell
particle records and builds local source trees as individual cells become ready.
Owned sink cells run once every near-source cell they need is ready, allowing
interior and ready boundary work to overlap outstanding cell fetches.
It adds no second periodic correction or distant-cell force:
the spherical FFT result supplies those contributions in either mode.

`COSMICTIGER_WITH_ORDER` sets the shared PMMM and near FMM expansion order.
The near FMM retains the current `gravity.openingAngle` and `compute.bucketSize`
controls. Tightening the opening angle makes the near result approach direct PP.
The cosmology driver can update the angle and bucket size during evolution, as
it does for ordinary FMM. Near FMM can reduce pair work in dense cells, but its
additional approximation and actual performance must be checked on the intended
distribution. With a PMMM-FMM build:

```sh
./build_pmmm/cosmictiger --gravity.meshSize=32 --gravity.pmmmNearWidth=2 \
  --gravity.openingAngle=0.4 --compute.bucketSize=32 \
  --hydro.enabled=false --compute.cuda=false
```

The compact gravity softening support must fit inside that near region:
the normalized support radius must be no larger than `w/n`. The existing
`gravity.supportRadius` and `gravity.effectiveRadius` input conventions still
apply; the parser checks this bound after converting the configured radius
to box units. Both near methods retain the selected compact gravity kernel for
direct pairs, including distinct coincident particles, and exclude primary self.
PMMM accuracy is controlled by mesh size, compiled expansion order, and near
width; the FMM opening angle controls only the optional near FMM, not the
far-field convolution.

## Source organization

PMMM interfaces are in `cosmictiger/gravity/nitadori/` and implementations in
`src/gravity/nitadori/`. `solver.cpp` coordinates the solve; `fft.cpp` owns the
persistent transform/cache lifecycle; `mesh_exchange.cpp` transfers moments and
local expansions. `ghost_storage.cpp` owns the reusable source arena,
`ghost_exchange.cpp` handles whole-cell requests, and `cell_schedule.cpp` launches
ready cell work. `cell_tree.cpp` supplies exact mesh-cell geometry to the common
tree builder. `fmm_batch.cpp` adapts ready cells to the common traversal, and
`evaluation.cpp` combines and consumes the fields. `near_pp.cpp`/`.cu` provide the
alternative direct near field, and `kernels.cpp` adapts the spherical operators.

Common Cartesian operators, tree construction, source views, and CPU/CUDA walks
remain directly in `gravity/`. PMMM near FMM uses this common code without a
dependency on the global solver in `gravity/dehnen/`. Owned particle arrays and
sorting remain in `particles/`. See [source organization](source_layout.md) for
the complete ownership and generated-code layout.

## Orders and coefficient conventions

PMMM uses the same application order convention and total-degree truncation
as the current FMM. For `COSMICTIGER_WITH_ORDER=q`, source multipoles have degree
`S=q-2`, local expansions have degree `L=q-1`, and a contraction retains only
terms with source degree plus local degree at most `L`. Each executable compiles
one selected order from 3 through 9; the default is seven.
This implementation does not retain the full
rectangular source/local contraction used by the experimental PMMM reference.

The spherical representation stores real and imaginary harmonic components
as real scalar fields. The channel counts are:

| Quantity | Scalar fields | At the default order 7 |
| --- | ---: | ---: |
| Source multipoles | `(q-1)^2` | 36 |
| Local expansions | `q^2` | 49 |
| Cached harmonic Green filters | `q^2` | 49 |

Let `h=1/n`, and let `a` be a particle's offset from its source cell center in
box units. P2M accumulates `mass * U(-a/h)` using the generated spherical
normalization. A degree-`d` Green coefficient is the unit-box physical
coefficient multiplied by `h^d`. The inverse transform produces local
coefficients for evaluation at `(position-cellCenter)/h`; acceleration is
divided by `h` to convert the local gradient back to box coordinates.
`gravity.gm` multiplies the complete far-plus-near result once.

The contraction table expresses harmonic conjugation as signed real
coefficient products. Spatial Fourier coefficients are complex, but they
are multiplied without Fourier conjugation. Each inverse transform is
normalized by `1/n^3`; source moments already contain particle masses and
need no density or cell-volume multiplier.

## Periodic Green function and near field

The Green filters use the generated spherical Ewald correction for the
periodic unit cube. Outside the direct neighbor cube, the primary free-space
Green function is added. Inside the cube, only that primary interaction is
omitted: periodic images remain in the convolution, including the finite
regular limit at the origin. Direct pair evaluation excludes the identical
particle's primary self interaction and retains distinct coincident
particles through the compact softened kernel.

Both near methods use the image selected by the two cell
centers. It is not independently wrapped to a different nearest particle
image; changing that image would no longer match the primary term omitted
from the Green filter.

The periodic uniform-background quadratic is kept separately from harmonic
coefficients. One small global reduction supplies total mass, the three sums
of cell-relative dipole components, and the radial second moment
`sum(mass*|a/h|^2)`. These provide the local trace, dipole correction, and
constant potential correction without additional FFTs. The radial second
moment enters the constant potential when the source degree includes two
(`q >= 4`), matching the generated FMM truncation. Trace-aware L2P includes
the quadratic contribution to both potential and acceleration.

The potential follows the Ewald periodic gauge and retains finite periodic
self images. The sampled, near-subtracted kernel's Fourier zero mode is
retained.

## FFT distribution, cache, and particle ownership

One FFTW3-MPI communicator contains every locality. All coefficient fields
use interleaved batched real-to-complex or complex-to-real transforms. Forward
plans produce transposed slabs, contraction operates directly on that layout,
and the inverse plan consumes it. There are no separate locality groups or
coefficient-group redistributions in this implementation.

The first solve creates three plans: source forward, Green-filter forward,
and local inverse. It also constructs and transforms the Green filters once.
Subsequent solves reuse those plans, buffers, and Green spectra. At order
seven, a normal solve performs 36 scalar-equivalent forward transforms and
49 inverse transforms through two batched execution calls; the 49 Green
transforms belong to initialization. This describes work, not measured speed.

The cache lasts until explicit PMMM destruction or application shutdown.
Mesh size, order, and near width cannot change while the cache is live.
Changing the near opening angle reuses the same FFT plans and Green spectra.
The solver, near method, and expansion order require a different build.
Plans use `FFTW_ESTIMATE`; there is no persistent disk-wisdom cache. MPI calls
run on native OS threads through HPX, use a communicator separate from HPX
parcel traffic, and require the application's existing `MPI_THREAD_MULTIPLE`
startup guarantee. FFTW planner operations share the process-wide planner
lock with other FFTW users.

PMMM uses the same spatial particle-domain decomposition as global FMM. Each
recursive cut balances the current particle count between two locality groups,
without requiring the cut to follow a mesh-cell plane. A mesh cell can therefore
have particle fragments on several localities, including when a dense cell
needs to be split for balance. Particle migration and checkpoint reload preserve
these continuous domains independently of the FFT mesh. Each solve sorts local
particle arrays in place into cell-fragment ranges, using parallel CPU
partitioning and preserving all attached particle fields.
The FFTW mesh retains its separate slab distribution, independent of particle
counts and particle-domain boundaries. Each particle owner forms partial cell
moments about the same mesh-cell center and sends them to that cell's FFTW slab
owner, which sums all fragments before the FFT. After the inverse FFT, every
particle owner fetches the completed expansion for each of its occupied cells
and evaluates it on its own particles. In PP mode, slab owners also hold copies
of near-source particles and return the requested neighboring records. In FMM
mode, a cell-owner directory identifies the sources needed by each locality's
occupied sink cells. Their owners send complete particle records for those
remote source-cell fragments; each arrival permits its source-tree construction.
All owner fragments of a required mesh cell are made available locally. A source
cell is copied once per recipient even when several sink cells need it.

Ghost records occupy a separate source arena; owned particle arrays retain their
original size. The counts and storage are reserved before arrivals, so building
or filling one cell cannot invalidate a running walk's data. Each owner builds
its cell tree before serving the cell's final particle permutation. Rungs and
other source metadata are captured for the current stage before any sink kick
can change them. Multiple owner fragments of one cell contribute local source
roots, and periodic source copies retain the existing image selection.

All owned and ghost source trees are local, so near walks fetch no remote tree
nodes or particles. A cell's sink task waits only for its required neighborhood,
not for every incoming ghost cell. Ready sinks may run separately or in batches:
`gravity.pmmmBatchParticles` is a positive target sink count (default 4096).
A cell larger than the target runs whole in its own batch. At most the worker
count of transient batches run concurrently. Ghost records and cell trees are
cleared after the solve; the reusable ghost arena retains its allocated capacity.
Ghosts never enter migration, checkpoints, or owned particle counts. Neither mode permanently migrates particles into FFT slabs.

`PmmmDiagnostics::ghostParticles`, `ghostCells`, and `ghostBytes` describe the
last solve's remote source copies, source cell-owner fragments, and particle-record payload
bytes, summed across localities. Payload bytes exclude directory and transport
metadata. `nearRemoteTreeFetches` and `nearRemoteParticleFetches` count remote
fetches during the near walk; both must remain zero. Near-field preparation
communicates to obtain whole-cell ghost sources while ready cells can walk.
`nearBatches` and `peakBatchParticles` report batching, and
`completedWhileGhostsPending` reports sink cells completing while other incoming
cells remain outstanding. `storedAccelerationParticles` and
`storedPotentialParticles` expose retained solve-result arrays: both are zero
for the ordinary streamed kick path.

SPH currently requires an FMM build whenever gravity is enabled.
PMMM with `hydro.enabled=true` is rejected at startup; set `hydro.enabled=false`
explicitly for cosmological PMMM runs, whose scenario otherwise enables SPH.

Moment formation, Green construction, FFTs, spectral contraction, and local
expansion evaluation run on CPUs. `compute.cuda=true` selects CUDA near-field
execution in a CUDA build and permits CUDA particle drift. For near FMM this
uses the existing FMM CPU/CUDA work distribution. Diagnostic fields named
`nearCpuAcceptedMultipoles` and `nearCpuDirectPairs` count CPU tree-walk work;
they do not claim to count GPU interactions.

The existing FMM near path accumulates potential and removes the softened
primary self term in single precision. That subtraction can lose precision
when the self potential is large compared with the remaining potential,
especially with a small softening support. Direct PMMM PP accumulates its
near field in double precision. This potential rounding difference is separate
from the opening-angle approximation and the force convergence checks.

## Memory and timestep costs

FFTW fields use single precision; moment accumulation, spectral product sums,
and spherical evaluation use double precision. If FFTW reports complex
allocation counts `A_s` and `A_l` for the batched source and local transforms,
and a rank owns `N_local` real cells, the principal retained mesh storage is
approximately

```text
16*A_s + 24*A_l + 8*(q-1)^2*N_local bytes.
```

This includes two padded real arrays, source/local/Green complex arrays, and
the double moment array. For a balanced large mesh on `P` ranks, its leading
term is `[16*(q-1)^2 + 12*q^2]*n^3/P` bytes. Order seven is about 1164 bytes
per cell across the job, or roughly 2.3 GiB for a `128^3` mesh before dividing
among ranks. FFT padding, allocation imbalance, and plan storage add overhead.
These estimates exclude particle arrays, source copies, cell forests and metadata,
communication packets, output fields, and near-field CPU/CUDA workspaces.

PMMM uses the same hierarchical particle timesteps as ordinary collisionless
FMM. Every ladder stage selects one particle range as both its sources and
sinks. Moment deposition, remote cell packets, and near FMM all use that same
selection; an inactive slow particle does not enter a fast stage's force.
The existing ascending/descending signed kicks and per-rung drifts are retained.
The timestep criterion uses the selected compact kernel's effective radius and
the complete near-plus-far acceleration.

Production calls `pmmmKick`: each completed cell's combined field is consumed
immediately for the kick and rung decision. It does not retain acceleration or
potential arrays for all particles. `analysis.saveAcceleration=true` and
`analysis.savePotential=true` independently request persistent particle
diagnostics; legacy `analysis.saveForce=true` requests both. These optional
fields are not checkpoint state. Explicit diagnostic calls to `pmmmSolve` still
collect full fields for force comparisons and do not advance particles.

PMMM and adaptive FMM have different error and work controls even at the same
nominal order. Mesh refinement, near width, and order convergence should be
checked on the intended particle distribution. Clustered cells can make the
direct near field expensive. No relative performance claim follows from the
coefficient counts or the FFT decomposition alone.

The application's FLOP accounting remains incomplete for PMMM: its FFT and
spherical far-field work are not included. Reported TFLOP/s therefore does not
describe the complete PMMM solve.

## Restart and regression checks

Checkpoints identify PMMM separately from FMM and record its mesh size, order,
near width, near method, and timestep compatibility parameters. Changing
these incompatible settings on restart is rejected, including restarting with a
binary compiled for a different solver, near method, or order. Historical ordinary PM
checkpoints are unsupported. Checkpoints do not store
FFTW plans or Green spectra: the first solve after restart reconstructs the
cache, then reuses it for later steps.
PMMM extension version two appends the near-method selection. Historical
version-one PMMM extensions identify the PP near method and are rejected by
PMMM-FMM builds. The checkpoint hierarchy flag must
also match: historical synchronous PMMM evolution checkpoints cannot resume
under the hierarchical driver. Opening angle and bucket size remain
mutable evolution controls rather than checkpoint identity.

The standalone `pmmm_kernels` tests check the numerical adapters and total
degree rule. Application tests compare complete fields against an independent
direct Ewald sum, exercise periodic boundaries,
and check cache reuse with the compiled near method. Run both PMMM build variants
to cover both near methods. In a PMMM-FMM build, a dense, unequal-mass fixture
proves that the CPU walk accepts multipoles and checks convergence when the near
opening angle is tightened. Distributed tests require ghost source exchange and zero remote
tree or particle fetches during the near walk. Repeated solves in each compiled PP/FMM variant check owned counts, exact position bits, velocities, rungs, and saved
field sentinels after ghost cleanup. The suite includes valid split cell ownership,
empty localities, periodic seams, distinct coincident sources, exact cell-center containment near
the fixed-point upper boundary, odd/even meshes, and CPU/CUDA agreement when
CUDA is requested. Streamed-stage tests compare signed kicks against an
independent Ewald sum over only the active sources, require unchanged inactive
sinks and no retained full-field result arrays, and exercise single-cell and
batched scheduling. A distributed observer gate holds one requested ghost
cell until an independent sink completes, checks that dependent sinks wait,
and compares the resulting fields with the ungated solve.
The evolution regression checks the compiled near method, hierarchical clocks,
exactly one scheduled drift per active particle, current restarts, historical
PP-extension acceptance or rejection, incompatible restarts, and startup rejection
of PMMM with SPH. Run it in both PMMM build variants for full near-method coverage.
A small CPU application
regression can be selected explicitly:

```sh
./build_pmmm/cosmictiger --run.test=pmmm --gravity.meshSize=8 \
  --initial.particlesPerDimension=8 --hydro.enabled=false \
  --compute.cuda=false --hpx:threads=2 --hpx:bind=none
```

Test definitions describe coverage; successful CPU, CUDA, and MPI execution
must be established separately for the build being used.
