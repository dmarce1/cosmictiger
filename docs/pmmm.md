# Periodic particle mesh multipoles

Select **PMMM-PP** or **PMMM-FMM** and the expansion order in `./build.sh`.
The default build is global FMM at order seven; selecting PMMM defaults to order
eight. An explicit order selection overrides the default. To configure PMMM
directly with CMake, use:

```sh
cmake -S . -B build_pmmm -DCOSMICTIGER_GRAVITY_METHOD=pmmm \
  -DCOSMICTIGER_PMMM_NEAR_METHOD=pp -DCOSMICTIGER_WITH_ORDER=8
cmake --build build_pmmm --parallel 12
```

Set `COSMICTIGER_PMMM_NEAR_METHOD=fmm` for an adaptive FMM near field. Each build
contains one solver variant and one order from 3 through 9. PMMM uses a periodic
uniform mesh, spherical multipoles about cell centers and FFT convolution for
distant interactions. Neighboring cells use the compiled direct-pair or adaptive
FMM near field.

PMMM's forward, inverse, and cached Green-function FFTs run their parallel jobs
on the existing HPX worker pool through FFTW's threading callback. They share
the workers selected by `--hpx:threads` with the near-field tasks; FFTW does not
create a separate worker pool. Small transforms may expose fewer parallel jobs.
MPI still distributes the mesh between processes, with blocking MPI operations
kept on HPX's native I/O threads. The build still requires `fftw3f_threads` for
the callback API alongside `fftw3f_mpi` and `fftw3f`. Cosmology's independent
one-dimensional transforms remain single-threaded inside FFTW and are
parallelized by HPX. Their planning and destruction also run on native I/O
threads so waiting for the shared planner lock cannot block HPX workers.

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

The former `gravity.method`, `gravity.pmmmNearMethod`, `gravity.fmmOrder`, and
`gravity.pmmmOrder` inputs are rejected on both the CLI and in INI files.
`--help` and the `build.*` startup fields report the selected build configuration.

`gravity.meshSize=n` requests `n` full-particle cells on each axis of the
normalized unit box, independently of the initial-condition particle count.
The default value zero is unset and is rejected for PMMM.
`gravity.pmmmMaxPrimeFactor` sets the largest allowed prime factor of each FFT
side length. Its default is `5`, allowing lengths of the form `2^a * 3^b * 5^c`
for nonnegative integer exponents. Each requested length rounds up to the
smallest allowed integer, and all three sides use that same length. For example,
requested lengths 7, 17, and 49 become 8, 18, and 50. Finite limits must be
integers at least two; the limit itself need not be prime. Set
`--gravity.pmmmMaxPrimeFactor=infinity` to retain unrestricted integer lengths.
The same key works in the `[gravity]` INI section. Resolved output reports both
the requested `gravity.meshSize` and the full-particle `gravity.effectiveMeshSize`.

`gravity.pmmmNearWidth=w` sets the ghost
width and accepts integers from one through four, defaulting to two. The near
region contains `(2*w+1)^3` cells (27, 125, 343, or 729, respectively),
including the sink cell. The mesh must satisfy `n >= 2*w+1`, so periodic near
cells are distinct. Odd mesh sizes are supported.

`COSMICTIGER_PMMM_NEAR_METHOD=pp|fmm` selects how those near cells are evaluated
at build time. The default `pp` sums all primary particle pairs. The `fmm` choice uses the
existing Cartesian FMM operators and adaptive tree acceptance within the same
near-cell region. It first exchanges complete remote source-cell particle
records and builds local source trees as individual cells become ready.
Owned sink cells run once every near-source cell they need is ready, allowing
interior and ready boundary work to overlap outstanding cell fetches.
It adds no second periodic correction or distant-cell force:
the spherical FFT result supplies those contributions in either mode.

`COSMICTIGER_WITH_ORDER` sets the shared PMMM and near FMM expansion order,
defaulting to eight in a fresh PMMM build. PMMM fixes `gravity.openingAngle` to
`sqrt(3)/(w+1)`, even when a scalar opening angle is supplied explicitly. At the
default ghost width of two, theta is approximately 0.577350269. This theta stays
fixed during cosmological evolution; global FMM opening schedules are not
accepted in PMMM mode. `compute.bucketSize` controls the near FMM tree bucket
size. Near FMM can reduce pair work in dense cells, but its additional
approximation and actual performance must be checked on the intended
distribution. With a PMMM-FMM build:

```sh
./build_pmmm/cosmictiger --gravity.meshSize=32 --gravity.pmmmNearWidth=2 \
  --compute.bucketSize=32 \
  --hydro.enabled=false --compute.cuda=false
```

The compact gravity softening support must fit inside that near region:
the normalized support radius must be no larger than `w/N`, where `N` is the
rounded full-particle side length. The existing
`gravity.supportRadius` and `gravity.effectiveRadius` input conventions still
apply; the parser checks this bound after converting the configured radius
to box units. Both near methods retain the selected compact gravity kernel for
direct pairs, including distinct coincident particles, and exclude primary self.
PMMM accuracy is controlled by mesh size, expansion order, and near width.
The derived opening angle controls only the optional near FMM; the near width
also changes which interactions are included in the far-field convolution.

## Global FMM opening schedules

For an FMM build with `run.scenario=cosmology`, an optional schedule replaces the
previous hardcoded cosmology opening angles:

```sh
./cosmictiger --cosmology.startRedshift=49 \
  --gravity.openingSchedule='49:0.41,20:0.51,2:0.61'
```

The corresponding INI entry is `openingSchedule = 49:0.41,20:0.51,2:0.61` in
`[gravity]`. Pairs may appear in any order. The initial theta comes from the pair
closest to the starting redshift; equal distances choose the higher redshift.
During expansion, crossing a listed redshift selects its theta. During
contraction, recrossing a boundary undoes that change and restores the theta
used before expansion crossed it. In the example above, expansion selects
0.41, then 0.51 at z=20, then 0.61 at z=2; contraction restores 0.51 just
above z=2 and 0.41 just above z=20. At an exact boundary, both branches use
that entry's theta. If a step crosses several entries, apply or undo each in
crossing order.

Redshifts must be distinct and finite with `z > -1`; theta values must be
positive and within the normal `float` range. An empty schedule uses the scalar
`gravity.openingAngle`, whose cosmology default is 0.8. CLI schedule input
overrides an INI schedule, including `--gravity.openingSchedule=none` to clear
the schedule and select the scalar angle. An empty INI entry also selects the
scalar angle. Nonempty schedules are rejected for PMMM and noncosmology
scenarios. Restart reconstructs theta from the original starting redshift and
subsequent crossings, so use the same configured schedule and original
`cosmology.startRedshift` when restarting.

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
one selected order from 3 through 9; PMMM defaults to eight.
This implementation does not retain the full
rectangular source/local contraction used by the experimental PMMM reference.

The spherical representation stores real and imaginary harmonic components
as real scalar fields. The channel counts are:

| Quantity | Scalar fields | At the default order 8 |
| --- | ---: | ---: |
| Source multipoles | `(q-1)^2` | 49 |
| Local expansions | `q^2` | 64 |
| Cached harmonic Green filters | `q^2` | 64 |

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

Each cached mesh has an FFTW3-MPI communicator containing every locality.
All coefficient fields use interleaved batched real-to-complex or complex-to-real
transforms. Forward
plans produce transposed slabs, contraction operates directly on that layout,
and the inverse plan consumes it. There are no separate locality groups or
coefficient-group redistributions in this implementation.

The first solve at a mesh size creates three plans: source forward, Green-filter
forward, and local inverse. It also constructs and transforms the Green filters once.
Subsequent solves reuse those plans, buffers, and Green spectra. At the default
order eight, a normal solve performs 49 scalar-equivalent forward transforms and
64 inverse transforms through two batched execution calls; the 64 Green
transforms belong to initialization. This describes work, not measured speed.

Each mesh cache lasts until eviction, explicit PMMM destruction, or application
shutdown. Changing the requested base mesh size, maximum prime factor, order,
or near width requires explicit PMMM destruction.
The near opening angle is fixed by the near width.
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

Both near methods group whole sink cells toward `gravity.pmmmBatchParticles`.
The PP path packs the cells into one submission, preserving a separate source
range and periodic image coordinates for each sink. CUDA buffers and streams
are reused across completed submissions and released by `pmmmDestroy()`.
PP uses the CPU when a complete batch has fewer than 128 sinks with nonempty
source ranges or fewer than 65,536 candidate pairs. These conservative dispatch
thresholds are heuristics, not measured crossover points. The PP source staging
budget is 64 MiB per batch; a single oversized cell runs whole.

The local FMM path also compares the generated M2L arithmetic estimate with
direct PP work for geometrically accepted leaf pairs, choosing PP when its
estimate is smaller. Non-leaf acceptance and global FMM behavior are unchanged.
Batching follows the particle target without a separate 32-cell limit, so sparse
mesh cells can fill a useful batch.

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
term is `[16*(q-1)^2 + 12*q^2]*n^3/P` bytes. Order eight is about 1552 bytes
per cell across the job, or roughly 3.0 GiB for a `128^3` mesh before dividing
among ranks. FFT padding, allocation imbalance, and plan storage add overhead.
These estimates exclude particle arrays, source copies, cell forests and metadata,
communication packets, output fields, and near-field CPU/CUDA workspaces.

`COSMICTIGER_TIME_INTEGRATOR=leapfrog` is the default. PMMM's leapfrog path uses
conservative synchronized kick-drift-kick steps: all particles are sources and
sinks in every mesh solve, and all particles advance on the most restrictive
requested rung across the job. It calls `pmmmSolve(true)`, retaining acceleration
and potential arrays for kick preparation, then applies a common rung before
the drift. This differs from the individual block timesteps used by leapfrog
with global FMM.

Build with `COSMICTIGER_TIME_INTEGRATOR=hierarchical` to use the hierarchical
particle timestep ladder for dark matter. Every ladder stage selects one
particle range as both its sources and sinks. Moment deposition, remote cell
packets, and near FMM all use that same selection; an inactive slow particle
does not enter a fast stage's force. Ascending/descending signed kicks and
per-rung drifts are retained. In both integrators, the timestep criterion uses
the selected compact kernel's effective radius and the complete near-plus-far
acceleration.

Each hierarchical stage scales the mesh volume with its active particle fraction.
With requested `N0 = gravity.meshSize`, `w = gravity.pmmmNearWidth`, and global
owned particle counts `active` and `total`, the effective side length is

```text
target = clamp(ceil(N0 * cbrt(active / total)), 2*w+1, N0)
N = nextAllowedLength(target, gravity.pmmmMaxPrimeFactor)
```

The count includes the entire stage-selected source-and-sink range, including
faster rungs, and excludes ghosts. Exact integer volume comparisons implement
the ceiling without floating-point ambiguity at cube boundaries. An empty
selection rounds the minimum side up to an allowed length. Scaling always
uses the requested base size, before prime-factor rounding. For example,
`N0=7` with limit five gives a full-particle side of eight, while a stage with
three sevenths of the particles uses side six. The base configuration remains
unchanged. With `infinity`, prime-factor rounding is disabled and the sizing
matches the unrestricted integer policy.
All ranks use the same effective size for cell sorting, FFTs, ghost exchange,
and near-field geometry. Plans and Green spectra are reused by effective mesh
size, retaining at most eight meshes and twice the rounded base mesh's global FFT
buffer allocation. Transient moments and FFTW's internal plan storage are
separate from this buffer cap. Mesh switches occur only after the previous
solve has completed. `PmmmDiagnostics` reports `meshSize`, `activeParticles`, and
`totalParticles` for the most recent solve.

The hierarchical path calls `pmmmKick`: each completed cell's combined field is
consumed immediately for the kick and rung decision, without retaining
acceleration or potential arrays for all particles.
`analysis.saveAcceleration=true` and
`analysis.savePotential=true` independently request persistent particle
diagnostics; legacy `analysis.saveForce=true` requests both. These optional
fields are not checkpoint state. Calls to `pmmmSolve` alone collect full fields
and do not advance particles.

PMMM and adaptive FMM have different error and work controls even at the same
nominal order. Mesh refinement, near width, and order convergence should be
checked on the intended particle distribution. Clustered cells can make the
direct near field expensive. No relative performance claim follows from the
coefficient counts or the FFT decomposition alone.

The `pmmm` regression separates finite-expansion accuracy from implementation
error. It compares the computed field with a direct double-precision expansion
in physical coordinates, bypassing mesh deposition, the Fourier contraction
table, FFTs, and cell-width rescaling. That comparison allows `2e-5` scaled
numerical error, with a separate allowance for near FMM's subtraction of softened
self potential in float. The independent direct Ewald sum remains the physical
reference for both the finite expansion and the computed field.

The off-center fixtures have the following potential and force truncation
budgets. These cover the unit-mass collisionless and mixed-mass SPH fixtures,
mesh sizes 5, 8, and 10, and near widths 1 through 4. They are regression limits
for these fixtures, not accuracy guarantees for arbitrary particle distributions.
Potential error is scaled by `max(1, abs(referencePotential))`; each force-component
error uses `max(1, norm(referenceAcceleration))`.

| Order | Potential budget | Force budget |
| ---: | ---: | ---: |
| 3 | `3e-2` | `6e-2` |
| 4 | `1.5e-3` | `2.5e-2` |
| 5 | `3.5e-4` | `7.5e-3` |
| 6 | `6e-5` | `2e-3` |
| 7 | `3e-5` | `5e-4` |
| 8 | `8e-6` | `1.25e-4` |
| 9 | `3e-6` | `5e-5` |

A direct double expansion reaches about `1.614e-3` force error at order six in
these fixtures, so the former `5e-4` limit for every order from six onward was
too strict even before FFT roundoff. The numerical allowance is checked
independently and added to the truncation budget for the computed-field Ewald
comparison.

PMMM counts moment deposition, spherical far-field operators, spectral
contraction, FFTW plan execution, near interactions, and particle kicks.
Reported TFLOP/s divides these source-operation counts by the force-advancement
wall time, including communication, staging, and scheduling overhead. See
`flop_accounting.md` for the counting convention.

## Restart and regression checks

Checkpoints identify PMMM separately from FMM and record its requested mesh size,
maximum prime factor, order, near width, near method, and timestep compatibility parameters. Changing
these incompatible settings on restart is rejected, including restarting with a
binary compiled for a different solver, near method, or order. Historical ordinary PM
checkpoints are unsupported. Checkpoints do not store
FFTW plans or Green spectra: the first solve after restart reconstructs the
cache, then reuses it for later steps.
PMMM extension version four adds the maximum prime factor. Versions one through
three used unrestricted lengths and require
`--gravity.pmmmMaxPrimeFactor=infinity` on restart. The policy is checkpoint
identity even when two limits give the same current full-particle mesh, because
their hierarchical stage meshes can differ.
PMMM extension version three appends the PMMM order and resolved opening angle
to the version-two metadata, which added the near-method selection. Historical
version-one and version-two extensions retain their legacy shared order
convention; their saved FMM order must match the compiled PMMM order. Version-one
extensions identify the PP near method and are rejected by PMMM-FMM builds. Version-three restarts also require the resolved opening angle
to match. The checkpoint hierarchy flag must match: historical synchronous PMMM
evolution checkpoints cannot resume under the hierarchical driver. Opening angle
is derived from the recorded near width; bucket size remains a mutable control
rather than checkpoint identity.

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
