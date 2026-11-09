# Periodic particle mesh multipoles

PMMM is CosmicTiger's gravity solver. `COSMICTIGER_ENABLE_GRAVITY` (default
`ON`) compiles it in `./build.sh` or with CMake feature options; every
configuration builds one `cosmictiger` executable containing the selected
capabilities. The default configuration enables PMMM gravity and SPH, particle
chemistry, the traditional integrator, and potential expansion degrees four
through six. A gravity-only configuration is:

```sh
cmake -S . -B build_pmmm -DCOSMICTIGER_ENABLE_SPH=OFF \
  -DCOSMICTIGER_EXPANSION_DEGREE=4-6
cmake --build build_pmmm --parallel 12
```

PMMM uses a periodic uniform mesh, spherical multipoles about cell centers and
FFT convolution for distant interactions. Neighboring cells use the adaptive FMM
near field (`gravity.method.near=fmm`, the default) or its fallback, the direct
particle-pair near field (`gravity.method.near=pp`). Both near fields are always
compiled.

Use `gravity.method.near=fmm|pp` and
`run.timeIntegrator=traditional|hierarchical|global` to select the run. An
integrator selector is required when more than one eligible integrator is
compiled; a sole eligible integrator is selected automatically.
`gravity.expansionDegree` selects a compiled potential degree and defaults to
the largest enabled degree, six in the default range. Near-field, integrator and
degree selections can differ between runs of one executable.

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
./build_pmmm/cosmictiger --gravity.method.near=pp \
  --gravity.expansionDegree=6 --run.timeIntegrator=traditional \
  --gravity.meshSize=32 --gravity.pmmmNearWidth=2 \
  --run.darkMatterParticleCount=32768 --run.gasParticleCount=32768 --hydro.enabled=false \
  --gravity.cuda=false --hpx:threads=12
```

The runtime settings can also be supplied in an INI file:

```ini
[gravity]
method.near = pp
expansionDegree = 6
meshSize = 32
pmmmNearWidth = 2

[run]
timeIntegrator = traditional

[hydro]
enabled = false
```

The former `gravity.method`, `gravity.pmmmNearMethod`, `gravity.fmmOrder`, and
`gravity.pmmmOrder` inputs are rejected on both the CLI and in INI files.
`--help` and the `build.*` startup fields report the enabled capabilities;
resolved option output reports the runtime selections.

`gravity.meshSize=n` requests `n` full-particle cells on each axis of the
normalized unit box, independently of the initial-condition particle count.
The default value zero is unset and is rejected for PMMM. `n` must be a power
of two; all three sides use that same length, and no rounding is applied.
Power-of-two sides make every cell boundary an exact fixed-point plane that
every coarser power-of-two mesh shares, which is what lets particle domains
avoid splitting cells (below).

`gravity.pmmmCoarseMeshSize=c` sets the coarsest mesh any hierarchical stage may
use and the granularity of particle-domain cuts. It must be a power of two no
smaller than the smallest power of two at least `2*w+1` (4, 8, 8, or 16 for
near widths one through four) and no larger than `gravity.meshSize`. The
default zero resolves to `n/8`, raised to that minimum when necessary, so a
`256^3` full mesh uses a `32^3` coarse mesh. The same keys work in the
`[gravity]` INI section. Resolved output reports `gravity.meshSize` and the
resolved `gravity.pmmmCoarseMeshSize`.

PMMM also requires a power-of-two locality count, and the coarse mesh must
have at least `2^ceil(log2(localities)/3)` cells per axis so every locality
receives whole coarse cells.

`gravity.pmmmNearWidth=w` sets the ghost
width and accepts integers from one through four, defaulting to two. The near
region contains `(2*w+1)^3` cells (27, 125, 343, or 729, respectively),
including the sink cell. The mesh must satisfy `n >= 2*w+1`, so periodic near
cells are distinct.

`gravity.method.near=fmm|pp` selects how those near cells are evaluated. The
`fmm` method (the default) uses the Cartesian FMM operators and adaptive tree
acceptance within the near-cell region. The `pp` method, its fallback, sums all
primary particle pairs. It first exchanges complete remote source-cell particle
records and builds local source trees as individual cells become ready.
Owned sink cells run once every near-source cell they need is ready, allowing
interior and ready boundary work to overlap outstanding cell fetches.
It adds no second periodic correction or distant-cell force:
the spherical FFT result supplies those contributions in either mode.

`gravity.expansionDegree` sets the shared PMMM and near FMM potential degree.
PMMM derives the near-field opening angle `gravity.openingAngle` as
`sqrt(3)/(w+1)`; it is not a configurable option. At the
default ghost width of two, theta is approximately 0.577350269. This theta stays
fixed during cosmological evolution. `compute.bucketSize` controls the near FMM tree bucket
size. Near FMM can reduce pair work in dense cells, but its additional
approximation and actual performance must be checked on the intended
distribution. With the near FMM capability enabled:

```sh
./build_pmmm/cosmictiger --gravity.method.near=fmm \
  --run.timeIntegrator=traditional --gravity.meshSize=32 --gravity.pmmmNearWidth=2 \
  --gravity.expansionDegree=6 --compute.bucketSize=32 \
  --hydro.enabled=false --gravity.cuda=false
```

The compact gravity softening support must fit inside that near region:
the normalized support radius must be no larger than `w/N`, where `N` is the
rounded full-particle side length. The existing
`gravity.smoothingLength` and `gravity.effectiveRadius` input conventions still
apply; the parser checks this bound after converting the configured radius
to box units. Both near methods retain the selected compact gravity kernel for
direct pairs, including distinct coincident particles, and exclude primary self.
PMMM accuracy is controlled by mesh size, expansion degree, and near width.
The derived opening angle controls only the optional near FMM; the near width
also changes which interactions are included in the far-field convolution.

## Source organization

PMMM interfaces are in `cosmictiger/gravity/` and implementations in
`src/gravity/`. `solver.cpp` coordinates the solve; `fft.cpp` owns the
persistent transform/cache lifecycle; `mesh_exchange.cpp` transfers moments and
local expansions. `ghost_storage.cpp` owns the reusable source arena,
`ghost_exchange.cpp` handles whole-cell requests, and `cell_schedule.cpp` launches
ready cell work. `cell_tree.cpp` supplies exact mesh-cell geometry to the common
tree builder. `fmm_batch.cpp` adapts ready cells to the common traversal, and
`evaluation.cpp` combines and consumes the fields. `near_pp.cpp`/`.cu` provide the
alternative direct near field, and `pmmm_kernels.cpp` adapts the spherical operators.

Common Cartesian operators, tree construction, source views, and CPU/CUDA walks
remain directly in `gravity/`; the PMMM near FMM is their only user. Owned
particle arrays and sorting remain in `particles/`. See [source organization](source_layout.md) for
the complete ownership and generated-code layout.

## Expansion degrees and coefficient conventions

The PMMM far field and near FMM share a potential-degree convention and total-degree truncation.
`COSMICTIGER_EXPANSION_DEGREE=d` compiles one degree; an inclusive range such as
`4-6` compiles every degree in that range. Supported degrees are two through
eight. For potential degree `d`, source multipoles have degree `S=d-1`, local
expansions have degree `L=d`, and a contraction retains only terms with source
degree plus local degree at most `d`. The internal historical order is
`q=d+1`, so the same limits are `S=q-2` and `L=q-1`.
This implementation does not retain the full
rectangular source/local contraction used by the experimental PMMM reference.

The spherical representation stores real and imaginary harmonic components
as real scalar fields. The channel counts are:

| Quantity | Scalar fields | At default potential degree 6 (`q=7`) |
| --- | ---: | ---: |
| Source multipoles | `d^2 = (q-1)^2` | 36 |
| Local expansions | `(d+1)^2 = q^2` | 49 |
| Cached harmonic Green filters | `(d+1)^2 = q^2` | 49 |

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
potential degree six, a normal solve performs 36 scalar-equivalent forward transforms and
49 inverse transforms through two batched execution calls; the 49 Green
transforms belong to initialization. This describes work, not measured speed.

Each mesh cache lasts until eviction, explicit PMMM destruction, or application
shutdown. Changing the requested base mesh size, coarse mesh size, degree,
or near width requires explicit PMMM destruction.
The near opening angle is fixed by the near width.
The runtime solver, near method, and degree remain fixed during an evolution.
Changing them between runs requires a new build only if the requested capability
or degree was not compiled.
Plans use `FFTW_ESTIMATE`; there is no persistent disk-wisdom cache. MPI calls
run on native OS threads through HPX, use a communicator separate from HPX
parcel traffic, and require the application's existing `MPI_THREAD_MULTIPLE`
startup guarantee. FFTW planner operations share the process-wide planner
lock with other FFTW users.

PMMM uses the common recursive bisection for its particle domains, but
every cut lies on a coarse-mesh plane. The locality count is a power of two;
depth `d` of the bisection halves the locality range and cuts axis `d mod 3`.
Among the coarse planes that leave each child enough whole coarse cells for the
cuts it still owes along that axis, the cut selects the plane whose particle
count comes closest to the balanced target, not the plane nearest the exact
balance coordinate. Ties prefer the plane nearest the middle of the parent.
Because every stage mesh is a power of two at least as fine as the coarse mesh,
no cell of any PMMM mesh ever has particles on two localities: each cell is
owned, with all its particles, by exactly one locality. Balance is therefore
only as fine as the coarse cells allow; a smaller coarse mesh (larger cells)
trades balance for a coarser minimum stage mesh. Gravity-free runs retain the
unrestricted continuous cuts. Particle migration and checkpoint reload preserve
the domains independently of the FFT slabs. Each solve sorts local particle
arrays in place into cell ranges, using parallel CPU partitioning and preserving
all attached particle fields. The hierarchical ladder keeps this ownership
exact for every active particle: a rung drifts only at its own level and is
migrated at that level before it next serves as a source.
The FFTW mesh retains its separate slab distribution, independent of particle
counts and particle-domain boundaries. Each cell's owner forms the cell's
moments about the mesh-cell center and sends them to that cell's FFTW slab
owner, which sums the packets before the FFT. After the inverse FFT, every
owner fetches the completed expansion for each of its occupied cells and
evaluates it on its own particles. Near-field sources never pass through the
slab owner: the owner of any cell is found from the particle-domain tree by the
cell center, and sinks read source particles directly from that owner. In PP
mode each sink cell requests its neighbor cells' particle records from their
owners, local ones without a message. In FMM mode each locality asks the remote
owners of its neighbor cells for their particle counts, reserves ghost storage,
and then fetches each occupied remote cell once; each arrival permits its
source-tree construction. A source cell is copied once per recipient even when
several sink cells need it.

Ghost records occupy a separate source arena; owned particle arrays retain their
original size. The counts and storage are reserved before arrivals, so building
or filling one cell cannot invalidate a running walk's data. Each owner builds
its cell tree before serving the cell's final particle permutation. Rungs and
other source metadata are captured for the current stage before any sink kick
can change them. Every source cell, owned or ghost, contributes exactly one
source root, and periodic source copies retain the existing image selection.

All owned and ghost source trees are local, so near walks fetch no remote tree
nodes or particles. A cell's sink task waits only for its required neighborhood,
not for every incoming ghost cell. Ready sinks may run separately or in batches:
`gravity.pmmmBatchParticles` is a positive target sink count (default 4096).
A cell larger than the target runs whole in its own batch. At most the worker
count of transient batches run concurrently. Ghost records and cell trees are
cleared after the solve; the reusable ghost arena retains its allocated capacity.
Ghosts never enter migration, checkpoints, or owned particle counts. Neither mode permanently migrates particles into FFT slabs.

`PmmmDiagnostics::ghostParticles`, `ghostCells`, and `ghostBytes` describe the
last solve's remote source copies, remote source cells, and particle-record payload
bytes, summed across localities. Payload bytes exclude directory and transport
metadata. `nearRemoteTreeFetches` and `nearRemoteParticleFetches` count remote
fetches during the near walk; both must remain zero. Near-field preparation
communicates to obtain whole-cell ghost sources while ready cells can walk.
`nearBatches` and `peakBatchParticles` report batching, and
`completedWhileGhostsPending` reports sink cells completing while other incoming
cells remain outstanding. `storedAccelerationParticles` and
`storedPotentialParticles` expose retained solve-result arrays: both are zero
for the ordinary streamed kick path.

SPH couples to PMMM self-gravity on the hierarchical integrator only (next section);
the traditional and global integrators run SPH without self-gravity. Set
`hydro.enabled=false` explicitly for gravity-only cosmological PMMM runs, whose scenario
otherwise enables SPH when compiled.
Particle chemistry additionally requires `COSMICTIGER_ENABLE_CHEMISTRY=ON`
(the default). Setting it to `OFF` retains SPH without particle chemistry;
homogeneous cosmological recombination remains available.

Moment formation, Green construction, FFTs, spectral contraction, and local
expansion evaluation run on CPUs. `gravity.cuda=true` selects CUDA near-field
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
estimate is smaller. Non-leaf acceptance is unchanged.
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
term is `[16*(q-1)^2 + 12*q^2]*n^3/P` bytes. Potential degree six (`q=7`) is about
1164 bytes per cell across the job, or roughly 2.3 GiB for a `128^3` mesh before dividing
among ranks. FFT padding, allocation imbalance, and plan storage add overhead.
These estimates exclude particle arrays, source copies, cell forests and metadata,
communication packets, output fields, and near-field CPU/CUDA workspaces.

`COSMICTIGER_ENABLE_INTEGRATOR_TRADITIONAL=ON` is the default;
`COSMICTIGER_ENABLE_INTEGRATOR_HIERARCHICAL` and
`COSMICTIGER_ENABLE_INTEGRATOR_GLOBAL` default to `OFF`. These capabilities are
independent. If several are eligible for the selected physics, choose one with
`run.timeIntegrator`.

PMMM's `traditional` path uses individual block timesteps, as FMM does. Every
substep rebuilds the mesh source field from all particles, including inactive
particles. Only sinks whose rung is active receive a gravity kick or a new rung;
inactive sinks retain their velocities and rungs. The gas acceleration is passed
to SPH for its own hydrodynamic update. `pmmmKick` consumes each
completed near-plus-far field directly; no common rung is imposed.

The `global` path advances every particle on one continuous timestep: the
collective minimum of the raw enabled gravity and SPH limits, capped by the
configured maximum step, cosmological limit, and remaining time to the endpoint.
It does not round that interval to a power of two, impose rung-growth limits, or
apply the minimum-rung floor. The previous interval's closing half-kick uses its
saved duration; the new opening half-kick uses the newly selected duration after
the collective reduction. The path supports gravity-only, SPH-only, and coupled
gravity/SPH evolution on CPU and CUDA. GADGET-4 cooling and star formation
run once after the completed SPH interval: star conversion uses the saved
previous star-formation rate, followed by cooling and the new rate. The source
does not retry or shorten that completed interval. Checkpoints preserve the
hydrodynamic closing half-kick and its exact interval; cooling has no pending
half-step state. Final diagnostics close the final hydrodynamic interval and
its cooling source without opening another interval. See
[equilibrium cooling](primordial_chemistry.md) and [star formation](star_formation.md).

Select `run.timeIntegrator=hierarchical` to use the hierarchical
particle timestep ladder for dark matter. Every ladder stage selects one
particle range as both its sources and sinks. Moment deposition, remote cell
packets, and near FMM all use that same selection; an inactive slow particle
does not enter a fast stage's force. Ascending/descending signed kicks and
per-rung drifts are retained. In all integrators, the gravity timestep criterion uses
the selected compact kernel's effective radius and the complete near-plus-far
acceleration.

Each hierarchical stage scales the mesh volume with its active particle fraction.
With `N0 = gravity.meshSize`, `C = gravity.pmmmCoarseMeshSize`, and global
owned particle counts `active` and `total`, the stage side length is the
smallest power of two `N` with

```text
C <= N <= N0  and  N^3 * total >= N0^3 * active
```

so the mean number of active particles per cell stays near the full-step value
while the side halves only when the active fraction drops below one eighth,
one sixty-fourth, and so on. The count includes the entire stage-selected
source-and-sink range, including faster rungs, and excludes ghosts. Exact
integer volume comparisons decide cube boundaries without floating-point
ambiguity. An empty selection uses the coarse mesh. The base configuration
remains unchanged.
All ranks use the same stage size for cell sorting, FFTs, ghost exchange,
and near-field geometry. Plans and Green spectra are reused by stage mesh
size, retaining at most eight meshes and twice the base mesh's global FFT
buffer allocation; with power-of-two sides every smaller stage mesh together
needs less than one seventh of the base workspace, so the byte cap is not
reached by stage meshes alone. Transient moments and FFTW's internal plan
storage are separate from this buffer cap. Mesh switches occur only after the
previous solve has completed. `PmmmDiagnostics` reports `meshSize`,
`activeParticles`, and `totalParticles` for the most recent solve. Here
`activeParticles` counts the selected sources (all particles for traditional
stepping); `evaluatedParticles` counts the sinks whose near field and local
expansion were actually evaluated. Traditional substep regressions check both
counts, inactive-source force contributions, unchanged inactive velocities and
rungs, and active sinks inside mixed tree leaves.

The hierarchical path calls `pmmmKick`: each completed cell's combined field is
consumed immediately for the kick and rung decision, without retaining
acceleration or potential arrays for all particles.
`analysis.saveAcceleration=true` and
`analysis.savePotential=true` independently request persistent particle
diagnostics; legacy `analysis.saveForce=true` requests both. These optional
fields are not checkpoint state. Calls to `pmmmSolve` alone collect full fields
and do not advance particles.

PMMM and adaptive FMM have different error and work controls even at the same
potential degree. Mesh refinement, near width, and degree convergence should be
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

| Potential degree (`q-1`) | Potential budget | Force budget |
| ---: | ---: | ---: |
| 2 | `3e-2` | `6e-2` |
| 3 | `1.5e-3` | `2.5e-2` |
| 4 | `3.5e-4` | `7.5e-3` |
| 5 | `6e-5` | `2e-3` |
| 6 | `3e-5` | `5e-4` |
| 7 | `8e-6` | `1.25e-4` |
| 8 | `3e-6` | `5e-5` |

A direct double expansion reaches about `1.614e-3` force error at potential
degree five (historical order six) in these fixtures, so the former `5e-4`
limit for every potential degree from five onward was
too strict even before FFT roundoff. The numerical allowance is checked
independently and added to the truncation budget for the computed-field Ewald
comparison.

PMMM counts moment deposition, spherical far-field operators, spectral
contraction, FFTW plan execution, near interactions, and particle kicks.
Reported TFLOP/s divides these source-operation counts by the force-advancement
wall time, including communication, staging, and scheduling overhead. See
`flop_accounting.md` for the counting convention.

## Outer interval with frozen near membership

`gravity.pmmmOuterInterval = true` (PMMM with `gravity.method.near = pp`,
`run.timeIntegrator = traditional` or `hierarchical`) integrates the far field
on a mandatory outer interval and the near field on the
inner rung clock. One mesh transform runs per outer boundary; no inner pass
deposits, transforms or evaluates a far expansion.

### The split

PMMM's far/near split is geometric: a pair interacts directly when the two
particles' cells lie within the `(2w+1)^3` neighbor cube, and through the mesh
otherwise. The outer interval freezes that assignment. When an interval opens,
every owned particle is sorted by position and its cell key is stored
(`particlesFrozenCell`); inside the interval the near passes group particles by
the stored key, wherever they have drifted, so

```text
H = T + V_far^P + V_near^P
```

is one fixed Hamiltonian split for the whole interval, with P the frozen
assignment. The pieces are smooth in the positions because membership does not
follow the particles, and the symmetric composition

```text
K_far(H/2) [ K_near(δ/2) D(δ) K_near(δ/2) ... ]  K_far(H/2)
```

keeps its second-order local error (Duncan, Levison & Lee 1998; Springel 2005,
eq. 33). Nothing limits how far a particle may move from its frozen cell; a
particle that left its cell keeps interacting directly with its frozen
neighbors and through the mesh with everything else. Near offsets are taken
from the image nearest the frozen cell center, which is exact as long as no
particle travels half a box inside one interval.

### Boundary equations

At a boundary with every particle at the synchronized position, with `S` the
near force and `L` the far force, `old` the frozen assignment being closed and
`new` the fresh assignment, `H` the outer durations and `δ` a particle's inner
durations:

```text
T_new   = L_new + S_new                      (one fresh far solve plus one near pass)
L_close = T_new - S_old                      (exact long complement of the old split)
Δp      = (δ_old/2) S_old + (H_old/2) L_close           closing
        + (H_new/2) L_new + (δ_new/2) S_new             opening
```

No far expansion is ever evaluated on the old assignment: the old long force
is the fresh total minus the old near force. A rung-zero particle (`δ_old =
H_old`) closes with `(H_old/2) T_new` exactly, independent of the assignment
change.

The traditional step performs this as three solver calls:

1. `frozenNearClose`: the old-membership pass, `Δp = (δ_old/2 - H_old/2) S_old`
   for every particle (skipped at `tau = 0`);
2. migration (with a rebound when the load imbalance exceeds its threshold),
   then `pmmmOuterBoundary`: sort by position, deposit, transform and evaluate
   the far field of every particle into `particlesLongAcceleration`, select the
   next interval;
3. `refreshMembership`: re-sort, store the new keys, near pass with the short
   closing half-kick spanning `H_old/2` for every rung
   (`closeShortOverInterval`), the long half-kicks `H_old/2` and `H_new/2` from
   the stored far acceleration, rung selection from `S_new` at the softening
   scale, and the `δ_new/2` opening.

The hierarchical ladder needs no correction pass: its ascending levels close
the fast subsets on the old membership, the top level migrates, runs the far
solve, refreshes the membership and kicks every particle with `(H_old/2)(S_new +
L_new)` and `(H_new/2)(S_new + L_new)`, and the descending levels open the fast
subsets on the new membership. Only the boundary top level migrates; inside an
interval every particle stays with its owner, so each frozen cell keeps one
owner and `cellOwner` resolves from the frozen cell center.

### Interval selection

```text
H_new = min( η sqrt(a w h / |L_new|) over particles, expansion bound,
             time to the next coarse output boundary, time to the end )
```

with `w h` the near width in box units, the smallest separation a far source
can have, `η = gravity.timestepEta` and the same caps as `pmSelectOuterInterval`.
Inner rungs come from the near force at the softening scale. The
potential energy at a boundary is the near potential of the boundary pass plus
the far potential of the far solve, counted once.

### State, restart and limits

Each particle carries a `uint64_t` frozen cell key (allocated only for this
mode) and, between the far solve and the boundary kick, the far acceleration in
`particlesLongAcceleration`. Checkpoints set header flag bit 23, write the keys
after the particle payload and the outer-interval extension (interval, start,
index) after the driver state; a restart inside an interval resumes the frozen
membership and a boundary checkpoint still holds the membership awaiting its
closing pass. `PmmmDiagnostics::transformCount` counts mesh transforms; the run
summary prints intervals, transforms and force passes.

Not implemented: `gravity.method.near = fmm` with the outer interval (the
near-FMM cell trees assume particles inside their cells and infer periodic
images from current positions) and the `global` integrator; both are rejected
at startup.

## SPH with PMMM self-gravity

Gas is kicked by the hierarchical (HOLD) ladder exactly as dark matter is: every
ladder stage selects one particle range as sources and sinks, and gas takes part in
the signed ascending and descending half-kicks and the long half-kicks of an outer
boundary (`src/gravity/pm_kick.cpp`). Two things differ, both because the SPH
midpoint integrator owns the gas rung and the hydro kick:

- Gas rungs change only at common boundaries. There the top level chooses each gas
  rung from the full gravity bound and the gas's hydro proposal,
  r = max(ceil(log₂(Δτ₀/Δτ_G)), r_old − 1, r_min, r_hydro), and the boundary rung
  balance closes it over touching neighbors. On all other levels the gas rung is held.
- The hydro kick of a rung-r interval is applied once, at its midpoint, for the whole
  interval (pair-once RK2). With dyadic rungs that midpoint is a boundary of every finer
  rung, so a coarse hydro event coincides with gravity half-kicks of finer gas. The
  hydro event must see the closing side only: gas opening half-kicks at that instant
  are collected (`gravityBeginDeferredGasOpening`) and applied right after it
  (`gravityFinishDeferredGasOpening`). An outer boundary never coincides with a hydro
  event, so no long half-kick is ever deferred.

Each SPH neighbor walk and pass is distributed with ghost leaves (see
`docs/sph_parent_tree_plan.md`). `tests/run_jeans_wave.py` (`ctest -R sph_jeans_wave`)
checks the coupling against the Jeans dispersion relation ω² = c²k² − 4πGρ₀ on 1, 2 and 4
localities; the measured frequency is within a few percent of ω_J at 20 particles per
wavelength, comparable to the SPH sound-speed error at that resolution, and the
localities agree to roundoff.

Gas also runs with the outer interval, in the cosmology and Plummer driver and in the
fixed-time scenario driver (`src/core/scenario_driver.cpp`), which follows the same
schedule: a boundary pass at every clock rollover, the outer interval as the rung base,
the exact interval clock and a terminal boundary. A boundary migrates every particle,
gas included (`domainsBegin(-1)`). A hydro rung
proposal is a rung number against the base step it was computed with, Δτ ≥ H 2^−r. The
first interval is chosen by the boundary pass of the first force step, after the initial
SPH timestep estimate, so that estimate takes the coarse interval as its base step;
every interval is at most that long, so those first rungs are never too coarse. At every
later boundary the proposals were made against the closing interval H_old, so a longer
opening interval H_new adds ceil(log₂(H_new/H_old)) rungs to them
(`hydroRungIntervalShift`).
`tests/run_sph_pmmm_interval.py` (`ctest -R sph_pmmm_interval_restart`) runs a small
cosmology with gas and dark matter and the Jeans wave, each on 1 and 2 localities,
restarts from a checkpoint inside an interval and requires the state after the next
boundary to match the uninterrupted run: bit for bit for the cosmology, within the
run-to-run roundoff for the Jeans wave, whose velocities are tiny. The Jeans frequency
with the interval agrees with the run without it to 0.03% of its error.

## Restart and regression checks

`tests/gravity/run_pmmm_interval.py` (`ctest -R pmmm_interval_cpu`) runs
the outer interval on the 512-particle Plummer fixture: one transform per
boundary plus the terminal pass and none inside, non-dyadic intervals selected
by the far acceleration bound, the interval clock, inner rungs from the near
force, frozen keys in every checkpoint, restart from inside an interval and
from a boundary, rejection of a restart with the mode switched off, and energy
and momentum against the per-substep PMMM run over the same time.

Checkpoints identify PMMM separately from FMM and record its requested mesh size,
coarse mesh size, expansion degree (in the historical internal order
convention), near width, near method, and timestep compatibility parameters.
Omitted runtime algorithm, degree, and integrator selectors inherit the saved
selection on restart. Explicit incompatible selections are rejected, as are
saved capabilities unavailable in the executable. A build may contain additional
algorithms or a larger degree range without invalidating a checkpoint.
Historical ordinary PM checkpoints are unsupported. Checkpoints do not store
FFTW plans or Green spectra: the first solve after restart reconstructs the
cache, then reuses it for later steps.
PMMM extension version five stores the resolved coarse mesh size; it is
checkpoint identity even when two coarse meshes give the same current
full-particle mesh, because their hierarchical stage meshes differ. Version
four stored the retired maximum FFT prime factor and versions one through three
used unrestricted lengths; those checkpoints restart under the requested coarse
mesh when their saved base mesh is a power of two, which the current option
validation requires.
PMMM extension version three appends the PMMM order and resolved opening angle
to the version-two metadata, which added the near-method selection. Historical
version-one and version-two extensions retain their legacy shared order
convention; their saved FMM order must match the selected PMMM order. Version-one
extensions identify the PP near method and require that capability and runtime
selection. Version-three restarts also require the resolved opening angle
to match. The checkpoint hierarchy flag must match: historical synchronous PMMM
evolution checkpoints cannot resume under the hierarchical driver. Global
evolution also records its exact previous interval and pending half-step phase;
it cannot restart under a different integrator. Opening angle
is derived from the recorded near width; bucket size remains a mutable control
rather than checkpoint identity.

The standalone `pmmm_kernels` tests check the numerical adapters and total
degree rule. Application tests compare complete fields against an independent
direct Ewald sum, exercise periodic boundaries,
and check cache reuse with the selected near method. Run with each near method
enabled, either in one combined executable or separate minimal builds. With
`gravity.method.near=fmm`, a dense, unequal-mass fixture
proves that the CPU walk accepts multipoles and checks convergence when the near
opening angle is tightened. Distributed tests require ghost source exchange and zero remote
tree or particle fetches during the near walk. Repeated solves with each PP/FMM
selection check owned counts, exact position bits, velocities, rungs, and saved
field sentinels after ghost cleanup. The suite includes valid split cell ownership,
empty localities, periodic seams, distinct coincident sources, exact cell-center containment near
the fixed-point upper boundary, power-of-two meshes, and CPU/CUDA agreement when
CUDA is requested. Streamed-stage tests compare signed kicks against an
independent Ewald sum over only the active sources, require unchanged inactive
sinks and no retained full-field result arrays, and exercise single-cell and
batched scheduling. A distributed observer gate holds one requested ghost
cell until an independent sink completes, checks that dependent sinks wait,
and compares the resulting fields with the ungated solve.
The evolution regression checks the selected near method, hierarchical clocks,
exactly one scheduled drift per active particle, current restarts, historical
PP-extension acceptance or rejection, and incompatible restarts. Run it with
both near methods for full near-method coverage. Global evolution checks cover
non-dyadic intervals, all-particle stepping, restart continuity, and coupled SPH.
A small CPU application
regression can be selected explicitly:

```sh
./build_pmmm/cosmictiger --run.test=pmmm \
  --gravity.method.near=pp --run.timeIntegrator=traditional --gravity.meshSize=8 \
  --run.darkMatterParticleCount=512 --run.gasParticleCount=512 --hydro.enabled=false \
  --gravity.cuda=false --hpx:threads=2 --hpx:bind=none
```

Test definitions describe coverage; successful CPU, CUDA, and MPI execution
must be established separately for the build being used.

The near-FMM cell forest reserves at most two local node slots per owned or
ghost source particle, plus domain joins and worker allocation-line slack. A
node slot costs `sizeof(TreeNode) + fmmPositionedMultipoleStride(order)` bytes;
the latter is the positioned multipole record padded to a 16-byte boundary.
Remote deserialized moments use a separate managed-memory chunk pool and do
not consume local node slots. The implementation checks the combined forest
capacity against the signed `TreeId` index range before allocation.
