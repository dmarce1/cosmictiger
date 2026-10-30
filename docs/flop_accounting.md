# Force-advancement FLOP accounting

The progress-table rate is the sum of instrumented CPU and CUDA solver operations
on every locality, divided by wall time between `resetFlops()` and completion of
force advancement. It includes gravity force **and potential** kernels and SPH.
Energy/momentum diagnostics, diagnostic reductions, signal-speed reporting,
assertion checks, and the counters themselves are excluded. PMMM FFT executions
are included using their cached FFTW plan counts. Chemistry and other FFT paths
are not instrumented; their elapsed time is included if they run in this interval.

These are source-operation estimates: arithmetic, square roots, divisions and
other elementary mathematical results cost one; FMA costs two. Comparisons,
selection/min/max, absolute values, unary signs, integer operations and copies
cost zero. Explicit fixed-point construction/conversion counts its floating
scaling multiplication. Compile-time constant expressions are excluded. Compiler
folding, instruction fusion and library implementations need not preserve these
source counts.

## SPH audit

Before this repair, no active SPH CPU or CUDA kernel submitted operations to the
global counter. The CUDA rung pass had a private estimate, and the SPH tree had a
partial return value; neither supplied the displayed total. A gravity-disabled
Rayleigh–Taylor baseline consequently reported zero TFLOP/s.

Each production function now counts the branches and iterations it executes.
`W` below denotes CPU SIMD width; `B(n)=3n(n+1)/2` is a Bernstein polynomial's
arithmetic count. Kernel degree and span count come from the selected table.

| Function | Counted work |
| --- | --- |
| `sphCpuNeighbors`, `Fixed32Range::contains`, `sphSupportIntersects` | Actual fixed-coordinate conversions and support-box construction, including short-circuit exits. |
| `solveSphCpuSupportRadius` | Precomputed distances: `9W` per packet. Each Newton iteration counts support-bound checks, reciprocal, `5W` plus density and derivative kernel costs per packet, reductions, residual/tolerance, and the selected update/bisection branch. |
| `solveSphSupportRadius` (CUDA) | Each iteration counts support checks and reciprocal, 10 per visited neighbor, another 4 plus density/derivative costs inside support, both block sums, residual/tolerance and the selected update. |
| `sphCpuHasActiveNeighbor` / CUDA semiactive scan | Distance, dot product and squared support tests for the packets/pairs actually visited. |
| `sphCpuPrehydro1`, `sphCudaPrehydro1` | Neighbor construction and every support-radius solve, including unsuccessful solves requiring a larger neighbor box. |
| `sphCpuPrehydro2`, `sphCudaPrehydro2` | Activity scan, support solve, mass/support normalization, every density-kernel evaluation and density reduction; grad-H sums, derivative kernels, reductions and correction evaluation. |
| `sphCpuHydro` | Scalar sink setup, including 5 for its directed pressure coefficient; `9W+1` per candidate packet; 79 per lane of interacting packets plus both derivative kernels and enabled viscosity switches; EOS fallback; four numerical sums; external accelerations, damping and cosmological thermal work; timestep/rung selection. Padded lanes count when executed. |
| `sphCudaHydro` | Sink setup per executing thread, including 5 for its directed pressure coefficient; 8 per candidate, another 2 for nonzero separation; 79 per accepted pair plus both derivative kernels and enabled viscosity switches; actual block sums; external accelerations, damping, thermal work and timestep/rung selection. |
| Hydro timestep selection | 25 per active phase-1 sink plus 13 when self-gravity constrains the rung; another 3 when the thermal limiter executes. Signal-speed reporting and gravity-only diagnostic calculations with self-gravity disabled are excluded. |
| `sphCpuRungs`, `sphCudaRungs` | Support squares, actual distance/dot products and neighbor geometry; integer rung propagation costs zero. |
| `sphParticlesApplyUpdates` | Two per visited particle for its half-step; active predictor/opening kick adds 8, corrector adds 13. |
| `sphTreeCreate` | Local splitting, fixed-coordinate bounds, support multipliers and active support boxes. Each node submits only its own work. |
| `sphTreeNeighbor` | Actual search intersections and refreshed support boxes. Self-box validation checks are excluded. |
| `SphRunWorkspace::run` | Gravity effective-radius division; staging/copies and dispatch have no arithmetic count. |

The reusable Monaghan viscosity helper costs are 9 (`viscosityMu`) and 10
(`artificialViscosity`). The variational pressure helpers cost 3
(`pressureVolumeFactor`), 2 (`directedCoefficient`), 3 (`pairForceFactor`) and 2
(`pairWork`). The hydro loops submit `39 + 9 + 10 + 21 = 79` per interacting lane
or pair before derivative kernels and viscosity switches. The 21-operation term
includes the neighbor's directed pressure coefficient, pressure force, reversible
work, viscous force and heating factors. Variable alpha adds 3, beta following
alpha adds 1, and the Balsara limiter adds 3 per interacting lane or pair.
A scalar density/derivative evaluation costs `5+B(degree-derivative)` inside support and
zero outside. SIMD evaluates every polynomial span, costing
`pieceCount * (1 + W*(4+B(degree-derivative)))` per call, including the scalar
span-width subtraction. CPU and CUDA totals therefore need not agree even when
their numerical results agree.

CUDA block reductions count the actual additions, including redundant additions
used to broadcast the result. The 96- and 160-thread blocks do not use a guessed
power-of-two cost. Every kernel submits its accumulated 64-bit integer count
once; the per-launch return value is informational and is not submitted again.

## Gravity audit

The existing gravity kernel counts were already substantially instrumented.
The audit checked each interaction family and its route to the global counter.

| Function/path | Counted work |
| --- | --- |
| `cpuGravityPp` | 21 per padded pair lane, selected softening work, setup and force/potential reductions. |
| `cudaGravityPpDirect` | 18 per equal-mass pair or 20 per mixed-mass pair, selected softening, setup and force/potential reductions. |
| `cpuGravityCc`, `cudaGravityCcDirect` | Generated M2L operations, coordinates and expansion accumulation/reductions. |
| `cpuGravityCp`, `cudaGravityCpDirect` | Generated Green-function operations, coordinates, mass weighting and expansion accumulation/reductions. |
| `cpuGravityPc`, `cudaGravityPcDirect` | Generated M2P operations, coordinates and force/potential accumulation. |
| Ewald CC paths | Actual accepted-image derivative work, generated Ewald M2L, coordinates and numerical reductions. |
| `treeComputeMultipole` (internal) | Two generated M2M evaluations, 12 coordinate operations and coefficient accumulation. |
| `treeComputeMultipole` (leaf) | Three setup operations plus padded particles times generated P2M, six coordinate operations and twice the coefficient count. |
| Tree/acceptance | Bounds, centers, radii and numerical acceptance tests; scalar acceptance costs 17–19 depending on short circuits, SIMD 19 per lane. |
| CPU/CUDA walk | L2L/L2P, interaction kernels, force/potential accumulation and numerical reductions. |
| CPU/CUDA particle kick | Timestep/rung selection and velocity integration. The ordinary non-gas update is 23 operations; its former 61 included 38 diagnostic operations. |

Diagnostic energy/momentum arithmetic and return-value reductions were removed
from the gravity counts. Potential evaluation remains included. Automatic CUDA
handoff sizing includes its four arithmetic/math operations.

## PMMM accounting

Both PMMM-PP and PMMM-FMM include moment deposition, fixed-position conversion,
moment-packet accumulation, periodic background terms, spectral contraction,
local evaluation, field combination, and particle kicks. The existing shared
FMM tree/walk counters cover the adaptive near solver; PMMM additionally counts
its cell bounds and center-clamping arithmetic. Diagnostic reductions and
representability checks remain excluded.

| Stage | Counted work |
| --- | --- |
| Spherical P2M, periodic Green, and Ewald L2P | Generated `kConventionalFlops_` uses FMA=2 and elementary results=1. Historical weighted `kFlops_` and kernel selection remain unchanged. Wrappers return counts for task-level batching. |
| Green initialization | Generated Ewald work, optional primary Green addition, displacement and degree scaling, and the Green FFT. Charged only when the cache is built. |
| Forward/inverse FFT | Cached `fftwf_flops` additions + multiplications + 2*FMAs for each executed plan. MPI plan queries are collective and already return a global sum, so only locality zero submits it. Planner work and copies are excluded. |
| Fourier contraction | Ten operations per executed real spherical contraction term per complex Fourier mode. |
| Background reduction | Four local additions per mesh cell, plus five logical sums of `localities-1` additions counted once globally. Internal MPI algorithm redundancy is not measured. |
| Direct PP, CPU and CUDA | Three setup operations per nonempty call. Every non-self pair counts 21 geometry/scaling/accumulation operations plus the selected radial force evaluation; potential adds three scaling/accumulation operations plus its selected radial evaluation. Identical rank/index pairs contribute zero. |
| Field application | Three acceleration scalings and optional potential scaling; actual timestep/rung and velocity-update branches. Energy/momentum diagnostics are excluded. |

PP uses 64-bit accumulation once per CPU call or CUDA thread, with the existing
warp aggregation for GPU submissions. Mesh work submits per packet, row, or
completed phase, and field evaluation/kicks submit per cell or ready-cell batch.
The FFT convention follows [FFTW's plan operation counts](https://www.fftw.org/fftw3_doc/Using-Plans.html).
These remain operation estimates divided by force-advancement wall time, not
hardware performance counters or isolated kernel throughput.

Regression coverage includes independent arithmetic-intercepting checks of the
generated operators, every scalar softening family/span and potential branch,
Fourier contraction, and exact production PP submissions on CPU/CUDA (including
partial warps and blocks). Production tests also check cold/warm Green caches,
an exact empty-mesh total across MPI localities, and hierarchical/nonhierarchical
kick branches with 64-bit accumulation.

## Cosmological evolution

The Friedmann derivative costs 17 operations, including unit normalization;
its conformal-time derivative costs 18. Each integration stage costs 76, so
`cosmosUpdate` submits 9,729 operations and `cosmosAinv` submits 10,369 for
their 128-stage calls. Counts are batched outside the integration loops.
`cosmosAinv` supplies the scale factor used during hierarchical force advancement.

Physical-age output and halo lightcone scale reconstruction submit no operations.
Time-table and lightcone-output callers explicitly disable accounting in helpers
also used by the solver. `cosmology_flops_test` checks analytic Einstein-de Sitter
evolution, exact counter submissions, and zero diagnostic submissions while
preserving numerical results.

## Regression coverage

`sph_flops_test`, `softening_flops_test` and `fmm_flops_test` use an independent
arithmetic-intercepting number type to check the actual helper expressions,
selected polynomial spans, centered/uncentered FMM kernels, potential branches,
Ewald image cutoffs and scalar acceptance branches. `sph_cpu_test` checks that
each production phase submits exactly its returned count. CUDA SPH fixtures
check the same property against the global device total and independently count
the production block-reduction operators for 64, 96, 160 and 256 threads.
`fmmWalk` verifies that production CPU interaction functions and CPU/CUDA walks
submit counts. The counter regressions separately cover concurrency, integer
precision above the exact-double range, partial/divergent warps and multiple
localities.

The earlier working-tree Release build passed 41 focused CTest cases, including CPU and CUDA
production gravity walks, a three-locality gravity walk and two-locality counter
aggregation. This build instantiates FMM orders 4–8; the arithmetic-oracle cases
covered those configured orders. CPU and CUDA SPH tests cover all supported radial
kernel families and the non-power-of-two block sums.

The isolated publication snapshot, prepared on top of the separate test-framework
migration, passed all 38 focused CTest cases covering the CPU/CUDA counters,
FMM and softening arithmetic, SPH helpers and production CPU/CUDA SPH phases.
Counter aggregation ran on three localities. Its configure, build and test logs
are `/tmp/cosmictiger_flops_publish_{configure,build,ctest}.log`.

An 8³-particle, one-step Rayleigh–Taylor run with gravity disabled reported
`0.000e+00` TFLOP/s before the repair, `1.728e-05` on CPU afterward and
`1.281e-05` on CUDA afterward. These small-run rates demonstrate accounting,
not a performance comparison. CPU energy and timestep output were unchanged
from the baseline. Run logs and full test output are in
`/tmp/cosmictiger_sph_flops_{before,after,cuda_after}/run.log` and
`/tmp/cosmictiger_flops_final_ctest.log`.

## SPH arena follow-up

The low Rayleigh-Taylor rate also exposed wall time unrelated to arithmetic:
the SPH driver freed its 2,097,152-node arena after every step. At 160 bytes per
node, the next step initialized 320 MiB even though this run used only 383 nodes.
Normal teardown now retains the arena, invalidates cached references, and
rebuilds only live nodes from fresh state. Storage grows when needed; explicit
`sphTreeDestroy(true)` still releases it.

In a matched four-step CPU run with two HPX workers, the steady-step allocation
phase fell from 0.188-0.196 seconds to 0.8-1.4 microseconds. Full SPH tree creation
fell from 0.189-0.196 seconds to 0.56-0.59 milliseconds. The first allocation is
still paid once. FLOP totals, energy, density and timestep output were identical;
the complete checkpoint differed only in saved wall time. These timings were
taken while War Thunder was on, with six-CPU affinity and no GPU kernels; they
are not a CUDA throughput benchmark.

The CUDA-enabled Release build and seven focused CPU tests passed, including
the new arena shrink/regrowth/release fixture and four cosmology accounting
cases. Distributed arena rebuilding and CUDA runtime behavior were not retested
in this follow-up. Logs and the before/after comparison are under
`/tmp/cosmictiger_flops_throughput/`.
