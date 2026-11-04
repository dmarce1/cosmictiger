# SPH octree validation

November 3, 2026. Branch `codex/sph_octree`, based on `dc0d3f0d9`.
See [the design and implementation notes](sph_octree_plan.md).

## Builds and focused checks

Local Release builds use FMM potential degree 4, fixed32 positions, and disabled
chemistry/stars. CPU and CUDA applications and the SPH kernel test executables
compile and link. A CPU SPH-only build with all gravity disabled compiles, links,
and completes a short Sod run. An FMM/SPH build with only the hierarchical
integrator enabled compiles, links, and passes the hierarchical application test.

The 41 CPU SPH kernel tests pass. The two new CUDA checks run on the GPU and pass:
arbitrary gravity-leaf target mappings agree with the existing CPU/CUDA operator,
and the separately returned hydro force survives a gravitational component of
`1e20` without subtractive cancellation. The 82 build-graph regression checks
and the configured CPU source-graph check pass.
The full grouped-option regression also passes with hierarchical FMM/SPH:
CLI/INI precedence, checkpoint metadata, output units/frames, parameter bounds,
and rejection of unsupported source combinations.

Focused application checks pass for the tree, SPH gravity, gas checkpoint state,
and Sod smoke test. Tree fixtures exercise the inclusive one-third threshold,
signed and periodic drifts, support growth, retained membership, selective-rung
drifts, invalidation, and node serialization.

## Integration and distributed checks

`tests/run_sph_octree.py` exercises global and traditional wave and mixed gas/DM
evolution, lifecycle decisions, and exact canonical checkpoint restart. These
cases pass with CPU gravity/SPH and CUDA gravity/SPH. Global wave and mixed cases
also pass with each mixed CPU/CUDA backend combination.

The two-rank fixtures use one worker per rank. They verify retained neighbors
against direct distances after gas crosses both rank boundaries, migration of
hierarchical clocks without changing inactive rung prefixes, and a threshold
rebuild with 6/16 active particles while ten inactive particles lie outside their
owners' domains. The mixed traditional MPI evolution and restart pass.

`tests/run_sph_hierarchical.py` passes on CPU, CUDA, and two MPI ranks. Its blast
uses rungs 1–5, retains the tree on eleven ladder events after one initial rebuild,
and exercises up to 3,455 sources stored at future drift endpoints. It checks
positive entropy, source clocks, and uniform-rung agreement with traditional
integration. Restart keeps positions, entropy, entropy rates, rungs, flags, and
clocks exact; rebuilding the tree can reorder floating-point sums. Only velocity,
support, density, force, and velocity-predictor fields permit bounded roundoff,
at four float epsilons times the corresponding field scale. The scripts write
the measured differences to `evidence.json`.

## Self-gravitating gas

`tests/run_sph_hierarchical_gravity.py` uses a gas sphere with comparable gravity
and pressure forces: maximum gravity/hydro ratio 0.904, natural rungs 11–12.
At a common physical time, forced rungs 4, 5, and 6 converge toward rung 7:

| Rung | RMS position error | RMS velocity error |
| --- | ---: | ---: |
| 4 | 2.2841e-8 | 1.4271e-11 |
| 5 | 4.2183e-9 | 3.0268e-12 |
| 6 | 1.4274e-9 | 5.8629e-13 |

In this damped stage, the relative energy change remains below 0.072%, and the
net-momentum fraction below 3.24e-8. At uniform rung 5, the traditional comparison
has RMS position/velocity differences 4.76e-11 and 1.83e-15. A complete run with
minimum rung 4 checks the terminal closing ladder and final diagnostic pass.

The same regression advances 64 gas and 64 dark-matter particles through eight
cosmological events. It exercises a gravity rung merge from 9 to 10 on event
three, expansion from scale factor 0.02 to 0.02002668, and the corresponding
changing inverse drift factors. Gas clocks, positive entropy, and nonzero
gravitational forces pass their checks. Restart from event seven reproduces
event eight's complete physical and stage state exactly.

## Limits

These are focused local checks, not the entire repository suite or hosted CI.
Performance, peak transient memory, long cosmological evolution, and fixed64
runtime have not been benchmarked or established by these checks. CUDA uses
temporary launch-range gravity storage followed by grouped hydro kernels.
Hierarchy-only predictor records add 64 bytes per gas particle in these builds;
removing the persistent gravity vector is not a net-memory saving claim for
hierarchical integration. The existing active-only neighbor limiter does not
wake inactive gas midinterval. Chemistry, stars, PM, and P3M are outside this
implementation's scope.

Gas checkpoint payloads use base version 50000, plus 1000 for hierarchical state
and the existing physics suffixes. Earlier gas payloads are rejected.
