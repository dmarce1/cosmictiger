# Particle-mesh gravity: PM, PP-PM and FMM-PM

`gravity.method.far=pm` selects pure particle-mesh gravity; `gravity.method.far=p3m`
selects the Gaussian-split solvers whose short range is evaluated either by
direct pairs (`gravity.method.near=pp`, PP-PM) or by a screened full-Cartesian
FMM (`gravity.method.near=fmm`, FMM-PM). The three variants share the scalar
PM/FFT solve in `gravity/pm/` and the mandatory outer PM interval described
below; they are compiled with `COSMICTIGER_ENABLE_PM`, `COSMICTIGER_ENABLE_P3M_PP`
and `COSMICTIGER_ENABLE_P3M_FMM` (all `OFF` by default) and may coexist with FMM
and PMMM in one executable. There is no long-range FMM: both P3M variants take
the long force from the mesh. The foundations (shared screened kernel, full
symmetric Cartesian operators, cloud weights) are described in
[gaussian_screening.md](gaussian_screening.md) and
[generated_cartesian.md](generated_cartesian.md).

```sh
cmake -S . -B build_p3m -DCOSMICTIGER_ENABLE_PM=ON -DCOSMICTIGER_ENABLE_P3M_PP=ON \
  -DCOSMICTIGER_ENABLE_P3M_FMM=ON -DCOSMICTIGER_EXPANSION_DEGREE=5
./build_p3m/cosmictiger --gravity.method.far=p3m --gravity.method.near=pp \
  --gravity.meshSize=64 --gravity.cloud=pcs --run.timeIntegrator=traditional \
  --initial.particlesPerDimension=32 --hydro.enabled=false
```

## Options

| Key | Default | Meaning |
| --- | --- | --- |
| `gravity.method.far` | build default | `pm` or `p3m` (also `fmm`, `pmmm`). |
| `gravity.method.near` | build default | `pp` (PP-PM) or `fmm` (FMM-PM) for `p3m`; unused by `pm`. |
| `gravity.meshSize` | 0 (required) | PM cells per axis, at least `max(2, cloud points)`; any size for `pm`, and large enough for a distinct-neighbor search mesh for `p3m` (32 at the default split and cutoff). |
| `gravity.cloud` | `pcs` | Mass assignment and force interpolation cloud: `ngp`, `cic`, `tsc`, `pcs`. |
| `gravity.p3mSplitCells` | 1.25 | Gaussian split radius `r_s` in PM cells (`p3m`). |
| `gravity.p3mCutoff` | 4.5 | Short-range pair cutoff in units of `r_s` (`p3m`). |
| `gravity.pmmmNearWidth` | 2 | Neighbor-cube radius of the short-range search mesh in cells (`p3m`). |
| `gravity.pmmmCoarseMeshSize` | 0 | Coarse mesh of the particle-domain cuts; resolves from the search mesh for `p3m`. |
| `gravity.openingAngle` | 0.35 for FMM-PM | Screened multipole acceptance of the short walk. |
| `gravity.timestepEta` | 0.204 | Shared by the outer PM interval and the inner short-range rungs. |

Resolved output adds the derived `gravity.p3mSplitRadius`, `gravity.p3mCutoffRadius`
and `gravity.p3mMeshSize` (box units and cells). `run.timeIntegrator` selects
the inner integration: `traditional` or `hierarchical`; the `global` integrator
is rejected for these solvers. Hopkins MFM/MFV is rejected because its tick
clock requires a fixed base interval; SPH is supported. `gravity.cuda=true`
runs the PP-PM pair batches on the device; the FMM-PM walk and the mesh solve are
CPU code. The same executable can run FMM, PMMM, PM and P3M on different runs.

## Scalar PM solve

Every owned particle is deposited through its cloud onto an `n^3` mesh with the
mesh-cell sort of `particles/particle_mesh_sort`, tiles of eight cells with a
two-cell halo collect the contributions and send them to the FFTW slab owners.
Fields are interpolated back with the same cloud. With spacing `h=1/n`, integer
modes `m`, physical wave number `k=2*pi*m` (unit box) and the window
`W(k)=prod_d [sin(k_d h/2)/(k_d h/2)]^p` of a `p`-point cloud, the solve applies

```text
Phi(k) = -4 pi gm / (|k|^2 W(k)^2) * exp(-|k|^2 r_s^2) * rho(k),     g_d(k) = -i k_d Phi(k).
```

The `1/W^2` compensates both assignment passes; the Gaussian factor is present
only for `p3m` (r_s in box units, `alpha=1/(2 r_s)` in the shared radial kernel),
so the mesh force of `p3m` is the convolution of the particle distribution with
`-erf(alpha r)/r`. The zero mode has zero potential and force, an even mesh's
Nyquist mode has zero derivative, and the inverse transforms are normalized by
`1/n^3`. Three inverse transforms produce the acceleration, a fourth the
potential when requested; the Green factor is built per mode and never stored.
The potential interpolated at a particle contains the particle's own discrete
cloud response; the solve subtracts that self term (the same compensated Green
function applied to the source and sink cloud, including the subcell
dependence) so the reported long potential approximates the pairwise
`-erf(alpha r)/r` sum. The reported long potential energy is
`sum(0.5 m phi)` scaled by `gm`. The gm-scaled acceleration is stored in the
per-particle long-acceleration field (`particlesLongAcceleration`), which
follows permutation and migration and is never checkpointed.

Plans use `FFTW_ESTIMATE` with FFTW-MPI slabs over one communicator per mesh;
all collectives run on native OS threads, and the three collective phases use
the PMMM collective dispatch so a failing locality aborts the MPI job instead
of deadlocking. FFT operation counts come from `fftwf_flops` once per plan.

## Short-range split

For a pair at separation `r` with softening support `H`, the short force added
by PP-PM or FMM-PM is

- `r >= H`: the screened Newtonian pair, force multiplier
  `erfc(q) + 2q/sqrt(pi) exp(-q^2)` with `q=alpha r`, potential `-m erfc(q)/r`;
- `r < H`: the physical softened force and potential minus the smooth long
  component (`-m erf(q)/r` and its gradient), so short plus mesh reproduces the
  softened law exactly; the softened force is never multiplied by the screening
  factor;
- `r = 0` between distinct particles: zero force and the finite potential
  `phi_soft(0) + 2 alpha m/sqrt(pi)`; identical primary particles are excluded;
- `r >= r_cut = p3mCutoff r_s`: dropped. At the default `4.5 r_s` the dropped
  multiplier is 1.7% of the Newtonian pair force; raise the cutoff for more.

The search mesh is the largest power of two `n_s` with `w/n_s >= r_cut`
(`w = gravity.pmmmNearWidth`), so the `(2w+1)^3` neighbor cube of a cell covers
every pair inside the cutoff; it must hold distinct periodic cells
(`n_s >= 2w+1`). Particle domains cut on coarse-mesh planes of that search
mesh (`pmmmCoarseMeshSize` semantics as for PMMM), so each cell with all its
particles belongs to one locality; remote neighbor cells are fetched once per
solve as plain source records with whole-box image shifts applied by the sink.
Sources are every owned particle in traditional steps and the rung-selected
range in hierarchical stages, as in PMMM; sinks are the active particles.

PP-PM batches sink cells toward `gravity.pmmmBatchParticles` and runs the
double-precision pair law on the CPU, or on CUDA above the same thresholds as
PMMM's near field. FMM-PM builds one kd-tree per source cell (owned or fetched)
with full symmetric Cartesian moments about node centers of mass at the runtime
`gravity.expansionDegree`, then walks each owned sink cell's tree against its
neighbor trees. A cell pair is accepted for M2L with the screened Green
derivatives when it lies entirely inside the cutoff, entirely outside the
softening support, and

```text
r_sink + r_source < theta * min(d, 2 r_s^2 / d),
```

where `d` is the center distance: the second length is where the Gaussian
factor's derivatives steepen faster than `1/d`. Rejected pairs open the larger
cell (or the source at a leaf sink); leaf pairs use the pair law above. Pairs
straddling the cutoff are therefore resolved to particles, so PP-PM and FMM-PM
drop exactly the same pairs. The acceptance error scales as `theta^order`;
with degree five, `theta=0.5` gives percent-level short-force errors and
`theta=0.35` about `1e-4` (see validation), so the FMM-PM default is 0.35.
Inactive sink particles in a cell are walked but not consumed; the walk is
CPU only and is efficient when many particles share a few `r_s`.

## Outer PM interval and inner integration

Every PM variant has a mandatory global outer interval. At an outer boundary all
particles are synchronized in position; the driver migrates domains, solves the
mesh with every gravitating particle as source, stores the long acceleration
and chooses the next interval

```text
T = min( eta sqrt(a r_eff / |g_L|) over particles, expansion bound,
         time to the next coarse output boundary, time to the end ),
```

with `r_eff = sqrt(order/3)/n` for `pm` and `sqrt(r_cloud^2 + 2 r_s^2)` for
`p3m` (the Gaussian force width). `T` is not a power of two; it never crosses a
coarse boundary (`run.numberOfSteps` keeps outputs on the nominal clock) and the
last interval ends exactly at the end time. The inner integer phase `itime`
subdivides the actual interval: rung `r` has substep `T/2^r`, and the existing
traditional or hierarchical machinery runs unchanged with `t0=T`. The outer
composition is

1. mesh solve at the boundary positions,
2. the force pass at `itime=0`: the short solver closes the previous substep
   with the previous interval's base (`KickParameters::closingBaseTimestep`),
   applies the closing long half-kick `g_L T_prev/2`, measures the synchronized
   diagnostics, selects the inner rungs from the short force at the softening
   scale, opens the next substep and applies the opening long half-kick
   `g_L T_next/2` (`longCloseTimestep`, `longOpenTimestep`);
3. inner drift / short-force / hydro passes until `itime` wraps; no FFT runs
   inside the interval.

Consecutive intervals of different length therefore close and open with their
own durations. Pure PM has no short force: the traditional inner kick keeps its
rung floor of one (two substeps per interval) and the hierarchical ladder keeps
every particle on rung zero, so inner substeps exist only for hydro. In the
hierarchical ladder every ascending level closes with the previous base, and
only the top level applies the long half-kicks. SPH gas receives the long
half-kicks directly and its short acceleration through the hydro integrator,
whose predictor/corrector close with the previous base
(`sphStep2(..., closingTimeStepScale)`); the diagnostic correction for the long
opening uses the dark-matter correction channels. Energies add the mesh
potential energy to the short potential at synchronized boundaries.

Checkpoints record the particle-mesh identity (version, mesh, cloud, short
solver, near width, coarse mesh, split cells, cutoff, eta, minimum rung, search
mesh, force convention) and the interval state (`outerInterval`, `outerStart`,
`outerIndex`) next to the interval-local `itime`; a restart inside or at the end
of an interval reproduces the uninterrupted evolution. Changing any identity
field, the far solver or the integrator is rejected. The end-of-run energy pass
solves the mesh once more and only closes.

## Validation

Results of 2026-11-02 in `build/p3m_cpu` (CPU, fixed32, degree 5, GCC 15, two
worker threads; `--run.test=pm`, `--run.test=p3m` and
`tests/gravity/pm/run_pm_evolution.py`):

- PM Fourier oracle: meshes 5, 6 and 24 with NGP/CIC/TSC/PCS against an
  independent direct DFT including both windows, Nyquist zeroing and, for `p3m`,
  the Gaussian factor; relative force errors `1e-7` to `2.5e-6`, self-subtracted
  potentials to `8e-5`, deposited mass conserved to `2e-6`, zero total force for
  unequal masses, zero force on a uniform lattice, exact whole-cell translation
  invariance (to `4e-6` of the force scale), and the stored particle long
  acceleration equal to the solve result. The outer-interval selector is checked
  for the acceleration, expansion, coarse-boundary and end-time limits.
- Split closure: 400 displacements inside and outside a support of 0.02 with
  `r_s=0.0125` close on the softened/Newtonian force and potential to `1e-10`;
  coincident pairs and the cutoff behave as specified.
- Screened walk against direct pairs (256 sources in a `0.004` ball, 48 sinks
  at `0.012`–`0.16`, `r_s=0.04`, bucket 4): maximum relative force error
  `6.4e-1`, `7.2e-2`, `1.7e-2`, `1.2e-4`, `1.4e-5` at `theta` 1.0, 0.7, 0.5,
  0.35, 0.25.
- Periodic references (`gm=0.125`, mesh 32, PCS, `r_s=1.25h`, cutoff `4.5 r_s`,
  support 0.02): 96 random particles against the direct softened Ewald
  reference, maximum relative total-force error `6.2e-4` (rms `1.5e-4`) for both
  PP-PM and FMM-PM; a softened close pair `3.7e-6`; 4096 clustered particles
  (two Gaussian clumps) `1.5e-3` on 64 sampled sinks for both, with FMM-PM
  accepting 158,977 multipoles and 2.25M direct pairs against 7.0M pairs for
  PP-PM; PP-PM versus FMM-PM short force differs by at most `2.6e-4` of the
  total force scale (rms `1e-5`) at `theta=0.35`; total momentum `1e-9`
  (PP-PM) and `2e-7` (FMM-PM) of `sum|m a|`.
- Evolution (Plummer, 512 particles, `run.numberOfSteps=4`, six force passes,
  `gravity.timestepEta` 0.05/0.03): one mesh solve per outer boundary and none
  inside intervals (summary line and verbose boundary lines agree), outer
  intervals selected by the acceleration bound (non-dyadic, changing, landing
  exactly on the coarse boundary), the interval-local clock, inner rungs for
  the short force, restart from a mid-interval and from a boundary checkpoint
  reproducing positions, velocities and rungs, identity rejection for mesh,
  cloud, eta, split, cutoff, short solver and far solver. The mixed SPH case
  (cosmology scenario) checks finite gas state and restart.

- CUDA (`build/p3m_cuda`, PM and PP-PM only, sm_86): the `p3m` regression with
  `gravity.cuda=true` ran its clustered pair batch on the device (diagnostic
  `cudaBatches` 1 of 1) and reproduced the CPU forces digit for digit in the
  printed errors; a bounded unit-test run, not a full CUDA application run.
- Two localities (`pm_mpi_2`, `p3m_mpi_2`, `pm_evolution_mpi`): the PM oracle,
  the P3M references and the pure-PM evolution/restart checks pass with the
  MPI parcelport; the search-mesh cells of each locality fetch their remote
  neighbor cells once per solve.
- The options regression (`tests/run_options.py --gravity-method p3m`) covers the
  new keys, the derived P3M geometry, the cloud/mesh/cutoff/support rejections,
  the rejected `global` integrator and the complete INI presets.

Limits: the FMM-PM walk is CPU only and has no CUDA counterpart; the `global`
inner integrator and Hopkins MFM/MFV are rejected for pm/p3m; no full CUDA
application run, production cosmology or more than two localities were tried.
Energy conservation over long evolutions and the time convergence of the outer
interval have not been measured beyond the restart-equivalence and clock checks
above; the standalone gtest binaries of the foundation stage were not rebuilt in
this tree.
