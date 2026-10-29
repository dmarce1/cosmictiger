# Gravity accuracy during evolution

Enable sampled direct-force comparisons with:

```sh
./cosmictiger --gravity.errorInterval=10 --gravity.errorSampleSize=100
```

These controls also work in INI files:

```ini
[gravity]
errorInterval = 10
errorSampleSize = 100
```

`gravity.errorInterval` is a nonnegative number of rung-0 (coarsest) timesteps.
The default, `0`, disables the diagnostic. A positive interval checks the initial
force evaluation at coarse step 0, then steps N, 2N, and so on. Substeps on finer
rungs do not trigger additional checks. Both ordinary leapfrog and hierarchical
integration use this same coarse-step clock, including after a checkpoint restart.

`gravity.errorSampleSize` defaults to 100 and must be positive while checks are
enabled. The actual count is capped at the global particle count. Each check selects
evenly spaced indices across the current global particle array. Selection is
independent of the Silo output controls `analysis.tracers` and
`analysis.tracerCount`. Temporary sample attributes follow particles through
sorting and migration. Checkpoints do not store these temporary attributes;
the next scheduled check selects a new sample from the current ordering.

For each selected particle, the diagnostic first computes and saves a direct
acceleration from all source particles at the current positions. It then compares
that reference with the gravity acceleration produced by the configured FMM or
PMMM solve when the kick finishes. The direct solve uses the configured softening
kernel and periodic gravity convention. Comparison does not require retaining
acceleration fields for every particle with `analysis.saveAcceleration`.

The reference uses CUDA when `compute.cuda=true`, and otherwise uses HPX CPU
tasks. The startup message identifies the reference backend separately from the
production gravity solver. CUDA distributes work across source tiles as well as
samples, so even a small sample can use multiple blocks. Both paths retain fixed
particle coordinates until forming double-precision separations, use the same
compact-support radial force function as the production PP kernels, and accumulate
mass-weighted forces in compensated double precision.

The smooth periodic correction is interpolated from an independent Ewald force
table, following the correction-plus-primary-force split in
[Hernquist, Bouchet & Suto (1991), section II](https://adsabs.harvard.edu/pdf/1991ApJS...75..231H).
The table is built once per process on first use. It uses 64 intervals per
half-box axis, three additional nodes beyond the half-box for centered
interpolation, and eight-point Lagrange interpolation in each direction.
Reflection and axis-permutation symmetry compress the table to 54,740 nodes,
each holding three doubles: 1,313,760 bytes (about 1.25 MiB). Only the periodic
correction is tabulated; the primary softened force is evaluated directly.

Table construction uses an independent double Ewald sum with alpha 2, real-space
radius 3.6, and reciprocal lattice radius 4. It does not use the production
FMM/PMMM coefficients. Numerical regressions check the interpolated correction
against explicit sums, including an alternative alpha, boundaries, near-zero
separations, and all sign/permutation symmetries. The table regressions require
component errors below `1e-11` for the unit-source correction in a unit box;
this is a tested tolerance, not a uniform analytic interpolation bound. Relative
accuracy cannot be guaranteed when the total reference force nearly cancels.
The diagnostic therefore retains absolute error fields.

The table remains in process memory; it creates no disk cache or checkpoint
payload. CUDA uses bounded temporary source batches and copies the same table
for each diagnostic invocation. Saved direct accelerations still scale with the
sample count, rather than requiring a force array for every particle.

The program prints a summary and writes `gravity_error.csv` under
`output.directory`. The CSV includes:

| Column | Meaning |
| --- | --- |
| `coarse_step` | Zero-based rung-0 step at the force evaluation |
| `time` | Simulation time of the sampled positions, before the kick |
| `scale_factor` | Scale factor at that force evaluation |
| `sample_count` | Number of sampled particles across all localities |
| `rms_absolute` | RMS magnitude of acceleration error |
| `rms_relative` | RMS error divided by RMS direct acceleration |
| `max_absolute` | Largest sampled acceleration-error magnitude |
| `max_relative` | Largest sampled relative acceleration error |
| `rms_reference` | RMS magnitude of the direct reference acceleration |
| `zero_reference_count` | Number of samples with exactly zero direct acceleration |

For the acceleration difference `e_i = g_solver,i - g_direct,i`, the absolute RMS
is `sqrt(sum(|e_i|^2) / sample_count)`. The relative RMS is
`sqrt(sum(|e_i|^2) / sum(|g_direct,i|^2))`; the maximum relative error is the largest
individual ratio `|e_i| / |g_direct,i|`. Exactly zero reference and error give a
relative error of zero; nonzero error against a zero reference gives infinity.
Absolute accelerations use the simulation's internal gravity units, before the
kick's time-integration factors. These are acceleration errors, not potential
errors.

Restarting appends to an existing CSV in the output directory. A restart in a new
directory starts a new report containing subsequent scheduled checks.

The diagnostic reports gravity solver accuracy at the sampled positions. It does
not estimate time-integration error. Its direct calculation costs work
proportional to the sample count times the total particle count, so use a modest
sample and an interval appropriate to the simulation size.
