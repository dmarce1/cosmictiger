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
