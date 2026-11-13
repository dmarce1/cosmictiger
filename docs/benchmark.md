# Static FMM benchmark

Select `run.scenario=benchmark`. The scenario reads particle positions without
cosmological initialization or time evolution and uses the production periodic
FMM on the backend selected by `gravity.cuda`.

| Option | Meaning |
| --- | --- |
| `benchmark.testFiles` | Comma-separated text filenames, relative to the working directory. |
| `benchmark.openingAngles` | Comma-separated angles in `(0,1]`, exactly one per file. |
| `benchmark.size` | Integer `n >= 1`; create `n^3` copies of each input distribution. Default 1. |
| `benchmark.repeats` | Positive number of timed solves per file. Default 3. |

Input has three whitespace-separated coordinates per particle, each in `[0,1)`.
Blank lines and `#` comments, including the exporter's redshift header, are ignored.
For tile indices `i,j,k` in `[0,n)`, positions become
`((x+i)/n, (y+j)/n, (z+k)/n)`. All particles are equal mass and active;
velocities and positions are not evolved. Softening uses the normal configured
mean-particle-spacing convention, recomputed for each tiled particle count.

Loading/tiling and initial distributed decomposition are timed separately.
Every repetition measures a fresh tree/multipole build, the completed force and
potential solve (including communication and GPU completion when enabled), and
tree/cache cleanup. No warmup solve is silently removed; the first repetition
can include lazy initialization. Counter reset/read and output are outside the
measured total. Existing counters supply aggregate CPU+GPU FLOP estimates across
all localities, separated into single, double and unclassified precision; these
are source-operation counts, not hardware instruction counters. FMA counts as two.
Each case prints its input and setup details, followed by aligned columns of
per-repeat timings in seconds and total GFLOP/s. Rows are printed as solves
finish, followed by a mean row for each timed phase and aggregate GFLOP/s.
The summary reports minimum, median, mean and maximum elapsed time and prints
the exact total FLOPs for one solve once, since the particle distribution and
work are unchanged across repetitions. The displayed count is not summed over
repetitions.
It also reports the total number of tree nodes across all localities, including
internal nodes and leaves, rather than allocated arena capacity. The final table
shows the particle count, leaf count and average particles per leaf (total
particles divided by all leaves, including empty leaves).

The prepared `configurations/benchmark.ini` uses 32^3-particle SILO exports at
z=0.117408174187551, 2.18800951546296 and 22.6499379613608: the snapshots strictly
above z=0,2,20. Their angles are 0.7,0.55,0.4. With `size=8`, each case contains
16,777,216 particles and runs 10 repeats. Run from the repository root, supplying
the executable path:

```sh
/path/to/cosmictiger --configuration=configurations/benchmark.ini --hpx:threads=2
```

The preset selects CUDA gravity. Override `--gravity.cuda=false` for CPU gravity.
The benchmark does not write evolution snapshots
or checkpoints. Existing startup/profile output still uses `output.directory`.
