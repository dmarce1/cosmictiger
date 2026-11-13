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
Each row reports counts and total GFLOP/s. Summaries report minimum, median, mean
and maximum elapsed time, summed counts and counts divided by summed time.

The prepared `configurations/benchmark.ini` uses 32^3-particle SILO exports at
z=22.6499379613608, 2.18800951546296 and 0.117408174187551: the snapshots strictly
above z=20,2,0. Their angles are 0.4,0.55,0.7. With `size=4`, each case contains
2,097,152 particles. Run from the repository root, supplying the executable path:

```sh
/path/to/cosmictiger --configuration=configurations/benchmark.ini --hpx:threads=2
```

The preset selects CPU gravity. Override `--gravity.cuda=true` for a CUDA build
when GPU resources are available. The benchmark does not write evolution snapshots
or checkpoints. Existing startup/profile output still uses `output.directory`.
