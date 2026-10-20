| binary | policy | backend | bucket | order | fixture | potential | kick ms (median) | min | max | pairs |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| baseline | auto | cuda | 512 | 7 | adaptive | 0 | 28.442 | 28.078 | 29.049 | 5 |
| baseline | auto | cuda | 512 | 7 | adaptive | 1 | 29.607 | 29.490 | 29.734 | 5 |
| baseline | auto | cuda | 512 | 7 | clustered | 0 | 9.651 | 9.573 | 9.920 | 5 |
| baseline | auto | cuda | 512 | 7 | clustered | 1 | 10.432 | 9.949 | 10.759 | 5 |
| baseline | auto | cuda | 512 | 7 | lattice | 0 | 10.340 | 10.188 | 11.784 | 5 |
| baseline | auto | cuda | 512 | 7 | lattice | 1 | 10.888 | 10.523 | 11.309 | 5 |
| ewald | auto | cuda | 512 | 7 | adaptive | 0 | 19.649 | 19.543 | 19.759 | 5 |
| ewald | auto | cuda | 512 | 7 | adaptive | 1 | 20.575 | 20.211 | 20.687 | 5 |
| ewald | auto | cuda | 512 | 7 | clustered | 0 | 5.223 | 5.161 | 5.327 | 5 |
| ewald | auto | cuda | 512 | 7 | clustered | 1 | 6.042 | 5.997 | 6.191 | 5 |
| ewald | auto | cuda | 512 | 7 | lattice | 0 | 7.630 | 7.516 | 7.757 | 5 |
| ewald | auto | cuda | 512 | 7 | lattice | 1 | 8.106 | 8.060 | 8.433 | 5 |
| head | auto | cuda | 512 | 7 | adaptive | 0 | 26.385 | 26.039 | 27.000 | 5 |
| head | auto | cuda | 512 | 7 | adaptive | 1 | 27.187 | 26.725 | 27.406 | 5 |
| head | auto | cuda | 512 | 7 | clustered | 0 | 5.469 | 5.333 | 5.610 | 5 |
| head | auto | cuda | 512 | 7 | clustered | 1 | 6.086 | 6.032 | 6.321 | 5 |
| head | auto | cuda | 512 | 7 | lattice | 0 | 9.244 | 8.974 | 9.701 | 5 |
| head | auto | cuda | 512 | 7 | lattice | 1 | 10.041 | 9.653 | 10.068 | 5 |

| binary | policy | backend | bucket | order | fixture | force rel L2 | force p99 | potential rel L2 | potential p99 |
|---|---|---|---:|---:|---|---:|---:|---:|---:|
| baseline | auto | cuda | 512 | 7 | lattice | 3.418e-01 | 1.857e+00 | 3.613e-02 | 1.180e-01 |
| baseline | auto | cuda | 512 | 7 | clustered | 1.767e-03 | 2.301e-02 | 5.584e-04 | 7.015e-03 |
| baseline | auto | cuda | 512 | 7 | adaptive | 1.001e-03 | 4.085e-02 | 9.754e-05 | 2.660e-02 |
| head | auto | cuda | 512 | 7 | lattice | 3.644e-01 | 2.101e+00 | 4.476e-02 | 1.435e-01 |
| head | auto | cuda | 512 | 7 | clustered | 2.146e-03 | 2.165e-02 | 1.120e-03 | 1.132e-02 |
| head | auto | cuda | 512 | 7 | adaptive | 1.015e-03 | 6.030e-02 | 1.108e-04 | 2.637e-02 |
| ewald | auto | cuda | 512 | 7 | lattice | 3.644e-01 | 2.101e+00 | 4.476e-02 | 1.435e-01 |
| ewald | auto | cuda | 512 | 7 | clustered | 2.146e-03 | 2.165e-02 | 1.120e-03 | 1.132e-02 |
| ewald | auto | cuda | 512 | 7 | adaptive | 1.015e-03 | 6.030e-02 | 1.108e-04 | 2.637e-02 |

| binary | policy | backend | bucket | order | kernel policy line |
|---|---|---|---:|---:|---|
| baseline | auto | cuda | 512 | 7 | `Occupancy is 6 shmem size = 15184 numregs = 255` |
| ewald | auto | cuda | 512 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 11760 bytes: control 648, expansion shared, forces shared tile 512, particle tile 224 (without mass plane), multipole tile 16 x 160 bytes; list slab 145792 bytes per block (4096 entries per list)` |
| head | auto | cuda | 512 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 11760 bytes: control 648, expansion shared, forces shared tile 512, particle tile 224 (without mass plane), multipole tile 16 x 160 bytes; list slab 145792 bytes per block (4096 entries per list)` |

| comparison | binary | policy | bucket | order | fixture | acceleration rel L2 | acceleration max pointwise | potential rel L2 |
|---|---|---|---:|---:|---|---:|---:|---:|
| vs_baseline | ewald | auto | 512 | 7 | adaptive | 9.163e-04 | 2.336e-01 | 6.926e-05 |
| vs_baseline | ewald | auto | 512 | 7 | clustered | 1.710e-03 | 9.544e-03 | 7.223e-04 |
| vs_baseline | ewald | auto | 512 | 7 | lattice | 1.200e-01 | 4.640e+00 | 2.273e-02 |
| vs_baseline | head | auto | 512 | 7 | adaptive | 9.163e-04 | 2.336e-01 | 6.926e-05 |
| vs_baseline | head | auto | 512 | 7 | clustered | 1.710e-03 | 9.544e-03 | 7.223e-04 |
| vs_baseline | head | auto | 512 | 7 | lattice | 1.200e-01 | 4.640e+00 | 2.273e-02 |
