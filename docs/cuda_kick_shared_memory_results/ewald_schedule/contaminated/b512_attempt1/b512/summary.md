| binary | policy | backend | bucket | order | fixture | potential | kick ms (median) | min | max | pairs |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| baseline | auto | cuda | 512 | 7 | adaptive | 0 | 29.585 | 28.950 | 30.440 | 5 |
| baseline | auto | cuda | 512 | 7 | adaptive | 1 | 29.624 | 28.862 | 31.344 | 5 |
| baseline | auto | cuda | 512 | 7 | clustered | 0 | 9.937 | 9.534 | 10.543 | 5 |
| baseline | auto | cuda | 512 | 7 | clustered | 1 | 10.950 | 10.771 | 11.058 | 5 |
| baseline | auto | cuda | 512 | 7 | lattice | 0 | 10.343 | 10.008 | 12.381 | 5 |
| baseline | auto | cuda | 512 | 7 | lattice | 1 | 11.341 | 10.862 | 11.540 | 5 |
| ewald | auto | cuda | 512 | 7 | adaptive | 0 | 19.620 | 19.395 | 19.870 | 5 |
| ewald | auto | cuda | 512 | 7 | adaptive | 1 | 20.325 | 20.209 | 20.609 | 5 |
| ewald | auto | cuda | 512 | 7 | clustered | 0 | 5.488 | 5.276 | 5.679 | 5 |
| ewald | auto | cuda | 512 | 7 | clustered | 1 | 6.015 | 5.952 | 6.237 | 5 |
| ewald | auto | cuda | 512 | 7 | lattice | 0 | 7.616 | 7.534 | 7.814 | 5 |
| ewald | auto | cuda | 512 | 7 | lattice | 1 | 8.109 | 7.905 | 8.265 | 5 |
| head | auto | cuda | 512 | 7 | adaptive | 0 | 26.707 | 26.126 | 29.357 | 5 |
| head | auto | cuda | 512 | 7 | adaptive | 1 | 27.626 | 26.907 | 30.191 | 5 |
| head | auto | cuda | 512 | 7 | clustered | 0 | 5.405 | 5.355 | 5.633 | 5 |
| head | auto | cuda | 512 | 7 | clustered | 1 | 6.184 | 6.086 | 6.402 | 5 |
| head | auto | cuda | 512 | 7 | lattice | 0 | 9.411 | 9.311 | 10.556 | 5 |
| head | auto | cuda | 512 | 7 | lattice | 1 | 10.039 | 9.779 | 10.488 | 5 |

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
