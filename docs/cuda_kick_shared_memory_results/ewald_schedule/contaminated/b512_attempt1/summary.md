| binary | policy | backend | bucket | order | fixture | potential | kick ms (median) | min | max | pairs |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| baseline | auto | cuda | 512 | 7 | adaptive | 0 | 29.786 | 29.110 | 30.660 | 3 |
| baseline | auto | cuda | 512 | 7 | adaptive | 1 | 29.953 | 28.914 | 31.475 | 3 |
| baseline | auto | cuda | 512 | 7 | clustered | 0 | 10.740 | 9.602 | 12.632 | 3 |
| baseline | auto | cuda | 512 | 7 | clustered | 1 | 11.046 | 10.820 | 11.550 | 3 |
| baseline | auto | cuda | 512 | 7 | lattice | 0 | 11.900 | 10.503 | 12.417 | 3 |
| baseline | auto | cuda | 512 | 7 | lattice | 1 | 11.403 | 11.305 | 11.914 | 3 |
| ewald | auto | cuda | 512 | 7 | adaptive | 0 | 20.863 | 19.973 | 21.105 | 3 |
| ewald | auto | cuda | 512 | 7 | adaptive | 1 | 21.108 | 20.399 | 21.519 | 3 |
| ewald | auto | cuda | 512 | 7 | clustered | 0 | 5.309 | 5.240 | 5.353 | 3 |
| ewald | auto | cuda | 512 | 7 | clustered | 1 | 6.060 | 6.038 | 6.124 | 3 |
| ewald | auto | cuda | 512 | 7 | lattice | 0 | 7.590 | 7.576 | 7.660 | 3 |
| ewald | auto | cuda | 512 | 7 | lattice | 1 | 7.969 | 7.932 | 8.350 | 3 |
| head | auto | cuda | 512 | 7 | adaptive | 0 | 27.079 | 26.350 | 29.201 | 3 |
| head | auto | cuda | 512 | 7 | adaptive | 1 | 28.981 | 26.998 | 30.080 | 3 |
| head | auto | cuda | 512 | 7 | clustered | 0 | 5.612 | 5.434 | 5.662 | 3 |
| head | auto | cuda | 512 | 7 | clustered | 1 | 6.206 | 6.069 | 6.315 | 3 |
| head | auto | cuda | 512 | 7 | lattice | 0 | 9.657 | 9.336 | 10.079 | 3 |
| head | auto | cuda | 512 | 7 | lattice | 1 | 10.798 | 9.953 | 10.834 | 3 |

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
