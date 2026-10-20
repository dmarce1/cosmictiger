| binary | policy | backend | bucket | order | fixture | potential | kick ms (median) | min | max | pairs |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| baseline | auto | cuda | 512 | 7 | adaptive | 0 | 29.114 | 28.312 | 30.916 | 3 |
| baseline | auto | cuda | 512 | 7 | adaptive | 1 | 29.710 | 29.708 | 31.983 | 3 |
| baseline | auto | cuda | 512 | 7 | clustered | 0 | 9.795 | 9.783 | 10.281 | 3 |
| baseline | auto | cuda | 512 | 7 | clustered | 1 | 10.880 | 10.348 | 12.730 | 3 |
| baseline | auto | cuda | 512 | 7 | lattice | 0 | 10.780 | 10.427 | 11.217 | 3 |
| baseline | auto | cuda | 512 | 7 | lattice | 1 | 11.333 | 11.083 | 13.093 | 3 |
| ewald | auto | cuda | 512 | 7 | adaptive | 0 | 19.611 | 19.464 | 19.820 | 3 |
| ewald | auto | cuda | 512 | 7 | adaptive | 1 | 20.267 | 20.003 | 20.709 | 3 |
| ewald | auto | cuda | 512 | 7 | clustered | 0 | 5.338 | 5.285 | 5.409 | 3 |
| ewald | auto | cuda | 512 | 7 | clustered | 1 | 6.105 | 6.044 | 6.300 | 3 |
| ewald | auto | cuda | 512 | 7 | lattice | 0 | 7.666 | 7.557 | 7.731 | 3 |
| ewald | auto | cuda | 512 | 7 | lattice | 1 | 8.268 | 8.218 | 8.365 | 3 |
| head | auto | cuda | 512 | 7 | adaptive | 0 | 26.733 | 25.987 | 28.189 | 3 |
| head | auto | cuda | 512 | 7 | adaptive | 1 | 29.540 | 27.807 | 29.727 | 3 |
| head | auto | cuda | 512 | 7 | clustered | 0 | 5.386 | 5.377 | 5.409 | 3 |
| head | auto | cuda | 512 | 7 | clustered | 1 | 6.013 | 5.993 | 6.164 | 3 |
| head | auto | cuda | 512 | 7 | lattice | 0 | 9.246 | 9.023 | 10.185 | 3 |
| head | auto | cuda | 512 | 7 | lattice | 1 | 10.055 | 9.961 | 10.655 | 3 |

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
