| binary | policy | backend | bucket | order | fixture | potential | kick ms (median) | min | max | pairs |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| baseline | auto | cuda | 64 | 7 | adaptive | 0 | 18.367 | 18.225 | 18.505 | 3 |
| baseline | auto | cuda | 64 | 7 | adaptive | 1 | 18.525 | 18.463 | 18.539 | 3 |
| baseline | auto | cuda | 64 | 7 | clustered | 0 | 8.481 | 8.261 | 8.619 | 3 |
| baseline | auto | cuda | 64 | 7 | clustered | 1 | 8.488 | 8.441 | 8.611 | 3 |
| baseline | auto | cuda | 64 | 7 | lattice | 0 | 32.341 | 32.236 | 32.952 | 3 |
| baseline | auto | cuda | 64 | 7 | lattice | 1 | 32.469 | 32.322 | 32.950 | 3 |
| ewald | auto | cuda | 64 | 7 | adaptive | 0 | 10.850 | 10.120 | 12.253 | 3 |
| ewald | auto | cuda | 64 | 7 | adaptive | 1 | 10.557 | 10.168 | 11.195 | 3 |
| ewald | auto | cuda | 64 | 7 | clustered | 0 | 5.562 | 5.517 | 5.741 | 3 |
| ewald | auto | cuda | 64 | 7 | clustered | 1 | 5.850 | 5.767 | 5.851 | 3 |
| ewald | auto | cuda | 64 | 7 | lattice | 0 | 14.197 | 14.132 | 14.261 | 3 |
| ewald | auto | cuda | 64 | 7 | lattice | 1 | 14.434 | 14.356 | 14.496 | 3 |
| head | auto | cuda | 64 | 7 | adaptive | 0 | 12.258 | 12.150 | 12.696 | 3 |
| head | auto | cuda | 64 | 7 | adaptive | 1 | 12.261 | 12.212 | 12.315 | 3 |
| head | auto | cuda | 64 | 7 | clustered | 0 | 5.574 | 5.498 | 5.602 | 3 |
| head | auto | cuda | 64 | 7 | clustered | 1 | 5.849 | 5.824 | 5.854 | 3 |
| head | auto | cuda | 64 | 7 | lattice | 0 | 14.632 | 14.519 | 14.691 | 3 |
| head | auto | cuda | 64 | 7 | lattice | 1 | 14.686 | 14.650 | 14.703 | 3 |

| binary | policy | backend | bucket | order | fixture | force rel L2 | force p99 | potential rel L2 | potential p99 |
|---|---|---|---:|---:|---|---:|---:|---:|---:|
| baseline | auto | cuda | 64 | 7 | lattice | 3.418e-01 | 1.857e+00 | 3.613e-02 | 1.181e-01 |
| baseline | auto | cuda | 64 | 7 | clustered | 4.815e-03 | 2.631e-02 | 5.752e-04 | 8.093e-03 |
| baseline | auto | cuda | 64 | 7 | adaptive | 2.111e-03 | 4.085e-02 | 1.025e-04 | 2.660e-02 |
| head | auto | cuda | 64 | 7 | lattice | 3.644e-01 | 2.101e+00 | 4.476e-02 | 1.435e-01 |
| head | auto | cuda | 64 | 7 | clustered | 5.305e-03 | 2.563e-02 | 1.096e-03 | 1.145e-02 |
| head | auto | cuda | 64 | 7 | adaptive | 2.262e-03 | 5.030e-02 | 1.163e-04 | 2.580e-02 |
| ewald | auto | cuda | 64 | 7 | lattice | 3.644e-01 | 2.101e+00 | 4.476e-02 | 1.435e-01 |
| ewald | auto | cuda | 64 | 7 | clustered | 5.305e-03 | 2.563e-02 | 1.096e-03 | 1.145e-02 |
| ewald | auto | cuda | 64 | 7 | adaptive | 2.262e-03 | 5.030e-02 | 1.163e-04 | 2.580e-02 |

| binary | policy | backend | bucket | order | kernel policy line |
|---|---|---|---:|---:|---|
| baseline | auto | cuda | 64 | 7 | `Occupancy is 6 shmem size = 15184 numregs = 255` |
| ewald | auto | cuda | 64 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8048 bytes: control 648, expansion shared, forces shared tile 64, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes; list slab 145792 bytes per block (4096 entries per list)` |
| head | auto | cuda | 64 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8048 bytes: control 648, expansion shared, forces shared tile 64, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes; list slab 145792 bytes per block (4096 entries per list)` |

| comparison | binary | policy | bucket | order | fixture | acceleration rel L2 | acceleration max pointwise | potential rel L2 |
|---|---|---|---:|---:|---|---:|---:|---:|
| vs_baseline | ewald | auto | 64 | 7 | adaptive | 1.669e-03 | 1.680e-01 | 7.101e-05 |
| vs_baseline | ewald | auto | 64 | 7 | clustered | 2.125e-03 | 9.649e-02 | 7.341e-04 |
| vs_baseline | ewald | auto | 64 | 7 | lattice | 1.200e-01 | 4.639e+00 | 2.273e-02 |
| vs_baseline | head | auto | 64 | 7 | adaptive | 1.669e-03 | 1.680e-01 | 7.101e-05 |
| vs_baseline | head | auto | 64 | 7 | clustered | 2.125e-03 | 9.649e-02 | 7.341e-04 |
| vs_baseline | head | auto | 64 | 7 | lattice | 1.200e-01 | 4.639e+00 | 2.273e-02 |
