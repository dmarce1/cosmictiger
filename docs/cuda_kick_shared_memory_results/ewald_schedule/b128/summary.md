| binary | policy | backend | bucket | order | fixture | potential | kick ms (median) | min | max | pairs |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| baseline | auto | cuda | 128 | 7 | adaptive | 0 | 16.121 | 16.103 | 16.125 | 3 |
| baseline | auto | cuda | 128 | 7 | adaptive | 1 | 16.262 | 16.250 | 16.379 | 3 |
| baseline | auto | cuda | 128 | 7 | clustered | 0 | 6.309 | 6.083 | 6.616 | 3 |
| baseline | auto | cuda | 128 | 7 | clustered | 1 | 6.411 | 6.404 | 6.490 | 3 |
| baseline | auto | cuda | 128 | 7 | lattice | 0 | 14.746 | 14.597 | 15.153 | 3 |
| baseline | auto | cuda | 128 | 7 | lattice | 1 | 15.010 | 14.881 | 15.566 | 3 |
| ewald | auto | cuda | 128 | 7 | adaptive | 0 | 10.513 | 10.286 | 10.747 | 3 |
| ewald | auto | cuda | 128 | 7 | adaptive | 1 | 10.805 | 10.778 | 10.875 | 3 |
| ewald | auto | cuda | 128 | 7 | clustered | 0 | 6.278 | 6.234 | 6.279 | 3 |
| ewald | auto | cuda | 128 | 7 | clustered | 1 | 6.548 | 6.541 | 6.630 | 3 |
| ewald | auto | cuda | 128 | 7 | lattice | 0 | 14.337 | 14.313 | 14.413 | 3 |
| ewald | auto | cuda | 128 | 7 | lattice | 1 | 14.543 | 14.485 | 14.579 | 3 |
| head | auto | cuda | 128 | 7 | adaptive | 0 | 12.389 | 12.246 | 12.891 | 3 |
| head | auto | cuda | 128 | 7 | adaptive | 1 | 12.717 | 12.469 | 12.833 | 3 |
| head | auto | cuda | 128 | 7 | clustered | 0 | 6.295 | 6.217 | 6.297 | 3 |
| head | auto | cuda | 128 | 7 | clustered | 1 | 6.667 | 6.457 | 6.673 | 3 |
| head | auto | cuda | 128 | 7 | lattice | 0 | 14.736 | 14.733 | 14.834 | 3 |
| head | auto | cuda | 128 | 7 | lattice | 1 | 14.796 | 14.733 | 14.837 | 3 |

| binary | policy | backend | bucket | order | fixture | force rel L2 | force p99 | potential rel L2 | potential p99 |
|---|---|---|---:|---:|---|---:|---:|---:|---:|
| baseline | auto | cuda | 128 | 7 | lattice | 3.418e-01 | 1.857e+00 | 3.613e-02 | 1.181e-01 |
| baseline | auto | cuda | 128 | 7 | clustered | 2.241e-03 | 1.023e-02 | 5.016e-04 | 8.356e-03 |
| baseline | auto | cuda | 128 | 7 | adaptive | 1.770e-03 | 4.085e-02 | 1.035e-04 | 2.660e-02 |
| head | auto | cuda | 128 | 7 | lattice | 3.644e-01 | 2.101e+00 | 4.476e-02 | 1.435e-01 |
| head | auto | cuda | 128 | 7 | clustered | 2.962e-03 | 1.798e-02 | 1.017e-03 | 1.036e-02 |
| head | auto | cuda | 128 | 7 | adaptive | 1.869e-03 | 6.030e-02 | 1.191e-04 | 2.637e-02 |
| ewald | auto | cuda | 128 | 7 | lattice | 3.644e-01 | 2.101e+00 | 4.476e-02 | 1.435e-01 |
| ewald | auto | cuda | 128 | 7 | clustered | 2.962e-03 | 1.798e-02 | 1.017e-03 | 1.036e-02 |
| ewald | auto | cuda | 128 | 7 | adaptive | 1.869e-03 | 6.030e-02 | 1.191e-04 | 2.637e-02 |

| binary | policy | backend | bucket | order | kernel policy line |
|---|---|---|---:|---:|---|
| baseline | auto | cuda | 128 | 7 | `Occupancy is 6 shmem size = 15184 numregs = 255` |
| ewald | auto | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 9072 bytes: control 648, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes; list slab 145792 bytes per block (4096 entries per list)` |
| head | auto | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 9072 bytes: control 648, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes; list slab 145792 bytes per block (4096 entries per list)` |

| comparison | binary | policy | bucket | order | fixture | acceleration rel L2 | acceleration max pointwise | potential rel L2 |
|---|---|---|---:|---:|---|---:|---:|---:|
| vs_baseline | ewald | auto | 128 | 7 | adaptive | 1.154e-03 | 2.336e-01 | 7.190e-05 |
| vs_baseline | ewald | auto | 128 | 7 | clustered | 2.486e-03 | 1.757e-01 | 7.306e-04 |
| vs_baseline | ewald | auto | 128 | 7 | lattice | 1.200e-01 | 4.639e+00 | 2.273e-02 |
| vs_baseline | head | auto | 128 | 7 | adaptive | 1.154e-03 | 2.336e-01 | 7.190e-05 |
| vs_baseline | head | auto | 128 | 7 | clustered | 2.486e-03 | 1.757e-01 | 7.306e-04 |
| vs_baseline | head | auto | 128 | 7 | lattice | 1.200e-01 | 4.639e+00 | 2.273e-02 |
