| binary | policy | backend | bucket | order | fixture | potential | kick ms (median) | min | max | pairs |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| baseline | auto | cuda | 512 | 7 | adaptive | 0 | 28.906 | 28.770 | 29.568 | 3 |
| baseline | auto | cuda | 512 | 7 | adaptive | 1 | 29.604 | 29.243 | 29.637 | 3 |
| baseline | auto | cuda | 512 | 7 | clustered | 0 | 9.687 | 9.549 | 9.711 | 3 |
| baseline | auto | cuda | 512 | 7 | clustered | 1 | 10.620 | 10.390 | 10.971 | 3 |
| baseline | auto | cuda | 512 | 7 | lattice | 0 | 10.746 | 10.252 | 12.021 | 3 |
| baseline | auto | cuda | 512 | 7 | lattice | 1 | 11.116 | 10.884 | 11.363 | 3 |
| ewald | auto | cuda | 512 | 7 | adaptive | 0 | 19.439 | 19.370 | 19.696 | 3 |
| ewald | auto | cuda | 512 | 7 | adaptive | 1 | 20.115 | 20.092 | 20.388 | 3 |
| ewald | auto | cuda | 512 | 7 | clustered | 0 | 5.295 | 5.160 | 5.325 | 3 |
| ewald | auto | cuda | 512 | 7 | clustered | 1 | 5.947 | 5.896 | 6.019 | 3 |
| ewald | auto | cuda | 512 | 7 | lattice | 0 | 7.570 | 7.487 | 7.650 | 3 |
| ewald | auto | cuda | 512 | 7 | lattice | 1 | 8.197 | 8.189 | 8.221 | 3 |
| head | auto | cuda | 512 | 7 | adaptive | 0 | 26.274 | 25.997 | 26.600 | 3 |
| head | auto | cuda | 512 | 7 | adaptive | 1 | 27.221 | 27.054 | 27.351 | 3 |
| head | auto | cuda | 512 | 7 | clustered | 0 | 5.646 | 5.349 | 5.756 | 3 |
| head | auto | cuda | 512 | 7 | clustered | 1 | 6.313 | 5.987 | 6.387 | 3 |
| head | auto | cuda | 512 | 7 | lattice | 0 | 9.506 | 9.391 | 9.512 | 3 |
| head | auto | cuda | 512 | 7 | lattice | 1 | 9.796 | 9.607 | 10.219 | 3 |

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
