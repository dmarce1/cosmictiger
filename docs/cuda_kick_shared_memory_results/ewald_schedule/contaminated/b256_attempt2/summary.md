| binary | policy | backend | bucket | order | fixture | potential | kick ms (median) | min | max | pairs |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| baseline | auto | cuda | 256 | 7 | adaptive | 0 | 18.568 | 18.526 | 18.638 | 3 |
| baseline | auto | cuda | 256 | 7 | adaptive | 1 | 18.682 | 18.157 | 19.157 | 3 |
| baseline | auto | cuda | 256 | 7 | clustered | 0 | 7.208 | 6.790 | 7.236 | 3 |
| baseline | auto | cuda | 256 | 7 | clustered | 1 | 7.267 | 7.257 | 7.397 | 3 |
| baseline | auto | cuda | 256 | 7 | lattice | 0 | 11.311 | 11.179 | 12.949 | 3 |
| baseline | auto | cuda | 256 | 7 | lattice | 1 | 12.128 | 11.881 | 13.211 | 3 |
| ewald | auto | cuda | 256 | 7 | adaptive | 0 | 13.675 | 13.481 | 23.302 | 3 |
| ewald | auto | cuda | 256 | 7 | adaptive | 1 | 13.841 | 13.717 | 23.236 | 3 |
| ewald | auto | cuda | 256 | 7 | clustered | 0 | 5.696 | 5.677 | 5.812 | 3 |
| ewald | auto | cuda | 256 | 7 | clustered | 1 | 6.144 | 6.093 | 11.338 | 3 |
| ewald | auto | cuda | 256 | 7 | lattice | 0 | 9.783 | 9.652 | 9.795 | 3 |
| ewald | auto | cuda | 256 | 7 | lattice | 1 | 9.784 | 9.753 | 10.107 | 3 |
| head | auto | cuda | 256 | 7 | adaptive | 0 | 17.370 | 16.963 | 17.508 | 3 |
| head | auto | cuda | 256 | 7 | adaptive | 1 | 17.770 | 17.636 | 17.969 | 3 |
| head | auto | cuda | 256 | 7 | clustered | 0 | 5.711 | 5.605 | 5.754 | 3 |
| head | auto | cuda | 256 | 7 | clustered | 1 | 6.104 | 6.052 | 6.136 | 3 |
| head | auto | cuda | 256 | 7 | lattice | 0 | 10.432 | 10.276 | 10.732 | 3 |
| head | auto | cuda | 256 | 7 | lattice | 1 | 10.863 | 10.577 | 11.144 | 3 |

| binary | policy | backend | bucket | order | fixture | force rel L2 | force p99 | potential rel L2 | potential p99 |
|---|---|---|---:|---:|---|---:|---:|---:|---:|
| baseline | auto | cuda | 256 | 7 | lattice | 3.418e-01 | 1.857e+00 | 3.613e-02 | 1.181e-01 |
| baseline | auto | cuda | 256 | 7 | clustered | 1.884e-03 | 2.339e-02 | 5.309e-04 | 6.867e-03 |
| baseline | auto | cuda | 256 | 7 | adaptive | 1.317e-03 | 4.085e-02 | 9.639e-05 | 2.661e-02 |
| head | auto | cuda | 256 | 7 | lattice | 3.644e-01 | 2.101e+00 | 4.476e-02 | 1.435e-01 |
| head | auto | cuda | 256 | 7 | clustered | 2.677e-03 | 2.816e-02 | 1.153e-03 | 1.118e-02 |
| head | auto | cuda | 256 | 7 | adaptive | 1.295e-03 | 6.030e-02 | 1.114e-04 | 2.637e-02 |
| ewald | auto | cuda | 256 | 7 | lattice | 3.644e-01 | 2.101e+00 | 4.476e-02 | 1.435e-01 |
| ewald | auto | cuda | 256 | 7 | clustered | 2.677e-03 | 2.816e-02 | 1.153e-03 | 1.118e-02 |
| ewald | auto | cuda | 256 | 7 | adaptive | 1.295e-03 | 6.030e-02 | 1.114e-04 | 2.637e-02 |

| binary | policy | backend | bucket | order | kernel policy line |
|---|---|---|---:|---:|---|
| baseline | auto | cuda | 256 | 7 | `Occupancy is 6 shmem size = 15184 numregs = 255` |
| ewald | auto | cuda | 256 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 11120 bytes: control 648, expansion shared, forces shared tile 256, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes; list slab 145792 bytes per block (4096 entries per list)` |
| head | auto | cuda | 256 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 11120 bytes: control 648, expansion shared, forces shared tile 256, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes; list slab 145792 bytes per block (4096 entries per list)` |

| comparison | binary | policy | bucket | order | fixture | acceleration rel L2 | acceleration max pointwise | potential rel L2 |
|---|---|---|---:|---:|---|---:|---:|---:|
| vs_baseline | ewald | auto | 256 | 7 | adaptive | 1.246e-03 | 2.336e-01 | 7.487e-05 |
| vs_baseline | ewald | auto | 256 | 7 | clustered | 1.848e-03 | 1.767e-01 | 7.305e-04 |
| vs_baseline | ewald | auto | 256 | 7 | lattice | 1.200e-01 | 4.639e+00 | 2.273e-02 |
| vs_baseline | head | auto | 256 | 7 | adaptive | 1.246e-03 | 2.336e-01 | 7.487e-05 |
| vs_baseline | head | auto | 256 | 7 | clustered | 1.848e-03 | 1.767e-01 | 7.305e-04 |
| vs_baseline | head | auto | 256 | 7 | lattice | 1.200e-01 | 4.639e+00 | 2.273e-02 |
