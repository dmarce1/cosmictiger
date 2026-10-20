| binary | policy | backend | bucket | order | fixture | potential | kick ms (median) | min | max | pairs |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| baseline | auto | cuda | 256 | 7 | adaptive | 0 | 18.257 | 17.970 | 18.338 | 3 |
| baseline | auto | cuda | 256 | 7 | adaptive | 1 | 18.471 | 18.193 | 18.627 | 3 |
| baseline | auto | cuda | 256 | 7 | clustered | 0 | 7.066 | 7.040 | 7.235 | 3 |
| baseline | auto | cuda | 256 | 7 | clustered | 1 | 7.427 | 7.383 | 7.457 | 3 |
| baseline | auto | cuda | 256 | 7 | lattice | 0 | 11.372 | 11.268 | 13.108 | 3 |
| baseline | auto | cuda | 256 | 7 | lattice | 1 | 11.761 | 11.722 | 13.127 | 3 |
| ewald | auto | cuda | 256 | 7 | adaptive | 0 | 13.264 | 13.199 | 13.796 | 3 |
| ewald | auto | cuda | 256 | 7 | adaptive | 1 | 14.247 | 14.028 | 14.356 | 3 |
| ewald | auto | cuda | 256 | 7 | clustered | 0 | 5.711 | 5.641 | 5.742 | 3 |
| ewald | auto | cuda | 256 | 7 | clustered | 1 | 6.128 | 6.099 | 6.137 | 3 |
| ewald | auto | cuda | 256 | 7 | lattice | 0 | 9.753 | 9.690 | 9.783 | 3 |
| ewald | auto | cuda | 256 | 7 | lattice | 1 | 10.016 | 9.978 | 10.063 | 3 |
| head | auto | cuda | 256 | 7 | adaptive | 0 | 17.319 | 17.244 | 17.649 | 3 |
| head | auto | cuda | 256 | 7 | adaptive | 1 | 17.811 | 17.537 | 18.092 | 3 |
| head | auto | cuda | 256 | 7 | clustered | 0 | 5.806 | 5.766 | 6.202 | 3 |
| head | auto | cuda | 256 | 7 | clustered | 1 | 6.156 | 6.143 | 6.300 | 3 |
| head | auto | cuda | 256 | 7 | lattice | 0 | 10.507 | 10.405 | 10.513 | 3 |
| head | auto | cuda | 256 | 7 | lattice | 1 | 10.735 | 10.696 | 10.836 | 3 |

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
