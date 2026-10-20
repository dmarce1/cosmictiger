| binary | policy | backend | bucket | order | fixture | potential | kick ms (median) | min | max | pairs |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| baseline | auto | cuda | 64 | 7 | adaptive | 0 | 18.059 | 17.965 | 18.269 | 3 |
| baseline | auto | cuda | 64 | 7 | adaptive | 1 | 18.395 | 18.194 | 18.549 | 3 |
| baseline | auto | cuda | 64 | 7 | clustered | 0 | 8.514 | 8.125 | 8.550 | 3 |
| baseline | auto | cuda | 64 | 7 | clustered | 1 | 8.386 | 8.250 | 8.677 | 3 |
| baseline | auto | cuda | 64 | 7 | lattice | 0 | 32.591 | 32.348 | 33.070 | 3 |
| baseline | auto | cuda | 64 | 7 | lattice | 1 | 32.707 | 32.569 | 32.907 | 3 |
| ewald | auto | cuda | 64 | 7 | adaptive | 0 | 10.359 | 10.284 | 11.108 | 3 |
| ewald | auto | cuda | 64 | 7 | adaptive | 1 | 10.562 | 10.496 | 10.596 | 3 |
| ewald | auto | cuda | 64 | 7 | clustered | 0 | 5.524 | 5.502 | 5.618 | 3 |
| ewald | auto | cuda | 64 | 7 | clustered | 1 | 5.827 | 5.792 | 6.013 | 3 |
| ewald | auto | cuda | 64 | 7 | lattice | 0 | 14.364 | 14.359 | 14.416 | 3 |
| ewald | auto | cuda | 64 | 7 | lattice | 1 | 14.479 | 14.383 | 14.757 | 3 |
| head | auto | cuda | 64 | 7 | adaptive | 0 | 12.129 | 12.082 | 12.287 | 3 |
| head | auto | cuda | 64 | 7 | adaptive | 1 | 12.182 | 11.938 | 12.320 | 3 |
| head | auto | cuda | 64 | 7 | clustered | 0 | 5.590 | 5.505 | 5.768 | 3 |
| head | auto | cuda | 64 | 7 | clustered | 1 | 5.887 | 5.882 | 5.943 | 3 |
| head | auto | cuda | 64 | 7 | lattice | 0 | 14.770 | 14.586 | 14.795 | 3 |
| head | auto | cuda | 64 | 7 | lattice | 1 | 14.688 | 14.648 | 14.875 | 3 |

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
