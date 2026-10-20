| binary | policy | backend | bucket | order | fixture | potential | kick ms (median) | min | max | pairs |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| baseline | auto | cuda | 256 | 7 | adaptive | 0 | 18.132 | 18.026 | 18.230 | 3 |
| baseline | auto | cuda | 256 | 7 | adaptive | 1 | 18.584 | 18.071 | 18.679 | 3 |
| baseline | auto | cuda | 256 | 7 | clustered | 0 | 6.780 | 6.751 | 7.285 | 3 |
| baseline | auto | cuda | 256 | 7 | clustered | 1 | 7.344 | 7.269 | 7.448 | 3 |
| baseline | auto | cuda | 256 | 7 | lattice | 0 | 11.413 | 11.366 | 12.522 | 3 |
| baseline | auto | cuda | 256 | 7 | lattice | 1 | 11.561 | 11.516 | 13.312 | 3 |
| ewald | auto | cuda | 256 | 7 | adaptive | 0 | 13.300 | 13.220 | 13.999 | 3 |
| ewald | auto | cuda | 256 | 7 | adaptive | 1 | 14.528 | 14.191 | 14.888 | 3 |
| ewald | auto | cuda | 256 | 7 | clustered | 0 | 5.730 | 5.689 | 5.770 | 3 |
| ewald | auto | cuda | 256 | 7 | clustered | 1 | 6.159 | 6.079 | 6.269 | 3 |
| ewald | auto | cuda | 256 | 7 | lattice | 0 | 9.939 | 9.764 | 9.982 | 3 |
| ewald | auto | cuda | 256 | 7 | lattice | 1 | 10.032 | 9.801 | 10.285 | 3 |
| head | auto | cuda | 256 | 7 | adaptive | 0 | 17.130 | 17.074 | 17.448 | 3 |
| head | auto | cuda | 256 | 7 | adaptive | 1 | 17.718 | 17.698 | 18.013 | 3 |
| head | auto | cuda | 256 | 7 | clustered | 0 | 5.687 | 5.660 | 5.695 | 3 |
| head | auto | cuda | 256 | 7 | clustered | 1 | 6.079 | 6.045 | 6.089 | 3 |
| head | auto | cuda | 256 | 7 | lattice | 0 | 10.441 | 10.387 | 10.460 | 3 |
| head | auto | cuda | 256 | 7 | lattice | 1 | 10.617 | 10.539 | 10.784 | 3 |

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
