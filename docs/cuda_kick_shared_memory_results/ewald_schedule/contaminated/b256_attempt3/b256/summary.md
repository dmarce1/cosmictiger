| binary | policy | backend | bucket | order | fixture | potential | kick ms (median) | min | max | pairs |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| baseline | auto | cuda | 256 | 7 | adaptive | 0 | 18.109 | 17.447 | 18.482 | 5 |
| baseline | auto | cuda | 256 | 7 | adaptive | 1 | 18.735 | 18.124 | 18.922 | 5 |
| baseline | auto | cuda | 256 | 7 | clustered | 0 | 6.904 | 6.859 | 6.985 | 5 |
| baseline | auto | cuda | 256 | 7 | clustered | 1 | 7.341 | 7.173 | 7.368 | 5 |
| baseline | auto | cuda | 256 | 7 | lattice | 0 | 11.354 | 11.279 | 13.722 | 5 |
| baseline | auto | cuda | 256 | 7 | lattice | 1 | 11.769 | 11.634 | 13.142 | 5 |
| ewald | auto | cuda | 256 | 7 | adaptive | 0 | 13.403 | 13.097 | 13.912 | 5 |
| ewald | auto | cuda | 256 | 7 | adaptive | 1 | 13.893 | 13.840 | 14.469 | 5 |
| ewald | auto | cuda | 256 | 7 | clustered | 0 | 5.727 | 5.640 | 5.822 | 5 |
| ewald | auto | cuda | 256 | 7 | clustered | 1 | 6.097 | 6.047 | 6.228 | 5 |
| ewald | auto | cuda | 256 | 7 | lattice | 0 | 9.699 | 9.562 | 9.794 | 5 |
| ewald | auto | cuda | 256 | 7 | lattice | 1 | 9.926 | 9.819 | 10.037 | 5 |
| head | auto | cuda | 256 | 7 | adaptive | 0 | 17.353 | 16.915 | 17.944 | 5 |
| head | auto | cuda | 256 | 7 | adaptive | 1 | 17.929 | 17.353 | 18.299 | 5 |
| head | auto | cuda | 256 | 7 | clustered | 0 | 5.697 | 5.646 | 5.719 | 5 |
| head | auto | cuda | 256 | 7 | clustered | 1 | 6.077 | 6.011 | 6.173 | 5 |
| head | auto | cuda | 256 | 7 | lattice | 0 | 10.385 | 10.247 | 10.520 | 5 |
| head | auto | cuda | 256 | 7 | lattice | 1 | 10.686 | 10.418 | 10.726 | 5 |

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
