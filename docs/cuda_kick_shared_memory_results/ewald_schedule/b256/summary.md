| binary | policy | backend | bucket | order | fixture | potential | kick ms (median) | min | max | pairs |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| baseline | auto | cuda | 256 | 7 | adaptive | 0 | 18.324 | 18.012 | 18.537 | 5 |
| baseline | auto | cuda | 256 | 7 | adaptive | 1 | 18.566 | 18.117 | 19.064 | 5 |
| baseline | auto | cuda | 256 | 7 | clustered | 0 | 6.842 | 6.791 | 6.998 | 5 |
| baseline | auto | cuda | 256 | 7 | clustered | 1 | 7.343 | 7.240 | 7.566 | 5 |
| baseline | auto | cuda | 256 | 7 | lattice | 0 | 11.369 | 11.284 | 12.174 | 5 |
| baseline | auto | cuda | 256 | 7 | lattice | 1 | 11.653 | 11.485 | 13.213 | 5 |
| ewald | auto | cuda | 256 | 7 | adaptive | 0 | 13.354 | 13.131 | 13.805 | 5 |
| ewald | auto | cuda | 256 | 7 | adaptive | 1 | 13.838 | 13.718 | 14.354 | 5 |
| ewald | auto | cuda | 256 | 7 | clustered | 0 | 5.688 | 5.659 | 5.749 | 5 |
| ewald | auto | cuda | 256 | 7 | clustered | 1 | 6.098 | 5.987 | 6.164 | 5 |
| ewald | auto | cuda | 256 | 7 | lattice | 0 | 9.735 | 9.666 | 9.805 | 5 |
| ewald | auto | cuda | 256 | 7 | lattice | 1 | 9.998 | 9.853 | 10.133 | 5 |
| head | auto | cuda | 256 | 7 | adaptive | 0 | 17.010 | 16.510 | 17.704 | 5 |
| head | auto | cuda | 256 | 7 | adaptive | 1 | 17.815 | 17.648 | 18.047 | 5 |
| head | auto | cuda | 256 | 7 | clustered | 0 | 5.672 | 5.627 | 5.717 | 5 |
| head | auto | cuda | 256 | 7 | clustered | 1 | 6.032 | 6.004 | 6.091 | 5 |
| head | auto | cuda | 256 | 7 | lattice | 0 | 10.413 | 10.340 | 10.541 | 5 |
| head | auto | cuda | 256 | 7 | lattice | 1 | 10.722 | 10.604 | 11.021 | 5 |

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
