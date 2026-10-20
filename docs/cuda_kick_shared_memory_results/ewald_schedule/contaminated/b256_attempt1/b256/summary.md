| binary | policy | backend | bucket | order | fixture | potential | kick ms (median) | min | max | pairs |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| baseline | auto | cuda | 256 | 7 | adaptive | 0 | 17.998 | 17.756 | 18.508 | 5 |
| baseline | auto | cuda | 256 | 7 | adaptive | 1 | 18.716 | 18.535 | 18.746 | 5 |
| baseline | auto | cuda | 256 | 7 | clustered | 0 | 6.908 | 6.754 | 7.147 | 5 |
| baseline | auto | cuda | 256 | 7 | clustered | 1 | 7.256 | 7.226 | 7.471 | 5 |
| baseline | auto | cuda | 256 | 7 | lattice | 0 | 11.241 | 11.173 | 13.071 | 5 |
| baseline | auto | cuda | 256 | 7 | lattice | 1 | 11.652 | 11.468 | 12.843 | 5 |
| ewald | auto | cuda | 256 | 7 | adaptive | 0 | 13.355 | 13.142 | 13.608 | 5 |
| ewald | auto | cuda | 256 | 7 | adaptive | 1 | 13.778 | 13.671 | 13.857 | 5 |
| ewald | auto | cuda | 256 | 7 | clustered | 0 | 5.702 | 5.660 | 5.845 | 5 |
| ewald | auto | cuda | 256 | 7 | clustered | 1 | 6.134 | 6.089 | 6.204 | 5 |
| ewald | auto | cuda | 256 | 7 | lattice | 0 | 9.679 | 9.628 | 9.683 | 5 |
| ewald | auto | cuda | 256 | 7 | lattice | 1 | 9.894 | 9.857 | 10.005 | 5 |
| head | auto | cuda | 256 | 7 | adaptive | 0 | 16.808 | 16.693 | 17.365 | 5 |
| head | auto | cuda | 256 | 7 | adaptive | 1 | 17.602 | 17.213 | 17.917 | 5 |
| head | auto | cuda | 256 | 7 | clustered | 0 | 5.650 | 5.624 | 5.776 | 5 |
| head | auto | cuda | 256 | 7 | clustered | 1 | 6.075 | 6.029 | 6.171 | 5 |
| head | auto | cuda | 256 | 7 | lattice | 0 | 10.450 | 10.357 | 10.536 | 5 |
| head | auto | cuda | 256 | 7 | lattice | 1 | 10.662 | 10.463 | 10.927 | 5 |

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
