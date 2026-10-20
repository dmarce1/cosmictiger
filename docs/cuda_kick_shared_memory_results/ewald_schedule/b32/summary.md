| binary | policy | backend | bucket | order | fixture | potential | kick ms (median) | min | max | pairs |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| baseline | auto | cuda | 32 | 7 | adaptive | 0 | 17.770 | 17.684 | 17.934 | 3 |
| baseline | auto | cuda | 32 | 7 | adaptive | 1 | 18.061 | 17.846 | 18.107 | 3 |
| baseline | auto | cuda | 32 | 7 | clustered | 0 | 8.597 | 8.560 | 8.973 | 3 |
| baseline | auto | cuda | 32 | 7 | clustered | 1 | 8.757 | 8.521 | 8.770 | 3 |
| baseline | auto | cuda | 32 | 7 | lattice | 0 | 32.946 | 32.808 | 33.460 | 3 |
| baseline | auto | cuda | 32 | 7 | lattice | 1 | 33.071 | 32.735 | 33.549 | 3 |
| ewald | auto | cuda | 32 | 7 | adaptive | 0 | 10.595 | 10.576 | 10.743 | 3 |
| ewald | auto | cuda | 32 | 7 | adaptive | 1 | 10.765 | 10.730 | 11.765 | 3 |
| ewald | auto | cuda | 32 | 7 | clustered | 0 | 5.888 | 5.849 | 5.905 | 3 |
| ewald | auto | cuda | 32 | 7 | clustered | 1 | 6.050 | 6.026 | 6.223 | 3 |
| ewald | auto | cuda | 32 | 7 | lattice | 0 | 14.714 | 14.576 | 14.743 | 3 |
| ewald | auto | cuda | 32 | 7 | lattice | 1 | 14.842 | 14.758 | 15.169 | 3 |
| head | auto | cuda | 32 | 7 | adaptive | 0 | 12.012 | 11.891 | 12.128 | 3 |
| head | auto | cuda | 32 | 7 | adaptive | 1 | 11.863 | 11.783 | 12.021 | 3 |
| head | auto | cuda | 32 | 7 | clustered | 0 | 5.735 | 5.700 | 5.958 | 3 |
| head | auto | cuda | 32 | 7 | clustered | 1 | 5.999 | 5.965 | 6.088 | 3 |
| head | auto | cuda | 32 | 7 | lattice | 0 | 15.054 | 15.002 | 15.125 | 3 |
| head | auto | cuda | 32 | 7 | lattice | 1 | 15.083 | 14.989 | 15.091 | 3 |

| binary | policy | backend | bucket | order | fixture | force rel L2 | force p99 | potential rel L2 | potential p99 |
|---|---|---|---:|---:|---|---:|---:|---:|---:|
| baseline | auto | cuda | 32 | 7 | lattice | 2.462e-01 | 1.367e+00 | 2.842e-02 | 9.277e-02 |
| baseline | auto | cuda | 32 | 7 | clustered | 3.901e-03 | 1.726e-02 | 5.182e-04 | 4.988e-03 |
| baseline | auto | cuda | 32 | 7 | adaptive | 2.669e-03 | 4.085e-02 | 1.058e-04 | 2.660e-02 |
| head | auto | cuda | 32 | 7 | lattice | 2.991e-01 | 1.656e+00 | 3.739e-02 | 1.249e-01 |
| head | auto | cuda | 32 | 7 | clustered | 4.526e-03 | 2.238e-02 | 1.022e-03 | 7.718e-03 |
| head | auto | cuda | 32 | 7 | adaptive | 2.927e-03 | 5.030e-02 | 1.253e-04 | 2.580e-02 |
| ewald | auto | cuda | 32 | 7 | lattice | 2.991e-01 | 1.656e+00 | 3.739e-02 | 1.249e-01 |
| ewald | auto | cuda | 32 | 7 | clustered | 4.526e-03 | 2.238e-02 | 1.022e-03 | 7.718e-03 |
| ewald | auto | cuda | 32 | 7 | adaptive | 2.927e-03 | 5.030e-02 | 1.253e-04 | 2.580e-02 |

| binary | policy | backend | bucket | order | kernel policy line |
|---|---|---|---:|---:|---|
| baseline | auto | cuda | 32 | 7 | `Occupancy is 6 shmem size = 15184 numregs = 255` |
| ewald | auto | cuda | 32 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 7536 bytes: control 648, expansion shared, forces shared tile 32, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes; list slab 145792 bytes per block (4096 entries per list)` |
| head | auto | cuda | 32 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 7536 bytes: control 648, expansion shared, forces shared tile 32, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes; list slab 145792 bytes per block (4096 entries per list)` |

| comparison | binary | policy | bucket | order | fixture | acceleration rel L2 | acceleration max pointwise | potential rel L2 |
|---|---|---|---:|---:|---|---:|---:|---:|
| vs_baseline | ewald | auto | 32 | 7 | adaptive | 1.864e-03 | 1.680e-01 | 7.141e-05 |
| vs_baseline | ewald | auto | 32 | 7 | clustered | 4.744e-03 | 2.028e-01 | 7.377e-04 |
| vs_baseline | ewald | auto | 32 | 7 | lattice | 2.791e-01 | 1.308e+01 | 3.153e-02 |
| vs_baseline | head | auto | 32 | 7 | adaptive | 1.864e-03 | 1.680e-01 | 7.141e-05 |
| vs_baseline | head | auto | 32 | 7 | clustered | 4.744e-03 | 2.028e-01 | 7.377e-04 |
| vs_baseline | head | auto | 32 | 7 | lattice | 2.791e-01 | 1.308e+01 | 3.153e-02 |
