| binary | policy | backend | bucket | order | fixture | potential | kick ms (median) | min | max | pairs |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| arena_cap1024 | auto | cuda | 64 | 7 | adaptive | 0 | 20.516 | 20.490 | 20.992 | 3 |
| arena_cap1024 | auto | cuda | 64 | 7 | adaptive | 1 | 20.806 | 20.629 | 21.180 | 3 |
| arena_cap1024 | auto | cuda | 64 | 7 | clustered | 0 | 8.678 | 8.380 | 8.680 | 3 |
| arena_cap1024 | auto | cuda | 64 | 7 | clustered | 1 | 8.753 | 8.541 | 8.890 | 3 |
| arena_cap1024 | auto | cuda | 64 | 7 | lattice | 0 | 31.635 | 31.515 | 31.965 | 3 |
| arena_cap1024 | auto | cuda | 64 | 7 | lattice | 1 | 31.665 | 31.625 | 32.208 | 3 |
| arena_cap128 | auto | cuda | 64 | 7 | adaptive | 0 | 20.986 | 20.701 | 21.218 | 3 |
| arena_cap128 | auto | cuda | 64 | 7 | adaptive | 1 | 21.038 | 20.872 | 21.175 | 3 |
| arena_cap128 | auto | cuda | 64 | 7 | clustered | 0 | 8.441 | 8.296 | 8.458 | 3 |
| arena_cap128 | auto | cuda | 64 | 7 | clustered | 1 | 8.638 | 8.557 | 8.804 | 3 |
| arena_cap128 | auto | cuda | 64 | 7 | lattice | 0 | 31.875 | 31.867 | 31.977 | 3 |
| arena_cap128 | auto | cuda | 64 | 7 | lattice | 1 | 32.120 | 31.735 | 32.173 | 3 |
| baseline | auto | cuda | 64 | 7 | adaptive | 0 | 19.396 | 19.363 | 19.449 | 3 |
| baseline | auto | cuda | 64 | 7 | adaptive | 1 | 19.472 | 19.093 | 19.862 | 3 |
| baseline | auto | cuda | 64 | 7 | clustered | 0 | 8.865 | 8.823 | 8.942 | 3 |
| baseline | auto | cuda | 64 | 7 | clustered | 1 | 8.977 | 8.872 | 9.178 | 3 |
| baseline | auto | cuda | 64 | 7 | lattice | 0 | 33.062 | 32.820 | 33.149 | 3 |
| baseline | auto | cuda | 64 | 7 | lattice | 1 | 32.940 | 32.852 | 33.185 | 3 |

| binary | policy | backend | bucket | order | kernel policy line |
|---|---|---|---:|---:|---|
| arena_cap1024 | auto | cuda | 64 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 7904 bytes: control 504, expansion shared, forces shared tile 64, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| arena_cap128 | auto | cuda | 64 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 7904 bytes: control 504, expansion shared, forces shared tile 64, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| baseline | auto | cuda | 64 | 7 | `Occupancy is 6 shmem size = 15184 numregs = 255` |

| comparison | binary | policy | bucket | order | fixture | acceleration rel L2 | acceleration max pointwise | potential rel L2 |
|---|---|---|---:|---:|---|---:|---:|---:|
| vs_baseline | arena_cap1024 | auto | 64 | 7 | adaptive | 5.051e-08 | 1.221e-06 | 3.883e-08 |
| vs_baseline | arena_cap1024 | auto | 64 | 7 | clustered | 2.326e-08 | 3.579e-06 | 3.596e-08 |
| vs_baseline | arena_cap1024 | auto | 64 | 7 | lattice | 1.004e-06 | 6.536e-05 | 3.584e-06 |
| vs_baseline | arena_cap128 | auto | 64 | 7 | adaptive | 5.051e-08 | 1.221e-06 | 3.883e-08 |
| vs_baseline | arena_cap128 | auto | 64 | 7 | clustered | 2.326e-08 | 3.579e-06 | 3.596e-08 |
| vs_baseline | arena_cap128 | auto | 64 | 7 | lattice | 1.004e-06 | 6.536e-05 | 3.584e-06 |
