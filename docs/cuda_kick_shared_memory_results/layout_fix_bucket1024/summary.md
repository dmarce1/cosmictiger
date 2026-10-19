| binary | policy | backend | bucket | order | fixture | potential | kick ms (median) | min | max | pairs |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| arena_fix | auto | cuda | 512 | 7 | adaptive | 0 | 33.811 | 33.727 | 33.895 | 2 |
| arena_fix | auto | cuda | 512 | 7 | adaptive | 1 | 35.629 | 35.253 | 36.005 | 2 |
| arena_fix | auto | cuda | 512 | 7 | clustered | 0 | 9.890 | 9.703 | 10.078 | 2 |
| arena_fix | auto | cuda | 512 | 7 | clustered | 1 | 10.577 | 10.398 | 10.755 | 2 |
| arena_fix | auto | cuda | 512 | 7 | lattice | 0 | 12.621 | 12.296 | 12.946 | 2 |
| arena_fix | auto | cuda | 512 | 7 | lattice | 1 | 12.487 | 11.879 | 13.094 | 2 |
| arena_fix | auto | cuda | 1024 | 7 | adaptive | 0 | 80.229 | 79.978 | 80.480 | 2 |
| arena_fix | auto | cuda | 1024 | 7 | adaptive | 1 | 81.257 | 80.518 | 81.996 | 2 |
| arena_fix | auto | cuda | 1024 | 7 | clustered | 0 | 16.486 | 16.466 | 16.506 | 2 |
| arena_fix | auto | cuda | 1024 | 7 | clustered | 1 | 17.985 | 17.919 | 18.051 | 2 |
| arena_fix | auto | cuda | 1024 | 7 | lattice | 0 | 19.001 | 18.832 | 19.170 | 2 |
| arena_fix | auto | cuda | 1024 | 7 | lattice | 1 | 20.466 | 20.456 | 20.476 | 2 |
| arena_fix | tile128 | cuda | 512 | 7 | adaptive | 0 | 34.597 | 34.589 | 34.605 | 2 |
| arena_fix | tile128 | cuda | 512 | 7 | adaptive | 1 | 35.029 | 34.981 | 35.076 | 2 |
| arena_fix | tile128 | cuda | 512 | 7 | clustered | 0 | 10.281 | 10.214 | 10.348 | 2 |
| arena_fix | tile128 | cuda | 512 | 7 | clustered | 1 | 11.009 | 10.984 | 11.035 | 2 |
| arena_fix | tile128 | cuda | 512 | 7 | lattice | 0 | 12.210 | 12.028 | 12.391 | 2 |
| arena_fix | tile128 | cuda | 512 | 7 | lattice | 1 | 12.601 | 11.883 | 13.320 | 2 |
| arena_fix | tile128 | cuda | 1024 | 7 | adaptive | 0 | 80.139 | 79.935 | 80.343 | 2 |
| arena_fix | tile128 | cuda | 1024 | 7 | adaptive | 1 | 81.911 | 81.811 | 82.012 | 2 |
| arena_fix | tile128 | cuda | 1024 | 7 | clustered | 0 | 16.800 | 16.747 | 16.852 | 2 |
| arena_fix | tile128 | cuda | 1024 | 7 | clustered | 1 | 17.819 | 17.581 | 18.057 | 2 |
| arena_fix | tile128 | cuda | 1024 | 7 | lattice | 0 | 19.768 | 19.061 | 20.474 | 2 |
| arena_fix | tile128 | cuda | 1024 | 7 | lattice | 1 | 20.312 | 20.229 | 20.396 | 2 |
| baseline | auto | cuda | 512 | 7 | adaptive | 0 | 28.971 | 28.863 | 29.078 | 2 |
| baseline | auto | cuda | 512 | 7 | adaptive | 1 | 29.298 | 29.087 | 29.509 | 2 |
| baseline | auto | cuda | 512 | 7 | clustered | 0 | 10.284 | 10.272 | 10.295 | 2 |
| baseline | auto | cuda | 512 | 7 | clustered | 1 | 10.287 | 9.912 | 10.661 | 2 |
| baseline | auto | cuda | 512 | 7 | lattice | 0 | 11.586 | 10.771 | 12.401 | 2 |
| baseline | auto | cuda | 512 | 7 | lattice | 1 | 11.086 | 11.040 | 11.131 | 2 |
| baseline | auto | cuda | 1024 | 7 | adaptive | 0 | 81.558 | 81.392 | 81.723 | 2 |
| baseline | auto | cuda | 1024 | 7 | adaptive | 1 | 82.981 | 82.775 | 83.186 | 2 |
| baseline | auto | cuda | 1024 | 7 | clustered | 0 | 16.415 | 16.113 | 16.717 | 2 |
| baseline | auto | cuda | 1024 | 7 | clustered | 1 | 17.788 | 17.441 | 18.136 | 2 |
| baseline | auto | cuda | 1024 | 7 | lattice | 0 | 16.399 | 16.294 | 16.504 | 2 |
| baseline | auto | cuda | 1024 | 7 | lattice | 1 | 17.382 | 17.332 | 17.433 | 2 |

| binary | policy | backend | bucket | order | fixture | force rel L2 | force p99 | potential rel L2 | potential p99 |
|---|---|---|---:|---:|---|---:|---:|---:|---:|
| baseline | auto | cuda | 512 | 7 | lattice | 3.418e-01 | 1.857e+00 | 3.613e-02 | 1.180e-01 |
| baseline | auto | cuda | 512 | 7 | clustered | 1.767e-03 | 2.301e-02 | 5.584e-04 | 7.015e-03 |
| baseline | auto | cuda | 512 | 7 | adaptive | 1.001e-03 | 4.085e-02 | 9.754e-05 | 2.660e-02 |
| baseline | auto | cuda | 1024 | 7 | lattice | 3.418e-01 | 1.857e+00 | 3.613e-02 | 1.180e-01 |
| baseline | auto | cuda | 1024 | 7 | clustered | 1.767e-03 | 2.301e-02 | 5.584e-04 | 7.015e-03 |
| baseline | auto | cuda | 1024 | 7 | adaptive | 7.528e-04 | 4.085e-02 | 9.124e-05 | 2.661e-02 |
| arena_fix | auto | cuda | 512 | 7 | lattice | 3.418e-01 | 1.857e+00 | 3.613e-02 | 1.180e-01 |
| arena_fix | auto | cuda | 512 | 7 | clustered | 1.767e-03 | 2.301e-02 | 5.584e-04 | 7.013e-03 |
| arena_fix | auto | cuda | 512 | 7 | adaptive | 1.001e-03 | 4.085e-02 | 9.754e-05 | 2.660e-02 |
| arena_fix | auto | cuda | 1024 | 7 | lattice | 3.418e-01 | 1.857e+00 | 3.613e-02 | 1.180e-01 |
| arena_fix | auto | cuda | 1024 | 7 | clustered | 1.767e-03 | 2.302e-02 | 5.584e-04 | 7.010e-03 |
| arena_fix | auto | cuda | 1024 | 7 | adaptive | 7.528e-04 | 4.085e-02 | 9.124e-05 | 2.661e-02 |
| arena_fix | tile128 | cuda | 512 | 7 | lattice | 3.418e-01 | 1.857e+00 | 3.613e-02 | 1.180e-01 |
| arena_fix | tile128 | cuda | 512 | 7 | clustered | 1.767e-03 | 2.302e-02 | 5.584e-04 | 7.010e-03 |
| arena_fix | tile128 | cuda | 512 | 7 | adaptive | 1.001e-03 | 4.085e-02 | 9.754e-05 | 2.660e-02 |
| arena_fix | tile128 | cuda | 1024 | 7 | lattice | 3.418e-01 | 1.857e+00 | 3.613e-02 | 1.180e-01 |
| arena_fix | tile128 | cuda | 1024 | 7 | clustered | 1.767e-03 | 2.302e-02 | 5.584e-04 | 7.010e-03 |
| arena_fix | tile128 | cuda | 1024 | 7 | adaptive | 7.528e-04 | 4.085e-02 | 9.124e-05 | 2.661e-02 |

| binary | policy | backend | bucket | order | kernel policy line |
|---|---|---|---:|---:|---|
| arena_fix | auto | cuda | 512 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 11648 bytes: control 504, expansion shared, forces shared tile 512, particle tile 224 (without mass plane), multipole tile 17 x 160 bytes` |
| arena_fix | auto | cuda | 1024 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8928 bytes: control 504, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| arena_fix | tile128 | cuda | 512 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8928 bytes: control 504, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| arena_fix | tile128 | cuda | 1024 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8928 bytes: control 504, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| baseline | auto | cuda | 512 | 7 | `Occupancy is 6 shmem size = 15184 numregs = 255` |
| baseline | auto | cuda | 1024 | 7 | `Occupancy is 6 shmem size = 15184 numregs = 255` |

| comparison | binary | policy | bucket | order | fixture | acceleration rel L2 | acceleration max pointwise | potential rel L2 |
|---|---|---|---:|---:|---|---:|---:|---:|
| vs_baseline | arena_fix | auto | 512 | 7 | adaptive | 2.500e-07 | 1.028e-05 | 1.385e-07 |
| vs_baseline | arena_fix | auto | 512 | 7 | clustered | 4.337e-07 | 1.197e-05 | 7.290e-07 |
| vs_baseline | arena_fix | auto | 512 | 7 | lattice | 6.603e-06 | 3.708e-04 | 2.360e-05 |
| vs_baseline | arena_fix | auto | 1024 | 7 | adaptive | 2.774e-07 | 7.465e-06 | 1.737e-07 |
| vs_baseline | arena_fix | auto | 1024 | 7 | clustered | 4.316e-07 | 1.391e-05 | 7.409e-07 |
| vs_baseline | arena_fix | auto | 1024 | 7 | lattice | 1.140e-06 | 6.535e-05 | 3.617e-06 |
| vs_baseline | arena_fix | tile128 | 512 | 7 | adaptive | 2.492e-07 | 1.086e-05 | 1.338e-07 |
| vs_baseline | arena_fix | tile128 | 512 | 7 | clustered | 4.316e-07 | 1.391e-05 | 7.409e-07 |
| vs_baseline | arena_fix | tile128 | 512 | 7 | lattice | 1.062e-06 | 6.536e-05 | 3.602e-06 |
| vs_baseline | arena_fix | tile128 | 1024 | 7 | adaptive | 2.774e-07 | 7.465e-06 | 1.737e-07 |
| vs_baseline | arena_fix | tile128 | 1024 | 7 | clustered | 4.316e-07 | 1.391e-05 | 7.409e-07 |
| vs_baseline | arena_fix | tile128 | 1024 | 7 | lattice | 1.140e-06 | 6.535e-05 | 3.617e-06 |
