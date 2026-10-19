| binary | policy | backend | bucket | order | fixture | potential | kick ms (median) | min | max | pairs |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| rule | auto | cuda | 128 | 7 | adaptive | 0 | 21.144 | 20.993 | 21.295 | 2 |
| rule | auto | cuda | 128 | 7 | adaptive | 1 | 23.261 | 22.878 | 23.643 | 2 |
| rule | auto | cuda | 128 | 7 | clustered | 0 | 14.560 | 14.248 | 14.872 | 2 |
| rule | auto | cuda | 128 | 7 | clustered | 1 | 16.488 | 16.163 | 16.812 | 2 |
| rule | auto | cuda | 128 | 7 | lattice | 0 | 23.521 | 22.785 | 24.257 | 2 |
| rule | auto | cuda | 128 | 7 | lattice | 1 | 24.863 | 24.037 | 25.688 | 2 |
| rule | handoff164 | cuda | 128 | 7 | adaptive | 0 | 26.372 | 25.626 | 27.117 | 2 |
| rule | handoff164 | cuda | 128 | 7 | adaptive | 1 | 28.637 | 28.596 | 28.678 | 2 |
| rule | handoff164 | cuda | 128 | 7 | clustered | 0 | 16.774 | 16.750 | 16.798 | 2 |
| rule | handoff164 | cuda | 128 | 7 | clustered | 1 | 19.393 | 18.728 | 20.058 | 2 |
| rule | handoff164 | cuda | 128 | 7 | lattice | 0 | 25.562 | 25.210 | 25.914 | 2 |
| rule | handoff164 | cuda | 128 | 7 | lattice | 1 | 24.812 | 24.736 | 24.887 | 2 |
| rule | handoff328 | cuda | 128 | 7 | adaptive | 0 | 21.669 | 21.490 | 21.849 | 2 |
| rule | handoff328 | cuda | 128 | 7 | adaptive | 1 | 23.356 | 23.342 | 23.371 | 2 |
| rule | handoff328 | cuda | 128 | 7 | clustered | 0 | 14.845 | 14.714 | 14.977 | 2 |
| rule | handoff328 | cuda | 128 | 7 | clustered | 1 | 16.204 | 15.904 | 16.504 | 2 |
| rule | handoff328 | cuda | 128 | 7 | lattice | 0 | 24.549 | 24.121 | 24.976 | 2 |
| rule | handoff328 | cuda | 128 | 7 | lattice | 1 | 24.371 | 24.064 | 24.679 | 2 |

| binary | policy | backend | bucket | order | kernel policy line |
|---|---|---|---:|---:|---|
| rule | auto | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| rule | handoff164 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| rule | handoff328 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |

| comparison | binary | policy | bucket | order | fixture | acceleration rel L2 | acceleration max pointwise | potential rel L2 |
|---|---|---|---:|---:|---|---:|---:|---:|
| vs_rule | rule | handoff164 | 128 | 7 | adaptive | 8.801e-08 | 5.574e-06 | 5.991e-08 |
| vs_rule | rule | handoff164 | 128 | 7 | clustered | 8.558e-08 | 4.857e-06 | 1.536e-07 |
| vs_rule | rule | handoff164 | 128 | 7 | lattice | 1.281e-05 | 1.048e-03 | 9.870e-05 |
| vs_rule | rule | handoff328 | 128 | 7 | adaptive | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| vs_rule | rule | handoff328 | 128 | 7 | clustered | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| vs_rule | rule | handoff328 | 128 | 7 | lattice | 0.000e+00 | 0.000e+00 | 0.000e+00 |
