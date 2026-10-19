| binary | policy | backend | bucket | order | fixture | potential | kick ms (median) | min | max | pairs |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| merged | auto | cuda | 128 | 7 | adaptive | 0 | 21.003 | 20.206 | 21.800 | 2 |
| merged | auto | cuda | 128 | 7 | adaptive | 1 | 22.098 | 21.589 | 22.607 | 2 |
| merged | auto | cuda | 128 | 7 | clustered | 0 | 15.050 | 14.992 | 15.109 | 2 |
| merged | auto | cuda | 128 | 7 | clustered | 1 | 15.783 | 15.697 | 15.868 | 2 |
| merged | auto | cuda | 128 | 7 | lattice | 0 | 23.521 | 22.794 | 24.248 | 2 |
| merged | auto | cuda | 128 | 7 | lattice | 1 | 24.288 | 23.985 | 24.591 | 2 |
| merged | handoff164 | cuda | 128 | 7 | adaptive | 0 | 24.593 | 23.790 | 25.396 | 2 |
| merged | handoff164 | cuda | 128 | 7 | adaptive | 1 | 26.827 | 25.701 | 27.953 | 2 |
| merged | handoff164 | cuda | 128 | 7 | clustered | 0 | 15.804 | 15.250 | 16.359 | 2 |
| merged | handoff164 | cuda | 128 | 7 | clustered | 1 | 18.179 | 17.597 | 18.761 | 2 |
| merged | handoff164 | cuda | 128 | 7 | lattice | 0 | 24.483 | 24.209 | 24.757 | 2 |
| merged | handoff164 | cuda | 128 | 7 | lattice | 1 | 24.462 | 24.335 | 24.590 | 2 |
| merged | handoff328 | cuda | 128 | 7 | adaptive | 0 | 20.607 | 20.195 | 21.018 | 2 |
| merged | handoff328 | cuda | 128 | 7 | adaptive | 1 | 22.748 | 22.590 | 22.906 | 2 |
| merged | handoff328 | cuda | 128 | 7 | clustered | 0 | 14.858 | 13.784 | 15.932 | 2 |
| merged | handoff328 | cuda | 128 | 7 | clustered | 1 | 15.263 | 15.080 | 15.446 | 2 |
| merged | handoff328 | cuda | 128 | 7 | lattice | 0 | 23.831 | 23.627 | 24.034 | 2 |
| merged | handoff328 | cuda | 128 | 7 | lattice | 1 | 25.065 | 24.884 | 25.245 | 2 |

| binary | policy | backend | bucket | order | kernel policy line |
|---|---|---|---:|---:|---|
| merged | auto | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| merged | handoff164 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| merged | handoff328 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |

| comparison | binary | policy | bucket | order | fixture | acceleration rel L2 | acceleration max pointwise | potential rel L2 |
|---|---|---|---:|---:|---|---:|---:|---:|
| vs_merged | merged | handoff164 | 128 | 7 | adaptive | 8.801e-08 | 5.574e-06 | 5.991e-08 |
| vs_merged | merged | handoff164 | 128 | 7 | clustered | 8.558e-08 | 4.857e-06 | 1.538e-07 |
| vs_merged | merged | handoff164 | 128 | 7 | lattice | 1.281e-05 | 1.048e-03 | 9.870e-05 |
| vs_merged | merged | handoff328 | 128 | 7 | adaptive | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| vs_merged | merged | handoff328 | 128 | 7 | clustered | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| vs_merged | merged | handoff328 | 128 | 7 | lattice | 0.000e+00 | 0.000e+00 | 0.000e+00 |
