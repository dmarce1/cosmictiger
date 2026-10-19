| binary | policy | backend | bucket | order | fixture | potential | kick ms (median) | min | max | pairs |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| rule | auto | cuda | 128 | 7 | adaptive | 0 | 13.948 | 13.737 | 14.158 | 2 |
| rule | auto | cuda | 128 | 7 | adaptive | 1 | 14.092 | 13.764 | 14.421 | 2 |
| rule | auto | cuda | 128 | 7 | clustered | 0 | 5.675 | 5.564 | 5.786 | 2 |
| rule | auto | cuda | 128 | 7 | clustered | 1 | 6.610 | 6.526 | 6.693 | 2 |
| rule | auto | cuda | 128 | 7 | lattice | 0 | 10.822 | 10.454 | 11.189 | 2 |
| rule | auto | cuda | 128 | 7 | lattice | 1 | 10.448 | 10.296 | 10.599 | 2 |
| rule | handoff164 | cuda | 128 | 7 | adaptive | 0 | 14.099 | 13.700 | 14.498 | 2 |
| rule | handoff164 | cuda | 128 | 7 | adaptive | 1 | 13.719 | 13.545 | 13.893 | 2 |
| rule | handoff164 | cuda | 128 | 7 | clustered | 0 | 5.720 | 5.707 | 5.732 | 2 |
| rule | handoff164 | cuda | 128 | 7 | clustered | 1 | 5.978 | 5.861 | 6.096 | 2 |
| rule | handoff164 | cuda | 128 | 7 | lattice | 0 | 10.137 | 10.130 | 10.144 | 2 |
| rule | handoff164 | cuda | 128 | 7 | lattice | 1 | 9.989 | 9.894 | 10.083 | 2 |
| rule | handoff328 | cuda | 128 | 7 | adaptive | 0 | 14.733 | 14.440 | 15.027 | 2 |
| rule | handoff328 | cuda | 128 | 7 | adaptive | 1 | 14.016 | 13.920 | 14.113 | 2 |
| rule | handoff328 | cuda | 128 | 7 | clustered | 0 | 5.768 | 5.710 | 5.826 | 2 |
| rule | handoff328 | cuda | 128 | 7 | clustered | 1 | 6.161 | 6.125 | 6.196 | 2 |
| rule | handoff328 | cuda | 128 | 7 | lattice | 0 | 10.082 | 9.971 | 10.192 | 2 |
| rule | handoff328 | cuda | 128 | 7 | lattice | 1 | 10.136 | 10.111 | 10.161 | 2 |

| binary | policy | backend | bucket | order | kernel policy line |
|---|---|---|---:|---:|---|
| rule | auto | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| rule | handoff164 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| rule | handoff328 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |

| comparison | binary | policy | bucket | order | fixture | acceleration rel L2 | acceleration max pointwise | potential rel L2 |
|---|---|---|---:|---:|---|---:|---:|---:|
| vs_rule | rule | handoff164 | 128 | 7 | adaptive | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| vs_rule | rule | handoff164 | 128 | 7 | clustered | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| vs_rule | rule | handoff164 | 128 | 7 | lattice | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| vs_rule | rule | handoff328 | 128 | 7 | adaptive | 1.181e-07 | 5.722e-06 | 6.288e-08 |
| vs_rule | rule | handoff328 | 128 | 7 | clustered | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| vs_rule | rule | handoff328 | 128 | 7 | lattice | 0.000e+00 | 0.000e+00 | 0.000e+00 |
