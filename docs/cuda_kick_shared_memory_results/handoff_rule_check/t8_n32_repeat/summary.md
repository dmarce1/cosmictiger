| binary | policy | backend | bucket | order | fixture | potential | kick ms (median) | min | max | pairs |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| rule | auto | cuda | 128 | 7 | adaptive | 0 | 13.493 | 13.392 | 13.670 | 3 |
| rule | auto | cuda | 128 | 7 | adaptive | 1 | 13.913 | 13.813 | 13.927 | 3 |
| rule | auto | cuda | 128 | 7 | clustered | 0 | 5.768 | 5.665 | 5.845 | 3 |
| rule | auto | cuda | 128 | 7 | clustered | 1 | 5.932 | 5.912 | 6.369 | 3 |
| rule | auto | cuda | 128 | 7 | lattice | 0 | 10.175 | 9.691 | 10.338 | 3 |
| rule | auto | cuda | 128 | 7 | lattice | 1 | 9.821 | 9.791 | 9.936 | 3 |
| rule | handoff164 | cuda | 128 | 7 | adaptive | 0 | 13.159 | 13.137 | 13.756 | 3 |
| rule | handoff164 | cuda | 128 | 7 | adaptive | 1 | 13.662 | 12.809 | 13.944 | 3 |
| rule | handoff164 | cuda | 128 | 7 | clustered | 0 | 5.728 | 5.591 | 5.805 | 3 |
| rule | handoff164 | cuda | 128 | 7 | clustered | 1 | 5.895 | 5.784 | 6.099 | 3 |
| rule | handoff164 | cuda | 128 | 7 | lattice | 0 | 10.609 | 10.503 | 12.712 | 3 |
| rule | handoff164 | cuda | 128 | 7 | lattice | 1 | 10.082 | 10.030 | 10.340 | 3 |
| rule | handoff328 | cuda | 128 | 7 | adaptive | 0 | 13.762 | 13.735 | 14.054 | 3 |
| rule | handoff328 | cuda | 128 | 7 | adaptive | 1 | 14.004 | 13.793 | 14.134 | 3 |
| rule | handoff328 | cuda | 128 | 7 | clustered | 0 | 5.817 | 5.608 | 6.166 | 3 |
| rule | handoff328 | cuda | 128 | 7 | clustered | 1 | 5.889 | 5.815 | 5.922 | 3 |
| rule | handoff328 | cuda | 128 | 7 | lattice | 0 | 10.000 | 9.584 | 10.356 | 3 |
| rule | handoff328 | cuda | 128 | 7 | lattice | 1 | 9.824 | 9.733 | 10.479 | 3 |

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
