| binary | policy | backend | bucket | order | fixture | potential | kick ms (median) | min | max | pairs |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| arena | handoff164 | cuda | 128 | 7 | adaptive | 0 | 16.519 | 16.273 | 16.765 | 2 |
| arena | handoff164 | cuda | 128 | 7 | adaptive | 1 | 15.585 | 15.367 | 15.802 | 2 |
| arena | handoff164 | cuda | 128 | 7 | clustered | 0 | 7.810 | 7.652 | 7.968 | 2 |
| arena | handoff164 | cuda | 128 | 7 | clustered | 1 | 8.230 | 8.203 | 8.257 | 2 |
| arena | handoff164 | cuda | 128 | 7 | lattice | 0 | 16.203 | 16.150 | 16.257 | 2 |
| arena | handoff164 | cuda | 128 | 7 | lattice | 1 | 16.186 | 16.091 | 16.281 | 2 |
| arena | handoff328 | cuda | 128 | 7 | adaptive | 0 | 19.839 | 19.658 | 20.021 | 2 |
| arena | handoff328 | cuda | 128 | 7 | adaptive | 1 | 18.570 | 18.487 | 18.653 | 2 |
| arena | handoff328 | cuda | 128 | 7 | clustered | 0 | 7.887 | 7.829 | 7.945 | 2 |
| arena | handoff328 | cuda | 128 | 7 | clustered | 1 | 8.269 | 8.002 | 8.536 | 2 |
| arena | handoff328 | cuda | 128 | 7 | lattice | 0 | 16.702 | 16.555 | 16.848 | 2 |
| arena | handoff328 | cuda | 128 | 7 | lattice | 1 | 16.562 | 16.561 | 16.563 | 2 |
| arena | handoff492 | cuda | 128 | 7 | adaptive | 0 | 19.860 | 19.781 | 19.939 | 2 |
| arena | handoff492 | cuda | 128 | 7 | adaptive | 1 | 18.565 | 18.197 | 18.933 | 2 |
| arena | handoff492 | cuda | 128 | 7 | clustered | 0 | 7.818 | 7.664 | 7.973 | 2 |
| arena | handoff492 | cuda | 128 | 7 | clustered | 1 | 8.110 | 8.001 | 8.220 | 2 |
| arena | handoff492 | cuda | 128 | 7 | lattice | 0 | 16.111 | 15.883 | 16.340 | 2 |
| arena | handoff492 | cuda | 128 | 7 | lattice | 1 | 16.092 | 16.076 | 16.108 | 2 |
| arena | handoff656 | cuda | 128 | 7 | adaptive | 0 | 19.960 | 19.916 | 20.004 | 2 |
| arena | handoff656 | cuda | 128 | 7 | adaptive | 1 | 18.658 | 18.631 | 18.684 | 2 |
| arena | handoff656 | cuda | 128 | 7 | clustered | 0 | 7.758 | 7.738 | 7.779 | 2 |
| arena | handoff656 | cuda | 128 | 7 | clustered | 1 | 8.126 | 8.075 | 8.178 | 2 |
| arena | handoff656 | cuda | 128 | 7 | lattice | 0 | 16.313 | 16.216 | 16.411 | 2 |
| arena | handoff656 | cuda | 128 | 7 | lattice | 1 | 16.177 | 15.925 | 16.429 | 2 |
| arena | handoff82 | cuda | 128 | 7 | adaptive | 0 | 19.917 | 19.747 | 20.087 | 2 |
| arena | handoff82 | cuda | 128 | 7 | adaptive | 1 | 19.931 | 19.660 | 20.202 | 2 |
| arena | handoff82 | cuda | 128 | 7 | clustered | 0 | 6.494 | 6.421 | 6.567 | 2 |
| arena | handoff82 | cuda | 128 | 7 | clustered | 1 | 6.934 | 6.916 | 6.951 | 2 |
| arena | handoff82 | cuda | 128 | 7 | lattice | 0 | 12.229 | 11.805 | 12.652 | 2 |
| arena | handoff82 | cuda | 128 | 7 | lattice | 1 | 12.823 | 12.522 | 13.123 | 2 |

| binary | policy | backend | bucket | order | kernel policy line |
|---|---|---|---:|---:|---|
| arena | handoff164 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| arena | handoff328 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| arena | handoff492 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| arena | handoff656 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| arena | handoff82 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |

| comparison | binary | policy | bucket | order | fixture | acceleration rel L2 | acceleration max pointwise | potential rel L2 |
|---|---|---|---:|---:|---|---:|---:|---:|
