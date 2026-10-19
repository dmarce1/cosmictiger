| binary | policy | backend | bucket | order | fixture | potential | kick ms (median) | min | max | pairs |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| arena | handoff164 | cuda | 128 | 7 | adaptive | 0 | 14.178 | 13.640 | 14.715 | 2 |
| arena | handoff164 | cuda | 128 | 7 | adaptive | 1 | 14.073 | 13.554 | 14.593 | 2 |
| arena | handoff164 | cuda | 128 | 7 | clustered | 0 | 6.177 | 6.160 | 6.193 | 2 |
| arena | handoff164 | cuda | 128 | 7 | clustered | 1 | 6.786 | 6.578 | 6.995 | 2 |
| arena | handoff164 | cuda | 128 | 7 | lattice | 0 | 11.815 | 11.525 | 12.106 | 2 |
| arena | handoff164 | cuda | 128 | 7 | lattice | 1 | 11.741 | 11.688 | 11.794 | 2 |
| arena | handoff328 | cuda | 128 | 7 | adaptive | 0 | 16.161 | 15.762 | 16.559 | 2 |
| arena | handoff328 | cuda | 128 | 7 | adaptive | 1 | 15.371 | 14.794 | 15.948 | 2 |
| arena | handoff328 | cuda | 128 | 7 | clustered | 0 | 6.336 | 6.122 | 6.550 | 2 |
| arena | handoff328 | cuda | 128 | 7 | clustered | 1 | 6.791 | 6.727 | 6.856 | 2 |
| arena | handoff328 | cuda | 128 | 7 | lattice | 0 | 11.500 | 11.203 | 11.797 | 2 |
| arena | handoff328 | cuda | 128 | 7 | lattice | 1 | 12.088 | 11.640 | 12.536 | 2 |
| arena | handoff492 | cuda | 128 | 7 | adaptive | 0 | 19.756 | 16.020 | 23.493 | 2 |
| arena | handoff492 | cuda | 128 | 7 | adaptive | 1 | 18.032 | 14.977 | 21.088 | 2 |
| arena | handoff492 | cuda | 128 | 7 | clustered | 0 | 6.120 | 5.954 | 6.286 | 2 |
| arena | handoff492 | cuda | 128 | 7 | clustered | 1 | 9.336 | 6.672 | 12.000 | 2 |
| arena | handoff492 | cuda | 128 | 7 | lattice | 0 | 12.010 | 11.729 | 12.292 | 2 |
| arena | handoff492 | cuda | 128 | 7 | lattice | 1 | 11.571 | 11.516 | 11.625 | 2 |
| arena | handoff656 | cuda | 128 | 7 | adaptive | 0 | 16.355 | 16.122 | 16.587 | 2 |
| arena | handoff656 | cuda | 128 | 7 | adaptive | 1 | 15.220 | 14.958 | 15.482 | 2 |
| arena | handoff656 | cuda | 128 | 7 | clustered | 0 | 6.032 | 5.932 | 6.133 | 2 |
| arena | handoff656 | cuda | 128 | 7 | clustered | 1 | 6.552 | 6.485 | 6.619 | 2 |
| arena | handoff656 | cuda | 128 | 7 | lattice | 0 | 11.708 | 11.588 | 11.828 | 2 |
| arena | handoff656 | cuda | 128 | 7 | lattice | 1 | 11.625 | 11.548 | 11.701 | 2 |
| arena | handoff82 | cuda | 128 | 7 | adaptive | 0 | 18.463 | 18.192 | 18.734 | 2 |
| arena | handoff82 | cuda | 128 | 7 | adaptive | 1 | 18.904 | 18.834 | 18.974 | 2 |
| arena | handoff82 | cuda | 128 | 7 | clustered | 0 | 5.799 | 5.733 | 5.866 | 2 |
| arena | handoff82 | cuda | 128 | 7 | clustered | 1 | 6.058 | 6.044 | 6.072 | 2 |
| arena | handoff82 | cuda | 128 | 7 | lattice | 0 | 10.072 | 9.816 | 10.328 | 2 |
| arena | handoff82 | cuda | 128 | 7 | lattice | 1 | 10.336 | 10.206 | 10.466 | 2 |

| binary | policy | backend | bucket | order | kernel policy line |
|---|---|---|---:|---:|---|
| arena | handoff164 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| arena | handoff328 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| arena | handoff492 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| arena | handoff656 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| arena | handoff82 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |

| comparison | binary | policy | bucket | order | fixture | acceleration rel L2 | acceleration max pointwise | potential rel L2 |
|---|---|---|---:|---:|---|---:|---:|---:|
