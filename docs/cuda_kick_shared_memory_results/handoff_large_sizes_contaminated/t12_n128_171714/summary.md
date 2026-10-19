| binary | policy | backend | bucket | order | fixture | potential | kick ms (median) | min | max | pairs |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| arena | handoff1312 | cuda | 128 | 7 | adaptive | 0 | 86.430 | 85.061 | 87.798 | 2 |
| arena | handoff1312 | cuda | 128 | 7 | adaptive | 1 | 98.272 | 98.199 | 98.345 | 2 |
| arena | handoff1312 | cuda | 128 | 7 | clustered | 0 | 75.074 | 74.288 | 75.860 | 2 |
| arena | handoff1312 | cuda | 128 | 7 | clustered | 1 | 85.781 | 84.131 | 87.431 | 2 |
| arena | handoff1312 | cuda | 128 | 7 | lattice | 0 | 107.268 | 106.961 | 107.575 | 2 |
| arena | handoff1312 | cuda | 128 | 7 | lattice | 1 | 117.178 | 113.518 | 120.837 | 2 |
| arena | handoff328 | cuda | 128 | 7 | adaptive | 0 | 98.493 | 88.022 | 108.963 | 2 |
| arena | handoff328 | cuda | 128 | 7 | adaptive | 1 | 116.588 | 103.864 | 129.312 | 2 |
| arena | handoff328 | cuda | 128 | 7 | clustered | 0 | 84.531 | 73.110 | 95.953 | 2 |
| arena | handoff328 | cuda | 128 | 7 | clustered | 1 | 102.576 | 89.518 | 115.634 | 2 |
| arena | handoff328 | cuda | 128 | 7 | lattice | 0 | 99.023 | 89.152 | 108.893 | 2 |
| arena | handoff328 | cuda | 128 | 7 | lattice | 1 | 108.676 | 94.561 | 122.790 | 2 |
| arena | handoff492 | cuda | 128 | 7 | adaptive | 0 | 95.204 | 94.937 | 95.471 | 2 |
| arena | handoff492 | cuda | 128 | 7 | adaptive | 1 | 109.833 | 109.493 | 110.174 | 2 |
| arena | handoff492 | cuda | 128 | 7 | clustered | 0 | 87.284 | 83.747 | 90.820 | 2 |
| arena | handoff492 | cuda | 128 | 7 | clustered | 1 | 105.606 | 102.211 | 109.001 | 2 |
| arena | handoff492 | cuda | 128 | 7 | lattice | 0 | 96.346 | 92.709 | 99.982 | 2 |
| arena | handoff492 | cuda | 128 | 7 | lattice | 1 | 110.008 | 108.626 | 111.390 | 2 |
| arena | handoff656 | cuda | 128 | 7 | adaptive | 0 | 87.102 | 83.157 | 91.046 | 2 |
| arena | handoff656 | cuda | 128 | 7 | adaptive | 1 | 102.563 | 97.916 | 107.210 | 2 |
| arena | handoff656 | cuda | 128 | 7 | clustered | 0 | 85.392 | 83.282 | 87.503 | 2 |
| arena | handoff656 | cuda | 128 | 7 | clustered | 1 | 99.106 | 96.294 | 101.918 | 2 |
| arena | handoff656 | cuda | 128 | 7 | lattice | 0 | 108.493 | 105.662 | 111.325 | 2 |
| arena | handoff656 | cuda | 128 | 7 | lattice | 1 | 122.353 | 116.982 | 127.723 | 2 |
| arena | handoff984 | cuda | 128 | 7 | adaptive | 0 | 89.024 | 88.655 | 89.393 | 2 |
| arena | handoff984 | cuda | 128 | 7 | adaptive | 1 | 102.429 | 102.144 | 102.714 | 2 |
| arena | handoff984 | cuda | 128 | 7 | clustered | 0 | 74.685 | 73.219 | 76.150 | 2 |
| arena | handoff984 | cuda | 128 | 7 | clustered | 1 | 86.543 | 84.809 | 88.276 | 2 |
| arena | handoff984 | cuda | 128 | 7 | lattice | 0 | 102.861 | 98.477 | 107.245 | 2 |
| arena | handoff984 | cuda | 128 | 7 | lattice | 1 | 113.398 | 110.144 | 116.652 | 2 |

| binary | policy | backend | bucket | order | kernel policy line |
|---|---|---|---:|---:|---|
| arena | handoff1312 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| arena | handoff328 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| arena | handoff492 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| arena | handoff656 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| arena | handoff984 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |

| comparison | binary | policy | bucket | order | fixture | acceleration rel L2 | acceleration max pointwise | potential rel L2 |
|---|---|---|---:|---:|---|---:|---:|---:|
