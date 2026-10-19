| binary | policy | backend | bucket | order | fixture | potential | kick ms (median) | min | max | pairs |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| arena | handoff1312 | cuda | 128 | 7 | adaptive | 0 | 79.677 | 78.232 | 81.122 | 2 |
| arena | handoff1312 | cuda | 128 | 7 | adaptive | 1 | 89.327 | 89.037 | 89.617 | 2 |
| arena | handoff1312 | cuda | 128 | 7 | clustered | 0 | 70.235 | 69.594 | 70.877 | 2 |
| arena | handoff1312 | cuda | 128 | 7 | clustered | 1 | 77.718 | 77.216 | 78.221 | 2 |
| arena | handoff1312 | cuda | 128 | 7 | lattice | 0 | 92.089 | 90.952 | 93.226 | 2 |
| arena | handoff1312 | cuda | 128 | 7 | lattice | 1 | 102.883 | 101.777 | 103.989 | 2 |
| arena | handoff328 | cuda | 128 | 7 | adaptive | 0 | 88.229 | 84.662 | 91.796 | 2 |
| arena | handoff328 | cuda | 128 | 7 | adaptive | 1 | 105.948 | 102.059 | 109.836 | 2 |
| arena | handoff328 | cuda | 128 | 7 | clustered | 0 | 76.518 | 72.902 | 80.134 | 2 |
| arena | handoff328 | cuda | 128 | 7 | clustered | 1 | 91.198 | 87.805 | 94.590 | 2 |
| arena | handoff328 | cuda | 128 | 7 | lattice | 0 | 84.763 | 78.662 | 90.864 | 2 |
| arena | handoff328 | cuda | 128 | 7 | lattice | 1 | 97.838 | 91.582 | 104.095 | 2 |
| arena | handoff492 | cuda | 128 | 7 | adaptive | 0 | 76.371 | 74.027 | 78.715 | 2 |
| arena | handoff492 | cuda | 128 | 7 | adaptive | 1 | 92.357 | 91.772 | 92.943 | 2 |
| arena | handoff492 | cuda | 128 | 7 | clustered | 0 | 72.906 | 69.802 | 76.010 | 2 |
| arena | handoff492 | cuda | 128 | 7 | clustered | 1 | 87.018 | 86.982 | 87.054 | 2 |
| arena | handoff492 | cuda | 128 | 7 | lattice | 0 | 77.912 | 76.516 | 79.309 | 2 |
| arena | handoff492 | cuda | 128 | 7 | lattice | 1 | 93.472 | 92.167 | 94.777 | 2 |
| arena | handoff656 | cuda | 128 | 7 | adaptive | 0 | 78.042 | 76.167 | 79.916 | 2 |
| arena | handoff656 | cuda | 128 | 7 | adaptive | 1 | 93.022 | 90.097 | 95.947 | 2 |
| arena | handoff656 | cuda | 128 | 7 | clustered | 0 | 71.406 | 67.546 | 75.266 | 2 |
| arena | handoff656 | cuda | 128 | 7 | clustered | 1 | 82.728 | 79.324 | 86.132 | 2 |
| arena | handoff656 | cuda | 128 | 7 | lattice | 0 | 95.761 | 91.159 | 100.362 | 2 |
| arena | handoff656 | cuda | 128 | 7 | lattice | 1 | 106.034 | 102.670 | 109.398 | 2 |
| arena | handoff984 | cuda | 128 | 7 | adaptive | 0 | 78.073 | 76.583 | 79.564 | 2 |
| arena | handoff984 | cuda | 128 | 7 | adaptive | 1 | 90.147 | 88.721 | 91.573 | 2 |
| arena | handoff984 | cuda | 128 | 7 | clustered | 0 | 66.653 | 62.082 | 71.224 | 2 |
| arena | handoff984 | cuda | 128 | 7 | clustered | 1 | 79.591 | 75.338 | 83.843 | 2 |
| arena | handoff984 | cuda | 128 | 7 | lattice | 0 | 95.905 | 92.985 | 98.826 | 2 |
| arena | handoff984 | cuda | 128 | 7 | lattice | 1 | 107.170 | 102.127 | 112.213 | 2 |

| binary | policy | backend | bucket | order | kernel policy line |
|---|---|---|---:|---:|---|
| arena | handoff1312 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| arena | handoff328 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| arena | handoff492 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| arena | handoff656 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| arena | handoff984 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |

| comparison | binary | policy | bucket | order | fixture | acceleration rel L2 | acceleration max pointwise | potential rel L2 |
|---|---|---|---:|---:|---|---:|---:|---:|
