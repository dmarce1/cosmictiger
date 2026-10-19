| binary | policy | backend | bucket | order | fixture | potential | kick ms (median) | min | max | pairs |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| arena | handoff1312 | cuda | 128 | 7 | adaptive | 0 | 78.944 | 78.908 | 78.980 | 2 |
| arena | handoff1312 | cuda | 128 | 7 | adaptive | 1 | 93.246 | 92.841 | 93.651 | 2 |
| arena | handoff1312 | cuda | 128 | 7 | clustered | 0 | 63.814 | 61.953 | 65.675 | 2 |
| arena | handoff1312 | cuda | 128 | 7 | clustered | 1 | 76.541 | 75.554 | 77.527 | 2 |
| arena | handoff1312 | cuda | 128 | 7 | lattice | 0 | 103.521 | 103.205 | 103.837 | 2 |
| arena | handoff1312 | cuda | 128 | 7 | lattice | 1 | 114.016 | 113.820 | 114.212 | 2 |
| arena | handoff328 | cuda | 128 | 7 | adaptive | 0 | 91.607 | 88.335 | 94.879 | 2 |
| arena | handoff328 | cuda | 128 | 7 | adaptive | 1 | 109.929 | 106.241 | 113.617 | 2 |
| arena | handoff328 | cuda | 128 | 7 | clustered | 0 | 76.399 | 76.105 | 76.694 | 2 |
| arena | handoff328 | cuda | 128 | 7 | clustered | 1 | 90.738 | 87.362 | 94.114 | 2 |
| arena | handoff328 | cuda | 128 | 7 | lattice | 0 | 81.812 | 79.867 | 83.756 | 2 |
| arena | handoff328 | cuda | 128 | 7 | lattice | 1 | 99.633 | 95.502 | 103.764 | 2 |
| arena | handoff492 | cuda | 128 | 7 | adaptive | 0 | 80.773 | 79.015 | 82.532 | 2 |
| arena | handoff492 | cuda | 128 | 7 | adaptive | 1 | 95.049 | 91.613 | 98.486 | 2 |
| arena | handoff492 | cuda | 128 | 7 | clustered | 0 | 77.028 | 76.728 | 77.328 | 2 |
| arena | handoff492 | cuda | 128 | 7 | clustered | 1 | 93.110 | 91.449 | 94.771 | 2 |
| arena | handoff492 | cuda | 128 | 7 | lattice | 0 | 83.285 | 83.226 | 83.345 | 2 |
| arena | handoff492 | cuda | 128 | 7 | lattice | 1 | 98.312 | 95.400 | 101.225 | 2 |
| arena | handoff656 | cuda | 128 | 7 | adaptive | 0 | 75.322 | 74.575 | 76.070 | 2 |
| arena | handoff656 | cuda | 128 | 7 | adaptive | 1 | 91.469 | 90.480 | 92.459 | 2 |
| arena | handoff656 | cuda | 128 | 7 | clustered | 0 | 67.948 | 65.370 | 70.526 | 2 |
| arena | handoff656 | cuda | 128 | 7 | clustered | 1 | 86.696 | 85.097 | 88.294 | 2 |
| arena | handoff656 | cuda | 128 | 7 | lattice | 0 | 100.783 | 100.315 | 101.251 | 2 |
| arena | handoff656 | cuda | 128 | 7 | lattice | 1 | 115.342 | 113.964 | 116.720 | 2 |
| arena | handoff984 | cuda | 128 | 7 | adaptive | 0 | 77.744 | 77.728 | 77.761 | 2 |
| arena | handoff984 | cuda | 128 | 7 | adaptive | 1 | 94.547 | 93.787 | 95.306 | 2 |
| arena | handoff984 | cuda | 128 | 7 | clustered | 0 | 65.225 | 64.629 | 65.821 | 2 |
| arena | handoff984 | cuda | 128 | 7 | clustered | 1 | 76.983 | 75.299 | 78.667 | 2 |
| arena | handoff984 | cuda | 128 | 7 | lattice | 0 | 104.357 | 102.617 | 106.096 | 2 |
| arena | handoff984 | cuda | 128 | 7 | lattice | 1 | 116.166 | 115.508 | 116.823 | 2 |

| binary | policy | backend | bucket | order | kernel policy line |
|---|---|---|---:|---:|---|
| arena | handoff1312 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| arena | handoff328 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| arena | handoff492 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| arena | handoff656 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| arena | handoff984 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |

| comparison | binary | policy | bucket | order | fixture | acceleration rel L2 | acceleration max pointwise | potential rel L2 |
|---|---|---|---:|---:|---|---:|---:|---:|
