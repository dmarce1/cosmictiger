| binary | policy | backend | bucket | order | fixture | potential | kick ms (median) | min | max | pairs |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| arena | handoff1312 | cuda | 128 | 7 | adaptive | 0 | 78.563 | 76.579 | 80.548 | 2 |
| arena | handoff1312 | cuda | 128 | 7 | adaptive | 1 | 88.944 | 88.469 | 89.418 | 2 |
| arena | handoff1312 | cuda | 128 | 7 | clustered | 0 | 67.125 | 64.847 | 69.402 | 2 |
| arena | handoff1312 | cuda | 128 | 7 | clustered | 1 | 79.108 | 74.609 | 83.608 | 2 |
| arena | handoff1312 | cuda | 128 | 7 | lattice | 0 | 93.286 | 90.241 | 96.330 | 2 |
| arena | handoff1312 | cuda | 128 | 7 | lattice | 1 | 107.089 | 104.047 | 110.131 | 2 |
| arena | handoff328 | cuda | 128 | 7 | adaptive | 0 | 87.703 | 83.075 | 92.330 | 2 |
| arena | handoff328 | cuda | 128 | 7 | adaptive | 1 | 106.528 | 100.139 | 112.917 | 2 |
| arena | handoff328 | cuda | 128 | 7 | clustered | 0 | 74.684 | 69.127 | 80.240 | 2 |
| arena | handoff328 | cuda | 128 | 7 | clustered | 1 | 90.446 | 84.207 | 96.684 | 2 |
| arena | handoff328 | cuda | 128 | 7 | lattice | 0 | 82.409 | 80.979 | 83.839 | 2 |
| arena | handoff328 | cuda | 128 | 7 | lattice | 1 | 100.734 | 97.992 | 103.475 | 2 |
| arena | handoff492 | cuda | 128 | 7 | adaptive | 0 | 80.028 | 74.527 | 85.530 | 2 |
| arena | handoff492 | cuda | 128 | 7 | adaptive | 1 | 90.563 | 90.486 | 90.640 | 2 |
| arena | handoff492 | cuda | 128 | 7 | clustered | 0 | 85.737 | 80.825 | 90.649 | 2 |
| arena | handoff492 | cuda | 128 | 7 | clustered | 1 | 98.436 | 90.244 | 106.628 | 2 |
| arena | handoff492 | cuda | 128 | 7 | lattice | 0 | 86.715 | 78.206 | 95.224 | 2 |
| arena | handoff492 | cuda | 128 | 7 | lattice | 1 | 104.442 | 101.278 | 107.606 | 2 |
| arena | handoff656 | cuda | 128 | 7 | adaptive | 0 | 74.845 | 72.642 | 77.049 | 2 |
| arena | handoff656 | cuda | 128 | 7 | adaptive | 1 | 89.805 | 89.568 | 90.042 | 2 |
| arena | handoff656 | cuda | 128 | 7 | clustered | 0 | 68.465 | 66.648 | 70.281 | 2 |
| arena | handoff656 | cuda | 128 | 7 | clustered | 1 | 80.451 | 80.243 | 80.658 | 2 |
| arena | handoff656 | cuda | 128 | 7 | lattice | 0 | 97.570 | 97.362 | 97.778 | 2 |
| arena | handoff656 | cuda | 128 | 7 | lattice | 1 | 110.285 | 109.677 | 110.893 | 2 |
| arena | handoff984 | cuda | 128 | 7 | adaptive | 0 | 80.683 | 76.729 | 84.638 | 2 |
| arena | handoff984 | cuda | 128 | 7 | adaptive | 1 | 93.414 | 89.886 | 96.942 | 2 |
| arena | handoff984 | cuda | 128 | 7 | clustered | 0 | 65.594 | 62.383 | 68.804 | 2 |
| arena | handoff984 | cuda | 128 | 7 | clustered | 1 | 77.397 | 75.541 | 79.253 | 2 |
| arena | handoff984 | cuda | 128 | 7 | lattice | 0 | 96.922 | 96.072 | 97.772 | 2 |
| arena | handoff984 | cuda | 128 | 7 | lattice | 1 | 112.102 | 108.865 | 115.339 | 2 |

| binary | policy | backend | bucket | order | kernel policy line |
|---|---|---|---:|---:|---|
| arena | handoff1312 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| arena | handoff328 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| arena | handoff492 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| arena | handoff656 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| arena | handoff984 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |

| comparison | binary | policy | bucket | order | fixture | acceleration rel L2 | acceleration max pointwise | potential rel L2 |
|---|---|---|---:|---:|---|---:|---:|---:|
