| binary | policy | backend | bucket | order | fixture | potential | kick ms (median) | min | max | pairs |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| counts | auto | cuda | 128 | 7 | adaptive | 0 | 72.024 | 72.024 | 72.024 | 1 |
| counts | auto | cuda | 128 | 7 | adaptive | 1 | 91.538 | 91.538 | 91.538 | 1 |
| counts | auto | cuda | 128 | 7 | clustered | 0 | 79.865 | 79.865 | 79.865 | 1 |
| counts | auto | cuda | 128 | 7 | clustered | 1 | 86.653 | 86.653 | 86.653 | 1 |
| counts | auto | cuda | 128 | 7 | lattice | 0 | 69.845 | 69.845 | 69.845 | 1 |
| counts | auto | cuda | 128 | 7 | lattice | 1 | 92.088 | 92.088 | 92.088 | 1 |
| counts | fine1312 | cuda | 128 | 7 | adaptive | 0 | 83.381 | 83.381 | 83.381 | 1 |
| counts | fine1312 | cuda | 128 | 7 | adaptive | 1 | 94.883 | 94.883 | 94.883 | 1 |
| counts | fine1312 | cuda | 128 | 7 | clustered | 0 | 60.286 | 60.286 | 60.286 | 1 |
| counts | fine1312 | cuda | 128 | 7 | clustered | 1 | 75.624 | 75.624 | 75.624 | 1 |
| counts | fine1312 | cuda | 128 | 7 | lattice | 0 | 65.060 | 65.060 | 65.060 | 1 |
| counts | fine1312 | cuda | 128 | 7 | lattice | 1 | 75.412 | 75.412 | 75.412 | 1 |
| counts | fine656 | cuda | 128 | 7 | adaptive | 0 | 78.159 | 78.159 | 78.159 | 1 |
| counts | fine656 | cuda | 128 | 7 | adaptive | 1 | 93.732 | 93.732 | 93.732 | 1 |
| counts | fine656 | cuda | 128 | 7 | clustered | 0 | 62.478 | 62.478 | 62.478 | 1 |
| counts | fine656 | cuda | 128 | 7 | clustered | 1 | 75.897 | 75.897 | 75.897 | 1 |
| counts | fine656 | cuda | 128 | 7 | lattice | 0 | 65.983 | 65.983 | 65.983 | 1 |
| counts | fine656 | cuda | 128 | 7 | lattice | 1 | 72.990 | 72.990 | 72.990 | 1 |
| counts | fine984 | cuda | 128 | 7 | adaptive | 0 | 82.226 | 82.226 | 82.226 | 1 |
| counts | fine984 | cuda | 128 | 7 | adaptive | 1 | 88.532 | 88.532 | 88.532 | 1 |
| counts | fine984 | cuda | 128 | 7 | clustered | 0 | 56.284 | 56.284 | 56.284 | 1 |
| counts | fine984 | cuda | 128 | 7 | clustered | 1 | 74.673 | 74.673 | 74.673 | 1 |
| counts | fine984 | cuda | 128 | 7 | lattice | 0 | 69.462 | 69.462 | 69.462 | 1 |
| counts | fine984 | cuda | 128 | 7 | lattice | 1 | 78.594 | 78.594 | 78.594 | 1 |
| counts | handoff492 | cuda | 128 | 7 | adaptive | 0 | 71.144 | 71.144 | 71.144 | 1 |
| counts | handoff492 | cuda | 128 | 7 | adaptive | 1 | 88.031 | 88.031 | 88.031 | 1 |
| counts | handoff492 | cuda | 128 | 7 | clustered | 0 | 63.990 | 63.990 | 63.990 | 1 |
| counts | handoff492 | cuda | 128 | 7 | clustered | 1 | 83.510 | 83.510 | 83.510 | 1 |
| counts | handoff492 | cuda | 128 | 7 | lattice | 0 | 65.628 | 65.628 | 65.628 | 1 |
| counts | handoff492 | cuda | 128 | 7 | lattice | 1 | 78.936 | 78.936 | 78.936 | 1 |
| counts | handoff656 | cuda | 128 | 7 | adaptive | 0 | 70.407 | 70.407 | 70.407 | 1 |
| counts | handoff656 | cuda | 128 | 7 | adaptive | 1 | 90.582 | 90.582 | 90.582 | 1 |
| counts | handoff656 | cuda | 128 | 7 | clustered | 0 | 62.077 | 62.077 | 62.077 | 1 |
| counts | handoff656 | cuda | 128 | 7 | clustered | 1 | 75.230 | 75.230 | 75.230 | 1 |
| counts | handoff656 | cuda | 128 | 7 | lattice | 0 | 89.106 | 89.106 | 89.106 | 1 |
| counts | handoff656 | cuda | 128 | 7 | lattice | 1 | 97.864 | 97.864 | 97.864 | 1 |

| binary | policy | backend | bucket | order | kernel policy line |
|---|---|---|---:|---:|---|
| counts | auto | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 9072 bytes: control 648, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes; list slab 145792 bytes per block (4096 entries per list)` |
| counts | fine1312 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 9072 bytes: control 648, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes; list slab 145792 bytes per block (4096 entries per list)` |
| counts | fine656 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 9072 bytes: control 648, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes; list slab 145792 bytes per block (4096 entries per list)` |
| counts | fine984 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 9072 bytes: control 648, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes; list slab 145792 bytes per block (4096 entries per list)` |
| counts | handoff492 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 9072 bytes: control 648, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes; list slab 145792 bytes per block (4096 entries per list)` |
| counts | handoff656 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 9072 bytes: control 648, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes; list slab 145792 bytes per block (4096 entries per list)` |

| comparison | binary | policy | bucket | order | fixture | acceleration rel L2 | acceleration max pointwise | potential rel L2 |
|---|---|---|---:|---:|---|---:|---:|---:|
| vs_counts | counts | fine1312 | 128 | 7 | adaptive | 9.715e-08 | 1.280e-06 | 8.774e-08 |
| vs_counts | counts | fine1312 | 128 | 7 | clustered | 8.664e-08 | 3.653e-06 | 8.512e-08 |
| vs_counts | counts | fine1312 | 128 | 7 | lattice | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| vs_counts | counts | fine656 | 128 | 7 | adaptive | 7.089e-08 | 3.053e-06 | 6.046e-08 |
| vs_counts | counts | fine656 | 128 | 7 | clustered | 5.693e-08 | 4.800e-07 | 5.780e-08 |
| vs_counts | counts | fine656 | 128 | 7 | lattice | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| vs_counts | counts | fine984 | 128 | 7 | adaptive | 7.581e-08 | 3.053e-06 | 6.666e-08 |
| vs_counts | counts | fine984 | 128 | 7 | clustered | 6.954e-08 | 3.653e-06 | 7.844e-08 |
| vs_counts | counts | fine984 | 128 | 7 | lattice | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| vs_counts | counts | handoff492 | 128 | 7 | adaptive | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| vs_counts | counts | handoff492 | 128 | 7 | clustered | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| vs_counts | counts | handoff492 | 128 | 7 | lattice | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| vs_counts | counts | handoff656 | 128 | 7 | adaptive | 7.100e-08 | 6.624e-06 | 6.098e-08 |
| vs_counts | counts | handoff656 | 128 | 7 | clustered | 5.693e-08 | 4.800e-07 | 5.780e-08 |
| vs_counts | counts | handoff656 | 128 | 7 | lattice | 1.792e-05 | 2.556e-03 | 4.388e-04 |
