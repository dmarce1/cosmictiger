| binary | policy | backend | bucket | order | fixture | potential | kick ms (median) | min | max | pairs |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| ewald | auto | cuda | 128 | 7 | adaptive | 0 | 63.661 | 63.529 | 63.794 | 2 |
| ewald | auto | cuda | 128 | 7 | adaptive | 1 | 80.467 | 79.642 | 81.293 | 2 |
| ewald | auto | cuda | 128 | 7 | clustered | 0 | 59.157 | 57.639 | 60.674 | 2 |
| ewald | auto | cuda | 128 | 7 | clustered | 1 | 75.436 | 75.376 | 75.495 | 2 |
| ewald | auto | cuda | 128 | 7 | lattice | 0 | 56.842 | 55.239 | 58.446 | 2 |
| ewald | auto | cuda | 128 | 7 | lattice | 1 | 66.315 | 66.033 | 66.596 | 2 |
| ewald | fine1312 | cuda | 128 | 7 | adaptive | 0 | 75.655 | 75.018 | 76.292 | 2 |
| ewald | fine1312 | cuda | 128 | 7 | adaptive | 1 | 91.232 | 91.196 | 91.267 | 2 |
| ewald | fine1312 | cuda | 128 | 7 | clustered | 0 | 56.761 | 56.756 | 56.767 | 2 |
| ewald | fine1312 | cuda | 128 | 7 | clustered | 1 | 71.345 | 70.669 | 72.022 | 2 |
| ewald | fine1312 | cuda | 128 | 7 | lattice | 0 | 58.000 | 56.631 | 59.370 | 2 |
| ewald | fine1312 | cuda | 128 | 7 | lattice | 1 | 69.784 | 69.465 | 70.104 | 2 |
| ewald | fine656 | cuda | 128 | 7 | adaptive | 0 | 73.520 | 72.260 | 74.781 | 2 |
| ewald | fine656 | cuda | 128 | 7 | adaptive | 1 | 90.352 | 90.008 | 90.697 | 2 |
| ewald | fine656 | cuda | 128 | 7 | clustered | 0 | 56.258 | 54.472 | 58.045 | 2 |
| ewald | fine656 | cuda | 128 | 7 | clustered | 1 | 70.240 | 70.076 | 70.405 | 2 |
| ewald | fine656 | cuda | 128 | 7 | lattice | 0 | 58.141 | 56.923 | 59.359 | 2 |
| ewald | fine656 | cuda | 128 | 7 | lattice | 1 | 69.467 | 69.445 | 69.489 | 2 |
| ewald | fine984 | cuda | 128 | 7 | adaptive | 0 | 74.108 | 73.970 | 74.247 | 2 |
| ewald | fine984 | cuda | 128 | 7 | adaptive | 1 | 94.030 | 93.365 | 94.695 | 2 |
| ewald | fine984 | cuda | 128 | 7 | clustered | 0 | 54.639 | 54.551 | 54.728 | 2 |
| ewald | fine984 | cuda | 128 | 7 | clustered | 1 | 69.931 | 67.538 | 72.325 | 2 |
| ewald | fine984 | cuda | 128 | 7 | lattice | 0 | 58.237 | 57.726 | 58.748 | 2 |
| ewald | fine984 | cuda | 128 | 7 | lattice | 1 | 68.458 | 68.182 | 68.735 | 2 |
| ewald | handoff492 | cuda | 128 | 7 | adaptive | 0 | 65.133 | 63.779 | 66.487 | 2 |
| ewald | handoff492 | cuda | 128 | 7 | adaptive | 1 | 82.394 | 80.355 | 84.433 | 2 |
| ewald | handoff492 | cuda | 128 | 7 | clustered | 0 | 60.162 | 57.158 | 63.167 | 2 |
| ewald | handoff492 | cuda | 128 | 7 | clustered | 1 | 78.117 | 75.175 | 81.058 | 2 |
| ewald | handoff492 | cuda | 128 | 7 | lattice | 0 | 58.538 | 58.501 | 58.574 | 2 |
| ewald | handoff492 | cuda | 128 | 7 | lattice | 1 | 68.283 | 66.624 | 69.941 | 2 |
| ewald | handoff656 | cuda | 128 | 7 | adaptive | 0 | 66.864 | 66.581 | 67.147 | 2 |
| ewald | handoff656 | cuda | 128 | 7 | adaptive | 1 | 86.543 | 86.298 | 86.788 | 2 |
| ewald | handoff656 | cuda | 128 | 7 | clustered | 0 | 55.010 | 54.214 | 55.806 | 2 |
| ewald | handoff656 | cuda | 128 | 7 | clustered | 1 | 70.128 | 69.932 | 70.324 | 2 |
| ewald | handoff656 | cuda | 128 | 7 | lattice | 0 | 69.298 | 67.634 | 70.963 | 2 |
| ewald | handoff656 | cuda | 128 | 7 | lattice | 1 | 81.889 | 81.587 | 82.190 | 2 |

| binary | policy | backend | bucket | order | kernel policy line |
|---|---|---|---:|---:|---|
| ewald | auto | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 9072 bytes: control 648, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes; list slab 145792 bytes per block (4096 entries per list)` |
| ewald | fine1312 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 9072 bytes: control 648, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes; list slab 145792 bytes per block (4096 entries per list)` |
| ewald | fine656 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 9072 bytes: control 648, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes; list slab 145792 bytes per block (4096 entries per list)` |
| ewald | fine984 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 9072 bytes: control 648, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes; list slab 145792 bytes per block (4096 entries per list)` |
| ewald | handoff492 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 9072 bytes: control 648, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes; list slab 145792 bytes per block (4096 entries per list)` |
| ewald | handoff656 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 9072 bytes: control 648, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes; list slab 145792 bytes per block (4096 entries per list)` |

| comparison | binary | policy | bucket | order | fixture | acceleration rel L2 | acceleration max pointwise | potential rel L2 |
|---|---|---|---:|---:|---|---:|---:|---:|
| vs_ewald | ewald | fine1312 | 128 | 7 | adaptive | 9.715e-08 | 1.280e-06 | 8.774e-08 |
| vs_ewald | ewald | fine1312 | 128 | 7 | clustered | 8.664e-08 | 3.653e-06 | 8.512e-08 |
| vs_ewald | ewald | fine1312 | 128 | 7 | lattice | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| vs_ewald | ewald | fine656 | 128 | 7 | adaptive | 7.089e-08 | 3.053e-06 | 6.046e-08 |
| vs_ewald | ewald | fine656 | 128 | 7 | clustered | 5.693e-08 | 4.800e-07 | 5.780e-08 |
| vs_ewald | ewald | fine656 | 128 | 7 | lattice | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| vs_ewald | ewald | fine984 | 128 | 7 | adaptive | 7.581e-08 | 3.053e-06 | 6.666e-08 |
| vs_ewald | ewald | fine984 | 128 | 7 | clustered | 6.954e-08 | 3.653e-06 | 7.844e-08 |
| vs_ewald | ewald | fine984 | 128 | 7 | lattice | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| vs_ewald | ewald | handoff492 | 128 | 7 | adaptive | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| vs_ewald | ewald | handoff492 | 128 | 7 | clustered | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| vs_ewald | ewald | handoff492 | 128 | 7 | lattice | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| vs_ewald | ewald | handoff656 | 128 | 7 | adaptive | 7.100e-08 | 6.624e-06 | 6.098e-08 |
| vs_ewald | ewald | handoff656 | 128 | 7 | clustered | 5.693e-08 | 4.800e-07 | 5.780e-08 |
| vs_ewald | ewald | handoff656 | 128 | 7 | lattice | 1.792e-05 | 2.556e-03 | 4.388e-04 |
