| binary | policy | backend | bucket | order | fixture | potential | kick ms (median) | min | max | pairs |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| ewald | auto | cuda | 128 | 7 | adaptive | 0 | 67.687 | 66.050 | 69.324 | 2 |
| ewald | auto | cuda | 128 | 7 | adaptive | 1 | 83.509 | 81.544 | 85.474 | 2 |
| ewald | auto | cuda | 128 | 7 | clustered | 0 | 63.149 | 63.021 | 63.277 | 2 |
| ewald | auto | cuda | 128 | 7 | clustered | 1 | 80.617 | 79.771 | 81.463 | 2 |
| ewald | auto | cuda | 128 | 7 | lattice | 0 | 66.065 | 62.682 | 69.447 | 2 |
| ewald | auto | cuda | 128 | 7 | lattice | 1 | 74.924 | 74.721 | 75.126 | 2 |
| ewald | fine1312 | cuda | 128 | 7 | adaptive | 0 | 78.116 | 75.801 | 80.430 | 2 |
| ewald | fine1312 | cuda | 128 | 7 | adaptive | 1 | 94.812 | 94.552 | 95.072 | 2 |
| ewald | fine1312 | cuda | 128 | 7 | clustered | 0 | 56.650 | 56.515 | 56.784 | 2 |
| ewald | fine1312 | cuda | 128 | 7 | clustered | 1 | 71.114 | 70.485 | 71.744 | 2 |
| ewald | fine1312 | cuda | 128 | 7 | lattice | 0 | 62.755 | 62.173 | 63.338 | 2 |
| ewald | fine1312 | cuda | 128 | 7 | lattice | 1 | 72.528 | 71.968 | 73.088 | 2 |
| ewald | fine656 | cuda | 128 | 7 | adaptive | 0 | 74.082 | 73.458 | 74.706 | 2 |
| ewald | fine656 | cuda | 128 | 7 | adaptive | 1 | 94.935 | 94.150 | 95.720 | 2 |
| ewald | fine656 | cuda | 128 | 7 | clustered | 0 | 58.264 | 57.346 | 59.182 | 2 |
| ewald | fine656 | cuda | 128 | 7 | clustered | 1 | 74.635 | 74.127 | 75.144 | 2 |
| ewald | fine656 | cuda | 128 | 7 | lattice | 0 | 62.247 | 61.639 | 62.855 | 2 |
| ewald | fine656 | cuda | 128 | 7 | lattice | 1 | 72.660 | 72.007 | 73.313 | 2 |
| ewald | fine984 | cuda | 128 | 7 | adaptive | 0 | 77.204 | 76.223 | 78.184 | 2 |
| ewald | fine984 | cuda | 128 | 7 | adaptive | 1 | 93.223 | 92.929 | 93.516 | 2 |
| ewald | fine984 | cuda | 128 | 7 | clustered | 0 | 54.673 | 54.268 | 55.077 | 2 |
| ewald | fine984 | cuda | 128 | 7 | clustered | 1 | 71.367 | 70.641 | 72.093 | 2 |
| ewald | fine984 | cuda | 128 | 7 | lattice | 0 | 62.113 | 61.482 | 62.744 | 2 |
| ewald | fine984 | cuda | 128 | 7 | lattice | 1 | 72.438 | 71.883 | 72.993 | 2 |
| ewald | handoff492 | cuda | 128 | 7 | adaptive | 0 | 70.714 | 67.516 | 73.912 | 2 |
| ewald | handoff492 | cuda | 128 | 7 | adaptive | 1 | 84.802 | 83.939 | 85.664 | 2 |
| ewald | handoff492 | cuda | 128 | 7 | clustered | 0 | 62.782 | 60.333 | 65.232 | 2 |
| ewald | handoff492 | cuda | 128 | 7 | clustered | 1 | 80.016 | 76.133 | 83.900 | 2 |
| ewald | handoff492 | cuda | 128 | 7 | lattice | 0 | 64.422 | 62.879 | 65.965 | 2 |
| ewald | handoff492 | cuda | 128 | 7 | lattice | 1 | 73.629 | 70.161 | 77.098 | 2 |
| ewald | handoff656 | cuda | 128 | 7 | adaptive | 0 | 72.707 | 71.181 | 74.234 | 2 |
| ewald | handoff656 | cuda | 128 | 7 | adaptive | 1 | 89.837 | 88.794 | 90.880 | 2 |
| ewald | handoff656 | cuda | 128 | 7 | clustered | 0 | 59.263 | 58.181 | 60.345 | 2 |
| ewald | handoff656 | cuda | 128 | 7 | clustered | 1 | 75.788 | 74.821 | 76.755 | 2 |
| ewald | handoff656 | cuda | 128 | 7 | lattice | 0 | 79.637 | 77.516 | 81.757 | 2 |
| ewald | handoff656 | cuda | 128 | 7 | lattice | 1 | 91.551 | 90.225 | 92.877 | 2 |

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
