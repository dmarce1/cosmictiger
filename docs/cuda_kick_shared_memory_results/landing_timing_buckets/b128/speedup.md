| backend | bucket | fixture | potential | binary | policy | order | kick ms | speedup vs reference | paired range | force rel L2 | potential rel L2 |
|---|---:|---|---:|---|---|---:|---:|---:|---|---:|---:|
| cuda | 128 | adaptive | 0 | merged | auto | 7 | 12.477 | 1.018 | 1.002-1.068 | 1.869e-03 | 1.191e-04 |
| cuda | 128 | adaptive | 0 | premerge | auto | 7 | 12.668 |  |  | 1.869e-03 | 1.191e-04 |
| cuda | 128 | adaptive | 1 | merged | auto | 7 | 12.725 | 0.997 | 0.982-1.010 | 1.869e-03 | 1.191e-04 |
| cuda | 128 | adaptive | 1 | premerge | auto | 7 | 12.725 |  |  | 1.869e-03 | 1.191e-04 |
| cuda | 128 | clustered | 0 | merged | auto | 7 | 6.262 | 1.082 | 1.026-1.112 | 2.962e-03 | 1.017e-03 |
| cuda | 128 | clustered | 0 | premerge | auto | 7 | 6.837 |  |  | 2.962e-03 | 1.017e-03 |
| cuda | 128 | clustered | 1 | merged | auto | 7 | 6.529 | 1.088 | 1.081-1.118 | 2.962e-03 | 1.017e-03 |
| cuda | 128 | clustered | 1 | premerge | auto | 7 | 7.075 |  |  | 2.962e-03 | 1.017e-03 |
| cuda | 128 | lattice | 0 | merged | auto | 7 | 15.208 | 1.008 | 0.993-1.013 | 3.644e-01 | 4.476e-02 |
| cuda | 128 | lattice | 0 | premerge | auto | 7 | 15.337 |  |  | 3.644e-01 | 4.476e-02 |
| cuda | 128 | lattice | 1 | merged | auto | 7 | 15.308 | 0.990 | 0.978-1.045 | 3.644e-01 | 4.476e-02 |
| cuda | 128 | lattice | 1 | premerge | auto | 7 | 15.253 |  |  | 3.644e-01 | 4.476e-02 |
