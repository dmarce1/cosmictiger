| backend | bucket | fixture | potential | binary | policy | order | kick ms | speedup vs reference | paired range | force rel L2 | potential rel L2 |
|---|---:|---|---:|---|---|---:|---:|---:|---|---:|---:|
| cuda | 128 | adaptive | 0 | baseline | auto | 7 | 16.121 | 0.768 | 0.760-0.801 | 1.770e-03 | 1.035e-04 |
| cuda | 128 | adaptive | 0 | ewald | auto | 7 | 10.513 | 1.204 | 1.139-1.226 | 1.869e-03 | 1.191e-04 |
| cuda | 128 | adaptive | 0 | head | auto | 7 | 12.389 |  |  | 1.869e-03 | 1.191e-04 |
| cuda | 128 | adaptive | 1 | baseline | auto | 7 | 16.262 | 0.782 | 0.767-0.784 | 1.770e-03 | 1.035e-04 |
| cuda | 128 | adaptive | 1 | ewald | auto | 7 | 10.805 | 1.180 | 1.154-1.180 | 1.869e-03 | 1.191e-04 |
| cuda | 128 | adaptive | 1 | head | auto | 7 | 12.717 |  |  | 1.869e-03 | 1.191e-04 |
| cuda | 128 | clustered | 0 | baseline | auto | 7 | 6.309 | 0.998 | 0.952-1.022 | 2.241e-03 | 5.016e-04 |
| cuda | 128 | clustered | 0 | ewald | auto | 7 | 6.278 | 1.003 | 0.990-1.010 | 2.962e-03 | 1.017e-03 |
| cuda | 128 | clustered | 0 | head | auto | 7 | 6.295 |  |  | 2.962e-03 | 1.017e-03 |
| cuda | 128 | clustered | 1 | baseline | auto | 7 | 6.411 | 1.028 | 1.008-1.040 | 2.241e-03 | 5.016e-04 |
| cuda | 128 | clustered | 1 | ewald | auto | 7 | 6.548 | 1.006 | 0.986-1.020 | 2.962e-03 | 1.017e-03 |
| cuda | 128 | clustered | 1 | head | auto | 7 | 6.667 |  |  | 2.962e-03 | 1.017e-03 |
| cuda | 128 | lattice | 0 | baseline | auto | 7 | 14.746 | 0.999 | 0.979-1.009 | 3.418e-01 | 3.613e-02 |
| cuda | 128 | lattice | 0 | ewald | auto | 7 | 14.337 | 1.029 | 1.028-1.030 | 3.644e-01 | 4.476e-02 |
| cuda | 128 | lattice | 0 | head | auto | 7 | 14.736 |  |  | 3.644e-01 | 4.476e-02 |
| cuda | 128 | lattice | 1 | baseline | auto | 7 | 15.010 | 0.989 | 0.946-0.994 | 3.418e-01 | 3.613e-02 |
| cuda | 128 | lattice | 1 | ewald | auto | 7 | 14.543 | 1.015 | 1.013-1.024 | 3.644e-01 | 4.476e-02 |
| cuda | 128 | lattice | 1 | head | auto | 7 | 14.796 |  |  | 3.644e-01 | 4.476e-02 |
