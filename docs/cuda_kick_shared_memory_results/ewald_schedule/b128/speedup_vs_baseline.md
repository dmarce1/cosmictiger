| backend | bucket | fixture | potential | binary | policy | order | kick ms | speedup vs reference | paired range | force rel L2 | potential rel L2 |
|---|---:|---|---:|---|---|---:|---:|---:|---|---:|---:|
| cuda | 128 | adaptive | 0 | baseline | auto | 7 | 16.121 |  |  | 1.770e-03 | 1.035e-04 |
| cuda | 128 | adaptive | 0 | ewald | auto | 7 | 10.513 | 1.532 | 1.500-1.568 | 1.869e-03 | 1.191e-04 |
| cuda | 128 | adaptive | 0 | head | auto | 7 | 12.389 | 1.302 | 1.249-1.316 | 1.869e-03 | 1.191e-04 |
| cuda | 128 | adaptive | 1 | baseline | auto | 7 | 16.262 |  |  | 1.770e-03 | 1.035e-04 |
| cuda | 128 | adaptive | 1 | ewald | auto | 7 | 10.805 | 1.506 | 1.504-1.509 | 1.869e-03 | 1.191e-04 |
| cuda | 128 | adaptive | 1 | head | auto | 7 | 12.717 | 1.279 | 1.276-1.303 | 1.869e-03 | 1.191e-04 |
| cuda | 128 | clustered | 0 | baseline | auto | 7 | 6.309 |  |  | 2.241e-03 | 5.016e-04 |
| cuda | 128 | clustered | 0 | ewald | auto | 7 | 6.278 | 1.012 | 0.969-1.054 | 2.962e-03 | 1.017e-03 |
| cuda | 128 | clustered | 0 | head | auto | 7 | 6.295 | 1.002 | 0.979-1.051 | 2.962e-03 | 1.017e-03 |
| cuda | 128 | clustered | 1 | baseline | auto | 7 | 6.411 |  |  | 2.241e-03 | 5.016e-04 |
| cuda | 128 | clustered | 1 | ewald | auto | 7 | 6.548 | 0.978 | 0.967-0.992 | 2.962e-03 | 1.017e-03 |
| cuda | 128 | clustered | 1 | head | auto | 7 | 6.667 | 0.973 | 0.962-0.992 | 2.962e-03 | 1.017e-03 |
| cuda | 128 | lattice | 0 | baseline | auto | 7 | 14.746 |  |  | 3.418e-01 | 3.613e-02 |
| cuda | 128 | lattice | 0 | ewald | auto | 7 | 14.337 | 1.030 | 1.018-1.051 | 3.644e-01 | 4.476e-02 |
| cuda | 128 | lattice | 0 | head | auto | 7 | 14.736 | 1.001 | 0.991-1.022 | 3.644e-01 | 4.476e-02 |
| cuda | 128 | lattice | 1 | baseline | auto | 7 | 15.010 |  |  | 3.418e-01 | 3.613e-02 |
| cuda | 128 | lattice | 1 | ewald | auto | 7 | 14.543 | 1.036 | 1.021-1.070 | 3.644e-01 | 4.476e-02 |
| cuda | 128 | lattice | 1 | head | auto | 7 | 14.796 | 1.012 | 1.006-1.057 | 3.644e-01 | 4.476e-02 |
