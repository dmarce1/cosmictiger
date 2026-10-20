| backend | bucket | fixture | potential | binary | policy | order | kick ms | speedup vs reference | paired range | force rel L2 | potential rel L2 |
|---|---:|---|---:|---|---|---:|---:|---:|---|---:|---:|
| cuda | 64 | adaptive | 0 | aligned | auto | 7 | 12.077 | 1.019 | 0.956-1.062 | 2.262e-03 | 1.163e-04 |
| cuda | 64 | adaptive | 0 | head | auto | 7 | 12.127 |  |  | 2.262e-03 | 1.163e-04 |
| cuda | 64 | adaptive | 1 | aligned | auto | 7 | 12.317 | 1.010 | 1.000-1.017 | 2.262e-03 | 1.163e-04 |
| cuda | 64 | adaptive | 1 | head | auto | 7 | 12.403 |  |  | 2.262e-03 | 1.163e-04 |
| cuda | 64 | clustered | 0 | aligned | auto | 7 | 5.551 | 0.996 | 0.983-1.008 | 5.305e-03 | 1.096e-03 |
| cuda | 64 | clustered | 0 | head | auto | 7 | 5.573 |  |  | 5.305e-03 | 1.096e-03 |
| cuda | 64 | clustered | 1 | aligned | auto | 7 | 5.880 | 0.988 | 0.960-1.003 | 5.305e-03 | 1.096e-03 |
| cuda | 64 | clustered | 1 | head | auto | 7 | 5.736 |  |  | 5.305e-03 | 1.096e-03 |
| cuda | 64 | lattice | 0 | aligned | auto | 7 | 14.677 | 1.000 | 0.985-1.023 | 3.644e-01 | 4.476e-02 |
| cuda | 64 | lattice | 0 | head | auto | 7 | 14.705 |  |  | 3.644e-01 | 4.476e-02 |
| cuda | 64 | lattice | 1 | aligned | auto | 7 | 14.728 | 1.003 | 0.988-1.026 | 3.644e-01 | 4.476e-02 |
| cuda | 64 | lattice | 1 | head | auto | 7 | 14.810 |  |  | 3.644e-01 | 4.476e-02 |
| cuda | 512 | adaptive | 0 | aligned | auto | 7 | 26.139 | 1.024 | 1.011-1.034 | 1.015e-03 | 1.108e-04 |
| cuda | 512 | adaptive | 0 | head | auto | 7 | 26.774 |  |  | 1.015e-03 | 1.108e-04 |
| cuda | 512 | adaptive | 1 | aligned | auto | 7 | 27.151 | 1.004 | 0.925-1.016 | 1.015e-03 | 1.108e-04 |
| cuda | 512 | adaptive | 1 | head | auto | 7 | 27.475 |  |  | 1.015e-03 | 1.108e-04 |
| cuda | 512 | clustered | 0 | aligned | auto | 7 | 5.376 | 0.996 | 0.927-1.015 | 2.146e-03 | 1.120e-03 |
| cuda | 512 | clustered | 0 | head | auto | 7 | 5.454 |  |  | 2.146e-03 | 1.120e-03 |
| cuda | 512 | clustered | 1 | aligned | auto | 7 | 6.110 | 0.996 | 0.962-1.023 | 2.146e-03 | 1.120e-03 |
| cuda | 512 | clustered | 1 | head | auto | 7 | 6.111 |  |  | 2.146e-03 | 1.120e-03 |
| cuda | 512 | lattice | 0 | aligned | auto | 7 | 9.494 | 1.008 | 0.997-1.035 | 3.644e-01 | 4.476e-02 |
| cuda | 512 | lattice | 0 | head | auto | 7 | 9.567 |  |  | 3.644e-01 | 4.476e-02 |
| cuda | 512 | lattice | 1 | aligned | auto | 7 | 10.103 | 1.002 | 0.981-1.032 | 3.644e-01 | 4.476e-02 |
| cuda | 512 | lattice | 1 | head | auto | 7 | 10.089 |  |  | 3.644e-01 | 4.476e-02 |
