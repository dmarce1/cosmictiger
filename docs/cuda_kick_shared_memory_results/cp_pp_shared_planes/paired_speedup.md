| backend | bucket | fixture | potential | binary | policy | order | kick ms | speedup vs reference | paired range | force rel L2 | potential rel L2 |
|---|---:|---|---:|---|---|---:|---:|---:|---|---:|---:|
| cuda | 64 | adaptive | 0 | head | auto | 7 | 10.434 |  |  | 2.262e-03 | 1.163e-04 |
| cuda | 64 | adaptive | 0 | shared | auto | 7 | 10.332 | 1.012 | 0.950-1.028 | 2.262e-03 | 1.163e-04 |
| cuda | 64 | adaptive | 1 | head | auto | 7 | 10.713 |  |  | 2.262e-03 | 1.163e-04 |
| cuda | 64 | adaptive | 1 | shared | auto | 7 | 10.474 | 1.020 | 0.996-1.038 | 2.262e-03 | 1.163e-04 |
| cuda | 64 | clustered | 0 | head | auto | 7 | 5.746 |  |  | 5.305e-03 | 1.096e-03 |
| cuda | 64 | clustered | 0 | shared | auto | 7 | 5.729 | 0.996 | 0.978-1.031 | 5.305e-03 | 1.096e-03 |
| cuda | 64 | clustered | 1 | head | auto | 7 | 5.866 |  |  | 5.305e-03 | 1.096e-03 |
| cuda | 64 | clustered | 1 | shared | auto | 7 | 5.971 | 0.979 | 0.967-1.016 | 5.305e-03 | 1.096e-03 |
| cuda | 64 | lattice | 0 | head | auto | 7 | 14.604 |  |  | 3.644e-01 | 4.476e-02 |
| cuda | 64 | lattice | 0 | shared | auto | 7 | 14.339 | 1.008 | 0.993-1.064 | 3.644e-01 | 4.476e-02 |
| cuda | 64 | lattice | 1 | head | auto | 7 | 14.679 |  |  | 3.644e-01 | 4.476e-02 |
| cuda | 64 | lattice | 1 | shared | auto | 7 | 14.574 | 1.012 | 0.996-1.014 | 3.644e-01 | 4.476e-02 |
| cuda | 512 | adaptive | 0 | head | auto | 7 | 19.480 |  |  | 1.015e-03 | 1.108e-04 |
| cuda | 512 | adaptive | 0 | shared | auto | 7 | 19.735 | 1.003 | 0.966-1.021 | 1.015e-03 | 1.108e-04 |
| cuda | 512 | adaptive | 1 | head | auto | 7 | 20.573 |  |  | 1.015e-03 | 1.108e-04 |
| cuda | 512 | adaptive | 1 | shared | auto | 7 | 20.455 | 0.996 | 0.975-1.035 | 1.015e-03 | 1.108e-04 |
| cuda | 512 | clustered | 0 | head | auto | 7 | 5.283 |  |  | 2.146e-03 | 1.120e-03 |
| cuda | 512 | clustered | 0 | shared | auto | 7 | 5.282 | 0.992 | 0.976-1.081 | 2.146e-03 | 1.120e-03 |
| cuda | 512 | clustered | 1 | head | auto | 7 | 6.038 |  |  | 2.146e-03 | 1.120e-03 |
| cuda | 512 | clustered | 1 | shared | auto | 7 | 5.987 | 0.984 | 0.978-1.057 | 2.146e-03 | 1.120e-03 |
| cuda | 512 | lattice | 0 | head | auto | 7 | 7.618 |  |  | 3.644e-01 | 4.476e-02 |
| cuda | 512 | lattice | 0 | shared | auto | 7 | 7.736 | 0.981 | 0.967-1.001 | 3.644e-01 | 4.476e-02 |
| cuda | 512 | lattice | 1 | head | auto | 7 | 8.322 |  |  | 3.644e-01 | 4.476e-02 |
| cuda | 512 | lattice | 1 | shared | auto | 7 | 8.315 | 1.029 | 0.966-1.038 | 3.644e-01 | 4.476e-02 |
