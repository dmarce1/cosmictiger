| backend | bucket | fixture | potential | binary | policy | order | kick ms | speedup vs reference | paired range | force rel L2 | potential rel L2 |
|---|---:|---|---:|---|---|---:|---:|---:|---|---:|---:|
| cuda | 64 | adaptive | 0 | baseline | auto | 7 | 18.059 | 0.669 | 0.664-0.684 | 2.111e-03 | 1.025e-04 |
| cuda | 64 | adaptive | 0 | ewald | auto | 7 | 10.359 | 1.175 | 1.092-1.186 | 2.262e-03 | 1.163e-04 |
| cuda | 64 | adaptive | 0 | head | auto | 7 | 12.129 |  |  | 2.262e-03 | 1.163e-04 |
| cuda | 64 | adaptive | 1 | baseline | auto | 7 | 18.395 | 0.664 | 0.649-0.670 | 2.111e-03 | 1.025e-04 |
| cuda | 64 | adaptive | 1 | ewald | auto | 7 | 10.562 | 1.161 | 1.130-1.163 | 2.262e-03 | 1.163e-04 |
| cuda | 64 | adaptive | 1 | head | auto | 7 | 12.182 |  |  | 2.262e-03 | 1.163e-04 |
| cuda | 64 | clustered | 0 | baseline | auto | 7 | 8.514 | 0.675 | 0.647-0.688 | 4.815e-03 | 5.752e-04 |
| cuda | 64 | clustered | 0 | ewald | auto | 7 | 5.524 | 0.997 | 0.995-1.048 | 5.305e-03 | 1.096e-03 |
| cuda | 64 | clustered | 0 | head | auto | 7 | 5.590 |  |  | 5.305e-03 | 1.096e-03 |
| cuda | 64 | clustered | 1 | baseline | auto | 7 | 8.386 | 0.702 | 0.685-0.713 | 4.815e-03 | 5.752e-04 |
| cuda | 64 | clustered | 1 | ewald | auto | 7 | 5.827 | 1.009 | 0.979-1.026 | 5.305e-03 | 1.096e-03 |
| cuda | 64 | clustered | 1 | head | auto | 7 | 5.887 |  |  | 5.305e-03 | 1.096e-03 |
| cuda | 64 | lattice | 0 | baseline | auto | 7 | 32.591 | 0.448 | 0.447-0.457 | 3.418e-01 | 3.613e-02 |
| cuda | 64 | lattice | 0 | ewald | auto | 7 | 14.364 | 1.026 | 1.016-1.028 | 3.644e-01 | 4.476e-02 |
| cuda | 64 | lattice | 0 | head | auto | 7 | 14.770 |  |  | 3.644e-01 | 4.476e-02 |
| cuda | 64 | lattice | 1 | baseline | auto | 7 | 32.707 | 0.451 | 0.448-0.452 | 3.418e-01 | 3.613e-02 |
| cuda | 64 | lattice | 1 | ewald | auto | 7 | 14.479 | 1.012 | 1.008-1.021 | 3.644e-01 | 4.476e-02 |
| cuda | 64 | lattice | 1 | head | auto | 7 | 14.688 |  |  | 3.644e-01 | 4.476e-02 |
