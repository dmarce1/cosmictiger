| backend | bucket | fixture | potential | binary | policy | order | kick ms | speedup vs reference | paired range | force rel L2 | potential rel L2 |
|---|---:|---|---:|---|---|---:|---:|---:|---|---:|---:|
| cuda | 64 | adaptive | 0 | baseline | auto | 7 | 18.059 |  |  | 2.111e-03 | 1.025e-04 |
| cuda | 64 | adaptive | 0 | ewald | auto | 7 | 10.359 | 1.734 | 1.645-1.756 | 2.262e-03 | 1.163e-04 |
| cuda | 64 | adaptive | 0 | head | auto | 7 | 12.129 | 1.495 | 1.462-1.506 | 2.262e-03 | 1.163e-04 |
| cuda | 64 | adaptive | 1 | baseline | auto | 7 | 18.395 |  |  | 2.111e-03 | 1.025e-04 |
| cuda | 64 | adaptive | 1 | ewald | auto | 7 | 10.562 | 1.742 | 1.733-1.751 | 2.262e-03 | 1.163e-04 |
| cuda | 64 | adaptive | 1 | head | auto | 7 | 12.182 | 1.506 | 1.494-1.541 | 2.262e-03 | 1.163e-04 |
| cuda | 64 | clustered | 0 | baseline | auto | 7 | 8.514 |  |  | 4.815e-03 | 5.752e-04 |
| cuda | 64 | clustered | 0 | ewald | auto | 7 | 5.524 | 1.541 | 1.446-1.554 | 5.305e-03 | 1.096e-03 |
| cuda | 64 | clustered | 0 | head | auto | 7 | 5.590 | 1.482 | 1.453-1.547 | 5.305e-03 | 1.096e-03 |
| cuda | 64 | clustered | 1 | baseline | auto | 7 | 8.386 |  |  | 4.815e-03 | 5.752e-04 |
| cuda | 64 | clustered | 1 | ewald | auto | 7 | 5.827 | 1.416 | 1.395-1.498 | 5.305e-03 | 1.096e-03 |
| cuda | 64 | clustered | 1 | head | auto | 7 | 5.887 | 1.424 | 1.403-1.460 | 5.305e-03 | 1.096e-03 |
| cuda | 64 | lattice | 0 | baseline | auto | 7 | 32.591 |  |  | 3.418e-01 | 3.613e-02 |
| cuda | 64 | lattice | 0 | ewald | auto | 7 | 14.364 | 2.270 | 2.252-2.294 | 3.644e-01 | 4.476e-02 |
| cuda | 64 | lattice | 0 | head | auto | 7 | 14.770 | 2.234 | 2.190-2.235 | 3.644e-01 | 4.476e-02 |
| cuda | 64 | lattice | 1 | baseline | auto | 7 | 32.707 |  |  | 3.418e-01 | 3.613e-02 |
| cuda | 64 | lattice | 1 | ewald | auto | 7 | 14.479 | 2.259 | 2.230-2.264 | 3.644e-01 | 4.476e-02 |
| cuda | 64 | lattice | 1 | head | auto | 7 | 14.688 | 2.217 | 2.212-2.233 | 3.644e-01 | 4.476e-02 |
