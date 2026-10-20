| backend | bucket | fixture | potential | binary | policy | order | kick ms | speedup vs reference | paired range | force rel L2 | potential rel L2 |
|---|---:|---|---:|---|---|---:|---:|---:|---|---:|---:|
| cuda | 32 | adaptive | 0 | baseline | auto | 7 | 17.770 | 0.676 | 0.669-0.679 | 2.669e-03 | 1.058e-04 |
| cuda | 32 | adaptive | 0 | ewald | auto | 7 | 10.595 | 1.124 | 1.118-1.145 | 2.927e-03 | 1.253e-04 |
| cuda | 32 | adaptive | 0 | head | auto | 7 | 12.012 |  |  | 2.927e-03 | 1.253e-04 |
| cuda | 32 | adaptive | 1 | baseline | auto | 7 | 18.061 | 0.664 | 0.652-0.665 | 2.669e-03 | 1.058e-04 |
| cuda | 32 | adaptive | 1 | ewald | auto | 7 | 10.765 | 1.095 | 1.008-1.120 | 2.927e-03 | 1.253e-04 |
| cuda | 32 | adaptive | 1 | head | auto | 7 | 11.863 |  |  | 2.927e-03 | 1.253e-04 |
| cuda | 32 | clustered | 0 | baseline | auto | 7 | 8.597 | 0.667 | 0.635-0.696 | 3.901e-03 | 5.182e-04 |
| cuda | 32 | clustered | 0 | ewald | auto | 7 | 5.888 | 0.975 | 0.974-1.009 | 4.526e-03 | 1.022e-03 |
| cuda | 32 | clustered | 0 | head | auto | 7 | 5.735 |  |  | 4.526e-03 | 1.022e-03 |
| cuda | 32 | clustered | 1 | baseline | auto | 7 | 8.757 | 0.694 | 0.685-0.700 | 3.901e-03 | 5.182e-04 |
| cuda | 32 | clustered | 1 | ewald | auto | 7 | 6.050 | 0.990 | 0.964-1.006 | 4.526e-03 | 1.022e-03 |
| cuda | 32 | clustered | 1 | head | auto | 7 | 5.999 |  |  | 4.526e-03 | 1.022e-03 |
| cuda | 32 | lattice | 0 | baseline | auto | 7 | 32.946 | 0.455 | 0.452-0.459 | 2.462e-01 | 2.842e-02 |
| cuda | 32 | lattice | 0 | ewald | auto | 7 | 14.714 | 1.026 | 1.020-1.033 | 2.991e-01 | 3.739e-02 |
| cuda | 32 | lattice | 0 | head | auto | 7 | 15.054 |  |  | 2.991e-01 | 3.739e-02 |
| cuda | 32 | lattice | 1 | baseline | auto | 7 | 33.071 | 0.453 | 0.450-0.461 | 2.462e-01 | 2.842e-02 |
| cuda | 32 | lattice | 1 | ewald | auto | 7 | 14.842 | 1.010 | 0.995-1.022 | 2.991e-01 | 3.739e-02 |
| cuda | 32 | lattice | 1 | head | auto | 7 | 15.083 |  |  | 2.991e-01 | 3.739e-02 |
