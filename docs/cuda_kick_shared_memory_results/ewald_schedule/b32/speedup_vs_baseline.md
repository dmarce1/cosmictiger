| backend | bucket | fixture | potential | binary | policy | order | kick ms | speedup vs reference | paired range | force rel L2 | potential rel L2 |
|---|---:|---|---:|---|---|---:|---:|---:|---|---:|---:|
| cuda | 32 | adaptive | 0 | baseline | auto | 7 | 17.770 |  |  | 2.669e-03 | 1.058e-04 |
| cuda | 32 | adaptive | 0 | ewald | auto | 7 | 10.595 | 1.680 | 1.646-1.693 | 2.927e-03 | 1.253e-04 |
| cuda | 32 | adaptive | 0 | head | auto | 7 | 12.012 | 1.479 | 1.472-1.494 | 2.927e-03 | 1.253e-04 |
| cuda | 32 | adaptive | 1 | baseline | auto | 7 | 18.061 |  |  | 2.669e-03 | 1.058e-04 |
| cuda | 32 | adaptive | 1 | ewald | auto | 7 | 10.765 | 1.678 | 1.517-1.688 | 2.927e-03 | 1.253e-04 |
| cuda | 32 | adaptive | 1 | head | auto | 7 | 11.863 | 1.506 | 1.504-1.533 | 2.927e-03 | 1.253e-04 |
| cuda | 32 | clustered | 0 | baseline | auto | 7 | 8.597 |  |  | 3.901e-03 | 5.182e-04 |
| cuda | 32 | clustered | 0 | ewald | auto | 7 | 5.888 | 1.460 | 1.450-1.534 | 4.526e-03 | 1.022e-03 |
| cuda | 32 | clustered | 0 | head | auto | 7 | 5.735 | 1.499 | 1.437-1.574 | 4.526e-03 | 1.022e-03 |
| cuda | 32 | clustered | 1 | baseline | auto | 7 | 8.757 |  |  | 3.901e-03 | 5.182e-04 |
| cuda | 32 | clustered | 1 | ewald | auto | 7 | 6.050 | 1.414 | 1.407-1.449 | 4.526e-03 | 1.022e-03 |
| cuda | 32 | clustered | 1 | head | auto | 7 | 5.999 | 1.440 | 1.429-1.460 | 4.526e-03 | 1.022e-03 |
| cuda | 32 | lattice | 0 | baseline | auto | 7 | 32.946 |  |  | 2.462e-01 | 2.842e-02 |
| cuda | 32 | lattice | 0 | ewald | auto | 7 | 14.714 | 2.251 | 2.239-2.269 | 2.991e-01 | 3.739e-02 |
| cuda | 32 | lattice | 0 | head | auto | 7 | 15.054 | 2.196 | 2.179-2.212 | 2.991e-01 | 3.739e-02 |
| cuda | 32 | lattice | 1 | baseline | auto | 7 | 33.071 |  |  | 2.462e-01 | 2.842e-02 |
| cuda | 32 | lattice | 1 | ewald | auto | 7 | 14.842 | 2.228 | 2.158-2.273 | 2.991e-01 | 3.739e-02 |
| cuda | 32 | lattice | 1 | head | auto | 7 | 15.083 | 2.206 | 2.169-2.224 | 2.991e-01 | 3.739e-02 |
