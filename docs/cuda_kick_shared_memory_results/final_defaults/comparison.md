| backend | bucket | fixture | potential | binary | policy | order | kick ms | speedup vs reference | paired range | force rel L2 | potential rel L2 |
|---|---:|---|---:|---|---|---:|---:|---:|---|---:|---:|
| cuda | 32 | adaptive | 0 | arena | auto | 7 | 14.425 | 1.273 | 1.215-1.350 | 2.669e-03 | 1.058e-04 |
| cuda | 32 | adaptive | 0 | arena | handoff492 | 7 | 18.086 | 1.013 | 1.002-1.041 | 2.669e-03 | 1.058e-04 |
| cuda | 32 | adaptive | 0 | baseline | auto | 7 | 18.170 |  |  | 2.669e-03 | 1.058e-04 |
| cuda | 32 | adaptive | 1 | arena | auto | 7 | 14.612 | 1.258 | 1.210-1.325 | 2.669e-03 | 1.058e-04 |
| cuda | 32 | adaptive | 1 | arena | handoff492 | 7 | 18.520 | 0.995 | 0.968-1.000 | 2.669e-03 | 1.058e-04 |
| cuda | 32 | adaptive | 1 | baseline | auto | 7 | 18.426 |  |  | 2.669e-03 | 1.058e-04 |
| cuda | 32 | clustered | 0 | arena | auto | 7 | 6.136 | 1.388 | 1.349-1.525 | 3.901e-03 | 5.182e-04 |
| cuda | 32 | clustered | 0 | arena | handoff492 | 7 | 8.065 | 1.082 | 1.026-1.163 | 3.901e-03 | 5.182e-04 |
| cuda | 32 | clustered | 0 | baseline | auto | 7 | 8.728 |  |  | 3.901e-03 | 5.182e-04 |
| cuda | 32 | clustered | 1 | arena | auto | 7 | 6.533 | 1.360 | 1.337-1.383 | 3.901e-03 | 5.182e-04 |
| cuda | 32 | clustered | 1 | arena | handoff492 | 7 | 8.117 | 1.076 | 1.012-1.145 | 3.901e-03 | 5.182e-04 |
| cuda | 32 | clustered | 1 | baseline | auto | 7 | 8.865 |  |  | 3.901e-03 | 5.182e-04 |
| cuda | 32 | lattice | 0 | arena | auto | 7 | 14.647 | 2.283 | 2.236-2.291 | 2.462e-01 | 2.842e-02 |
| cuda | 32 | lattice | 0 | arena | handoff492 | 7 | 31.280 | 1.061 | 1.036-1.076 | 2.462e-01 | 2.842e-02 |
| cuda | 32 | lattice | 0 | baseline | auto | 7 | 33.196 |  |  | 2.462e-01 | 2.842e-02 |
| cuda | 32 | lattice | 1 | arena | auto | 7 | 14.835 | 2.259 | 2.210-2.285 | 2.462e-01 | 2.842e-02 |
| cuda | 32 | lattice | 1 | arena | handoff492 | 7 | 31.580 | 1.064 | 1.034-1.071 | 2.462e-01 | 2.842e-02 |
| cuda | 32 | lattice | 1 | baseline | auto | 7 | 33.600 |  |  | 2.462e-01 | 2.842e-02 |
| cuda | 64 | adaptive | 0 | arena | auto | 7 | 13.796 | 1.347 | 1.283-1.367 | 2.111e-03 | 1.025e-04 |
| cuda | 64 | adaptive | 0 | arena | handoff492 | 7 | 19.055 | 0.973 | 0.946-0.978 | 2.111e-03 | 1.025e-04 |
| cuda | 64 | adaptive | 0 | baseline | auto | 7 | 18.351 |  |  | 2.111e-03 | 1.025e-04 |
| cuda | 64 | adaptive | 1 | arena | auto | 7 | 14.563 | 1.288 | 1.250-1.308 | 2.111e-03 | 1.025e-04 |
| cuda | 64 | adaptive | 1 | arena | handoff492 | 7 | 19.653 | 0.952 | 0.930-0.959 | 2.111e-03 | 1.025e-04 |
| cuda | 64 | adaptive | 1 | baseline | auto | 7 | 18.558 |  |  | 2.111e-03 | 1.025e-04 |
| cuda | 64 | clustered | 0 | arena | auto | 7 | 5.879 | 1.380 | 1.379-1.532 | 4.815e-03 | 5.752e-04 |
| cuda | 64 | clustered | 0 | arena | handoff492 | 7 | 7.880 | 1.053 | 1.009-1.168 | 4.815e-03 | 5.752e-04 |
| cuda | 64 | clustered | 0 | baseline | auto | 7 | 8.107 |  |  | 4.815e-03 | 5.752e-04 |
| cuda | 64 | clustered | 1 | arena | auto | 7 | 6.078 | 1.380 | 1.344-1.385 | 4.815e-03 | 5.752e-04 |
| cuda | 64 | clustered | 1 | arena | handoff492 | 7 | 7.921 | 1.063 | 1.026-1.066 | 4.815e-03 | 5.752e-04 |
| cuda | 64 | clustered | 1 | baseline | auto | 7 | 8.388 |  |  | 4.815e-03 | 5.752e-04 |
| cuda | 64 | lattice | 0 | arena | auto | 7 | 14.450 | 2.225 | 2.216-2.257 | 3.418e-01 | 3.613e-02 |
| cuda | 64 | lattice | 0 | arena | handoff492 | 7 | 30.972 | 1.039 | 1.034-1.041 | 3.418e-01 | 3.613e-02 |
| cuda | 64 | lattice | 0 | baseline | auto | 7 | 32.156 |  |  | 3.418e-01 | 3.613e-02 |
| cuda | 64 | lattice | 1 | arena | auto | 7 | 14.396 | 2.242 | 2.226-2.297 | 3.418e-01 | 3.613e-02 |
| cuda | 64 | lattice | 1 | arena | handoff492 | 7 | 30.891 | 1.045 | 1.035-1.064 | 3.418e-01 | 3.613e-02 |
| cuda | 64 | lattice | 1 | baseline | auto | 7 | 32.281 |  |  | 3.418e-01 | 3.613e-02 |
| cuda | 128 | adaptive | 0 | arena | auto | 7 | 13.589 | 1.170 | 1.147-1.189 | 1.770e-03 | 1.035e-04 |
| cuda | 128 | adaptive | 0 | arena | handoff492 | 7 | 17.341 | 0.919 | 0.871-0.929 | 1.770e-03 | 1.035e-04 |
| cuda | 128 | adaptive | 0 | baseline | auto | 7 | 15.767 |  |  | 1.770e-03 | 1.035e-04 |
| cuda | 128 | adaptive | 1 | arena | auto | 7 | 13.820 | 1.157 | 1.136-1.187 | 1.770e-03 | 1.035e-04 |
| cuda | 128 | adaptive | 1 | arena | handoff492 | 7 | 17.694 | 0.908 | 0.890-0.910 | 1.770e-03 | 1.035e-04 |
| cuda | 128 | adaptive | 1 | baseline | auto | 7 | 16.104 |  |  | 1.770e-03 | 1.035e-04 |
| cuda | 128 | clustered | 0 | arena | auto | 7 | 6.084 | 1.007 | 1.005-1.046 | 2.241e-03 | 5.016e-04 |
| cuda | 128 | clustered | 0 | arena | handoff492 | 7 | 5.973 | 1.025 | 1.008-1.057 | 2.241e-03 | 5.016e-04 |
| cuda | 128 | clustered | 0 | baseline | auto | 7 | 6.125 |  |  | 2.241e-03 | 5.016e-04 |
| cuda | 128 | clustered | 1 | arena | auto | 7 | 6.308 | 1.007 | 1.000-1.039 | 2.241e-03 | 5.016e-04 |
| cuda | 128 | clustered | 1 | arena | handoff492 | 7 | 6.304 | 1.007 | 1.001-1.060 | 2.241e-03 | 5.016e-04 |
| cuda | 128 | clustered | 1 | baseline | auto | 7 | 6.393 |  |  | 2.241e-03 | 5.016e-04 |
| cuda | 128 | lattice | 0 | arena | auto | 7 | 14.430 | 1.024 | 1.007-1.069 | 3.418e-01 | 3.613e-02 |
| cuda | 128 | lattice | 0 | arena | handoff492 | 7 | 14.213 | 1.027 | 1.018-1.085 | 3.418e-01 | 3.613e-02 |
| cuda | 128 | lattice | 0 | baseline | auto | 7 | 14.661 |  |  | 3.418e-01 | 3.613e-02 |
| cuda | 128 | lattice | 1 | arena | auto | 7 | 14.393 | 1.022 | 1.019-1.036 | 3.418e-01 | 3.613e-02 |
| cuda | 128 | lattice | 1 | arena | handoff492 | 7 | 14.584 | 1.013 | 1.011-1.022 | 3.418e-01 | 3.613e-02 |
| cuda | 128 | lattice | 1 | baseline | auto | 7 | 14.849 |  |  | 3.418e-01 | 3.613e-02 |
| cuda | 256 | adaptive | 0 | arena | auto | 7 | 20.530 | 0.884 | 0.861-0.886 | 1.317e-03 | 9.639e-05 |
| cuda | 256 | adaptive | 0 | arena | handoff492 | 7 | 21.419 | 0.825 | 0.824-0.903 | 1.317e-03 | 9.639e-05 |
| cuda | 256 | adaptive | 0 | baseline | auto | 7 | 17.908 |  |  | 1.317e-03 | 9.639e-05 |
| cuda | 256 | adaptive | 1 | arena | auto | 7 | 20.956 | 0.886 | 0.869-0.886 | 1.317e-03 | 9.639e-05 |
| cuda | 256 | adaptive | 1 | arena | handoff492 | 7 | 22.081 | 0.826 | 0.810-0.874 | 1.317e-03 | 9.639e-05 |
| cuda | 256 | adaptive | 1 | baseline | auto | 7 | 18.365 |  |  | 1.317e-03 | 9.639e-05 |
| cuda | 256 | clustered | 0 | arena | auto | 7 | 6.708 | 1.029 | 1.015-1.099 | 1.884e-03 | 5.309e-04 |
| cuda | 256 | clustered | 0 | arena | handoff492 | 7 | 6.826 | 0.995 | 0.986-1.164 | 1.884e-03 | 5.309e-04 |
| cuda | 256 | clustered | 0 | baseline | auto | 7 | 6.806 |  |  | 1.884e-03 | 5.309e-04 |
| cuda | 256 | clustered | 1 | arena | auto | 7 | 7.371 | 0.971 | 0.926-0.979 | 1.884e-03 | 5.309e-04 |
| cuda | 256 | clustered | 1 | arena | handoff492 | 7 | 7.481 | 0.965 | 0.936-1.060 | 1.884e-03 | 5.309e-04 |
| cuda | 256 | clustered | 1 | baseline | auto | 7 | 7.217 |  |  | 1.884e-03 | 5.309e-04 |
| cuda | 256 | lattice | 0 | arena | auto | 7 | 11.754 | 0.963 | 0.912-1.005 | 3.418e-01 | 3.613e-02 |
| cuda | 256 | lattice | 0 | arena | handoff492 | 7 | 11.721 | 0.966 | 0.909-0.999 | 3.418e-01 | 3.613e-02 |
| cuda | 256 | lattice | 0 | baseline | auto | 7 | 11.583 |  |  | 3.418e-01 | 3.613e-02 |
| cuda | 256 | lattice | 1 | arena | auto | 7 | 12.505 | 0.940 | 0.914-0.972 | 3.418e-01 | 3.613e-02 |
| cuda | 256 | lattice | 1 | arena | handoff492 | 7 | 12.364 | 0.941 | 0.897-0.947 | 3.418e-01 | 3.613e-02 |
| cuda | 256 | lattice | 1 | baseline | auto | 7 | 11.637 |  |  | 3.418e-01 | 3.613e-02 |
