| backend | bucket | fixture | potential | binary | policy | order | kick ms | speedup vs reference | paired range | force rel L2 | potential rel L2 |
|---|---:|---|---:|---|---|---:|---:|---:|---|---:|---:|
| cuda | 32 | adaptive | 0 | arena | auto | 7 | 20.633 | 0.871 | 0.806-0.882 | 2.669e-03 | 1.058e-04 |
| cuda | 32 | adaptive | 0 | arena | global | 7 | 20.505 | 0.877 | 0.861-0.879 | 2.669e-03 | 1.058e-04 |
| cuda | 32 | adaptive | 0 | arena | tile128 | 7 | 20.740 | 0.864 | 0.852-0.878 | 2.669e-03 | 1.058e-04 |
| cuda | 32 | adaptive | 0 | arena | tile64 | 7 | 20.440 | 0.880 | 0.831-0.889 | 2.669e-03 | 1.058e-04 |
| cuda | 32 | adaptive | 0 | baseline | auto | 7 | 17.911 |  |  | 2.669e-03 | 1.058e-04 |
| cuda | 32 | adaptive | 1 | arena | auto | 7 | 20.793 | 0.863 | 0.848-0.885 | 2.669e-03 | 1.058e-04 |
| cuda | 32 | adaptive | 1 | arena | global | 7 | 20.546 | 0.892 | 0.874-0.904 | 2.669e-03 | 1.058e-04 |
| cuda | 32 | adaptive | 1 | arena | tile128 | 7 | 20.857 | 0.892 | 0.857-0.894 | 2.669e-03 | 1.058e-04 |
| cuda | 32 | adaptive | 1 | arena | tile64 | 7 | 20.460 | 0.887 | 0.878-0.895 | 2.669e-03 | 1.058e-04 |
| cuda | 32 | adaptive | 1 | baseline | auto | 7 | 18.212 |  |  | 2.669e-03 | 1.058e-04 |
| cuda | 32 | clustered | 0 | arena | auto | 7 | 11.337 | 0.744 | 0.724-0.817 | 3.901e-03 | 5.183e-04 |
| cuda | 32 | clustered | 0 | arena | global | 7 | 11.003 | 0.803 | 0.797-0.813 | 3.901e-03 | 5.183e-04 |
| cuda | 32 | clustered | 0 | arena | tile128 | 7 | 11.099 | 0.780 | 0.769-0.796 | 3.901e-03 | 5.183e-04 |
| cuda | 32 | clustered | 0 | arena | tile64 | 7 | 10.666 | 0.814 | 0.796-0.828 | 3.901e-03 | 5.183e-04 |
| cuda | 32 | clustered | 0 | baseline | auto | 7 | 8.800 |  |  | 3.901e-03 | 5.182e-04 |
| cuda | 32 | clustered | 1 | arena | auto | 7 | 11.410 | 0.755 | 0.746-0.791 | 3.901e-03 | 5.183e-04 |
| cuda | 32 | clustered | 1 | arena | global | 7 | 10.919 | 0.786 | 0.776-0.818 | 3.901e-03 | 5.183e-04 |
| cuda | 32 | clustered | 1 | arena | tile128 | 7 | 11.189 | 0.767 | 0.744-0.804 | 3.901e-03 | 5.183e-04 |
| cuda | 32 | clustered | 1 | arena | tile64 | 7 | 10.771 | 0.796 | 0.771-0.822 | 3.901e-03 | 5.183e-04 |
| cuda | 32 | clustered | 1 | baseline | auto | 7 | 8.577 |  |  | 3.901e-03 | 5.182e-04 |
| cuda | 32 | lattice | 0 | arena | auto | 7 | 90.734 | 0.363 | 0.352-0.373 | 2.462e-01 | 2.842e-02 |
| cuda | 32 | lattice | 0 | arena | global | 7 | 90.611 | 0.365 | 0.364-0.374 | 2.462e-01 | 2.842e-02 |
| cuda | 32 | lattice | 0 | arena | tile128 | 7 | 90.779 | 0.364 | 0.363-0.371 | 2.462e-01 | 2.842e-02 |
| cuda | 32 | lattice | 0 | arena | tile64 | 7 | 90.402 | 0.364 | 0.364-0.374 | 2.462e-01 | 2.842e-02 |
| cuda | 32 | lattice | 0 | baseline | auto | 7 | 33.104 |  |  | 2.462e-01 | 2.842e-02 |
| cuda | 32 | lattice | 1 | arena | auto | 7 | 90.772 | 0.362 | 0.351-0.372 | 2.462e-01 | 2.842e-02 |
| cuda | 32 | lattice | 1 | arena | global | 7 | 90.998 | 0.366 | 0.361-0.372 | 2.462e-01 | 2.842e-02 |
| cuda | 32 | lattice | 1 | arena | tile128 | 7 | 91.292 | 0.365 | 0.360-0.371 | 2.462e-01 | 2.842e-02 |
| cuda | 32 | lattice | 1 | arena | tile64 | 7 | 90.688 | 0.364 | 0.363-0.373 | 2.462e-01 | 2.842e-02 |
| cuda | 32 | lattice | 1 | baseline | auto | 7 | 33.287 |  |  | 2.462e-01 | 2.842e-02 |
| cuda | 64 | adaptive | 0 | arena | auto | 7 | 19.526 | 0.948 | 0.905-0.961 | 2.111e-03 | 1.025e-04 |
| cuda | 64 | adaptive | 0 | arena | global | 7 | 20.063 | 0.924 | 0.917-0.927 | 2.111e-03 | 1.025e-04 |
| cuda | 64 | adaptive | 0 | arena | tile128 | 7 | 20.260 | 0.917 | 0.908-0.922 | 2.111e-03 | 1.025e-04 |
| cuda | 64 | adaptive | 0 | arena | tile64 | 7 | 19.934 | 0.942 | 0.907-0.964 | 2.111e-03 | 1.025e-04 |
| cuda | 64 | adaptive | 0 | baseline | auto | 7 | 18.490 |  |  | 2.111e-03 | 1.025e-04 |
| cuda | 64 | adaptive | 1 | arena | auto | 7 | 20.247 | 0.927 | 0.915-0.935 | 2.111e-03 | 1.025e-04 |
| cuda | 64 | adaptive | 1 | arena | global | 7 | 20.286 | 0.921 | 0.913-0.952 | 2.111e-03 | 1.025e-04 |
| cuda | 64 | adaptive | 1 | arena | tile128 | 7 | 20.431 | 0.915 | 0.910-0.919 | 2.111e-03 | 1.025e-04 |
| cuda | 64 | adaptive | 1 | arena | tile64 | 7 | 19.796 | 0.949 | 0.899-0.960 | 2.111e-03 | 1.025e-04 |
| cuda | 64 | adaptive | 1 | baseline | auto | 7 | 18.676 |  |  | 2.111e-03 | 1.025e-04 |
| cuda | 64 | clustered | 0 | arena | auto | 7 | 8.175 | 1.005 | 0.991-1.045 | 4.815e-03 | 5.752e-04 |
| cuda | 64 | clustered | 0 | arena | global | 7 | 8.131 | 1.043 | 1.005-1.060 | 4.815e-03 | 5.752e-04 |
| cuda | 64 | clustered | 0 | arena | tile128 | 7 | 7.910 | 1.058 | 0.993-1.090 | 4.815e-03 | 5.752e-04 |
| cuda | 64 | clustered | 0 | arena | tile64 | 7 | 8.029 | 1.043 | 1.040-1.056 | 4.815e-03 | 5.752e-04 |
| cuda | 64 | clustered | 0 | baseline | auto | 7 | 8.372 |  |  | 4.815e-03 | 5.752e-04 |
| cuda | 64 | clustered | 1 | arena | auto | 7 | 8.090 | 1.024 | 1.014-1.034 | 4.815e-03 | 5.752e-04 |
| cuda | 64 | clustered | 1 | arena | global | 7 | 8.307 | 1.011 | 0.997-1.024 | 4.815e-03 | 5.752e-04 |
| cuda | 64 | clustered | 1 | arena | tile128 | 7 | 8.307 | 0.997 | 0.982-1.020 | 4.815e-03 | 5.752e-04 |
| cuda | 64 | clustered | 1 | arena | tile64 | 7 | 7.878 | 1.051 | 1.004-1.070 | 4.815e-03 | 5.752e-04 |
| cuda | 64 | clustered | 1 | baseline | auto | 7 | 8.319 |  |  | 4.815e-03 | 5.752e-04 |
| cuda | 64 | lattice | 0 | arena | auto | 7 | 31.033 | 1.048 | 1.027-1.052 | 3.418e-01 | 3.613e-02 |
| cuda | 64 | lattice | 0 | arena | global | 7 | 31.180 | 1.040 | 1.033-1.043 | 3.418e-01 | 3.613e-02 |
| cuda | 64 | lattice | 0 | arena | tile128 | 7 | 31.275 | 1.036 | 1.025-1.042 | 3.418e-01 | 3.613e-02 |
| cuda | 64 | lattice | 0 | arena | tile64 | 7 | 30.829 | 1.053 | 1.035-1.055 | 3.418e-01 | 3.613e-02 |
| cuda | 64 | lattice | 0 | baseline | auto | 7 | 32.387 |  |  | 3.418e-01 | 3.613e-02 |
| cuda | 64 | lattice | 1 | arena | auto | 7 | 31.341 | 1.043 | 1.036-1.045 | 3.418e-01 | 3.613e-02 |
| cuda | 64 | lattice | 1 | arena | global | 7 | 31.224 | 1.049 | 1.038-1.053 | 3.418e-01 | 3.613e-02 |
| cuda | 64 | lattice | 1 | arena | tile128 | 7 | 31.221 | 1.046 | 1.036-1.049 | 3.418e-01 | 3.613e-02 |
| cuda | 64 | lattice | 1 | arena | tile64 | 7 | 30.800 | 1.059 | 1.038-1.070 | 3.418e-01 | 3.613e-02 |
| cuda | 64 | lattice | 1 | baseline | auto | 7 | 32.613 |  |  | 3.418e-01 | 3.613e-02 |
| cuda | 128 | adaptive | 0 | arena | auto | 7 | 17.826 | 0.911 | 0.881-0.915 | 1.770e-03 | 1.035e-04 |
| cuda | 128 | adaptive | 0 | arena | global | 7 | 17.473 | 0.927 | 0.915-0.929 | 1.770e-03 | 1.035e-04 |
| cuda | 128 | adaptive | 0 | arena | tile128 | 7 | 17.791 | 0.908 | 0.902-0.913 | 1.770e-03 | 1.035e-04 |
| cuda | 128 | adaptive | 0 | arena | tile64 | 7 | 17.201 | 0.944 | 0.906-0.954 | 1.770e-03 | 1.035e-04 |
| cuda | 128 | adaptive | 0 | baseline | auto | 7 | 16.182 |  |  | 1.770e-03 | 1.035e-04 |
| cuda | 128 | adaptive | 1 | arena | auto | 7 | 17.774 | 0.935 | 0.905-0.963 | 1.770e-03 | 1.035e-04 |
| cuda | 128 | adaptive | 1 | arena | global | 7 | 17.613 | 0.953 | 0.928-0.962 | 1.770e-03 | 1.035e-04 |
| cuda | 128 | adaptive | 1 | arena | tile128 | 7 | 17.961 | 0.918 | 0.917-0.935 | 1.770e-03 | 1.035e-04 |
| cuda | 128 | adaptive | 1 | arena | tile64 | 7 | 17.766 | 0.945 | 0.925-0.971 | 1.770e-03 | 1.035e-04 |
| cuda | 128 | adaptive | 1 | baseline | auto | 7 | 16.622 |  |  | 1.770e-03 | 1.035e-04 |
| cuda | 128 | clustered | 0 | arena | auto | 7 | 6.176 | 0.991 | 0.983-1.031 | 2.241e-03 | 5.016e-04 |
| cuda | 128 | clustered | 0 | arena | global | 7 | 6.396 | 0.957 | 0.950-1.006 | 2.241e-03 | 5.016e-04 |
| cuda | 128 | clustered | 0 | arena | tile128 | 7 | 6.119 | 1.014 | 1.000-1.032 | 2.241e-03 | 5.016e-04 |
| cuda | 128 | clustered | 0 | arena | tile64 | 7 | 6.121 | 1.019 | 1.012-1.032 | 2.241e-03 | 5.016e-04 |
| cuda | 128 | clustered | 0 | baseline | auto | 7 | 6.195 |  |  | 2.241e-03 | 5.016e-04 |
| cuda | 128 | clustered | 1 | arena | auto | 7 | 6.432 | 1.014 | 1.005-1.014 | 2.241e-03 | 5.016e-04 |
| cuda | 128 | clustered | 1 | arena | global | 7 | 6.575 | 0.988 | 0.965-1.004 | 2.241e-03 | 5.016e-04 |
| cuda | 128 | clustered | 1 | arena | tile128 | 7 | 6.405 | 1.009 | 1.004-1.029 | 2.241e-03 | 5.016e-04 |
| cuda | 128 | clustered | 1 | arena | tile64 | 7 | 6.485 | 1.015 | 0.994-1.027 | 2.241e-03 | 5.016e-04 |
| cuda | 128 | clustered | 1 | baseline | auto | 7 | 6.495 |  |  | 2.241e-03 | 5.016e-04 |
| cuda | 128 | lattice | 0 | arena | auto | 7 | 14.368 | 1.034 | 0.988-1.065 | 3.418e-01 | 3.613e-02 |
| cuda | 128 | lattice | 0 | arena | global | 7 | 14.312 | 1.047 | 1.016-1.069 | 3.418e-01 | 3.613e-02 |
| cuda | 128 | lattice | 0 | arena | tile128 | 7 | 14.485 | 1.026 | 1.000-1.053 | 3.418e-01 | 3.613e-02 |
| cuda | 128 | lattice | 0 | arena | tile64 | 7 | 14.370 | 1.044 | 0.994-1.060 | 3.418e-01 | 3.613e-02 |
| cuda | 128 | lattice | 0 | baseline | auto | 7 | 14.979 |  |  | 3.418e-01 | 3.613e-02 |
| cuda | 128 | lattice | 1 | arena | auto | 7 | 14.685 | 1.022 | 1.020-1.039 | 3.418e-01 | 3.613e-02 |
| cuda | 128 | lattice | 1 | arena | global | 7 | 14.373 | 1.039 | 1.033-1.073 | 3.418e-01 | 3.613e-02 |
| cuda | 128 | lattice | 1 | arena | tile128 | 7 | 14.586 | 1.031 | 1.024-1.051 | 3.418e-01 | 3.613e-02 |
| cuda | 128 | lattice | 1 | arena | tile64 | 7 | 14.397 | 1.047 | 1.036-1.065 | 3.418e-01 | 3.613e-02 |
| cuda | 128 | lattice | 1 | baseline | auto | 7 | 14.975 |  |  | 3.418e-01 | 3.613e-02 |
| cuda | 256 | adaptive | 0 | arena | auto | 7 | 21.041 | 0.878 | 0.877-0.920 | 1.317e-03 | 9.639e-05 |
| cuda | 256 | adaptive | 0 | arena | global | 7 | 20.352 | 0.928 | 0.891-0.936 | 1.317e-03 | 9.639e-05 |
| cuda | 256 | adaptive | 0 | arena | tile128 | 7 | 21.174 | 0.876 | 0.870-0.892 | 1.317e-03 | 9.639e-05 |
| cuda | 256 | adaptive | 0 | arena | tile64 | 7 | 20.988 | 0.882 | 0.879-0.947 | 1.317e-03 | 9.639e-05 |
| cuda | 256 | adaptive | 0 | baseline | auto | 7 | 18.504 |  |  | 1.317e-03 | 9.639e-05 |
| cuda | 256 | adaptive | 1 | arena | auto | 7 | 21.643 | 0.883 | 0.864-0.889 | 1.317e-03 | 9.639e-05 |
| cuda | 256 | adaptive | 1 | arena | global | 7 | 21.721 | 0.877 | 0.872-0.935 | 1.317e-03 | 9.639e-05 |
| cuda | 256 | adaptive | 1 | arena | tile128 | 7 | 21.800 | 0.877 | 0.862-0.886 | 1.317e-03 | 9.639e-05 |
| cuda | 256 | adaptive | 1 | arena | tile64 | 7 | 21.819 | 0.873 | 0.863-0.938 | 1.317e-03 | 9.639e-05 |
| cuda | 256 | adaptive | 1 | baseline | auto | 7 | 19.057 |  |  | 1.317e-03 | 9.639e-05 |
| cuda | 256 | clustered | 0 | arena | auto | 7 | 7.000 | 1.033 | 0.992-1.041 | 1.884e-03 | 5.309e-04 |
| cuda | 256 | clustered | 0 | arena | global | 7 | 7.039 | 1.001 | 0.977-1.018 | 1.884e-03 | 5.309e-04 |
| cuda | 256 | clustered | 0 | arena | tile128 | 7 | 7.215 | 0.995 | 0.936-1.002 | 1.884e-03 | 5.309e-04 |
| cuda | 256 | clustered | 0 | arena | tile64 | 7 | 6.815 | 1.020 | 1.009-1.066 | 1.884e-03 | 5.309e-04 |
| cuda | 256 | clustered | 0 | baseline | auto | 7 | 7.165 |  |  | 1.884e-03 | 5.309e-04 |
| cuda | 256 | clustered | 1 | arena | auto | 7 | 7.493 | 0.999 | 0.987-1.024 | 1.884e-03 | 5.309e-04 |
| cuda | 256 | clustered | 1 | arena | global | 7 | 7.629 | 0.988 | 0.984-1.014 | 1.884e-03 | 5.309e-04 |
| cuda | 256 | clustered | 1 | arena | tile128 | 7 | 7.723 | 0.962 | 0.952-0.987 | 1.884e-03 | 5.309e-04 |
| cuda | 256 | clustered | 1 | arena | tile64 | 7 | 7.336 | 1.017 | 1.009-1.062 | 1.884e-03 | 5.309e-04 |
| cuda | 256 | clustered | 1 | baseline | auto | 7 | 7.540 |  |  | 1.884e-03 | 5.309e-04 |
| cuda | 256 | lattice | 0 | arena | auto | 7 | 11.714 | 0.977 | 0.941-0.994 | 3.418e-01 | 3.613e-02 |
| cuda | 256 | lattice | 0 | arena | global | 7 | 11.765 | 0.977 | 0.926-1.004 | 3.418e-01 | 3.613e-02 |
| cuda | 256 | lattice | 0 | arena | tile128 | 7 | 11.887 | 0.967 | 0.918-0.978 | 3.418e-01 | 3.613e-02 |
| cuda | 256 | lattice | 0 | arena | tile64 | 7 | 11.832 | 0.964 | 0.963-1.013 | 3.418e-01 | 3.613e-02 |
| cuda | 256 | lattice | 0 | baseline | auto | 7 | 11.442 |  |  | 3.418e-01 | 3.613e-02 |
| cuda | 256 | lattice | 1 | arena | auto | 7 | 12.568 | 0.951 | 0.933-0.957 | 3.418e-01 | 3.613e-02 |
| cuda | 256 | lattice | 1 | arena | global | 7 | 12.103 | 0.993 | 0.961-1.010 | 3.418e-01 | 3.613e-02 |
| cuda | 256 | lattice | 1 | arena | tile128 | 7 | 12.818 | 0.932 | 0.932-0.935 | 3.418e-01 | 3.613e-02 |
| cuda | 256 | lattice | 1 | arena | tile64 | 7 | 12.129 | 0.991 | 0.920-1.029 | 3.418e-01 | 3.613e-02 |
| cuda | 256 | lattice | 1 | baseline | auto | 7 | 11.951 |  |  | 3.418e-01 | 3.613e-02 |
| cuda | 512 | adaptive | 0 | arena | auto | 7 | 34.640 | 0.823 | 0.785-0.907 | 1.001e-03 | 9.754e-05 |
| cuda | 512 | adaptive | 0 | arena | global | 7 | 34.195 | 0.880 | 0.818-0.909 | 1.001e-03 | 9.754e-05 |
| cuda | 512 | adaptive | 0 | arena | tile128 | 7 | 35.008 | 0.823 | 0.809-0.857 | 1.001e-03 | 9.754e-05 |
| cuda | 512 | adaptive | 0 | arena | tile64 | 7 | 34.951 | 0.860 | 0.805-0.904 | 1.001e-03 | 9.754e-05 |
| cuda | 512 | adaptive | 0 | baseline | auto | 7 | 28.505 |  |  | 1.001e-03 | 9.754e-05 |
| cuda | 512 | adaptive | 1 | arena | auto | 7 | 35.282 | 0.843 | 0.795-0.850 | 1.001e-03 | 9.754e-05 |
| cuda | 512 | adaptive | 1 | arena | global | 7 | 35.223 | 0.824 | 0.824-0.896 | 1.001e-03 | 9.754e-05 |
| cuda | 512 | adaptive | 1 | arena | tile128 | 7 | 35.942 | 0.807 | 0.799-0.834 | 1.001e-03 | 9.754e-05 |
| cuda | 512 | adaptive | 1 | arena | tile64 | 7 | 35.099 | 0.818 | 0.813-0.909 | 1.001e-03 | 9.754e-05 |
| cuda | 512 | adaptive | 1 | baseline | auto | 7 | 29.037 |  |  | 1.001e-03 | 9.754e-05 |
| cuda | 512 | clustered | 0 | arena | auto | 7 | 10.017 | 0.991 | 0.927-1.034 | 1.767e-03 | 5.584e-04 |
| cuda | 512 | clustered | 0 | arena | global | 7 | 9.981 | 1.008 | 0.994-1.034 | 1.767e-03 | 5.584e-04 |
| cuda | 512 | clustered | 0 | arena | tile128 | 7 | 10.139 | 0.979 | 0.941-1.036 | 1.767e-03 | 5.584e-04 |
| cuda | 512 | clustered | 0 | arena | tile64 | 7 | 10.257 | 1.003 | 0.929-1.067 | 1.767e-03 | 5.584e-04 |
| cuda | 512 | clustered | 0 | baseline | auto | 7 | 9.925 |  |  | 1.767e-03 | 5.584e-04 |
| cuda | 512 | clustered | 1 | arena | auto | 7 | 11.079 | 1.007 | 0.934-1.019 | 1.767e-03 | 5.584e-04 |
| cuda | 512 | clustered | 1 | arena | global | 7 | 10.501 | 1.007 | 1.007-1.127 | 1.767e-03 | 5.584e-04 |
| cuda | 512 | clustered | 1 | arena | tile128 | 7 | 10.853 | 1.012 | 0.975-1.023 | 1.767e-03 | 5.584e-04 |
| cuda | 512 | clustered | 1 | arena | tile64 | 7 | 10.623 | 1.035 | 1.009-1.064 | 1.767e-03 | 5.584e-04 |
| cuda | 512 | clustered | 1 | baseline | auto | 7 | 10.992 |  |  | 1.767e-03 | 5.584e-04 |
| cuda | 512 | lattice | 0 | arena | auto | 7 | 12.596 | 0.853 | 0.836-0.899 | 3.418e-01 | 3.613e-02 |
| cuda | 512 | lattice | 0 | arena | global | 7 | 12.246 | 0.868 | 0.866-0.959 | 3.418e-01 | 3.613e-02 |
| cuda | 512 | lattice | 0 | arena | tile128 | 7 | 12.473 | 0.849 | 0.847-0.871 | 3.418e-01 | 3.613e-02 |
| cuda | 512 | lattice | 0 | arena | tile64 | 7 | 12.391 | 0.854 | 0.845-0.954 | 3.418e-01 | 3.613e-02 |
| cuda | 512 | lattice | 0 | baseline | auto | 7 | 10.603 |  |  | 3.418e-01 | 3.613e-02 |
| cuda | 512 | lattice | 1 | arena | auto | 7 | 13.228 | 0.843 | 0.826-0.936 | 3.418e-01 | 3.613e-02 |
| cuda | 512 | lattice | 1 | arena | global | 7 | 12.696 | 0.905 | 0.871-0.955 | 3.418e-01 | 3.613e-02 |
| cuda | 512 | lattice | 1 | arena | tile128 | 7 | 13.064 | 0.869 | 0.854-0.897 | 3.418e-01 | 3.613e-02 |
| cuda | 512 | lattice | 1 | arena | tile64 | 7 | 12.608 | 0.911 | 0.853-0.963 | 3.418e-01 | 3.613e-02 |
| cuda | 512 | lattice | 1 | baseline | auto | 7 | 11.151 |  |  | 3.418e-01 | 3.613e-02 |
| cuda | 1024 | adaptive | 0 | arena | auto | 7 | 98.092 | 0.844 | 0.818-0.851 | 7.528e-04 | 9.124e-05 |
| cuda | 1024 | adaptive | 0 | arena | global | 7 | 80.158 | 1.033 | 1.022-1.085 | 7.528e-04 | 9.124e-05 |
| cuda | 1024 | adaptive | 0 | arena | tile128 | 7 | 80.807 | 1.017 | 1.003-1.127 | 7.528e-04 | 9.124e-05 |
| cuda | 1024 | adaptive | 0 | arena | tile64 | 7 | 81.684 | 1.013 | 0.976-1.110 | 7.528e-04 | 9.124e-05 |
| cuda | 1024 | adaptive | 0 | baseline | auto | 7 | 82.778 |  |  | 7.528e-04 | 9.124e-05 |
| cuda | 1024 | adaptive | 1 | arena | auto | 7 | 102.599 | 0.834 | 0.797-0.843 | 7.528e-04 | 9.124e-05 |
| cuda | 1024 | adaptive | 1 | arena | global | 7 | 82.848 | 1.044 | 1.010-1.094 | 7.528e-04 | 9.124e-05 |
| cuda | 1024 | adaptive | 1 | arena | tile128 | 7 | 82.116 | 1.054 | 0.985-1.125 | 7.528e-04 | 9.124e-05 |
| cuda | 1024 | adaptive | 1 | arena | tile64 | 7 | 83.880 | 1.028 | 0.975-1.107 | 7.528e-04 | 9.124e-05 |
| cuda | 1024 | adaptive | 1 | baseline | auto | 7 | 84.111 |  |  | 7.528e-04 | 9.124e-05 |
| cuda | 1024 | clustered | 0 | arena | auto | 7 | 18.409 | 0.917 | 0.888-0.917 | 1.767e-03 | 5.584e-04 |
| cuda | 1024 | clustered | 0 | arena | global | 7 | 16.427 | 1.023 | 1.015-1.099 | 1.767e-03 | 5.584e-04 |
| cuda | 1024 | clustered | 0 | arena | tile128 | 7 | 16.707 | 1.010 | 0.976-1.010 | 1.767e-03 | 5.584e-04 |
| cuda | 1024 | clustered | 0 | arena | tile64 | 7 | 16.954 | 0.995 | 0.978-1.060 | 1.767e-03 | 5.584e-04 |
| cuda | 1024 | clustered | 0 | baseline | auto | 7 | 16.799 |  |  | 1.767e-03 | 5.584e-04 |
| cuda | 1024 | clustered | 1 | arena | auto | 7 | 19.740 | 0.903 | 0.871-0.911 | 1.767e-03 | 5.584e-04 |
| cuda | 1024 | clustered | 1 | arena | global | 7 | 17.873 | 1.014 | 0.956-1.079 | 1.767e-03 | 5.584e-04 |
| cuda | 1024 | clustered | 1 | arena | tile128 | 7 | 18.332 | 0.968 | 0.938-1.107 | 1.767e-03 | 5.584e-04 |
| cuda | 1024 | clustered | 1 | arena | tile64 | 7 | 18.397 | 0.966 | 0.934-1.091 | 1.767e-03 | 5.584e-04 |
| cuda | 1024 | clustered | 1 | baseline | auto | 7 | 17.763 |  |  | 1.767e-03 | 5.584e-04 |
| cuda | 1024 | lattice | 0 | arena | auto | 7 | 19.223 | 0.854 | 0.841-0.883 | 3.418e-01 | 3.613e-02 |
| cuda | 1024 | lattice | 0 | arena | global | 7 | 19.178 | 0.879 | 0.852-0.943 | 3.418e-01 | 3.613e-02 |
| cuda | 1024 | lattice | 0 | arena | tile128 | 7 | 19.245 | 0.862 | 0.853-0.871 | 3.418e-01 | 3.613e-02 |
| cuda | 1024 | lattice | 0 | arena | tile64 | 7 | 19.330 | 0.869 | 0.853-0.937 | 3.418e-01 | 3.613e-02 |
| cuda | 1024 | lattice | 0 | baseline | auto | 7 | 16.481 |  |  | 3.418e-01 | 3.613e-02 |
| cuda | 1024 | lattice | 1 | arena | auto | 7 | 20.720 | 0.861 | 0.841-0.867 | 3.418e-01 | 3.613e-02 |
| cuda | 1024 | lattice | 1 | arena | global | 7 | 20.405 | 0.874 | 0.869-0.951 | 3.418e-01 | 3.613e-02 |
| cuda | 1024 | lattice | 1 | arena | tile128 | 7 | 20.661 | 0.859 | 0.838-0.863 | 3.418e-01 | 3.613e-02 |
| cuda | 1024 | lattice | 1 | arena | tile64 | 7 | 20.417 | 0.874 | 0.855-0.944 | 3.418e-01 | 3.613e-02 |
| cuda | 1024 | lattice | 1 | baseline | auto | 7 | 17.733 |  |  | 3.418e-01 | 3.613e-02 |
