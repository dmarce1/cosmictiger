| backend | bucket | fixture | potential | binary | policy | order | kick ms | speedup vs reference | paired range | force rel L2 | potential rel L2 |
|---|---:|---|---:|---|---|---:|---:|---:|---|---:|---:|
| cuda | 32 | adaptive | 0 | arena | auto | 7 | 21.004 | 0.898 | 0.890-0.906 |  |  |
| cuda | 32 | adaptive | 0 | arena | handoff492 | 7 | 19.189 | 0.983 | 0.969-0.997 |  |  |
| cuda | 32 | adaptive | 0 | arena | res6 | 7 | 21.141 | 0.892 | 0.880-0.905 |  |  |
| cuda | 32 | adaptive | 0 | arena | reserve0 | 7 | 21.946 | 0.859 | 0.854-0.864 |  |  |
| cuda | 32 | adaptive | 0 | baseline | auto | 7 | 18.859 |  |  |  |  |
| cuda | 32 | adaptive | 1 | arena | auto | 7 | 21.278 | 0.882 | 0.870-0.893 |  |  |
| cuda | 32 | adaptive | 1 | arena | handoff492 | 7 | 19.438 | 0.965 | 0.961-0.969 |  |  |
| cuda | 32 | adaptive | 1 | arena | res6 | 7 | 21.305 | 0.880 | 0.871-0.890 |  |  |
| cuda | 32 | adaptive | 1 | arena | reserve0 | 7 | 21.755 | 0.862 | 0.859-0.865 |  |  |
| cuda | 32 | adaptive | 1 | baseline | auto | 7 | 18.757 |  |  |  |  |
| cuda | 32 | clustered | 0 | arena | auto | 7 | 11.464 | 0.824 | 0.788-0.861 |  |  |
| cuda | 32 | clustered | 0 | arena | handoff492 | 7 | 8.528 | 1.108 | 1.062-1.154 |  |  |
| cuda | 32 | clustered | 0 | arena | res6 | 7 | 11.155 | 0.847 | 0.823-0.871 |  |  |
| cuda | 32 | clustered | 0 | arena | reserve0 | 7 | 10.934 | 0.865 | 0.813-0.918 |  |  |
| cuda | 32 | clustered | 0 | baseline | auto | 7 | 9.454 |  |  |  |  |
| cuda | 32 | clustered | 1 | arena | auto | 7 | 11.974 | 0.797 | 0.782-0.812 |  |  |
| cuda | 32 | clustered | 1 | arena | handoff492 | 7 | 8.890 | 1.073 | 1.061-1.086 |  |  |
| cuda | 32 | clustered | 1 | arena | res6 | 7 | 11.768 | 0.812 | 0.775-0.848 |  |  |
| cuda | 32 | clustered | 1 | arena | reserve0 | 7 | 11.819 | 0.807 | 0.784-0.830 |  |  |
| cuda | 32 | clustered | 1 | baseline | auto | 7 | 9.539 |  |  |  |  |
| cuda | 32 | lattice | 0 | arena | auto | 7 | 91.291 | 0.369 | 0.368-0.370 |  |  |
| cuda | 32 | lattice | 0 | arena | handoff492 | 7 | 31.817 | 1.060 | 1.054-1.065 |  |  |
| cuda | 32 | lattice | 0 | arena | res6 | 7 | 91.287 | 0.369 | 0.368-0.371 |  |  |
| cuda | 32 | lattice | 0 | arena | reserve0 | 7 | 90.621 | 0.372 | 0.369-0.375 |  |  |
| cuda | 32 | lattice | 0 | baseline | auto | 7 | 33.713 |  |  |  |  |
| cuda | 32 | lattice | 1 | arena | auto | 7 | 90.933 | 0.371 | 0.370-0.372 |  |  |
| cuda | 32 | lattice | 1 | arena | handoff492 | 7 | 32.006 | 1.054 | 1.040-1.068 |  |  |
| cuda | 32 | lattice | 1 | arena | res6 | 7 | 91.177 | 0.370 | 0.369-0.371 |  |  |
| cuda | 32 | lattice | 1 | arena | reserve0 | 7 | 90.595 | 0.372 | 0.371-0.374 |  |  |
| cuda | 32 | lattice | 1 | baseline | auto | 7 | 33.740 |  |  |  |  |
| cuda | 64 | adaptive | 0 | arena | auto | 7 | 20.700 | 0.923 | 0.908-0.938 |  |  |
| cuda | 64 | adaptive | 0 | arena | handoff492 | 7 | 19.943 | 0.958 | 0.954-0.962 |  |  |
| cuda | 64 | adaptive | 0 | arena | res6 | 7 | 20.454 | 0.934 | 0.920-0.948 |  |  |
| cuda | 64 | adaptive | 0 | arena | reserve0 | 7 | 21.093 | 0.906 | 0.893-0.919 |  |  |
| cuda | 64 | adaptive | 0 | baseline | auto | 7 | 19.101 |  |  |  |  |
| cuda | 64 | adaptive | 1 | arena | auto | 7 | 20.972 | 0.926 | 0.923-0.928 |  |  |
| cuda | 64 | adaptive | 1 | arena | handoff492 | 7 | 20.108 | 0.966 | 0.958-0.974 |  |  |
| cuda | 64 | adaptive | 1 | arena | res6 | 7 | 20.831 | 0.932 | 0.929-0.935 |  |  |
| cuda | 64 | adaptive | 1 | arena | reserve0 | 7 | 21.135 | 0.919 | 0.905-0.933 |  |  |
| cuda | 64 | adaptive | 1 | baseline | auto | 7 | 19.418 |  |  |  |  |
| cuda | 64 | clustered | 0 | arena | auto | 7 | 8.387 | 1.051 | 1.038-1.065 |  |  |
| cuda | 64 | clustered | 0 | arena | handoff492 | 7 | 8.422 | 1.047 | 1.046-1.048 |  |  |
| cuda | 64 | clustered | 0 | arena | res6 | 7 | 8.317 | 1.060 | 1.042-1.078 |  |  |
| cuda | 64 | clustered | 0 | arena | reserve0 | 7 | 8.516 | 1.036 | 1.017-1.055 |  |  |
| cuda | 64 | clustered | 0 | baseline | auto | 7 | 8.818 |  |  |  |  |
| cuda | 64 | clustered | 1 | arena | auto | 7 | 8.504 | 1.061 | 1.039-1.083 |  |  |
| cuda | 64 | clustered | 1 | arena | handoff492 | 7 | 8.407 | 1.073 | 1.048-1.099 |  |  |
| cuda | 64 | clustered | 1 | arena | res6 | 7 | 8.517 | 1.059 | 1.047-1.072 |  |  |
| cuda | 64 | clustered | 1 | arena | reserve0 | 7 | 8.738 | 1.032 | 1.021-1.044 |  |  |
| cuda | 64 | clustered | 1 | baseline | auto | 7 | 9.021 |  |  |  |  |
| cuda | 64 | lattice | 0 | arena | auto | 7 | 31.731 | 1.045 | 1.040-1.049 |  |  |
| cuda | 64 | lattice | 0 | arena | handoff492 | 7 | 31.840 | 1.041 | 1.025-1.058 |  |  |
| cuda | 64 | lattice | 0 | arena | res6 | 7 | 31.534 | 1.051 | 1.046-1.057 |  |  |
| cuda | 64 | lattice | 0 | arena | reserve0 | 7 | 31.891 | 1.040 | 1.037-1.042 |  |  |
| cuda | 64 | lattice | 0 | baseline | auto | 7 | 33.152 |  |  |  |  |
| cuda | 64 | lattice | 1 | arena | auto | 7 | 31.560 | 1.053 | 1.048-1.057 |  |  |
| cuda | 64 | lattice | 1 | arena | handoff492 | 7 | 31.605 | 1.051 | 1.051-1.051 |  |  |
| cuda | 64 | lattice | 1 | arena | res6 | 7 | 31.421 | 1.057 | 1.045-1.070 |  |  |
| cuda | 64 | lattice | 1 | arena | reserve0 | 7 | 31.716 | 1.048 | 1.045-1.050 |  |  |
| cuda | 64 | lattice | 1 | baseline | auto | 7 | 33.225 |  |  |  |  |
