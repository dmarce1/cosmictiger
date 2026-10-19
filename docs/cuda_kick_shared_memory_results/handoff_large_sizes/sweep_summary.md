### 2,097,152 particles, 8 threads (`handoff_large_sizes/t8_n128/`)

| bucket | fixture | potential | handoff328 | handoff492 | handoff656 | handoff984 | handoff1312 | fastest |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 128 | adaptive | 0 | 91.61 ms | 80.77 ms, 1.136x (1.070-1.201) | 75.32 ms, 1.217x (1.161-1.272) | 77.74 ms, 1.178x (1.136-1.221) | 78.94 ms, 1.160x (1.118-1.202) | handoff656 |
| 128 | adaptive | 1 | 109.93 ms | 95.05 ms, 1.157x (1.154-1.160) | 91.47 ms, 1.202x (1.149-1.256) | 94.55 ms, 1.163x (1.115-1.211) | 93.25 ms, 1.179x (1.134-1.224) | handoff656 |
| 128 | clustered | 0 | 76.40 ms | 77.03 ms, 0.992x (0.992-0.992) | 67.95 ms, 1.126x (1.087-1.164) | 65.22 ms, 1.171x (1.156-1.187) | 63.81 ms, 1.198x (1.168-1.228) | handoff1312 |
| 128 | clustered | 1 | 90.74 ms | 93.11 ms, 0.975x (0.922-1.029) | 86.70 ms, 1.046x (1.027-1.066) | 76.98 ms, 1.178x (1.160-1.196) | 76.54 ms, 1.186x (1.127-1.246) | handoff1312 |
| 128 | lattice | 0 | 81.81 ms | 83.29 ms, 0.982x (0.960-1.005) | 100.78 ms, 0.812x (0.789-0.835) | 104.36 ms, 0.784x (0.753-0.816) | 103.52 ms, 0.790x (0.774-0.807) | handoff328 |
| 128 | lattice | 1 | 99.63 ms | 98.31 ms, 1.013x (1.001-1.025) | 115.34 ms, 0.864x (0.818-0.910) | 116.17 ms, 0.858x (0.817-0.898) | 114.02 ms, 0.874x (0.836-0.912) | handoff492 |

### 2,097,152 particles, 12 threads (`handoff_large_sizes/t12_n128/`)

| bucket | fixture | potential | handoff328 | handoff492 | handoff656 | handoff984 | handoff1312 | fastest |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 128 | adaptive | 0 | 88.23 ms | 76.37 ms, 1.155x (1.144-1.166) | 78.04 ms, 1.130x (1.112-1.149) | 78.07 ms, 1.130x (1.105-1.154) | 79.68 ms, 1.107x (1.082-1.132) | handoff492 |
| 128 | adaptive | 1 | 105.95 ms | 92.36 ms, 1.147x (1.098-1.197) | 93.02 ms, 1.139x (1.133-1.145) | 90.15 ms, 1.175x (1.150-1.199) | 89.33 ms, 1.186x (1.146-1.226) | handoff1312 |
| 128 | clustered | 0 | 76.52 ms | 72.91 ms, 1.054x (0.959-1.148) | 71.41 ms, 1.072x (1.065-1.079) | 66.65 ms, 1.150x (1.125-1.174) | 70.24 ms, 1.090x (1.029-1.151) | handoff984 |
| 128 | clustered | 1 | 91.20 ms | 87.02 ms, 1.048x (1.009-1.087) | 82.73 ms, 1.103x (1.098-1.107) | 79.59 ms, 1.147x (1.128-1.165) | 77.72 ms, 1.173x (1.137-1.209) | handoff1312 |
| 128 | lattice | 0 | 84.76 ms | 77.91 ms, 1.090x (0.992-1.188) | 95.76 ms, 0.884x (0.863-0.905) | 95.91 ms, 0.883x (0.846-0.919) | 92.09 ms, 0.920x (0.865-0.975) | handoff492 |
| 128 | lattice | 1 | 97.84 ms | 93.47 ms, 1.046x (0.994-1.098) | 106.03 ms, 0.922x (0.892-0.952) | 107.17 ms, 0.912x (0.897-0.928) | 102.88 ms, 0.950x (0.900-1.001) | handoff492 |

### Time relative to handoff328, geometric mean over cases (worst case)

| particles | threads | handoff328 | handoff492 | handoff656 | handoff984 | handoff1312 |
|---:|---:|---:|---:|---:|---:|---:|
| 2,097,152 | 8 | 1.000 (1.000) | 0.962 (1.026) | 0.969 (1.232) | 0.961 (1.276) | 0.952 (1.265) |
| 2,097,152 | 12 | 1.000 (1.000) | 0.919 (0.955) | 0.964 (1.130) | 0.944 (1.131) | 0.938 (1.086) |

### Fastest policy per case (reference handoff328 time over the fastest time)

| particles | threads | bucket | fixture | potential | fastest | reference / fastest |
|---:|---:|---:|---|---:|---|---:|
| 2,097,152 | 8 | 128 | adaptive | 0 | handoff656 | 1.216 |
| 2,097,152 | 8 | 128 | adaptive | 1 | handoff656 | 1.202 |
| 2,097,152 | 8 | 128 | clustered | 0 | handoff1312 | 1.197 |
| 2,097,152 | 8 | 128 | clustered | 1 | handoff1312 | 1.185 |
| 2,097,152 | 8 | 128 | lattice | 0 | handoff328 | 1.000 |
| 2,097,152 | 8 | 128 | lattice | 1 | handoff492 | 1.013 |
| 2,097,152 | 12 | 128 | adaptive | 0 | handoff492 | 1.155 |
| 2,097,152 | 12 | 128 | adaptive | 1 | handoff1312 | 1.186 |
| 2,097,152 | 12 | 128 | clustered | 0 | handoff984 | 1.148 |
| 2,097,152 | 12 | 128 | clustered | 1 | handoff1312 | 1.173 |
| 2,097,152 | 12 | 128 | lattice | 0 | handoff492 | 1.088 |
| 2,097,152 | 12 | 128 | lattice | 1 | handoff492 | 1.047 |

### Stage conditions

| particles | threads | load average at start | worst pair spread | process wall s |
|---:|---:|---:|---:|---:|
| 2,097,152 | 8 | 6.10 | 1.087 | 35.2-44.0 |
| 2,097,152 | 12 | 8.25 | 1.155 | 38.5-48.0 |
