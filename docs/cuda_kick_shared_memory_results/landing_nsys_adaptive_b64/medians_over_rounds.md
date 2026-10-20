Medians over the three rounds of the per-phase table in summary.md (whole kick, kick kernel and host share = whole kick minus its kernel, forces only and with the potential). matched = compute.cudaKickHandoffBlocks=492 and gravity.openLargerCell=false on both binaries.

| fixture, bucket | variant | whole kick (ms) | kick kernel (ms) | host (ms) | with potential: whole kick | kernel | host |
|---|---|---:|---:|---:|---:|---:|---:|
| adaptive, bucket 64 | merged (3 rounds) | 13.25 | 6.46 | 6.76 | 12.45 | 5.79 | 6.54 |
| adaptive, bucket 64 | merged_matched (3 rounds) | 19.99 | 5.10 | 14.57 | 19.94 | 5.18 | 14.66 |
| adaptive, bucket 64 | premerge (3 rounds) | 13.54 | 7.11 | 6.76 | 12.74 | 6.14 | 6.36 |
| adaptive, bucket 64 | premerge_matched (3 rounds) | 20.34 | 5.90 | 14.33 | 20.51 | 5.87 | 14.25 |
