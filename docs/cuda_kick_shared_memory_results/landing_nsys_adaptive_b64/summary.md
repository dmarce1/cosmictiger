All twelve runs (23:15:25 to 23:16:21 CDT, under the shared GPU lock after 60 s
of quiet) ran back to back with no other GPU client and no compiler or profiler
on the machine; the idle wait before each process delayed none of them; the
one-minute load was 5.2 to 5.8 (status.txt), the idle desktop's level. Every
round is used. premerge = 00909d26, merged = 819d0013 (the frozen 1024-bucket
binaries of landing_timing_order7_b64/); "matched" pins both to the baseline's
work division and opening rule (compute.cudaKickHandoffBlocks=492,
gravity.openLargerCell=false). Medians over the rounds: medians_over_rounds.md.

| variant | round | whole kick forces (ms) | whole kick forces+potential (ms) | kernel launches | mean kernel (ms) | host share forces (ms) |
|---|---:|---:|---:|---:|---:|---:|
| adaptive64_merged | 1 | 13.972 | 13.063 | 32 | 6.521 | 7.451 |
| adaptive64_merged | 2 | 12.972 | 12.217 | 32 | 6.326 | 6.646 |
| adaptive64_merged | 3 | 13.246 | 12.448 | 32 | 6.280 | 6.966 |
| adaptive64_merged_matched | 1 | 19.994 | 20.123 | 32 | 5.469 | 14.525 |
| adaptive64_merged_matched | 2 | 19.899 | 19.780 | 32 | 5.333 | 14.566 |
| adaptive64_merged_matched | 3 | 20.003 | 19.940 | 32 | 5.313 | 14.691 |
| adaptive64_premerge | 1 | 13.544 | 12.744 | 32 | 6.942 | 6.602 |
| adaptive64_premerge | 2 | 14.337 | 12.803 | 32 | 7.257 | 7.080 |
| adaptive64_premerge | 3 | 13.341 | 12.695 | 32 | 6.947 | 6.394 |
| adaptive64_premerge_matched | 1 | 20.522 | 20.904 | 32 | 6.793 | 13.729 |
| adaptive64_premerge_matched | 2 | 20.101 | 20.429 | 32 | 6.519 | 13.581 |
| adaptive64_premerge_matched | 3 | 20.337 | 20.510 | 32 | 6.537 | 13.800 |

Per-phase medians over the samples (kernel launches paired with kicks in order):

| variant | round | forces: kick (ms) | kernel (ms) | host (ms) | forces+potential: kick (ms) | kernel (ms) | host (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| adaptive64_merged | 1 | 13.972 | 6.749 | 6.825 | 13.063 | 6.492 | 6.733 |
| adaptive64_merged | 2 | 12.972 | 6.191 | 6.630 | 12.217 | 5.789 | 6.391 |
| adaptive64_merged | 3 | 13.246 | 6.465 | 6.757 | 12.448 | 5.776 | 6.535 |
| adaptive64_merged_matched | 1 | 19.994 | 5.066 | 14.496 | 20.123 | 5.181 | 14.693 |
| adaptive64_merged_matched | 2 | 19.899 | 5.349 | 14.573 | 19.780 | 5.121 | 14.473 |
| adaptive64_merged_matched | 3 | 20.003 | 5.098 | 14.783 | 19.940 | 5.199 | 14.665 |
| adaptive64_premerge | 1 | 13.544 | 7.110 | 6.632 | 12.744 | 6.142 | 6.410 |
| adaptive64_premerge | 2 | 14.337 | 7.311 | 6.778 | 12.803 | 6.115 | 6.357 |
| adaptive64_premerge | 3 | 13.341 | 6.666 | 6.756 | 12.695 | 6.278 | 6.339 |
| adaptive64_premerge_matched | 1 | 20.522 | 6.298 | 14.280 | 20.904 | 6.499 | 14.211 |
| adaptive64_premerge_matched | 2 | 20.101 | 5.900 | 14.438 | 20.429 | 5.865 | 14.250 |
| adaptive64_premerge_matched | 3 | 20.337 | 5.879 | 14.330 | 20.510 | 5.788 | 14.561 |
