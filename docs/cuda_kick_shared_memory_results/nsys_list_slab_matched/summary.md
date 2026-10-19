Run of 14:49:57 to 14:52:22 CDT under the shared GPU lock. A 12-job ninja build
of another session ran during the first 15 s (the first one or two runs of
round 1 at bucket 64), and in round 2 at bucket 128 the host share of
head_matched (18.8 ms) and slab (15.4 ms) is twice that of the other rounds,
CPU contention the GPU-process sampler does not see; the kernel durations of
every variant are consistent across the three rounds, and the document's table
uses medians over the three rounds, which discard those single outliers.

| variant | round | whole kick forces (ms) | whole kick forces+potential (ms) | kernel launches | mean kernel (ms) | host share forces (ms) |
|---|---:|---:|---:|---:|---:|---:|
| adaptive128_baseline | 1 | 18.205 | 16.700 | 32 | 7.939 | 10.266 |
| adaptive128_baseline | 2 | 17.417 | 16.921 | 32 | 7.779 | 9.638 |
| adaptive128_baseline | 3 | 18.358 | 17.230 | 32 | 8.196 | 10.162 |
| adaptive128_head | 1 | 14.177 | 12.828 | 32 | 6.977 | 7.200 |
| adaptive128_head | 2 | 14.870 | 13.524 | 32 | 7.435 | 7.435 |
| adaptive128_head | 3 | 14.041 | 13.802 | 32 | 7.351 | 6.690 |
| adaptive128_head_matched | 1 | 19.832 | 18.299 | 32 | 9.509 | 10.323 |
| adaptive128_head_matched | 2 | 27.376 | 28.221 | 32 | 9.324 | 18.052 |
| adaptive128_head_matched | 3 | 19.722 | 17.995 | 32 | 9.623 | 10.099 |
| adaptive128_slab | 1 | 13.473 | 12.792 | 32 | 6.283 | 7.190 |
| adaptive128_slab | 2 | 21.966 | 23.344 | 32 | 7.155 | 14.811 |
| adaptive128_slab | 3 | 13.739 | 12.213 | 32 | 6.266 | 7.472 |
| adaptive128_slab_matched | 1 | 19.728 | 18.203 | 32 | 8.999 | 10.730 |
| adaptive128_slab_matched | 2 | 19.006 | 18.183 | 32 | 9.160 | 9.847 |
| adaptive128_slab_matched | 3 | 18.694 | 17.868 | 32 | 8.985 | 9.709 |
| adaptive64_baseline | 1 | 19.948 | 20.060 | 32 | 5.614 | 14.335 |
| adaptive64_baseline | 2 | 20.216 | 19.785 | 32 | 5.388 | 14.828 |
| adaptive64_baseline | 3 | 19.847 | 19.612 | 32 | 5.371 | 14.476 |
| adaptive64_head | 1 | 14.170 | 13.382 | 32 | 7.289 | 6.881 |
| adaptive64_head | 2 | 13.363 | 13.149 | 32 | 7.180 | 6.184 |
| adaptive64_head | 3 | 13.604 | 13.384 | 32 | 7.235 | 6.369 |
| adaptive64_head_matched | 1 | 21.335 | 21.186 | 32 | 6.972 | 14.363 |
| adaptive64_head_matched | 2 | 21.006 | 20.609 | 32 | 6.545 | 14.462 |
| adaptive64_head_matched | 3 | 21.362 | 21.052 | 32 | 6.865 | 14.498 |
| adaptive64_slab | 1 | 13.870 | 12.652 | 32 | 6.599 | 7.271 |
| adaptive64_slab | 2 | 13.760 | 13.002 | 32 | 6.646 | 7.113 |
| adaptive64_slab | 3 | 13.801 | 12.741 | 32 | 6.709 | 7.092 |
| adaptive64_slab_matched | 1 | 20.015 | 19.355 | 32 | 5.098 | 14.917 |
| adaptive64_slab_matched | 2 | 20.395 | 20.027 | 32 | 5.459 | 14.935 |
| adaptive64_slab_matched | 3 | 20.693 | 20.226 | 32 | 5.820 | 14.873 |

Per-phase medians over the samples (kernel launches paired with kicks in order):

| variant | round | forces: kick (ms) | kernel (ms) | host (ms) | forces+potential: kick (ms) | kernel (ms) | host (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| adaptive128_baseline | 1 | 18.205 | 8.411 | 9.787 | 16.700 | 6.843 | 9.535 |
| adaptive128_baseline | 2 | 17.417 | 7.442 | 9.630 | 16.921 | 7.404 | 9.446 |
| adaptive128_baseline | 3 | 18.358 | 8.271 | 9.606 | 17.230 | 7.234 | 9.500 |
| adaptive128_head | 1 | 14.177 | 6.653 | 6.864 | 12.828 | 5.770 | 6.890 |
| adaptive128_head | 2 | 14.870 | 7.622 | 7.424 | 13.524 | 6.456 | 7.057 |
| adaptive128_head | 3 | 14.041 | 6.841 | 7.045 | 13.802 | 7.024 | 6.773 |
| adaptive128_head_matched | 1 | 19.832 | 9.525 | 9.902 | 18.299 | 8.608 | 9.889 |
| adaptive128_head_matched | 2 | 27.376 | 9.309 | 18.806 | 28.221 | 8.891 | 19.210 |
| adaptive128_head_matched | 3 | 19.722 | 10.183 | 9.620 | 17.995 | 8.654 | 9.585 |
| adaptive128_slab | 1 | 13.473 | 6.241 | 7.230 | 12.792 | 5.847 | 6.889 |
| adaptive128_slab | 2 | 21.966 | 6.905 | 15.400 | 23.344 | 6.866 | 16.646 |
| adaptive128_slab | 3 | 13.739 | 6.485 | 6.940 | 12.213 | 5.509 | 6.690 |
| adaptive128_slab_matched | 1 | 19.728 | 9.635 | 9.971 | 18.203 | 8.627 | 9.591 |
| adaptive128_slab_matched | 2 | 19.006 | 9.118 | 9.762 | 18.183 | 8.671 | 9.522 |
| adaptive128_slab_matched | 3 | 18.694 | 8.971 | 9.537 | 17.868 | 8.083 | 9.312 |
| adaptive64_baseline | 1 | 19.948 | 4.956 | 14.916 | 20.060 | 5.075 | 14.950 |
| adaptive64_baseline | 2 | 20.216 | 5.107 | 14.897 | 19.785 | 4.664 | 14.644 |
| adaptive64_baseline | 3 | 19.847 | 5.308 | 14.701 | 19.612 | 5.050 | 14.571 |
| adaptive64_head | 1 | 14.170 | 7.159 | 7.019 | 13.382 | 6.338 | 6.721 |
| adaptive64_head | 2 | 13.363 | 6.609 | 6.779 | 13.149 | 6.641 | 6.534 |
| adaptive64_head | 3 | 13.604 | 7.108 | 6.966 | 13.384 | 6.696 | 6.587 |
| adaptive64_head_matched | 1 | 21.335 | 6.637 | 14.868 | 21.186 | 6.312 | 14.872 |
| adaptive64_head_matched | 2 | 21.006 | 5.649 | 15.025 | 20.609 | 5.849 | 14.929 |
| adaptive64_head_matched | 3 | 21.362 | 6.830 | 14.620 | 21.052 | 6.288 | 14.507 |
| adaptive64_slab | 1 | 13.870 | 6.772 | 6.982 | 12.652 | 5.704 | 6.818 |
| adaptive64_slab | 2 | 13.760 | 6.749 | 7.069 | 13.002 | 6.114 | 6.722 |
| adaptive64_slab | 3 | 13.801 | 6.684 | 6.950 | 12.741 | 6.104 | 6.573 |
| adaptive64_slab_matched | 1 | 20.015 | 5.158 | 14.544 | 19.355 | 4.619 | 14.576 |
| adaptive64_slab_matched | 2 | 20.395 | 5.517 | 14.893 | 20.027 | 5.074 | 14.766 |
| adaptive64_slab_matched | 3 | 20.693 | 5.450 | 14.607 | 20.226 | 5.623 | 14.463 |
