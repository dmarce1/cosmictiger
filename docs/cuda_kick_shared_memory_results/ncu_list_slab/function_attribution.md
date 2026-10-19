| device function | head | slab | difference |
|---|---:|---:|---:|
| `void cudaKickKernel<(int)7>` | 128,944,938 | 128,912,577 | -32,361 |
| `cudaAllocate` | 17,144,740 | 0 | -17,144,740 |
| `cudaRelease` | 1,075,614 | 0 | -1,075,614 |
| `addGpuFlops` | 641,480 | 641,480 | +0 |
| `cosmictigerCudaAssert` | 35,070 | 35,070 | +0 |
| `total` | 147,841,842 | 129,589,127 | -18,252,715 |
