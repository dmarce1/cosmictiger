| device function | baseline warp instructions | arena warp instructions | difference |
|---|---:|---:|---:|
| `cudaGravityCcEwald` | 125,981,625 | 125,232,892 | -748,733 |
| `cudaAllocate` | 42,000,359 | 64,255,418 | +22,255,059 |
| `cudaGravityPpDirect<` | 0 | 30,949,237 | +30,949,237 |
| `cudaGravityPpDirect` | 29,534,872 | 0 | -29,534,872 |
| `cudaRelease` | 9,418,301 | 14,921,484 | +5,503,183 |
| `cudaKickKernel` | 7,625,648 | 7,628,695 | +3,047 |
| `cudaGravityPcDirect` | 4,217,217 | 4,200,592 | -16,625 |
| `cudaGravityCcDirect` | 3,182,189 | 3,119,561 | -62,628 |
| `EwaldConst::realIndex` | 2,951,952 | 2,937,712 | -14,240 |
| `cudaGravityCpDirect<` | 0 | 4,152,040 | +4,152,040 |
| `cudaGravityCpDirect` | 3,886,445 | 0 | -3,886,445 |
| `EwaldConst::fourExpansion` | 1,907,160 | 1,897,960 | -9,200 |
| `EwaldConst::fourIndex` | 1,907,160 | 1,897,960 | -9,200 |
| `addGpuFlops` | 578,620 | 573,860 | -4,760 |
| `EwaldConst::d0` | 350,337 | 348,647 | -1,690 |
| total | 233,568,846 | 262,142,559 | +28,573,713 |
