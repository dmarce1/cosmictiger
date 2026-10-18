#pragma once

#ifdef USE_CUDA
void healpixInit();
#endif
#ifdef __CUDACC__
__device__ void vec2pixNest(long const nside, double* vec, long* ipix);

#endif
