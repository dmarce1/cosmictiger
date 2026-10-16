#pragma once

void healpixInit();
__device__ void vec2pixNest(long const nside, double* vec, long* ipix);
