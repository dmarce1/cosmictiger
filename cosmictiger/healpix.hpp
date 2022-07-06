#pragma once


void healpix_init();
__device__ void vec2pix_nest(const long nside, double *vec, long *ipix);
