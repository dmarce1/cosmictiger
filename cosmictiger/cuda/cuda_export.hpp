/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#pragma once

#ifdef __CUDACC__
#define CUDA_EXPORT __host__ __device__
#else
#define CUDA_EXPORT
#endif
