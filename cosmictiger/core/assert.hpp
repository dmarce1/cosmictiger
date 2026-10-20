/*
CosmicTiger - A cosmological N-Body code
Copyright (C) 2021  Dominic C. Marcello

Distributed under the Boost Software License, Version 1.0.
See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
*/

#pragma once

#ifdef NDEBUG
#define ASSERT(a)
#else
#ifdef __CUDA_ARCH__
#define ASSERT(a) cosmictigerCudaAssert(#a, a, __FILE__, __LINE__)
#else
#define ASSERT(a) cosmictigerAssert(#a, a, __FILE__, __LINE__)
#endif
#endif

#ifdef __CUDA_ARCH__
#define ALWAYS_ASSERT(a) cosmictigerCudaAssert(#a, a, __FILE__, __LINE__)
#else
#define ALWAYS_ASSERT(a) cosmictigerAssert(#a, a, __FILE__, __LINE__)
#endif

#ifdef __CUDA_ARCH__
__device__ void cosmictigerCudaAssert(char const*, bool, char const*, int);
#else
void cosmictigerAssert(char const*, bool, char const*, int);
#endif
