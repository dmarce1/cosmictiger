/*
CosmicTiger - A cosmological N-Body code
Copyright (C) 2021  Dominic C. Marcello

Distributed under the Boost Software License, Version 1.0.
See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
*/

#include <cosmictiger/core/assert.hpp>
#include <cosmictiger/core/safe_io.hpp>

#ifdef __CUDA_ARCH__
__device__ void cosmictigerCudaAssert(
    char const* condStr, bool cond, char const* filename, int line) {
    if (!cond) {
        PRINT("Condition \"%s\" failed in %s on line %i in CUDA_ARCH\n", condStr, filename, line);
        __trap();
    }
}
#endif
