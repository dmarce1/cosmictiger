/*
CosmicTiger - A cosmological N-Body code
Copyright (C) 2021  Dominic C. Marcello

Distributed under the Boost Software License, Version 1.0.
See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
*/

#ifndef SAFE_IO_HPP_
#define SAFE_IO_HPP_

#include <stdio.h>
#include <stdlib.h>

#include <cosmictiger/core/assert.hpp>

#define PRINT(...) print(__VA_ARGS__)

#define PRINT_BOTH(fp, ...) printBoth(fp, __VA_ARGS__)

#define THROW_ERROR(...) throwError(__FILE__, __LINE__, __VA_ARGS__)

template <class... Args>
#ifdef __CUDA_ARCH__
__device__
#endif
    inline void print(char const* fmt, Args... args) {
    printf(fmt, args...);
#ifndef __CUDA_ARCH__
    fflush(stdout);
#endif
}

template <class... Args>
inline void printBoth(FILE* fp, char const* fmt, Args... args) {
    printf(fmt, args...);
    fprintf(fp, fmt, args...);
    fflush(stdout);
    fflush(fp);
}

#ifdef __CUDA_ARCH__
__device__
#endif
    inline void print(char const* str) {
    printf("%s", str);
#ifndef __CUDA_ARCH__
    fflush(stdout);
#endif
}
template <class... Args>
#ifdef __CUDA_ARCH__
__device__
#endif
    inline void throwError(char const* file, int line, char const* fmt, Args... args) {
    printf(fmt, args...);
    printf("Error in %s on line %i\n", file, line);
#ifndef __CUDA_ARCH__
    fflush(stdout);
    ALWAYS_ASSERT(false);
#else
    __trap();
#endif
}

inline void throwError(char const* file, int line, char const* str) {
    printf("%s", str);
    printf("Error in %s on line %i\n", file, line);
#ifndef __CUDA_ARCH__
    fflush(stdout);
    ALWAYS_ASSERT(false);
#else
    __trap();
#endif
}

#define FREAD(a, b, c, d) safeFread(a, b, c, d, __LINE__, __FILE__)
#define FWRITE(a, b, c, d) safeFwrite(a, b, c, d, __LINE__, __FILE__)

static void safeFread(void* src, size_t size, size_t count, FILE* fp, int line, char const* file) {
    auto read = fread(src, size, count, fp);
    if (read != count) {
        PRINT("Attempt to read %li elements of size %li in %s on line %i failed - only %li "
              "elements read.\n",
            count, size, file, line, read);
        abort();
    }
}

static void safeFwrite(void* src, size_t size, size_t count, FILE* fp, int line, char const* file) {
    auto write = fwrite(src, size, count, fp);
    if (write != count) {
        PRINT("Attempt to write %li elements of size %li in %s on line %i failed - only %li "
              "elements written.\n",
            count, size, file, line, write);
        abort();
    }
}

#define ASPRINTF(...)               \
    if (asprintf(__VA_ARGS__) == 0) \
    printf("asprintf failed\n")
#endif /* SAFE_IO_HPP_ */
