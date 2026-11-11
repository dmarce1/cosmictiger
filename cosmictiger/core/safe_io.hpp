/*
CosmicTiger - A cosmological N-Body code
Copyright (C) 2021  Dominic C. Marcello

Distributed under the Boost Software License, Version 1.0.
See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
*/

#ifndef SAFE_IO_HPP_
#define SAFE_IO_HPP_

#include <cstdio>
#include <format>
#include <ios>

#define FREAD(a, b, c, d) safeFread(a, b, c, d, __LINE__, __FILE__)
#define FWRITE(a, b, c, d) safeFwrite(a, b, c, d, __LINE__, __FILE__)

/// Read exactly count binary elements, reporting a source-located error on a short read.
static void safeFread(void* src, size_t size, size_t count, FILE* fp, int line, char const* file) {
    auto read = fread(src, size, count, fp);
    if (read != count) {
        throw std::ios_base::failure(
            std::format("Attempt to read {} elements of size {} failed - only {} elements read.\n"
                        "Error in {} on line {}",
                count, size, read, file, line));
    }
}

/// Write exactly count binary elements, reporting a source-located error on a short write.
static void safeFwrite(
    void const* src, size_t size, size_t count, FILE* fp, int line, char const* file) {
    auto write = fwrite(src, size, count, fp);
    if (write != count) {
        throw std::ios_base::failure(std::format(
            "Attempt to write {} elements of size {} failed - only {} elements written.\n"
            "Error in {} on line {}",
            count, size, write, file, line));
    }
}

#define ASPRINTF(...)               \
    if (asprintf(__VA_ARGS__) == 0) \
    printf("asprintf failed\n")
#endif /* SAFE_IO_HPP_ */
