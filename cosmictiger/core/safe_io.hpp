/*
CosmicTiger - A cosmological N-Body code
Copyright (C) 2021  Dominic C. Marcello

Distributed under the Boost Software License, Version 1.0.
See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
*/

#ifndef SAFE_IO_HPP_
#define SAFE_IO_HPP_

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include <cosmictiger/core/assert.hpp>

#define THROW_ERROR(...) throwError(__FILE__, __LINE__, __VA_ARGS__)

#ifndef __CUDA_ARCH__
namespace detail {

[[noreturn]] inline void throwRuntimeError(char const* file, int line, std::string message) {
    if (!message.empty() && message.back() != '\n') {
        message += '\n';
    }
    message += "Error in ";
    message += file;
    message += " on line ";
    message += std::to_string(line);
    throw std::runtime_error(message);
}

}    // namespace detail
#endif

// CUDA device code cannot throw C++ exceptions; retain its diagnostic and trap.
template <class... Args>
[[noreturn]]
#ifdef __CUDA_ARCH__
__device__
#endif
    inline void throwError(char const* file, int line, char const* fmt, Args... args) {
#ifdef __CUDA_ARCH__
    printf(fmt, args...);
    printf("Error in %s on line %i\n", file, line);
    __trap();
#else
    int const length = std::snprintf(nullptr, 0, fmt, args...);
    if (length < 0) {
        detail::throwRuntimeError(file, line, "Unable to format error message");
    }
    std::string message(static_cast<size_t>(length) + 1, '\0');
    std::snprintf(message.data(), message.size(), fmt, args...);
    message.resize(static_cast<size_t>(length));
    detail::throwRuntimeError(file, line, message);
#endif
}

[[noreturn]]
#ifdef __CUDA_ARCH__
__device__
#endif
    inline void throwError(char const* file, int line, char const* str) {
#ifdef __CUDA_ARCH__
    printf("%s", str);
    printf("Error in %s on line %i\n", file, line);
    __trap();
#else
    detail::throwRuntimeError(file, line, str);
#endif
}

#define FREAD(a, b, c, d) safeFread(a, b, c, d, __LINE__, __FILE__)
#define FWRITE(a, b, c, d) safeFwrite(a, b, c, d, __LINE__, __FILE__)

static void safeFread(void* src, size_t size, size_t count, FILE* fp, int line, char const* file) {
    auto read = fread(src, size, count, fp);
    if (read != count) {
        throwError(file, line,
            "Attempt to read %zu elements of size %zu failed - only %zu elements read.\n", count,
            size, read);
    }
}

static void safeFwrite(
    void const* src, size_t size, size_t count, FILE* fp, int line, char const* file) {
    auto write = fwrite(src, size, count, fp);
    if (write != count) {
        throwError(file, line,
            "Attempt to write %zu elements of size %zu failed - only %zu elements written.\n",
            count, size, write);
    }
}

#define ASPRINTF(...)               \
    if (asprintf(__VA_ARGS__) == 0) \
    printf("asprintf failed\n")
#endif /* SAFE_IO_HPP_ */
