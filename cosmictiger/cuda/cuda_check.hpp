#pragma once

#include <cstdio>
#include <format>
#include <stdexcept>

// CUDA status checks evaluate expressions once and behave as one statement.
#define CUDA_CHECK(expression)                                                       \
    do {                                                                             \
        cudaError_t const cudaCheckStatus = (expression);                            \
        if (cudaCheckStatus != cudaSuccess) {                                        \
            throw std::runtime_error(                                                \
                std::format("CUDA error: {}", cudaGetErrorString(cudaCheckStatus))); \
        }                                                                            \
    } while (false)

#define CUDA_WARN(expression)                                                  \
    do {                                                                       \
        cudaError_t const cudaWarnStatus = (expression);                       \
        if (cudaWarnStatus != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA warning at %s:%i: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(cudaWarnStatus));                           \
        }                                                                      \
    } while (false)
