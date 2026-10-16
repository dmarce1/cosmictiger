/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2021  Dominic C. Marcello

 This program is free software; you can redistribute it and/or
 modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation; either version 2
 of the License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */

#include <cosmictiger/cuda.hpp>
#include <cosmictiger/cuda_memory.hpp>
#include <cosmictiger/ewald_indices.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/particles.hpp>

#ifdef USE_CUDA

void cudaAllocate(void** ptr, size_t size, char const* file, int line) {
    static MutexType mutex;
    std::lock_guard<MutexType> lock(mutex);
    double freeMem = cudaFreeMemory();
    double totalMem = cudaTotalMemory();
    if ((freeMem - (double) size) / totalMem < 0.15) {
        PRINT("Attempt to allocate %li bytes on rank %i in %s on line %i leaves less than 15%% "
              "memory\n",
            size, hpxRank(), file, line);
        abort();
    }
    CUDA_CHECK(cudaMalloc(ptr, size));
}

void cudaSetDevice() {
    int count;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    int const deviceNum = hpxRank() % count;
    CUDA_CHECK(cudaSetDevice(deviceNum));
}

int cudaGetDevice() {
    int count;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    int const deviceNum = hpxRank() % count;
    return deviceNum;
}

size_t cudaTotalMemory() {
    size_t total;
    size_t free;
    cudaSetDevice();
    CUDA_CHECK(cudaMemGetInfo(&free, &total));
    return total;
}

size_t cudaFreeMemory() {
    size_t total;
    size_t free;
    cudaSetDevice();
    CUDA_CHECK(cudaMemGetInfo(&free, &total));
    return free;
}

int cudaSmpCount() {
    int count;
    cudaSetDevice();
    CUDA_CHECK(cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, cudaGetDevice()));
    return count;
}

cudaStream_t cudaGetStream() {
    cudaStream_t stream;
    cudaSetDevice();
    CUDA_CHECK(cudaStreamCreate(&stream));
    return stream;
}

void synchronizeCudaStream(cudaStream_t stream) {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    while (cudaStreamQuery(stream) != cudaSuccess) {
        hpxYield();
    }
    CUDA_CHECK(cudaSetDevice(device));
}

void cudaEndStream(cudaStream_t stream) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

void cudaInit() {
    cudaSetDevice();
    CUDA_CHECK(cudaDeviceReset());
    /*	size_t value = 64;
        size_t oval = value;
        CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMaxL2FetchGranularity , value));
        CUDA_CHECK(cudaDeviceGetLimit(&value, cudaLimitMaxL2FetchGranularity ));
        if (value != oval) {
            THROW_ERROR("Unable to set stack size to %li\n", oval);
        }*/
    cudaMemoryInitialize(HEAP_SIZE);
}

#endif
