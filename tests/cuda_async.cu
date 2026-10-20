/* CosmicTiger - bounded CUDA completion test kernel; BSL-1.0. */

#include <cuda_runtime.h>

namespace detail {

__global__ void cudaAsyncTransformKernel(int* values, int count) {
    int const index = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < count) {
        values[index] = 2 * values[index] + 7;
    }
}

}    // namespace detail

void cudaAsyncTestLaunch(int* values, int count, cudaStream_t stream, bool invalid) {
    unsigned const blocks = invalid ? 0u : 1u;
    detail::cudaAsyncTransformKernel<<<blocks, 64, 0, stream>>>(values, count);
}
