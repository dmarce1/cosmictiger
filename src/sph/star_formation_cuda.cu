/* CosmicTiger - parallel SH03 source, EOS, and timestep batches; BSL-1.0. */
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

#include <cosmictiger/sph/star_formation_cuda.hpp>

namespace star_formation_backend {
namespace detail {
    __global__ void batchKernel(Operation operation, Input const* inputs, Output* outputs,
        size_t count, Parameters parameters) {
        size_t const stride = size_t(blockDim.x) * gridDim.x;
        for (size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
            index += stride) {
            outputs[index] = evaluate(operation, inputs[index], outputs[index], parameters);
        }
    }
    inline void check(cudaError_t error) {
        if (error != cudaSuccess) {
            throw std::runtime_error(std::string("SH03 CUDA: ") + cudaGetErrorString(error));
        }
    }
    template <class T>
    class Buffer {
        T* pointer_{};

    public:
        explicit Buffer(size_t count) {
            check(cudaMalloc(reinterpret_cast<void**>(&pointer_), count * sizeof(T)));
        }
        ~Buffer() {
            cudaFree(pointer_);
        }
        T* get() const {
            return pointer_;
        }
        Buffer(Buffer const&) = delete;
        Buffer& operator=(Buffer const&) = delete;
    };
}    // namespace detail
void runCuda(Operation operation, Input const* inputs, Output* outputs, size_t count,
    Parameters const& parameters) {
    if (count == 0) {
        return;
    }
    if (inputs == nullptr || outputs == nullptr ||
        count > std::numeric_limits<size_t>::max() / sizeof(Output) ||
        count > std::numeric_limits<size_t>::max() / sizeof(Input)) {
        throw std::invalid_argument("Invalid SH03 CUDA batch buffers");
    }
    detail::Buffer<Input> deviceInputs(count);
    detail::Buffer<Output> deviceOutputs(count);
    detail::check(
        cudaMemcpy(deviceInputs.get(), inputs, count * sizeof(Input), cudaMemcpyHostToDevice));
    detail::check(
        cudaMemcpy(deviceOutputs.get(), outputs, count * sizeof(Output), cudaMemcpyHostToDevice));
    constexpr unsigned kBlockSize = 64;
    unsigned const blocks = unsigned(std::min<size_t>(65535, 1 + (count - 1) / kBlockSize));
    detail::batchKernel<<<blocks, kBlockSize>>>(
        operation, deviceInputs.get(), deviceOutputs.get(), count, parameters);
    detail::check(cudaGetLastError());
    detail::check(
        cudaMemcpy(outputs, deviceOutputs.get(), count * sizeof(Output), cudaMemcpyDeviceToHost));
    for (size_t index = 0; index < count; ++index) {
        if (outputs[index].status != 0) {
            throw std::runtime_error("SH03 CUDA particle " + std::to_string(index) +
                " failed with status " + std::to_string(outputs[index].status) +
                "; no source state committed");
        }
    }
}
}    // namespace star_formation_backend
