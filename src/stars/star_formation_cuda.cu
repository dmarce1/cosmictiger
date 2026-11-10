/* CosmicTiger - parallel SH03 source, EOS, and timestep batches; BSL-1.0. */
#include <algorithm>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include <cosmictiger/stars/star_formation_cuda.hpp>

namespace star_formation_backend {
namespace detail {
    /// Evaluate the selected star-formation operation for each input in a grid-stride
    /// CUDA loop.
    __global__ void batchKernel(Operation operation, Input const* inputs, Output* outputs,
        size_t count, Parameters parameters) {
        size_t const stride = size_t(blockDim.x) * gridDim.x;
        for (size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
            index += stride) {
            outputs[index] = evaluate(operation, inputs[index], outputs[index], parameters);
        }
    }
    /// Translate a CUDA failure into a star-formation backend exception.
    inline void check(cudaError_t error) {
        if (error != cudaSuccess) {
            throw std::runtime_error(std::string("SH03 CUDA: ") + cudaGetErrorString(error));
        }
    }
    template <class T>
    class Buffer {
        T* pointer_{};

    public:
        /// Allocate an owned device array for the requested number of elements.
        explicit Buffer(size_t count) {
            check(cudaMalloc(reinterpret_cast<void**>(&pointer_), count * sizeof(T)));
        }
        /// Release the owned device allocation.
        ~Buffer() {
            cudaFree(pointer_);
        }
        /// Expose the device pointer without transferring ownership.
        T* get() const {
            return pointer_;
        }
        Buffer(Buffer const&) = delete;
        Buffer& operator=(Buffer const&) = delete;
    };
}    // namespace detail
/// Validate and transfer a batch, execute it, and publish host outputs only after all
/// results pass validation.
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
    auto deviceParameters = parameters;
    if (operation == Operation::advance || operation == Operation::diagnostics) {
        std::string error;
        deviceParameters.cooling.rates = gadget4::deviceRateTable(&error);
        if (deviceParameters.cooling.rates == nullptr) {
            throw std::runtime_error("GADGET-4 SF CUDA table: " + error);
        }
    }
    detail::batchKernel<<<blocks, kBlockSize>>>(
        operation, deviceInputs.get(), deviceOutputs.get(), count, deviceParameters);
    detail::check(cudaGetLastError());
    std::vector<Output> staged(count);
    detail::check(cudaMemcpy(
        staged.data(), deviceOutputs.get(), count * sizeof(Output), cudaMemcpyDeviceToHost));
    for (size_t index = 0; index < count; ++index) {
        if (staged[index].status != 0) {
            auto const& input = inputs[index];
            double const step = parameters.explicitTimestep >= 0.0 ?
                parameters.explicitTimestep :
                std::ldexp(parameters.baseTimestep, -input.rung);
            std::ostringstream message;
            message << std::scientific << std::setprecision(17)
                    << "GADGET-4 SF CUDA particle_id=" << input.state.id << " batch_index=" << index
                    << " status=" << statusName(staged[index].status)
                    << " rho_g_cm3=" << detail::density(input, parameters)
                    << " u_erg_g=" << detail::energy(input.energy, parameters)
                    << " dt_seconds=" << parameters.scaleFactor * step * parameters.timeSeconds
                    << "; no source state committed";
            throw std::runtime_error(message.str());
        }
    }
    std::copy(staged.begin(), staged.end(), outputs);
}
}    // namespace star_formation_backend
