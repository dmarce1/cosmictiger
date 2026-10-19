#include <atomic>
#include <memory>
#include <mutex>
#include <utility>

#ifdef USE_CUDA
#include <hpx/async_cuda/cuda_event.hpp>
#include <hpx/async_cuda/cuda_exception.hpp>
#include <hpx/async_cuda/cuda_polling_helper.hpp>
#include <hpx/async_cuda/target.hpp>
#include <hpx/runtime_local/shutdown_function.hpp>
#include <hpx/threading_base/threading_base_fwd.hpp>
#endif

#include <cosmictiger/cuda/cuda_stream.hpp>

#ifdef USE_CUDA

namespace detail {

struct CudaPollingState {
    std::mutex mutex;
    std::shared_ptr<hpx::cuda::experimental::enable_user_polling> polling;
    std::atomic<bool> eventsInitialized{false};
    bool shutdownRegistered = false;
    bool shuttingDown = false;
};

CudaPollingState& cudaPollingState() {
    // The pre-shutdown hook owns teardown; keep this small control object valid
    // for late stream destructors and avoid static destruction ordering.
    static auto* state = new CudaPollingState;
    return *state;
}

void stopCudaStreamRuntime() {
    auto& state = cudaPollingState();
    std::shared_ptr<hpx::cuda::experimental::enable_user_polling> polling;
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        state.shuttingDown = true;
        polling = std::move(state.polling);
    }
    // Each pending event wait retains its own reference until completion.
    polling.reset();
}

void initializeCudaStreamRuntime() {
    if (hpx::threads::get_self_ptr() == nullptr) {
        return;
    }
    auto& state = cudaPollingState();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.polling || state.shuttingDown) {
        return;
    }
    int previousDevice;
    hpx::cuda::experimental::check_cuda_error(cudaGetDevice(&previousDevice));
    try {
        // HPX's event pool survives runtime shutdown and cannot survive a CUDA
        // device reset. Warm it only after production CUDA initialization/reset.
        state.eventsInitialized.store(true, std::memory_order_release);
        hpx::cuda::experimental::cuda_event_pool::get_event_pool();
        if (!state.shutdownRegistered) {
            hpx::register_pre_shutdown_function(stopCudaStreamRuntime);
            state.shutdownRegistered = true;
        }
        state.polling = std::make_shared<hpx::cuda::experimental::enable_user_polling>();
    } catch (...) {
        cudaSetDevice(previousDevice);
        throw;
    }
}

bool cudaStreamEventsInitialized() noexcept {
    return cudaPollingState().eventsInitialized.load(std::memory_order_acquire);
}

std::shared_ptr<hpx::cuda::experimental::enable_user_polling> cudaStreamPolling() {
    if (hpx::threads::get_self_ptr() == nullptr) {
        return {};
    }
    auto& state = cudaPollingState();
    std::lock_guard<std::mutex> lock(state.mutex);
    return state.polling;
}

}    // namespace detail

struct CudaStream::Impl {
    int device;
    std::unique_ptr<hpx::cuda::experimental::target> target;
    mutable bool pending = false;
    mutable bool launchCheckPending = false;

    explicit Impl(int selectedDevice)
      : device(selectedDevice)
      , target(std::make_unique<hpx::cuda::experimental::target>(selectedDevice)) {
        target->native_handle().get_stream();
        detail::initializeCudaStreamRuntime();
    }

    ~Impl() noexcept {
        int previousDevice = device;
        bool const restoreDevice = cudaGetDevice(&previousDevice) == cudaSuccess;
        cudaSetDevice(device);
        drain();
        cudaSetDevice(device);
        target.reset();
        if (restoreDevice) {
            cudaSetDevice(previousDevice);
        }
    }

    cudaStream_t handle() const {
        hpx::cuda::experimental::check_cuda_error(cudaSetDevice(device));
        pending = true;
        return target->native_handle().get_stream();
    }

    void drain() noexcept {
        if (!pending) {
            return;
        }
        cudaSetDevice(device);
        // Clear an unchecked failed launch before recording completion of any
        // earlier successful submissions on this stream.
        cudaGetLastError();
        launchCheckPending = false;
        try {
            auto const polling = detail::cudaStreamPolling();
            if (polling) {
                target->get_future_with_event().get();
            } else {
                target->synchronize();
            }
            pending = false;
        } catch (...) {
            // Event registration itself can fail. Preserve the original
            // exception while making a final attempt to drain the stream.
            try {
                cudaSetDevice(device);
                target->synchronize();
                pending = false;
            } catch (...) {
            }
        }
    }
};

CudaStream::CudaStream(int device)
  : implementation_(std::make_unique<Impl>(device)) {}

CudaStream::~CudaStream() noexcept = default;

cudaStream_t CudaStream::nativeHandle() const {
    implementation_->launchCheckPending = true;
    return implementation_->handle();
}

void CudaStream::copyAsync(
    void* destination, void const* source, size_t bytes, cudaMemcpyKind kind) {
    hpx::cuda::experimental::check_cuda_error(
        cudaMemcpyAsync(destination, source, bytes, kind, implementation_->handle()));
}

void CudaStream::checkLaunch() {
    implementation_->launchCheckPending = false;
    hpx::cuda::experimental::check_cuda_error(cudaGetLastError());
}

void CudaStream::prefetchAsync(void const* pointer, size_t bytes, int destinationDevice) {
    hpx::cuda::experimental::check_cuda_error(
        cudaMemPrefetchAsync(pointer, bytes, destinationDevice, implementation_->handle()));
}

void CudaStream::wait() {
    if (!implementation_->pending) {
        return;
    }
    int previousDevice = implementation_->device;
    bool restoreDevice = false;
    try {
        hpx::cuda::experimental::check_cuda_error(cudaGetDevice(&previousDevice));
        restoreDevice = true;
        hpx::cuda::experimental::check_cuda_error(cudaSetDevice(implementation_->device));
        if (implementation_->launchCheckPending) {
            checkLaunch();
        }
        auto const polling = detail::cudaStreamPolling();
        if (polling) {
            implementation_->target->get_future_with_event().get();
        } else {
            implementation_->target->synchronize();
        }
        implementation_->pending = false;
        // An HPX continuation can resume on another worker with another device.
        hpx::cuda::experimental::check_cuda_error(cudaSetDevice(previousDevice));
    } catch (...) {
        implementation_->drain();
        if (restoreDevice) {
            cudaSetDevice(previousDevice);
        }
        throw;
    }
}

#endif
