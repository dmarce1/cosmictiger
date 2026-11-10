#pragma once

#include <type_traits>

// First-error transport for kernels that must return normally. Put this state
// in shared host/device storage, clear it before launch, and call throwIfFailed
// only after the producing stream has completed. Payloads are read by the host
// after synchronization; pending() alone is safe to poll from other GPU threads.
// Cancellation stays explicit at block-uniform boundaries: this class never
// skips barriers, unwinds device frames, or rolls back partially written output.
template <class Error>
class CudaKernelError {
    static_assert(std::is_trivially_copyable_v<Error>);
    int reported_{};
    Error error_{};

public:
    /// Reset the recorded-error flag and payload before launching a kernel.
    void clear() {
        reported_ = 0;
        error_ = {};
    }

#ifdef __CUDACC__
    // The winning thread owns the diagnostic payload and optional device print.
    __device__ bool record(Error const& error) {
        if (atomicCAS(&reported_, 0, 1) != 0) {
            return false;
        }
        error_ = error;
        return true;
    }

    __host__ __device__
#endif
        /// Test whether an error has been recorded, using an atomic read on the
        /// device.
        bool pending() const {
#ifdef __CUDA_ARCH__
        return atomicAdd(const_cast<int*>(&reported_), 0) != 0;
#else
        return reported_ != 0;
#endif
    }

    /// Invoke the supplied host error handler with the recorded payload when a kernel
    /// reported failure.
    template <class Handler>
    void throwIfFailed(Handler&& handler) const {
        if (reported_ != 0) {
            static_cast<Handler&&>(handler)(error_);
        }
    }
};
