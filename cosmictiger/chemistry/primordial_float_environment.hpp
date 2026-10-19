/* CosmicTiger - host floating-point contract; GPL-2.0-or-later. */
#pragma once

#include <bit>
#include <cfenv>
#include <cstdint>
#include <limits>

#if defined(__SSE__)
#include <xmmintrin.h>
#endif

namespace primordial {

// FloatFloat error-free transforms require nearest rounding and gradual
// underflow. Save the caller environment, establish those conditions locally,
// and restore it on every exit. Exceptions are nontrapping inside the scope;
// nonfinite results remain explicit solver failures. Each worker needs a scope.
// Intel SDM Vol1 section10.2.3 documents MXCSR FTZ(bit15)/DAZ(bit6):
// https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html
class StrictFloatEnvironment {
    std::fenv_t savedEnvironment_{};
    bool captured_{};
    bool valid_{};
#if defined(__SSE__)
    unsigned savedControl_{};
#elif defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
    uint64_t savedControl_{};
#endif
public:
    StrictFloatEnvironment() noexcept {
#if defined(__SSE__)
        savedControl_ = _mm_getcsr();
#elif defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
        asm volatile("mrs %0, fpcr" : "=r"(savedControl_));
#endif
        captured_ = std::feholdexcept(&savedEnvironment_) == 0;
        valid_ = captured_ && std::fesetround(FE_TONEAREST) == 0;
        if (!valid_) {
            return;
        }
#if defined(__SSE__)
        _mm_setcsr(_mm_getcsr() & ~((1U << 15) | (1U << 6)));
#elif defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
        uint64_t control;
        asm volatile("mrs %0, fpcr" : "=r"(control));
        control &= ~((uint64_t(1) << 24) | (uint64_t(1) << 19));
        asm volatile("msr fpcr, %0" : : "r"(control));
#endif
        // Also detect unsupported targets/modes instead of silently assuming
        // gradual underflow. Volatile and bit inspection prevent folding.
        float volatile tiny = std::numeric_limits<float>::denorm_min();
        float volatile twice = tiny + tiny;
        valid_ = std::bit_cast<uint32_t>(static_cast<float>(twice)) == 2U;
    }
    ~StrictFloatEnvironment() {
        if (captured_) {
            std::fesetenv(&savedEnvironment_);
        }
#if defined(__SSE__)
        _mm_setcsr(savedControl_);
#elif defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
        asm volatile("msr fpcr, %0" : : "r"(savedControl_));
#endif
    }
    StrictFloatEnvironment(StrictFloatEnvironment const&) = delete;
    StrictFloatEnvironment& operator=(StrictFloatEnvironment const&) = delete;
    bool valid() const noexcept {
        return valid_;
    }
};
}    // namespace primordial
