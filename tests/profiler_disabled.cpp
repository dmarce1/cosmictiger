/* CosmicTiger - disabled profiler macro regression; BSL-1.0. */

#include <cosmictiger/core/profiler.hpp>

#if !defined(PROFILE_DISABLE) && !defined(__CUDA_ARCH__)
#error This regression requires the disabled or CUDA-device PROFILE macro.
#endif

namespace detail {

constexpr int profiledConstant(int value) {
    PROFILE();
    return value + 1;
}

consteval int profiledImmediate() {
    PROFILE();
    return 7;
}

template <typename Value>
constexpr Value profiledTemplate(Value value) {
    PROFILE();
    return value + 2;
}

static_assert(profiledConstant(4) == 5);
static_assert(profiledImmediate() == 7);
static_assert(profiledTemplate(4) == 6);

}    // namespace detail

// This target does not link the profiler library: an accidental runtime
// registration would fail to link as well as violating the disabled contract.
int main() {
    int volatile runtimeValue = 4;
    return detail::profiledConstant(runtimeValue) == 5 &&
            detail::profiledTemplate(runtimeValue) == 6 &&
            detail::profiledTemplate(static_cast<long>(runtimeValue)) == 6 ?
        0 :
        1;
}
