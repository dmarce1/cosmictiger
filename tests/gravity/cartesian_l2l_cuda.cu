// Exercise the production cooperative L2L wrapper without running timings.
#define COSMICTIGER_CARTESIAN_L2L_VALIDATE_ONLY
#include <tests/../benchmarks/gravity/cartesian_l2l_cuda.cu>

#include <tests/cuda_test.hpp>

using CartesianL2lCuda = detail::CudaTest;

TEST_F(CartesianL2lCuda, CooperativeTranslationMatchesIndependentReference) {
    detail::forEachOrder(
        [](auto order) { detail::runOrder<order.value>(true, detail::kValidationSamples, 0, 0); });
}
