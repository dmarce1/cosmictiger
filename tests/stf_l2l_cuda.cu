// Exercise the production cooperative L2L wrapper without running timings.
#define COSMICTIGER_STF_L2L_VALIDATE_ONLY
#include "../benchmarks/stf_l2l_cuda.cu"

#include "cuda_test.hpp"

using StfL2lCuda = detail::CudaTest;

TEST_F(StfL2lCuda, CooperativeTranslationMatchesIndependentReference) {
    detail::forEachOrder(
        [](auto order) { detail::runOrder<order.value>(true, detail::kValidationSamples, 0, 0); });
}
