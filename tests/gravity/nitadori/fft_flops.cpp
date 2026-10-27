/* CosmicTiger - PMMM Fourier accounting regression; BSL-1.0. */

#include <array>
#include <memory>

#include <fftw3.h>
#include <gtest/gtest.h>

#include <cosmictiger/gravity/nitadori/fft_flops.hpp>
#include <tests/flop_counting_real.hpp>

TEST(PmmmFftFlops, ComplexContractionMatchesExecutedArithmetic) {
    using Real = detail::CountingReal;
    for (double weight : {-2.0, -1.0, 1.0, 2.0}) {
        Real real = 0.25;
        Real imaginary = -0.75;
        Real::reset();
        pmmm::detail::accumulateFourierTerm(
            real, imaginary, Real(weight), Real(2.0), Real(3.0), Real(4.0), Real(-5.0));
        EXPECT_EQ(Real::count(), pmmm::detail::kFourierTermFlops);
        EXPECT_DOUBLE_EQ(real.value(), 0.25 + 23.0 * weight);
        EXPECT_DOUBLE_EQ(imaginary.value(), -0.75 + 2.0 * weight);
    }
}

TEST(PmmmFftFlops, CachedPlansCountEachExecutionWithFmaAsTwo) {
    int constexpr kSize = 8;
    std::array<float, kSize * kSize*(kSize + 2)> real{};
    std::array<fftwf_complex, kSize * kSize*(kSize / 2 + 1)> spectrum{};
    for (bool inverse : {false, true}) {
        auto* plan = inverse ?
            fftwf_plan_dft_c2r_3d(
                kSize, kSize, kSize, spectrum.data(), real.data(), FFTW_ESTIMATE) :
            fftwf_plan_dft_r2c_3d(kSize, kSize, kSize, real.data(), spectrum.data(), FFTW_ESTIMATE);
        ASSERT_NE(plan, nullptr);
        std::unique_ptr<fftwf_plan_s, decltype(&fftwf_destroy_plan)> owner(
            plan, fftwf_destroy_plan);
        double additions = 0.0;
        double multiplications = 0.0;
        double fusedMultiplyAdds = 0.0;
        fftwf_flops(plan, &additions, &multiplications, &fusedMultiplyAdds);
        auto const count = pmmm::detail::fftPlanFlops(plan);
        EXPECT_GT(count, 0u);
        EXPECT_EQ(double(count), additions + multiplications + 2 * fusedMultiplyAdds);
        fftwf_execute(plan);
        EXPECT_EQ(pmmm::detail::fftPlanFlops(plan), count);
    }
}
