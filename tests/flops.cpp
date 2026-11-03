/* CosmicTiger - FLOP accounting regression; BSL-1.0. */

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/core/hpx.hpp>

namespace detail {

template <class Count>
concept HasImplicitPrecisionSubmission = requires(Count count) { addCpuFlops(count); };

static_assert(!HasImplicitPrecisionSubmission<FlopCount>);
static_assert(HasImplicitPrecisionSubmission<FlopCounts>);

}    // namespace detail

Vector<hpx::id_type> const& hpxChildren() {
    static auto const children = []() {
        Vector<hpx::id_type> result;
        auto const localities = hpx::find_all_localities();
        auto const rank = hpx::get_locality_id();
        for (size_t index = 2 * rank + 1; index < localities.size() && index <= 2 * rank + 2;
            ++index) {
            result.push_back(localities[index]);
        }
        return result;
    }();
    return children;
}

void contributeFlops() {
    // A single contribution already exceeds both int32 and exact double range.
    addCpuFlops((FlopCount(1) << 54) + 7, FlopPrecision::Single);
    addCpuFlops(FlopCounts{(FlopCount(1) << 53) + 5, (FlopCount(1) << 55) + 11, 0});
}
HPX_PLAIN_ACTION(contributeFlops, ContributeFlopsAction);

TEST(Flops, WideCounterAndDisabledCounter) {
    FlopCounter<> counter;
    EXPECT_EQ(FlopCount(counter), 0);
    (counter = FlopCount(1) << 40) += 23;
    counter -= 5;
    EXPECT_EQ(FlopCount(counter), (FlopCount(1) << 40) + 18);
    FlopCounter<FlopCount, false> disabled(99);
    (disabled = 123) += 456;
    disabled -= 7;
    EXPECT_EQ(FlopCount(disabled), 0);
}

TEST(Flops, AggregateCountersRespectDisabledInstrumentation) {
    FlopCounter<FlopCounts> counter(FlopCounts{11, 13, 17});
    counter += FlopCounts{19, 23, 29};
    FlopCounts const counts = counter;
    EXPECT_EQ(counts.singlePrecision, 30);
    EXPECT_EQ(counts.doublePrecision, 36);
    EXPECT_EQ(counts.unclassified, 46);
    EXPECT_EQ(counts.total(), 112);

    FlopCounter<FlopCounts, false> disabled(FlopCounts{11, 13, 17});
    disabled = FlopCounts{19, 23, 29};
    disabled += FlopCounts{31, 37, 41};
    FlopCounts const disabledCounts = disabled;
    EXPECT_EQ(disabledCounts.singlePrecision, 0);
    EXPECT_EQ(disabledCounts.doublePrecision, 0);
    EXPECT_EQ(disabledCounts.unclassified, 0);
    EXPECT_EQ(disabledCounts.total(), 0);
}

TEST(Flops, FloatingConversionsCountAtHigherPrecision) {
    constexpr auto kWidened = floatingPointConversionFlops<float, double>();
    constexpr auto kNarrowed = floatingPointConversionFlops<double, float>();
    static_assert(kWidened.singlePrecision == 0 && kWidened.doublePrecision == 1);
    static_assert(kNarrowed.singlePrecision == 0 && kNarrowed.doublePrecision == 1);
    static_assert(floatingPointConversionFlops<float, float>().total() == 0);
    static_assert(floatingPointConversionFlops<double const, double&>().total() == 0);
    static_assert(floatingPointConversionFlops<float, int>().total() == 0);
    static_assert(floatingPointConversionFlops<int, float>().total() == 0);
    static_assert(floatingPointConversionFlops<int, double>().doublePrecision == 1);
    static_assert(floatingPointConversionFlops<double, int>().doublePrecision == 1);
    FlopCount const count = (FlopCount(1) << 54) + 7;
    auto totals = floatingPointConversionFlops<float, double>(count);
    totals += floatingPointConversionFlops<double, float>(count);
    EXPECT_EQ(totals.singlePrecision, 0);
    EXPECT_EQ(totals.doublePrecision, 2 * count);
    EXPECT_EQ(totals.unclassified, 0);
    EXPECT_EQ(totals.total(), 2 * count);
}

TEST(Flops, ConcurrentAndDistributedAggregation) {
    auto const localities = hpx::find_all_localities();
    for (int pass = 0; pass < 2; ++pass) {
        resetFlops();
        ASSERT_EQ(readFlops(), 0);
        std::vector<std::thread> workers;
        for (int thread = 0; thread < 4; ++thread) {
            workers.emplace_back([]() {
                for (int i = 0; i < 10000; ++i) {
                    addCpuFlops(1234567, FlopPrecision::Single);
                    addCpuFlops(2468, FlopPrecision::Single);
                    addCpuFlops(913, FlopPrecision::Double);
                }
            });
        }
        for (auto& worker : workers) {
            worker.join();
        }
        for (auto const& locality : localities) {
            hpx::async<ContributeFlopsAction>(locality).get();
        }
        FlopCounts const expected{FlopCount(4) * 10000 * (1234567 + 2468) +
                localities.size() * ((FlopCount(1) << 54) + (FlopCount(1) << 53) + 12),
            FlopCount(4) * 10000 * 913 + localities.size() * ((FlopCount(1) << 55) + 11), 0};
        auto const actual = readFlopCounts();
        EXPECT_EQ(actual.singlePrecision, expected.singlePrecision);
        EXPECT_EQ(actual.doublePrecision, expected.doublePrecision);
        EXPECT_EQ(actual.unclassified, 0);
        EXPECT_EQ(readFlops(), expected.total());
        auto const rates = flopRatesPerSecond();
        EXPECT_NEAR(rates.singlePrecision / rates.doublePrecision,
            double(expected.singlePrecision) / double(expected.doublePrecision), 1.e-15);
        EXPECT_EQ(rates.unclassified, 0.0);
        double const first = flopsPerSecond();
        double const second = flopsPerSecond();
        EXPECT_TRUE(std::isfinite(first));
        EXPECT_GT(second, 0.0);
        EXPECT_LE(second, first);
    }
}

TEST(Flops, ExplicitUnclassifiedDiagnosticIsPreserved) {
    resetFlops();
    addCpuFlops(17, FlopPrecision::Unclassified);
    addCpuFlops(FlopCounts{2, 3, 19});
    auto const counts = readFlopCounts();
    EXPECT_EQ(counts.singlePrecision, 2);
    EXPECT_EQ(counts.doublePrecision, 3);
    EXPECT_EQ(counts.unclassified, 36);
    EXPECT_EQ(counts.total(), 41);
}

TEST(Flops, EmptyRate) {
    resetFlops();
    auto const counts = readFlopCounts();
    EXPECT_EQ(counts.singlePrecision, 0);
    EXPECT_EQ(counts.doublePrecision, 0);
    EXPECT_EQ(counts.unclassified, 0);
    auto const rates = flopRatesPerSecond();
    EXPECT_EQ(rates.singlePrecision, 0.0);
    EXPECT_EQ(rates.doublePrecision, 0.0);
    EXPECT_EQ(rates.unclassified, 0.0);
    EXPECT_EQ(flopsPerSecond(), 0.0);
}

int hpx_main(int, char*[]) {
    int const status = RUN_ALL_TESTS();
    hpx::finalize();
    return status;
}

int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    if (GTEST_FLAG_GET(list_tests)) {
        return RUN_ALL_TESTS();
    }
    return hpx::init(argc, argv);
}
