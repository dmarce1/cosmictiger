/* CosmicTiger - FLOP accounting regression; BSL-1.0. */

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/core/hpx.hpp>

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
    addCpuFlops((FlopCount(1) << 54) + 7);
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

TEST(Flops, ConcurrentAndDistributedAggregation) {
    auto const localities = hpx::find_all_localities();
    for (int pass = 0; pass < 2; ++pass) {
        resetFlops();
        ASSERT_EQ(readFlops(), 0);
        std::vector<std::thread> workers;
        for (int thread = 0; thread < 4; ++thread) {
            workers.emplace_back([]() {
                for (int i = 0; i < 10000; ++i) {
                    addCpuFlops(1234567);
                }
            });
        }
        for (auto& worker : workers) {
            worker.join();
        }
        for (auto const& locality : localities) {
            hpx::async<ContributeFlopsAction>(locality).get();
        }
        FlopCount const expected =
            FlopCount(4) * 10000 * 1234567 + localities.size() * ((FlopCount(1) << 54) + 7);
        EXPECT_EQ(readFlops(), expected);
        double const first = flopsPerSecond();
        double const second = flopsPerSecond();
        EXPECT_TRUE(std::isfinite(first));
        EXPECT_GT(second, 0.0);
        EXPECT_LE(second, first);
    }
}

TEST(Flops, EmptyRate) {
    resetFlops();
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
