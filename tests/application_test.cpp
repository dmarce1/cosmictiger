/* CosmicTiger - GoogleTest bridge for HPX application checks; BSL-1.0. */

#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include <tests/application_test.hpp>

namespace detail {

class ApplicationCheck : public ::testing::Test {
    void (*run_)();

public:
    explicit ApplicationCheck(void (*run)())
      : run_(run) {}

    void TestBody() override {
        run_();
    }
};

}    // namespace detail

void runApplicationTest(char const* name, void (*run)()) {
    // Each invocation has its own configured HPX/particle state. Register only
    // that check so the run.test CLI and its integration runners retain isolation.
    int argc = 1;
    char program[] = "cosmictiger";
    char* argv[] = {program, nullptr};
    ::testing::InitGoogleTest(&argc, argv);
    // An inherited filter must never silently suppress the requested check.
    GTEST_FLAG_SET(filter, "*");
    GTEST_FLAG_SET(list_tests, false);
    GTEST_FLAG_SET(repeat, 1);
    ::testing::RegisterTest("Application", name, nullptr, nullptr, __FILE__, __LINE__,
        [run]() -> detail::ApplicationCheck* { return new detail::ApplicationCheck(run); });
    if (RUN_ALL_TESTS() != 0) {
        throw std::runtime_error(std::string("Application regression failed: ") + name);
    }
}
