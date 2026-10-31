#include <chrono>
#include <cstdio>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

#include <gtest/gtest.h>

#include <cosmictiger/core/section_timing.hpp>

namespace {
using Clock = section_timing::detail::Clock;
using Section = section_timing::Section;
using ScopedSection = section_timing::ScopedSection;

class SectionTiming : public ::testing::Test {
protected:
    void SetUp() override {
        auto& state = section_timing::detail::registry();
        std::lock_guard<std::mutex> lock(state.mutex);
        state.enabled.store(false);
        state.started = false;
        state.finished = false;
        state.reportPath.clear();
        state.entries = {};
    }

    static section_timing::detail::Entry entry(Section section) {
        auto& state = section_timing::detail::registry();
        std::lock_guard<std::mutex> lock(state.mutex);
        return state.entries[static_cast<size_t>(section)];
    }

    static Clock::duration sourceTotal() {
        return entry(Section::hydrodynamics).elapsed + entry(Section::starFormation).elapsed +
            entry(Section::chemistry).elapsed;
    }

    static void expectPartition(Clock::time_point before, Clock::time_point innerBegin,
        Clock::time_point innerEnd, Clock::time_point after) {
        // The three exclusive pieces must sum to the outer scope's exact wall
        // interval, bracketed by timestamps immediately inside/outside it. This
        // uses no scheduling-dependent upper bounds on individual phase times.
        EXPECT_GE(sourceTotal(), innerEnd - innerBegin);
        EXPECT_LE(sourceTotal(), after - before);
    }
};

TEST_F(SectionTiming, NestedSourcesPartitionHydrodynamics) {
    ASSERT_TRUE(section_timing::start(""));
    Clock::time_point innerBegin;
    Clock::time_point innerEnd;
    auto const before = Clock::now();
    {
        ScopedSection hydro(Section::hydrodynamics);
        innerBegin = Clock::now();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        {
            ScopedSection stars(Section::starFormation, &hydro);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            {
                ScopedSection chemistry(Section::chemistry, &stars);
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        {
            ScopedSection chemistry(Section::chemistry, &hydro);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        innerEnd = Clock::now();
    }
    expectPartition(before, innerBegin, innerEnd, Clock::now());
    EXPECT_EQ(entry(Section::hydrodynamics).calls, 1);
    EXPECT_EQ(entry(Section::starFormation).calls, 1);
    EXPECT_EQ(entry(Section::chemistry).calls, 2);
}

TEST_F(SectionTiming, ExceptionResumesEveryParent) {
    ASSERT_TRUE(section_timing::start(""));
    Clock::time_point innerBegin;
    Clock::time_point innerEnd;
    auto const before = Clock::now();
    {
        ScopedSection hydro(Section::hydrodynamics);
        innerBegin = Clock::now();
        try {
            ScopedSection stars(Section::starFormation, &hydro);
            ScopedSection chemistry(Section::chemistry, &stars);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            throw std::runtime_error("source failure");
        } catch (std::runtime_error const&) {
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        innerEnd = Clock::now();
    }
    expectPartition(before, innerBegin, innerEnd, Clock::now());
    EXPECT_EQ(entry(Section::hydrodynamics).calls, 1);
    EXPECT_EQ(entry(Section::starFormation).calls, 1);
    EXPECT_EQ(entry(Section::chemistry).calls, 1);
}

TEST_F(SectionTiming, LogicalTaskCanResumeOnAnotherThread) {
    ASSERT_TRUE(section_timing::start(""));
    auto const before = Clock::now();
    auto hydro = std::make_unique<ScopedSection>(Section::hydrodynamics);
    auto const innerBegin = Clock::now();
    Clock::time_point innerEnd;
    std::thread continuation([hydro = std::move(hydro), &innerEnd]() mutable {
        {
            ScopedSection stars(Section::starFormation, hydro.get());
            ScopedSection chemistry(Section::chemistry, &stars);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        innerEnd = Clock::now();
        hydro.reset();
    });
    continuation.join();
    expectPartition(before, innerBegin, innerEnd, Clock::now());
    EXPECT_EQ(entry(Section::hydrodynamics).calls, 1);
}

TEST_F(SectionTiming, UnrelatedScopesStayInclusive) {
    ASSERT_TRUE(section_timing::start(""));
    {
        ScopedSection initialization(Section::initialization);
        ScopedSection transfer(Section::transferFunctions);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_GE(entry(Section::initialization).elapsed, entry(Section::transferFunctions).elapsed);
    EXPECT_EQ(entry(Section::initialization).calls, 1);
    EXPECT_EQ(entry(Section::transferFunctions).calls, 1);
}

TEST_F(SectionTiming, DisabledLocalityDoesNotRecordSources) {
    {
        ScopedSection hydro(Section::hydrodynamics);
        ScopedSection stars(Section::starFormation, &hydro);
        ScopedSection chemistry(Section::chemistry, &stars);
    }
    EXPECT_EQ(sourceTotal(), Clock::duration::zero());
    EXPECT_EQ(entry(Section::hydrodynamics).calls, 0);
    EXPECT_EQ(entry(Section::starFormation).calls, 0);
    EXPECT_EQ(entry(Section::chemistry).calls, 0);
}

TEST_F(SectionTiming, ReportExplainsExclusiveSourcesAndCoordinatorBoundary) {
    ASSERT_TRUE(section_timing::start(""));
    {
        ScopedSection hydro(Section::hydrodynamics);
        ScopedSection stars(Section::starFormation, &hydro);
        ScopedSection chemistry(Section::chemistry, &stars);
    }
    auto closeFile = [](std::FILE* stream) { std::fclose(stream); };
    auto file = std::unique_ptr<std::FILE, decltype(closeFile)>(std::tmpfile(), closeFile);
    ASSERT_NE(file, nullptr);
    section_timing::detail::writeReport(file.get(), section_timing::detail::registry());
    std::rewind(file.get());
    std::string report;
    char line[512];
    while (std::fgets(line, sizeof(line), file.get()) != nullptr) {
        report += line;
    }
    EXPECT_NE(report.find("Hydrodynamics, Chemistry, and Star formation exclude one another"),
        std::string::npos);
    EXPECT_NE(report.find("remote waits remain Star formation"), std::string::npos);
    EXPECT_NE(report.find("Other sections are inclusive"), std::string::npos);
    for (auto const* label : {"Hydrodynamics", "Chemistry", "Star formation"}) {
        EXPECT_NE(report.find(std::string("\n") + label), std::string::npos);
    }
}
}    // namespace
