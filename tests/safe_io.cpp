#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include <cosmictiger/core/safe_io.hpp>

TEST(SafeIo, FormattedErrorCarriesMessageAndLocation) {
    int const line = __LINE__ + 2;
    try {
        THROW_ERROR("%s %i\n", "Tree arena full", 42);
        FAIL() << "THROW_ERROR returned";
    } catch (std::runtime_error const& error) {
        EXPECT_EQ(std::string(error.what()),
            "Tree arena full 42\nError in " + std::string(__FILE__) + " on line " +
                std::to_string(line));
    }
}

TEST(SafeIo, PlainErrorPreservesPercentCharacters) {
    try {
        throwError("caller.cpp", 17, "100% full (%s)");
        FAIL() << "throwError returned";
    } catch (std::runtime_error const& error) {
        EXPECT_STREQ(error.what(), "100% full (%s)\nError in caller.cpp on line 17");
    }
}

TEST(SafeIo, LongFormattedErrorIsNotTruncated) {
    std::string const payload(16384, 'x');
    try {
        throwError("caller.cpp", 23, "%s: %zu", payload.c_str(), payload.size());
        FAIL() << "throwError returned";
    } catch (std::runtime_error const& error) {
        EXPECT_EQ(std::string(error.what()), payload + ": 16384\nError in caller.cpp on line 23");
    }
}

TEST(SafeIo, ErrorUnwindsStack) {
    auto owner = std::make_shared<int>(42);
    EXPECT_THROW(
        {
            auto localOwner = owner;
            THROW_ERROR("Tree arena full");
        },
        std::runtime_error);
    EXPECT_EQ(owner.use_count(), 1);
}

TEST(SafeIo, ShortReadThrowsWithCallerLocation) {
    std::unique_ptr<FILE, int (*)(FILE*)> file(std::tmpfile(), &std::fclose);
    ASSERT_NE(file, nullptr);
    int value = 0;
    try {
        safeFread(&value, sizeof(value), 1, file.get(), 31, "reader.cpp");
        FAIL() << "Short read returned";
    } catch (std::runtime_error const& error) {
        std::string const message = error.what();
        EXPECT_NE(message.find("only 0 elements read"), std::string::npos);
        EXPECT_NE(message.find("Error in reader.cpp on line 31"), std::string::npos);
    }
}

TEST(SafeIo, FailedWriteThrowsWithCallerLocation) {
    std::unique_ptr<FILE, int (*)(FILE*)> file(std::fopen("/dev/null", "r"), &std::fclose);
    ASSERT_NE(file, nullptr);
    int const value = 42;
    try {
        safeFwrite(&value, sizeof(value), 1, file.get(), 37, "writer.cpp");
        FAIL() << "Failed write returned";
    } catch (std::runtime_error const& error) {
        std::string const message = error.what();
        EXPECT_NE(message.find("only 0 elements written"), std::string::npos);
        EXPECT_NE(message.find("Error in writer.cpp on line 37"), std::string::npos);
    }
}
