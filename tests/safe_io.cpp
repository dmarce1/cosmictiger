#include <cstdio>
#include <ios>
#include <memory>
#include <string>

#include <gtest/gtest.h>

#include <cosmictiger/core/safe_io.hpp>

TEST(SafeIo, ShortReadThrowsWithCallerLocation) {
    std::unique_ptr<FILE, int (*)(FILE*)> file(std::tmpfile(), &std::fclose);
    ASSERT_NE(file, nullptr);
    int value = 0;
    try {
        safeFread(&value, sizeof(value), 1, file.get(), 31, "reader.cpp");
        FAIL() << "Short read returned";
    } catch (std::ios_base::failure const& error) {
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
    } catch (std::ios_base::failure const& error) {
        std::string const message = error.what();
        EXPECT_NE(message.find("only 0 elements written"), std::string::npos);
        EXPECT_NE(message.find("Error in writer.cpp on line 37"), std::string::npos);
    }
}

TEST(SafeIo, ShortReadPreservesLongFilenameAndUnwindsStack) {
    std::unique_ptr<FILE, int (*)(FILE*)> file(std::tmpfile(), &std::fclose);
    ASSERT_NE(file, nullptr);
    std::string const filename = std::string(16384, 'x') + "100%full";
    auto owner = std::make_shared<int>(42);
    int value = 0;
    try {
        auto localOwner = owner;
        safeFread(&value, sizeof(value), 1, file.get(), 41, filename.c_str());
        FAIL() << "Short read returned";
    } catch (std::ios_base::failure const& error) {
        EXPECT_NE(std::string(error.what()).find(filename + " on line 41"), std::string::npos);
    }
    EXPECT_EQ(owner.use_count(), 1);
}

TEST(SafeIo, BinaryRoundTrip) {
    std::unique_ptr<FILE, int (*)(FILE*)> file(std::tmpfile(), &std::fclose);
    ASSERT_NE(file, nullptr);
    int const expected = 42;
    FWRITE(&expected, sizeof(expected), 1, file.get());
    std::rewind(file.get());
    int actual = 0;
    FREAD(&actual, sizeof(actual), 1, file.get());
    EXPECT_EQ(actual, expected);
}
