#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <streambuf>
#include <string>

#include <cosmictiger/recfast.hpp>

namespace detail {

void require(bool condition, char const* label) {
    if (!condition) {
        throw std::runtime_error(label);
    }
}

void close(double actual, double expected, double tolerance, char const* label) {
    auto const bits = std::bit_cast<std::uint64_t>(actual);
    constexpr std::uint64_t kExponentMask = UINT64_C(0x7ff0000000000000);
    if ((bits & kExponentMask) == kExponentMask ||
        std::abs(actual - expected) > tolerance * std::max(std::abs(expected), 1.0e-12)) {
        throw std::runtime_error(
            std::string(label) + ": " + std::to_string(actual) + " != " + std::to_string(expected));
    }
}

template <class Function>
void expectFailure(Function function, char const* label) {
    try {
        function();
    } catch (std::exception const&) {
        return;
    }
    throw std::runtime_error(std::string("Expected rejection: ") + label);
}

std::string makeTable(std::function<double(int)> fraction, int rows = recfast::kSampleCount) {
    std::ostringstream output;
    output << "  z          x_e   \n" << std::setprecision(17);
    for (int row = 0; row < rows; ++row) {
        int const index = recfast::kSampleCount - 1 - row;
        output << index * recfast::kRedshiftStep << "  " << fraction(index) << '\n';
    }
    return output.str();
}

std::string replaceLine(std::string text, int lineNumber, std::string const& replacement) {
    size_t begin = 0;
    for (int line = 0; line < lineNumber; ++line) {
        begin = text.find('\n', begin) + 1;
    }
    size_t const end = text.find('\n', begin);
    text.replace(begin, end - begin, replacement);
    return text;
}

recfast::IonizationHistory readTable(std::string const& text) {
    std::istringstream input(text);
    return recfast::IonizationHistory(input);
}

// Simulate an I/O error after a valid prefix, including after the final row.
// This distinguishes read failures from an ordinary, permitted end of file.
class FailingBuffer : public std::streambuf {
    std::string text_;
    size_t position_ = 0;
    size_t failurePosition_;
    char current_ = '\0';

protected:
    int_type underflow() override {
        if (position_ == failurePosition_) {
            throw std::runtime_error("simulated read failure");
        }
        if (position_ == text_.size()) {
            return traits_type::eof();
        }
        current_ = text_[position_++];
        setg(&current_, &current_, &current_ + 1);
        return traits_type::to_int_type(current_);
    }

public:
    FailingBuffer(std::string const& text, size_t failurePosition)
      : text_(text)
      , failurePosition_(failurePosition) {}
};

void testInterpolation() {
    auto const reciprocal = [](int index) { return 1.0 / (index + 1); };
    auto const history = readTable(makeTable(reciprocal));
    close(history(1.0), reciprocal(0), 0, "low-redshift endpoint");
    close(history(1.0 / (1.0 + recfast::kMaximumRedshift)), reciprocal(recfast::kSampleCount - 1),
        0, "high-redshift endpoint");
    close(history(1.0 / 6.0), 0.5 * (reciprocal(0) + reciprocal(1)), 2e-14,
        "first interval interpolation");
    close(history(1.0 / 9986.0), 0.5 * (reciprocal(998) + reciprocal(999)), 2e-14,
        "last interval interpolation");
    close(history(1.1), reciprocal(0), 0, "future endpoint extension");
    close(
        history(std::numeric_limits<double>::max()), reciprocal(0), 0, "large finite scale factor");
    close(history(std::numeric_limits<double>::min()), reciprocal(999), 0,
        "early endpoint extension");
    close(history(std::numeric_limits<double>::denorm_min()), reciprocal(999), 0,
        "subnormal scale factor");
    for (double const boundary : {1.0, 1.0 / 9991.0}) {
        for (double const direction : {0.0, 2.0}) {
            double const scaleFactor = std::nextafter(boundary, direction);
            double const expected = boundary == 1.0 ? reciprocal(0) : reciprocal(999);
            close(history(scaleFactor), expected, 2e-12, "endpoint continuity");
        }
    }
    for (int index = 0; index < recfast::kSampleCount; ++index) {
        close(history(1.0 / (1.0 + index * recfast::kRedshiftStep)), reciprocal(index), 2e-12,
            "table knot");
    }
    // A four-point centered cubic must reproduce a quadratic in the interior.
    auto const quadratic = [](double coordinate) {
        return 0.0001 + coordinate * coordinate * 1e-6;
    };
    auto const curved = readTable(makeTable([&](int index) { return quadratic(index); }));
    for (double const coordinate : {1.5, 9.25, 57.8, 501.2, 997.75}) {
        close(curved(1.0 / (1.0 + coordinate * recfast::kRedshiftStep)), quadratic(coordinate),
            2e-12, "interior quadratic reproduction");
    }
    auto const neutral = readTable(makeTable([](int) { return 0.0; }));
    close(neutral(1.0 / 101.5), 0.0, 0, "zero electron fraction is allowed");
    for (double const scaleFactor : {0.0, -0.0, -1.0, -std::numeric_limits<double>::denorm_min(),
             std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(),
             std::numeric_limits<double>::quiet_NaN()}) {
        expectFailure([&] { history(scaleFactor); }, "invalid scale factor");
    }
}

void testMalformedInput() {
    auto const fraction = [](int) { return 0.1; };
    std::string const valid = makeTable(fraction);
    for (std::string const& header : {std::string(), std::string("z"), std::string("redshift xe"),
             std::string("z x_e extra"), std::string(1000, 'x')}) {
        expectFailure([&] { readTable(replaceLine(valid, 0, header)); }, "invalid header");
    }
    expectFailure([&] { readTable(""); }, "empty input");
    expectFailure(
        [&] { readTable(makeTable(fraction, recfast::kSampleCount - 1)); }, "truncated table");
    expectFailure([&] { readTable("z x_e\n9990"); }, "partial table and row");
    for (std::string const& row :
        {std::string(), std::string("9990"), std::string("9990 bad"), std::string("9990 0.1 extra"),
            std::string("9990 -0.1"), std::string("9990 nan"), std::string("9990 inf"),
            std::string("9990 1e309"), std::string("nan 0.1"), std::string("inf 0.1"),
            std::string("1e309 0.1"), std::string("9991 0.1"), std::string("0 0.1")}) {
        expectFailure([&] { readTable(replaceLine(valid, 1, row)); }, "invalid data row");
    }
    expectFailure([&] { readTable(replaceLine(valid, 2, "9990 0.1")); }, "duplicate redshift");
    expectFailure([&] { readTable(valid + "0 0.1\n"); }, "extra row");
    expectFailure([&] { readTable(valid + "trailing garbage"); }, "trailing nonnumeric data");
    close(readTable(valid + " \n\t\r\n")(1.0), 0.1, 0, "trailing whitespace");
    expectFailure([] { recfast::loadIonizationHistory(""); }, "missing output file");
    for (size_t const failurePosition : {size_t(0), valid.size() / 2, valid.size()}) {
        FailingBuffer buffer(valid, failurePosition);
        std::istream input(&buffer);
        expectFailure([&] { recfast::IonizationHistory history(input); }, "stream read failure");
    }
}

void testFile(std::string const& filename) {
    auto const history = recfast::loadIonizationHistory(filename);
    std::ifstream expected(filename);
    std::string header;
    std::getline(expected, header);
    double redshift;
    double electronFraction;
    int count = 0;
    while (expected >> redshift >> electronFraction) {
        close(
            history(1.0 / (1.0 + redshift)), electronFraction, 2e-12, "actual RECFAST output knot");
        if (count == 0) {
            close(history(std::numeric_limits<double>::min()), electronFraction, 0,
                "actual RECFAST early endpoint");
        }
        if (redshift == 0) {
            close(history(1.0), electronFraction, 0, "actual RECFAST z=0 endpoint");
            close(history(1.1), electronFraction, 0, "actual RECFAST future endpoint");
        }
        ++count;
    }
    require(count == recfast::kSampleCount, "actual RECFAST row count");
}

}    // namespace detail

int main(int argc, char** argv) {
    try {
        if (argc > 2) {
            throw std::runtime_error("Usage: recfast_test [RECFAST output filename]");
        }
        detail::testInterpolation();
        detail::testMalformedInput();
        if (argc == 2) {
            detail::testFile(argv[1]);
        }
        std::puts("RECFAST parsing and interpolation tests passed");
    } catch (std::exception const& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    return 0;
}
