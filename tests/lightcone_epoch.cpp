#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#include <cosmictiger/lightcone_entry.hpp>

void cosmictigerAssert(char const* expression, bool condition, char const* file, int line) {
    if (!condition) {
        std::fprintf(stderr, "Assertion failed: %s at %s:%d\n", expression, file, line);
        std::abort();
    }
}

namespace detail {

void requireClose(double actual, double expected, char const* message) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > 2.e-14) {
        throw std::runtime_error(message);
    }
}

}    // namespace detail

int main() {
    try {
        // Conformal observation time is 1.0. Radius 0.4 is emitted at tau=0.6,
        // exactly the scale-factor reference epoch, and must need no correction.
        detail::requireClose(lightconeEmissionScale(0.8, 0.1, 0.6, 1.0, 0.4), 0.8,
            "Emission at the reference epoch must preserve the scale factor");
        detail::requireClose(lightconeEmissionScale(0.8, 0.1, 0.6, 1.0, 0.45), 0.795,
            "A more distant particle must use an earlier emission scale factor");
        detail::requireClose(lightconeEmissionScale(0.8, 0.1, 0.6, 1.0, 0.35), 0.805,
            "A closer particle must use a later emission scale factor");
        detail::requireClose(lightconeEmissionScale(0.79, 0.1, 0.5, 1.0, 0.4), 0.8,
            "Equivalent reference epochs must give the same linear emission scale");
        detail::requireClose(lightconeEmissionScale(0.8, 0.1, 0.6, 1.2, 0.6), 0.8,
            "Equal observation-minus-radius values must give equal emission epochs");
        detail::requireClose(lightconeEmissionScale(0.8, 0.0, 0.6, 1.0, 0.9), 0.8,
            "A static scale factor must be independent of distance");
        double const storedVelocity = 0.16;
        double const scale = lightconeEmissionScale(0.8, 0.1, 0.6, 1.0, 0.4);
        detail::requireClose(storedVelocity / scale, 0.2,
            "Peculiar velocity conversion must use the emission scale factor");
        std::puts("Lightcone emission epoch regressions passed");
    } catch (std::exception const& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    return 0;
}
