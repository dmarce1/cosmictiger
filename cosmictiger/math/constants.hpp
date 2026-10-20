/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2021  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

namespace constants {
constexpr double kC = 2.99792458e10;
constexpr double kG = 6.67259e-8;
constexpr double kH = 6.6260755e-27;
constexpr double kMe = 9.1093897e-28;
constexpr double kMh = 1.6733e-24;
constexpr double kSigma = 5.67051e-5;
constexpr double kH0 = 1e7 / 3.086e24;
constexpr double kEvtoerg = 1.6021772e-12;
constexpr double kB = 1.380658e-16;
constexpr double kHbar = 1.05457266e-27;
constexpr double kTcmb = 2.73;
constexpr double kSigmaT = 6.6524587158e-25;
constexpr double kEvtoK = 11604.5250061657;
constexpr double kSecondsToYears = 1.0 / (365.24 * 24 * 60 * 60);
constexpr double kMpcToCm = 3.085678e24;
constexpr double kSpyr = 365.24 * 24.0 * 60.0 * 60.0;
constexpr double kM0 = 1.989e33;
constexpr double kAvo = 6.0221409e+23;
constexpr double kE = 4.8032068e-10;
}    // namespace constants

namespace cosmic_constants {
constexpr double kH0 = 1e7 / constants::kC;
}
