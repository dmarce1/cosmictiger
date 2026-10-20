/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2021  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#ifndef TIME_HPP_
#define TIME_HPP_

#include <cstdint>

using RungType = std::int8_t;
using TimeType = std::uint64_t;

inline TimeType inc(TimeType t, RungType maxRung) {
    if (maxRung > 0) {
        t += (TimeType(1) << TimeType(64 - maxRung));
    }
    return t;
}

inline RungType minRung(TimeType t) {
    RungType minRung = 64;
    while (((t & 1) == 0) && (minRung != 0)) {
        minRung--;
        t >>= 1;
    }
    return minRung;
}

#endif /* TIME_HPP_ */
