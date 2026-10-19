/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2021  Dominic C. Marcello

 This program is free software; you can redistribute it and/or
 modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation; either version 2
 of the License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
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
