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

using rung_type = std::int8_t;
using time_type = std::uint64_t;

inline time_type inc(time_type t, rung_type max_rung) {
	t += (time_type(1) << time_type(64 - max_rung));
	return t;
}

inline rung_type min_rung(time_type t) {
	rung_type min_rung = 64;
	while (((t & 1) == 0) && (min_rung != 0)) {
		min_rung--;
		t >>= 1;
	}
	return min_rung;
}


#endif /* TIME_HPP_ */
