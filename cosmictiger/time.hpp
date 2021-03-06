#pragma once

#include <cosmictiger/defs.hpp>

#include <limits>

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
