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

#ifndef RA22ND_HPP_
#define RA22ND_HPP_

class rnd_gen {
	//L’Ecuyer, Blouin and Coutre
	static constexpr unsigned a1 = 107374182;
	static constexpr unsigned a5 = 104480;
	static constexpr unsigned m = (1U << 31) - 1;
	unsigned x1;
	unsigned x2;
	unsigned x3;
	unsigned x4;
	unsigned x5;
public:
	void reseed(int seed) {
		const unsigned a0 = 48271;
		x1 = seed;
		x2 = (x1 * a0) % m;
		x3 = (x2 * a0) % m;
		x4 = (x3 * a0) % m;
		x5 = (x4 * a0) % m;
	}
	rnd_gen(int seed) {
		reseed(seed);
	}
	int rnd_number() {
		const long long x0 = (a1 * x1 + a5 * x5) % m;
		x5 = x4;
		x4 = x3;
		x3 = x2;
		x2 = x1;
		x1 = x0;
		return (int) x0;
	}
	int rnd_max() const {
		return m - 1;
	}
	float rnd_one() {
		return ((float) rnd_number() + 0.5f) / (float) rnd_max();
	}
};

#endif /* RAND_HPP_ */
