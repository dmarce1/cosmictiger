/*
 * float40.hpp
 *
 *  Created on: Jun 12, 2022
 *      Author: dmarce1
 */

#ifndef FLOAT40_HPP_
#define FLOAT40_HPP_

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

class float40 {
	struct {
		unsigned m : 31;
		unsigned s : 1;
	};
	char e;
public:
	inline float40& operator+(const float40& other) {
		unsigned Am = m;
		unsigned As = s;
		signed Ae = e;
		unsigned Bm = m;
		unsigned Bs = s;
		signed Be = e;
		const int dif = Ae - Be;
		if( Ae > Be) {
			Bm >>= dif;
			Ae = Be;
		} else {
			Am >>= -dif;
			Be = Ae;
		}

	}
};

#endif /* FLOAT40_HPP_ */
