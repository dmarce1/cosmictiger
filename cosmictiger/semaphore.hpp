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


#ifndef SEMAPHORE_HPP_
#define SEMAPHORE_HPP_


#include <cosmictiger/hpx.hpp>

class semaphore {
	std::atomic<int> available;
public:
	semaphore(int cnt) {
		available = cnt;
	}
	semaphore() {
		available = 0;
	}
	void wait() {
		while( --available < 0 ) {
			available++;
			hpx_yield();
		}
	}
	void signal() {
		available++;
	}
};


#endif /* SEMAPHORE_HPP_ */
