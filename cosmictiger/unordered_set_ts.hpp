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


#ifndef UNORDERED_SET_TS_HPP_
#define UNORDERED_SET_TS_HPP_

#include <cosmictiger/containers.hpp>
#include <cosmictiger/defs.hpp>
#include <cosmictiger/hpx.hpp>

#include <unordered_set>

template<class T, class HASH>
class unordered_set_ts {
	struct hash_lo {
		size_t operator()(const T& member) const {
			HASH hash;
			return hash(member) % UNORDERED_SET_SIZE;
		}
	};
	struct hash_hi {
		size_t operator()(const T& member) const {
			HASH hash;
			return hash(member) / UNORDERED_SET_SIZE;
		}
	};
	vector<std::unordered_set<T, hash_hi>> sets;
	vector<std::shared_ptr<spinlock_type>> mutexes;
public:
	unordered_set_ts() {
		sets.resize(UNORDERED_SET_SIZE);
		mutexes.resize(UNORDERED_SET_SIZE, std::make_shared<spinlock_type>());
	}
	void insert(const T& member) {
		PRINT( "Inserting\n");
		hash_lo hashlo;
		const int set_index = hashlo(member);
		std::lock_guard<spinlock_type> lock(*mutexes[set_index]);
		sets[set_index].insert(member);
	}
	vector<T> to_vector() const {
		vector<T> members;
		int size = 0;
		for (int i = 0; i < UNORDERED_SET_SIZE; i++) {
			size += sets[i].size();
		}
		members.reserve(size);
		for (int i = 0; i < UNORDERED_SET_SIZE; i++) {
			for (auto j = sets[i].begin(); j != sets[i].end(); j++) {
				members.push_back(*j);
			}
		}
		return members;
	}
};

#endif /* UNORDERED_SET_TS_HPP_ */
