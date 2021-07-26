/*
 * unordered_map_ts.hpp
 *
 *  Created on: Jul 25, 2021
 *      Author: dmarce1
 */

#ifndef UNORDERED_MAP_TS_HPP_
#define UNORDERED_MAP_TS_HPP_

#include <tigerfmm/defs.hpp>

#include <unordered_map>

template<class T, class V, class HASH>
class unordered_map_ts {
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
	vector<std::unordered_map<T, V, hash_hi>> maps;
	vector<std::shared_ptr<spinlock_type>> mutexes;
public:
	unordered_map_ts() {
		maps.resize(UNORDERED_MAP_SIZE);
		mutexes.resize(UNORDERED_MAP_SIZE, std::make_shared<spinlock_type>());
	}
	void insert(std::pair<T, V>&& element) {
		hash_lo hash;
		const int map_index = hash(element.first);
		std::lock_guard<spinlock_type> lock(*mutexes[map_index]);
		maps[map_index].insert(std::move(element));
	}
	bool exists(const T& key) const {
		hash_lo hash;
		const int map_index = hash(key);
		std::lock_guard<spinlock_type> lock(*mutexes[map_index]);
		return maps[map_index].find(key) != maps[map_index].end();
	}
	V operator[](const T& key) const {
		hash_lo hash;
		const int map_index = hash(key);
		return maps[map_index][key];
	}
	V& operator[](const T& key) {
		hash_lo hash;
		const int map_index = hash(key);
		return maps[map_index][key];
	}
	vector<std::pair<T,V>> to_vector() const {
		vector<std::pair<T,V>> members;
		int size = 0;
		for (int i = 0; i < UNORDERED_SET_SIZE; i++) {
			size += maps[i].size();
		}
		members.reserve(size);
		for (int i = 0; i < UNORDERED_SET_SIZE; i++) {
			for (auto j = maps[i].begin(); j != maps[i].end(); j++) {
				members.push_back(*j);
			}
		}
		return members;
	}

};

#endif /* UNORDERED_MAP_TS_HPP_ */
