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

#ifndef CUDA_UNORDERED_MAP_HPP_
#define CUDA_UNORDERED_MAP_HPP_

#include <cosmictiger/device_vector.hpp>
#include <cuda/std/climits>

template<class T>
class cuda_unordered_map {
	device_vector<device_vector<pair<int, T>>> buckets;
	int sz;
	CUDA_EXPORT
	void initialize() {
		sz = 0;
	}
	CUDA_EXPORT
	void rehash() {
		int new_mod = std::max(buckets.size(),8);
		while( size() / new_mod > 2 ) {
			new_mod *= 2;
		}
		if( new_mod > buckets.size()) {
			device_vector<device_vector<pair<int, T>>> new_buckets;
			new_buckets.resize(new_mod);
			for( int i = 0; i < buckets.size(); i++) {
				for( int j = 0; j < buckets[i].size(); j++) {
					new_buckets[buckets[i][j].first % new_mod].push_back(std::move(buckets[i][j]));
				}
			}
			buckets.swap(new_buckets);
		}
	}
public:
	class iterator {
		int bucket;
		int index;
		cuda_unordered_map<T>* map;
	public:
		CUDA_EXPORT
		T& operator*() {
			return map->buckets[bucket][index];
		}
		CUDA_EXPORT
		pair<int,T>* operator->() {
			return map->buckets[bucket].data() + index;
		}
		CUDA_EXPORT
		bool operator==(const iterator& other) {
			return bucket == other.bucket && index == other.index;
		}
		CUDA_EXPORT
		bool operator!=(const iterator& other) {
			return bucket != other.bucket || index != other.index;
		}
		CUDA_EXPORT
		iterator& operator++() {
			index++;
			if( index >= map->buckets[index].size()) {
				index = 0;
				bucket++;
			}
			return *this;
		}
		CUDA_EXPORT
		iterator operator++(int) {
			auto old = *this;
			operator++();
			return *this;
		}
		friend class cuda_unordered_map<T>;
	};
	CUDA_EXPORT
	iterator find( int key) {
		if( buckets.size()) {
			int bucket_index = key % buckets.size();
			auto& bucket = buckets[bucket_index];
			for( int i = 0; i < bucket.size(); i++) {
				if( bucket[i].first == key ) {
					iterator iter;
					iter.bucket = bucket_index;
					iter.index = i;
					iter.map = this;
					return iter;
				}
			}
			return end();
		} else {
			return end();
		}
	}
	CUDA_EXPORT
	iterator begin() {
		iterator i;
		i.bucket = 0;
		i.index = 0;
		i.map = this;
		return i;
	}
	CUDA_EXPORT
	iterator end() {
		iterator i;
		i.bucket = buckets.size();
		i.index = 0;
		i.map = this;
		return i;
	}
	CUDA_EXPORT
	cuda_unordered_map() {
		initialize();
	}
	CUDA_EXPORT
	void clear() {
		for( int i = 0; i < buckets.size(); i++) {
			buckets[i].resize(0);
		}
	}
	CUDA_EXPORT
	iterator insert(pair<int,T>&& entry ) {
#ifdef __CUDA_ARCH__
		ALWAYS_ASSERT(false);
		return end();
#else
		rehash();
		int bucket_index = entry.first % buckets.size();
		auto& bucket = buckets[bucket_index];
		bucket.push_back(std::move(entry));
		sz++;
		iterator iter;
		iter.map = this;
		iter.bucket = bucket_index;
		iter.index = bucket.size() - 1;
		return iter;
#endif
	}
	CUDA_EXPORT
	void erase(int key) {
		int bucket_index = key % buckets.size();
		auto& bucket = buckets[bucket_index];
		for( int i = 0; i < bucket.size(); i++) {
			if( bucket[i].first == key ) {
				bucket.back() = bucket[i];
				bucket.pop_back();
			}
		}
		sz--;
	}
	CUDA_EXPORT
	int size() const {
		return sz;
	}
	CUDA_EXPORT
	T& operator[]( int key ) {
		auto iter = find(key);
		if(iter == end()) {
			pair<int,T> I;
			I.first = key;
			iter = insert(std::move(I));
		}
		return iter->second;
	}
};

#endif /* CUDA_UNORDERED_MAP_HPP_ */
