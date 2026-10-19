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

#include <unordered_set>

#include <cosmictiger/core/containers.hpp>
#include <cosmictiger/core/defs.hpp>
#include <cosmictiger/core/hpx.hpp>

template <class T, class HASH>
class UnorderedSetTs {
    struct HashLo {
        size_t operator()(T const& member) const {
            HASH hash;
            return hash(member) % UNORDERED_SET_SIZE;
        }
    };
    struct HashHi {
        size_t operator()(T const& member) const {
            HASH hash;
            return hash(member) / UNORDERED_SET_SIZE;
        }
    };
    Vector<std::unordered_set<T, HashHi>> sets_;
    Vector<std::shared_ptr<SpinlockType>> mutexes_;

public:
    UnorderedSetTs() {
        sets_.resize(UNORDERED_SET_SIZE);
        mutexes_.resize(UNORDERED_SET_SIZE, std::make_shared<SpinlockType>());
    }
    void insert(T const& member) {
        PRINT("Inserting\n");
        HashLo hashlo;
        int const setIndex = hashlo(member);
        std::lock_guard<SpinlockType> lock(*mutexes_[setIndex]);
        sets_[setIndex].insert(member);
    }
    Vector<T> toVector() const {
        Vector<T> members;
        int size = 0;
        for (int i = 0; i < UNORDERED_SET_SIZE; i++) {
            size += sets_[i].size();
        }
        members.reserve(size);
        for (int i = 0; i < UNORDERED_SET_SIZE; i++) {
            for (auto j = sets_[i].begin(); j != sets_[i].end(); j++) {
                members.push_back(*j);
            }
        }
        return members;
    }
};

#endif /* UNORDERED_SET_TS_HPP_ */
