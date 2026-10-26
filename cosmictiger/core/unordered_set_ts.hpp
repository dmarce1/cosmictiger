/*
CosmicTiger - A cosmological N-Body code
Copyright (C) 2021  Dominic C. Marcello

Distributed under the Boost Software License, Version 1.0.
See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
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
            return hash(member) % kUnorderedSetSize;
        }
    };
    struct HashHi {
        size_t operator()(T const& member) const {
            HASH hash;
            return hash(member) / kUnorderedSetSize;
        }
    };
    Vector<std::unordered_set<T, HashHi>> sets_;
    Vector<std::shared_ptr<SpinlockType>> mutexes_;

public:
    UnorderedSetTs() {
        sets_.resize(kUnorderedSetSize);
        mutexes_.resize(kUnorderedSetSize, std::make_shared<SpinlockType>());
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
        for (int i = 0; i < kUnorderedSetSize; i++) {
            size += sets_[i].size();
        }
        members.reserve(size);
        for (int i = 0; i < kUnorderedSetSize; i++) {
            for (auto j = sets_[i].begin(); j != sets_[i].end(); j++) {
                members.push_back(*j);
            }
        }
        return members;
    }
};

#endif /* UNORDERED_SET_TS_HPP_ */
