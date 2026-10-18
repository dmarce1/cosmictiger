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

#include <climits>

#include <cosmictiger/device_vector.hpp>

template <class T>
class CudaUnorderedMap {
    DeviceVector<DeviceVector<Pair<int, T>>> buckets_;
    int size_;
    CUDA_EXPORT
    void initialize() {
        size_ = 0;
    }
    CUDA_EXPORT
    void rehash() {
        int newMod = std::max(buckets_.size(), 8);
        while (size() / newMod > 2) {
            newMod *= 2;
        }
        if (newMod > buckets_.size()) {
            DeviceVector<DeviceVector<Pair<int, T>>> newBuckets;
            newBuckets.resize(newMod);
            for (int i = 0; i < buckets_.size(); i++) {
                for (int j = 0; j < buckets_[i].size(); j++) {
                    newBuckets[buckets_[i][j].first % newMod].pushBack(std::move(buckets_[i][j]));
                }
            }
            buckets_.swap(newBuckets);
        }
    }

public:
    class Iterator {
        int bucket_;
        int index_;
        CudaUnorderedMap<T>* map_;

    public:
        CUDA_EXPORT
        T& operator*() {
            return map_->buckets_[bucket_][index_];
        }
        CUDA_EXPORT
        Pair<int, T>* operator->() {
            return map_->buckets_[bucket_].data() + index_;
        }
        CUDA_EXPORT
        bool operator==(Iterator const& other) {
            return bucket_ == other.bucket_ && index_ == other.index_;
        }
        CUDA_EXPORT
        bool operator!=(Iterator const& other) {
            return bucket_ != other.bucket_ || index_ != other.index_;
        }
        CUDA_EXPORT
        Iterator& operator++() {
            index_++;
            if (index_ >= map_->buckets_[index_].size()) {
                index_ = 0;
                bucket_++;
            }
            return *this;
        }
        CUDA_EXPORT
        Iterator operator++(int) {
            auto old = *this;
            operator++();
            return *this;
        }
        friend class CudaUnorderedMap<T>;
    };
    CUDA_EXPORT
    Iterator find(int key) {
        if (buckets_.size()) {
            int bucketIndex = key % buckets_.size();
            auto& bucket = buckets_[bucketIndex];
            for (int i = 0; i < bucket.size(); i++) {
                if (bucket[i].first == key) {
                    Iterator iter;
                    iter.bucket_ = bucketIndex;
                    iter.index_ = i;
                    iter.map_ = this;
                    return iter;
                }
            }
            return end();
        } else {
            return end();
        }
    }
    CUDA_EXPORT
    Iterator begin() {
        Iterator i;
        i.bucket_ = 0;
        i.index_ = 0;
        i.map_ = this;
        return i;
    }
    CUDA_EXPORT
    Iterator end() {
        Iterator i;
        i.bucket_ = buckets_.size();
        i.index_ = 0;
        i.map_ = this;
        return i;
    }
    CUDA_EXPORT
    CudaUnorderedMap() {
        initialize();
    }
    CUDA_EXPORT
    void clear() {
        for (int i = 0; i < buckets_.size(); i++) {
            buckets_[i].resize(0);
        }
    }
    CUDA_EXPORT
    Iterator insert(Pair<int, T>&& entry) {
#ifdef __CUDA_ARCH__
        ALWAYS_ASSERT(false);
        return end();
#else
        rehash();
        int bucketIndex = entry.first % buckets_.size();
        auto& bucket = buckets_[bucketIndex];
        bucket.pushBack(std::move(entry));
        size_++;
        Iterator iter;
        iter.map_ = this;
        iter.bucket_ = bucketIndex;
        iter.index_ = bucket.size() - 1;
        return iter;
#endif
    }
    CUDA_EXPORT
    void erase(int key) {
        int bucketIndex = key % buckets_.size();
        auto& bucket = buckets_[bucketIndex];
        for (int i = 0; i < bucket.size(); i++) {
            if (bucket[i].first == key) {
                bucket.back() = bucket[i];
                bucket.popBack();
            }
        }
        size_--;
    }
    CUDA_EXPORT
    int size() const {
        return size_;
    }
    CUDA_EXPORT
    T& operator[](int key) {
        auto iter = find(key);
        if (iter == end()) {
            Pair<int, T> iValue;
            iValue.first = key;
            iter = insert(std::move(iValue));
        }
        return iter->second;
    }
};

#endif /* CUDA_UNORDERED_MAP_HPP_ */
