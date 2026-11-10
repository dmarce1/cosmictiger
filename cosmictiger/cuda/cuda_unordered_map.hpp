/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2021  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#ifndef CUDA_UNORDERED_MAP_HPP_
#define CUDA_UNORDERED_MAP_HPP_

#include <climits>

#include <cosmictiger/cuda/device_vector.hpp>

template <class T>
class CudaUnorderedMap {
    DeviceVector<DeviceVector<Pair<int, T>>> buckets_;
    int size_;
    /// Initialize the logical entry count to zero.
    CUDA_EXPORT
    void initialize() {
        size_ = 0;
    }
    /// Grow the bucket table when needed and redistribute entries by integer-key
    /// modulus.
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
        /// Access the key/value entry at the iterator's current bucket and offset.
        CUDA_EXPORT
        Pair<int, T>& operator*() {
            return map_->buckets_[bucket_][index_];
        }
        /// Return a pointer to the iterator's current key/value entry.
        CUDA_EXPORT
        Pair<int, T>* operator->() {
            return map_->buckets_[bucket_].data() + index_;
        }
        /// Compare bucket and offset positions; iterators must belong to the same
        /// map.
        CUDA_EXPORT
        bool operator==(Iterator const& other) {
            return bucket_ == other.bucket_ && index_ == other.index_;
        }
        /// Test whether bucket or offset positions differ within the same map.
        CUDA_EXPORT
        bool operator!=(Iterator const& other) {
            return bucket_ != other.bucket_ || index_ != other.index_;
        }
        /// Advance to the next entry, skipping exhausted buckets.
        CUDA_EXPORT
        Iterator& operator++() {
            index_++;
            while (bucket_ < map_->buckets_.size() && index_ >= map_->buckets_[bucket_].size()) {
                index_ = 0;
                bucket_++;
            }
            return *this;
        }
        /// Advance to the next entry and return the iterator's previous position.
        CUDA_EXPORT
        Iterator operator++(int) {
            auto old = *this;
            operator++();
            return old;
        }
        friend class CudaUnorderedMap<T>;
    };
    /// Find a key in its bucket, returning end() when it is absent.
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
    /// Return an iterator to the first entry, skipping empty buckets.
    CUDA_EXPORT
    Iterator begin() {
        Iterator i;
        i.bucket_ = 0;
        i.index_ = 0;
        i.map_ = this;
        while (i.bucket_ < buckets_.size() && buckets_[i.bucket_].empty()) {
            ++i.bucket_;
        }
        return i;
    }
    /// Return the sentinel iterator positioned after the bucket table.
    CUDA_EXPORT
    Iterator end() {
        Iterator i;
        i.bucket_ = buckets_.size();
        i.index_ = 0;
        i.map_ = this;
        return i;
    }
    /// Initialize an empty integer-key map.
    CUDA_EXPORT
    CudaUnorderedMap() {
        initialize();
    }
    /// Remove all entries while retaining bucket storage for reuse.
    CUDA_EXPORT
    void clear() {
        for (int i = 0; i < buckets_.size(); i++) {
            buckets_[i].resize(0);
        }
        size_ = 0;
    }
    /// Append an entry on the host, growing buckets as needed; device insertion
    /// triggers an assertion.
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
    /// Remove the matching key by moving the bucket's last entry into its slot.
    CUDA_EXPORT
    void erase(int key) {
        if (buckets_.empty()) {
            return;
        }
        auto& bucket = buckets_[key % buckets_.size()];
        for (int i = 0; i < bucket.size(); ++i) {
            if (bucket[i].first == key) {
                if (i != bucket.size() - 1) {
                    bucket[i] = std::move(bucket.back());
                }
                bucket.popBack();
                --size_;
                return;
            }
        }
    }
    /// Return the number of stored key/value entries.
    CUDA_EXPORT
    int size() const {
        return size_;
    }
    /// Access a key's value, inserting a new entry on the host when the key is absent.
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
