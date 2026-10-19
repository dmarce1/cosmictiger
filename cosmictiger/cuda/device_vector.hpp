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

#ifndef DEVICE_VECTOR_HPP_
#define DEVICE_VECTOR_HPP_

#include <cosmictiger/cuda/cuda_memory.hpp>

struct ThreadIndex {
    CUDA_EXPORT
    inline bool operator==(int const tid) const {
#ifdef __CUDA_ARCH__
        return threadIdx.x == tid;
#else
        return true;
#endif
    }
    CUDA_EXPORT
    inline operator int() const {
#ifdef __CUDA_ARCH__
        return threadIdx.x;
#else
        return 0;
#endif
    }
};

struct BlockSize {
    CUDA_EXPORT
    inline operator int() const {
#ifdef __CUDA_ARCH__
        return blockDim.x;
#else
        return 1;
#endif
    }
};

template <class T>
class DeviceVector {
    T* ptr_;
    T* newPtr_;
    int size_;
    int capacity_;

    CUDA_EXPORT
    inline void construct(int a, int b) {
        ThreadIndex tid;
        BlockSize bsz;
        syncthreads();
        for (int i = a + tid; i < b; i += (int) bsz) {
            new (ptr_ + i) T;
        }
        syncthreads();
    }

    CUDA_EXPORT
    inline void destruct(int a, int b) {
        ThreadIndex tid;
        BlockSize bsz;
        syncthreads();
        for (int i = a + tid; i < b; i += (int) bsz) {
            (ptr_ + i)->~T();
        }
        syncthreads();
    }

    CUDA_EXPORT
    inline void initialize() {
        ThreadIndex tid;
        syncthreads();
        if (tid == 0) {
            size_ = 0;
            capacity_ = 0;
            ptr_ = nullptr;
        }
        syncthreads();
    }
    CUDA_EXPORT
    inline void destroy() {
        ThreadIndex tid;
        syncthreads();
        if (tid == 0) {
            if (ptr_) {
                cudaRelease(ptr_);
            }
        }
        syncthreads();
    }

public:
    CUDA_EXPORT
    inline int capacity() const {
        return capacity_;
    }
    CUDA_EXPORT
    inline void swap(DeviceVector& other) {
        auto const previousData = other.ptr_;
        auto const previousCapacity = other.capacity_;
        auto const previousSize = other.size_;
        other.ptr_ = ptr_;
        other.capacity_ = capacity_;
        other.size_ = size_;
        ptr_ = previousData;
        capacity_ = previousCapacity;
        size_ = previousSize;
    }
    CUDA_EXPORT
    inline T* begin() {
        return ptr_;
    }
    CUDA_EXPORT
    inline T* end() {
        return ptr_ + size_;
    }
    CUDA_EXPORT
    inline T const* begin() const {
        return ptr_;
    }
    CUDA_EXPORT
    inline T const* end() const {
        return ptr_ + size_;
    }
    CUDA_EXPORT
    inline bool empty() const {
        return size_ == 0;
    }
    CUDA_EXPORT
    inline DeviceVector() {
        initialize();
    }
    CUDA_EXPORT
    inline DeviceVector(DeviceVector const& other) {
        initialize();
        *this = other;
    }
    CUDA_EXPORT
    inline DeviceVector& operator=(DeviceVector const& other) {
        resize(other.size());
        copyCudaMemory(ptr_, other.ptr_, size() * sizeof(T));
        return *this;
    }
    CUDA_EXPORT
    inline DeviceVector& operator=(DeviceVector&& other) {
        destroy();
        initialize();
        swap(other);
        return *this;
    }
    CUDA_EXPORT
    inline DeviceVector(DeviceVector&& other) {
        initialize();
        swap(other);
    }
    CUDA_EXPORT
    inline DeviceVector(int initialSize) {
        initialize();
        resize(initialSize);
    }
    CUDA_EXPORT
    inline ~DeviceVector() {
        destroy();
    }
    CUDA_EXPORT
    inline T* data() {
        return ptr_;
    }
    CUDA_EXPORT
    inline T const* data() const {
        return ptr_;
    }
    // Collective: move the live elements into a pool block of at least
    // minimumCapacity elements (grown by doubling from the default block).
    CUDA_EXPORT
    inline void grow(int minimumCapacity) {
        ThreadIndex tid;
        syncthreads();
        int newCapacity = 1024 / sizeof(T);
        if (newCapacity < 1) {
            newCapacity = 1;
        }
        while (newCapacity < minimumCapacity) {
            newCapacity *= 2;
        }
        if (tid == 0) {
            newPtr_ = (T*) cudaAllocate(sizeof(T) * newCapacity);
            if (newPtr_ == nullptr) {
                PRINT("OOM in device_vector while requesting %i! \n", newCapacity);
                ALWAYS_ASSERT(false);
            }
        }
        syncthreads();
        if (ptr_) {
            copyCudaMemory(newPtr_, ptr_, sizeof(T) * size_);
        }
        syncthreads();
        destroy();
        if (tid == 0) {
            ptr_ = newPtr_;
            capacity_ = newCapacity;
        }
        syncthreads();
    }
    // Collective: guarantee capacity for newCapacity elements without
    // changing the size, so later growth needs no pool traffic.
    CUDA_EXPORT
    inline void reserve(int newCapacity) {
        if (newCapacity > capacity_) {
            grow(newCapacity);
        }
    }
    CUDA_EXPORT
    inline void resize(int newSize) {
        ThreadIndex tid;
        destruct(newSize, size_);
        auto const previousSize = size_;
        if (newSize > capacity_) {
            grow(newSize);
        }
        syncthreads();
        if (tid == 0) {
            size_ = newSize;
        }
        syncthreads();
        construct(previousSize, size_);
    }
    CUDA_EXPORT
    inline T& back() {
        return ptr_[size_ - 1];
    }
    CUDA_EXPORT
    inline T const& back() const {
        return ptr_[size_ - 1];
    }
    CUDA_EXPORT
    inline void popBack() {
        ThreadIndex tid;
        if (tid == 0) {
            size_--;
        }
        syncthreads();
    }
    CUDA_EXPORT
    inline void pushBack(T const& item) {
        ThreadIndex tid;
        resize(size() + 1);
        if (tid == 0) {
            back() = item;
        }
        syncthreads();
    }
    CUDA_EXPORT
    inline void pushBack(T&& item) {
        ThreadIndex tid;
        resize(size() + 1);
        if (tid == 0) {
            back() = std::move(item);
        }
        syncthreads();
    }
    CUDA_EXPORT
    inline int size() const {
        return size_;
    }
    CUDA_EXPORT
    inline T& operator[](int i) {
#ifdef CHECK_BOUNDS
        if (i >= size_) {
            PRINT("A) Bound exceeded in device_vector %i %i %i\n", i, size_, sizeof(T));
            ALWAYS_ASSERT(false);
        }
#endif
        return ptr_[i];
    }
    CUDA_EXPORT
    inline T const& operator[](int i) const {
#ifdef CHECK_BOUNDS
        if (i >= size_) {
            PRINT("B) Bound exceeded in device_vector %i %i %i\n", i, size_, sizeof(T));
            ALWAYS_ASSERT(false);
        }
#endif
        return ptr_[i];
    }
    template <class A>
    void serialize(A&& arc, unsigned) {
        int size = size_;
        arc & size;
        resize(size);
        for (int i = 0; i < size; i++) {
            arc&(*this)[i];
        }
    }
};

#include <cosmictiger/cuda/cuda_memory.hpp>

#endif /* DEVICE_VECTOR_HPP_ */
