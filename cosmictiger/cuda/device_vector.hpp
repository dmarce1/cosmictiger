/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2021  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
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

// Growable array whose storage normally comes from the cudaAllocate pool. A
// vector may instead be attached to caller-provided storage (a region of a
// per-launch slab); it then touches the pool only if it outgrows that region,
// and never releases the attached storage.
template <class T>
class DeviceVector {
    T* ptr_;
    T* newPtr_;
    int size_;
    int capacity_;
    bool pooled_;    // ptr_ came from the pool and is released by destroy()

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
            pooled_ = true;
        }
        syncthreads();
    }
    CUDA_EXPORT
    inline void destroy() {
        ThreadIndex tid;
        syncthreads();
        if (tid == 0) {
            if (ptr_ && pooled_) {
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
        auto const previousPooled = other.pooled_;
        other.ptr_ = ptr_;
        other.capacity_ = capacity_;
        other.size_ = size_;
        other.pooled_ = pooled_;
        ptr_ = previousData;
        capacity_ = previousCapacity;
        size_ = previousSize;
        pooled_ = previousPooled;
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
    // Collective: an empty vector on caller-provided storage; see attach().
    CUDA_EXPORT
    inline DeviceVector(T* storage, int storageCapacity) {
        initialize();
        attach(storage, storageCapacity);
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
    // Collective: use caller-provided storage for up to storageCapacity
    // elements. The live elements move there; the previous pool block, if any,
    // is released. Growth beyond storageCapacity falls back to the pool, and
    // the attached storage is never released by this vector.
    CUDA_EXPORT
    inline void attach(T* storage, int storageCapacity) {
        ThreadIndex tid;
        syncthreads();
        if (size_ > storageCapacity) {
            PRINT("device_vector attach: %i live elements exceed the storage of %i\n", size_,
                storageCapacity);
            ALWAYS_ASSERT(false);
        }
        if (ptr_ && size_) {
            copyCudaMemory(storage, ptr_, sizeof(T) * size_);
        }
        syncthreads();
        destroy();
        if (tid == 0) {
            ptr_ = storage;
            capacity_ = storageCapacity;
            pooled_ = false;
        }
        syncthreads();
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
            pooled_ = true;
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
