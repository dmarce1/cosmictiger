/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2021  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#ifndef DEVICE_VECTOR_HPP_
#define DEVICE_VECTOR_HPP_

#include <algorithm>
#include <climits>
#include <cstdint>
#include <format>
#include <iostream>
#include <memory>
#include <type_traits>
#include <utility>

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
// and never releases the attached storage. On the device, mutations are
// collective over a one-dimensional block and element types must be trivially
// copyable. Owning nested element types are constructed/moved/destroyed on the host.
template <class T>
class DeviceVector {
    static_assert(alignof(T) <= kCudaPoolAlignment);
    T* ptr_;
    T* newPtr_;
    int size_;
    int capacity_;
    bool pooled_;    // ptr_ came from the pool and is released by destroy()

    CUDA_EXPORT
    inline void construct(int a, int b) {
#ifdef __CUDA_ARCH__
        // Nontrivial owners may contain block-collective constructors. They
        // are managed on the host; per-thread placement construction is unsafe.
        ALWAYS_ASSERT(std::is_trivially_copyable_v<T>);
        ThreadIndex tid;
        BlockSize bsz;
        syncthreads();
        for (int i = a + tid; i < b; i += (int) bsz) {
            new (ptr_ + i) T;
        }
        syncthreads();
#else
        if (a < b) {
            // The standard algorithm destroys the constructed prefix on failure.
            std::uninitialized_default_construct(ptr_ + a, ptr_ + b);
        }
#endif
    }

    CUDA_EXPORT
    inline int elementIndex(T const& item) const {
        // Compare addresses without ordering or subtracting unrelated pointers.
        auto const offset = reinterpret_cast<uintptr_t>(&item) - reinterpret_cast<uintptr_t>(ptr_);
        return offset < sizeof(T) * size_t(size_) ? int(offset / sizeof(T)) : -1;
    }

    CUDA_EXPORT
    inline void destruct(int a, int b) {
        ThreadIndex tid;
        BlockSize bsz;
        syncthreads();
        if constexpr (!std::is_trivially_destructible_v<T>) {
#ifdef __CUDA_ARCH__
            ALWAYS_ASSERT(a >= b);
#else
            for (int i = a; i < b; ++i) {
                (ptr_ + i)->~T();
            }
#endif
        }
        syncthreads();
    }

    CUDA_EXPORT
    inline void initialize() {
#ifdef __CUDA_ARCH__
        ALWAYS_ASSERT(blockDim.y == 1 && blockDim.z == 1);
#endif
        ThreadIndex tid;
        syncthreads();
        if (tid == 0) {
            size_ = 0;
            capacity_ = 0;
            ptr_ = nullptr;
            newPtr_ = nullptr;
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

    // Relocate live elements into uninitialized storage. Owning elements need
    // real moves so their old lifetimes can end without aliasing pool buffers.
    CUDA_EXPORT
    inline void relocate(T* storage) {
        if (size_ == 0) {
            return;
        }
        if constexpr (std::is_trivially_copyable_v<T>) {
            copyCudaMemory(storage, ptr_, sizeof(T) * size_t(size_));
        } else {
#ifdef __CUDA_ARCH__
            ALWAYS_ASSERT(false);
#else
            std::uninitialized_move(begin(), end(), storage);
            destruct(0, size_);
#endif
        }
    }

public:
    CUDA_EXPORT
    inline int capacity() const {
        return capacity_;
    }
    CUDA_EXPORT
    inline void swap(DeviceVector& other) {
        syncthreads();
        ThreadIndex tid;
        if (tid == 0) {
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
        syncthreads();
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
    inline DeviceVector(DeviceVector const& other)
      : DeviceVector() {
        *this = other;
    }
    CUDA_EXPORT
    inline DeviceVector& operator=(DeviceVector const& other) {
        if (this == &other) {
            return *this;
        }
        if constexpr (std::is_trivially_copyable_v<T>) {
            resize(other.size());
            copyCudaMemory(ptr_, other.ptr_, size_t(size()) * sizeof(T));
        } else {
#ifdef __CUDA_ARCH__
            ALWAYS_ASSERT(false);
#else
            DeviceVector replacement(other.size());
            std::copy(other.begin(), other.end(), replacement.begin());
            swap(replacement);
#endif
        }
        return *this;
    }
    CUDA_EXPORT
    inline DeviceVector& operator=(DeviceVector&& other) {
        if (this == &other) {
            return *this;
        }
        destruct(0, size_);
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
    inline DeviceVector(int initialSize)
      : DeviceVector() {
        // Delegation makes the destructor run if element construction throws.
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
        destruct(0, size_);
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
#ifdef __CUDA_ARCH__
            printf("device_vector attach: %i live elements exceed the storage of %i\n", size_,
                storageCapacity);
#else
            std::cout
                << std::format(
                       "device_vector attach: {:d} live elements exceed the storage of {:d}\n",
                       size_, storageCapacity)
                << std::flush;
#endif
            ALWAYS_ASSERT(false);
        }
        ALWAYS_ASSERT(storageCapacity >= 0);
        if (storage == ptr_) {
            ALWAYS_ASSERT(storageCapacity == capacity_);
            return;
        }
        relocate(storage);
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
        ALWAYS_ASSERT(minimumCapacity >= size_ && minimumCapacity >= 0);
        while (newCapacity < minimumCapacity) {
            if (newCapacity > INT_MAX / 2) {
                newCapacity = minimumCapacity;
                break;
            }
            newCapacity *= 2;
        }
        if (tid == 0) {
            newPtr_ = (T*) cudaAllocate(sizeof(T) * newCapacity);
            if (newPtr_ == nullptr) {
#ifdef __CUDA_ARCH__
                printf("OOM in device_vector while requesting %i! \n", newCapacity);
#else
                std::cout << std::format(
                                 "OOM in device_vector while requesting {:d}! \n", newCapacity)
                          << std::flush;
#endif
                ALWAYS_ASSERT(false);
            }
        }
        syncthreads();
#ifndef __CUDA_ARCH__
        try {
            relocate(newPtr_);
        } catch (...) {
            cudaRelease(newPtr_);
            throw;
        }
#else
        relocate(newPtr_);
#endif
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
        ALWAYS_ASSERT(newSize >= 0);
        ThreadIndex tid;
        destruct(newSize, size_);
        auto const previousSize = size_;
        if (newSize > capacity_) {
            grow(newSize);
        }
        construct(previousSize, newSize);
        syncthreads();
        if (tid == 0) {
            size_ = newSize;
        }
        syncthreads();
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
        syncthreads();
        ALWAYS_ASSERT(size_ > 0);
        destruct(size_ - 1, size_);
        ThreadIndex tid;
        if (tid == 0) {
            size_--;
        }
        syncthreads();
    }
    CUDA_EXPORT
    inline void pushBack(T const& item) {
        ThreadIndex tid;
        ALWAYS_ASSERT(size() < INT_MAX);
        int const sourceIndex = elementIndex(item);
        resize(size() + 1);
        if (tid == 0) {
            // Growth invalidates references into the previous storage.
            back() = sourceIndex >= 0 ? ptr_[sourceIndex] : item;
        }
        syncthreads();
    }
    CUDA_EXPORT
    inline void pushBack(T&& item) {
        ThreadIndex tid;
        ALWAYS_ASSERT(size() < INT_MAX);
        int const sourceIndex = elementIndex(item);
        resize(size() + 1);
        if (tid == 0) {
            back() = std::move(sourceIndex >= 0 ? ptr_[sourceIndex] : item);
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
        if (i < 0 || i >= size_) {
#ifdef __CUDA_ARCH__
            printf("A) Bound exceeded in device_vector %i %i %i\n", i, size_, sizeof(T));
#else
            std::cout << std::format("A) Bound exceeded in device_vector {:d} {:d} {:d}\n", i,
                             size_, sizeof(T))
                      << std::flush;
#endif
            ALWAYS_ASSERT(false);
        }
#endif
        return ptr_[i];
    }
    CUDA_EXPORT
    inline T const& operator[](int i) const {
#ifdef CHECK_BOUNDS
        if (i < 0 || i >= size_) {
#ifdef __CUDA_ARCH__
            printf("B) Bound exceeded in device_vector %i %i %i\n", i, size_, sizeof(T));
#else
            std::cout << std::format("B) Bound exceeded in device_vector {:d} {:d} {:d}\n", i,
                             size_, sizeof(T))
                      << std::flush;
#endif
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

#endif /* DEVICE_VECTOR_HPP_ */
