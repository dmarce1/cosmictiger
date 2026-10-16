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

#include <cosmictiger/cuda_mem.hpp>

struct Threadid {
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

struct Blocksz {
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
    int sz_;
    int cap_;

    CUDA_EXPORT
    inline void construct(int a, int b) {
        Threadid tid;
        Blocksz bsz;
        syncthreads();
        for (int i = a + tid; i < b; i += (int) bsz) {
            new (ptr_ + i) T;
        }
        syncthreads();
    }

    CUDA_EXPORT
    inline void destruct(int a, int b) {
        Threadid tid;
        Blocksz bsz;
        syncthreads();
        for (int i = a + tid; i < b; i += (int) bsz) {
            (ptr_ + i)->~T();
        }
        syncthreads();
    }

    CUDA_EXPORT
    inline void initialize() {
        Threadid tid;
        syncthreads();
        if (tid == 0) {
            sz_ = 0;
            cap_ = 0;
            ptr_ = nullptr;
        }
        syncthreads();
    }
    CUDA_EXPORT
    inline void destroy() {
        Threadid tid;
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
        return cap_;
    }
    CUDA_EXPORT
    inline void swap(DeviceVector& other) {
        auto const optr = other.ptr_;
        auto const ocap = other.cap_;
        auto const osz = other.sz_;
        other.ptr_ = ptr_;
        other.cap_ = cap_;
        other.sz_ = sz_;
        ptr_ = optr;
        cap_ = ocap;
        sz_ = osz;
    }
    CUDA_EXPORT
    inline T* begin() {
        return ptr_;
    }
    CUDA_EXPORT
    inline T* end() {
        return ptr_ + sz_;
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
    inline DeviceVector(int sz0) {
        initialize();
        resize(sz0);
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
    CUDA_EXPORT
    inline void resize(int newSz) {
        Threadid tid;
        destruct(newSz, sz_);
        auto const osz = sz_;
        if (newSz <= cap_) {
            syncthreads();
            if (tid == 0) {
                sz_ = newSz;
            }
            syncthreads();
        } else {
            syncthreads();
            int newCap = 1024 / sizeof(T);
            if (newCap < 1) {
                newCap = 1;
            }
            if (tid == 0) {
                while (newCap < newSz) {
                    newCap *= 2;
                }
                newPtr_ = (T*) cudaAllocate(sizeof(T) * newCap);
                if (newPtr_ == nullptr) {
                    PRINT("OOM in device_vector while requesting %i! \n", newCap);
                    ALWAYS_ASSERT(false);
                }
            }
            syncthreads();
            if (ptr_) {
                copyCudaMemory(newPtr_, ptr_, sizeof(T) * sz_);
            }
            syncthreads();
            destroy();
            if (tid == 0) {
                ptr_ = newPtr_;
                cap_ = newCap;
                sz_ = newSz;
            }
            syncthreads();
        }
        construct(osz, sz_);
    }
    CUDA_EXPORT
    inline T& back() {
        return ptr_[sz_ - 1];
    }
    CUDA_EXPORT
    inline T const& back() const {
        return ptr_[sz_ - 1];
    }
    CUDA_EXPORT
    inline void popBack() {
        Threadid tid;
        if (tid == 0) {
            sz_--;
        }
        syncthreads();
    }
    CUDA_EXPORT
    inline void pushBack(T const& item) {
        Threadid tid;
        resize(size() + 1);
        if (tid == 0) {
            back() = item;
        }
        syncthreads();
    }
    CUDA_EXPORT
    inline void pushBack(T&& item) {
        Threadid tid;
        resize(size() + 1);
        if (tid == 0) {
            back() = std::move(item);
        }
        syncthreads();
    }
    CUDA_EXPORT
    inline int size() const {
        return sz_;
    }
    CUDA_EXPORT
    inline T& operator[](int i) {
#ifdef CHECK_BOUNDS
        if (i >= sz_) {
            PRINT("A) Bound exceeded in device_vector %i %i %i\n", i, sz_, sizeof(T));
            ALWAYS_ASSERT(false);
        }
#endif
        return ptr_[i];
    }
    CUDA_EXPORT
    inline T const& operator[](int i) const {
#ifdef CHECK_BOUNDS
        if (i >= sz_) {
            PRINT("B) Bound exceeded in device_vector %i %i %i\n", i, sz_, sizeof(T));
            ALWAYS_ASSERT(false);
        }
#endif
        return ptr_[i];
    }
    template <class A>
    void serialize(A&& arc, unsigned) {
        int size = sz_;
        arc & size;
        resize(size);
        for (int i = 0; i < size; i++) {
            arc&(*this)[i];
        }
    }
};

#include <cosmictiger/cuda_mem.hpp>

#endif /* DEVICE_VECTOR_HPP_ */
