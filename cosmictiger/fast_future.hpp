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

#pragma once

#include <cosmictiger/hpx.hpp>

#ifndef __CUDACC__

template <class T>
class FastFuture {
    T data_;
    hpx::future<T> fut_;
    bool hasData_;

public:
    FastFuture() {
        hasData_ = false;
    }
    FastFuture(FastFuture<T> const&) = delete;
    FastFuture(FastFuture<T>&&) = default;
    FastFuture& operator=(FastFuture<T> const&) = delete;
    FastFuture& operator=(FastFuture<T>&&) = default;
    FastFuture& operator=(T&& dataArg) {
        hasData_ = true;
        data_ = std::move(dataArg);
        return *this;
    }
    inline FastFuture(T const& dataArg) {
        hasData_ = true;
        data_ = dataArg;
    }
    inline FastFuture(T&& dataArg) {
        hasData_ = true;
        data_ = std::move(dataArg);
    }
    inline FastFuture& operator=(hpx::future<T>&& futArg) {
        hasData_ = false;
        fut_ = std::move(futArg);
        return *this;
    }
    inline FastFuture(hpx::future<T>&& futArg) {
        hasData_ = false;
        fut_ = std::move(futArg);
    }
    inline void setValue(T&& thisData) {
        hasData_ = true;
        data_ = std::move(thisData);
    }
    inline bool valid() const {
        return (hasData_ || fut_.valid());
    }
    inline T get() {
        if (hasData_) {
            return std::move(data_);
        } else {
            return fut_.get();
        }
    }
    operator hpx::future<T>() {
        if (hasData_) {
            return hpx::make_ready_future(data_);
        } else {
            return std::move(fut_);
        }
    }
};

#endif
