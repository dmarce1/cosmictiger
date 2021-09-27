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

template<class T>
class fast_future {
   T data;
   hpx::future<T> fut;
   bool has_data;
public:
   fast_future() {
      has_data = false;
   }
   fast_future(const fast_future<T>&) = delete;
   fast_future(fast_future<T>&&) = default;
   fast_future& operator=(const fast_future<T>&) = delete;
   fast_future& operator=(fast_future<T>&&) = default;
   fast_future& operator=(T&& data_) {
      has_data = true;
      data = std::move(data_);
      return *this;
   }
   inline fast_future(const T& data_) {
      has_data = true;
      data = data_;
   }
   inline fast_future(T&& data_) {
      has_data = true;
      data = std::move(data_);
   }
   inline fast_future& operator=(hpx::future<T> &&fut_) {
      has_data = false;
      fut = std::move(fut_);
      return *this;
   }
   inline fast_future(hpx::future<T> &&fut_) {
      has_data = false;
      fut = std::move(fut_);
   }
   inline void set_value(T&& this_data) {
      has_data = true;
      data = std::move(this_data);
   }
   inline bool valid() const {
      return( has_data || fut.valid() );
   }
   inline T get() {
      if( has_data ) {
         return std::move(data);
      } else {
         return fut.get();
      }
   }
   operator hpx::future<T>() {
      if( has_data) {
         return hpx::make_ready_future(data);
      } else{
         return std::move(fut);
      }
   }

};


#endif
