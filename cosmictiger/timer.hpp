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

#include <chrono>

class timer {
   std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
   double time;
public:
   inline timer() {
      time = 0.0;
   }
   inline void stop() {
      std::chrono::time_point<std::chrono::high_resolution_clock> stop_time = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> dur = stop_time - start_time;
      time += dur.count();
   }
   inline void start() {
      start_time = std::chrono::high_resolution_clock::now();
   }
   inline void reset() {
      time = 0.0;
   }
   inline double read() {
      return time;
   }
};
