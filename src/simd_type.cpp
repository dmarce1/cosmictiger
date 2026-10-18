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

#include <cstdio>

#include <cosmictiger/simd_backend.hpp>

int main() {
#if COSMICTIGER_SIMD_STANDARD
    std::printf("Using std::simd");
#else
    std::printf("Using std::experimental::simd");
#endif
    std::printf(" (%d float lanes, %d double lanes)\n", SIMD_FLOAT_SIZE, int(SimdDouble::size()));
}
