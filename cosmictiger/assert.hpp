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

#ifndef CODE_GEN_CPP

#ifdef NDEBUG
#define ASSERT(a)
#else
#ifdef __CUDA_ARCH__
#define ASSERT(a) cosmictiger_cuda_assert( #a, a, __FILE__, __LINE__)
#else
#define ASSERT(a) cosmictiger_assert( #a, a, __FILE__, __LINE__)
#endif
#endif

#ifdef __CUDA_ARCH__
#define ALWAYS_ASSERT(a) cosmictiger_cuda_assert( #a, a, __FILE__, __LINE__)
#else
#define ALWAYS_ASSERT(a) cosmictiger_assert( #a, a, __FILE__, __LINE__)
#endif

#ifdef __CUDA_ARCH__
__device__ void cosmictiger_cuda_assert( const char*, bool, const char*, int);
#else
void cosmictiger_assert( const char*, bool, const char*, int);
#endif

#else

#include <cassert>

#define ASSERT assert
#define ALWAYS_ASSERT assert

#endif
