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


#ifndef SAFE_IO_HPP_
#define SAFE_IO_HPP_

#include <cosmictiger/assert.hpp>
#include <stdio.h>
#include <stdlib.h>

#define PRINT(...) print(__VA_ARGS__)

#define PRINT_BOTH(fp, ...) print_both(fp, __VA_ARGS__)

#define THROW_ERROR(...) throw_error(__FILE__, __LINE__, __VA_ARGS__)

template<class ...Args>
#ifdef __CUDA_ARCH__
__device__
#endif
inline void print(const char* fmt, Args ...args) {
	printf(fmt, args...);
#ifndef __CUDA_ARCH__
	fflush(stdout);
#endif
}

template<class ...Args>
inline void print_both(FILE* fp, const char* fmt, Args ...args) {
	printf(fmt, args...);
	fprintf(fp, fmt, args...);
	fflush(stdout);
	fflush(fp);
}



#ifdef __CUDA_ARCH__
__device__
#endif
inline void print(const char* str) {
	printf("%s", str);
#ifndef __CUDA_ARCH__
	fflush(stdout);
#endif
}
template<class ...Args>
#ifdef __CUDA_ARCH__
__device__
#endif
inline void throw_error(const char* file, int line, const char* fmt, Args ...args) {
	printf(fmt, args...);
	printf("Error in %s on line %i\n", file, line);
#ifndef __CUDA_ARCH__
	fflush(stdout);
	ALWAYS_ASSERT(false);
#else
	__trap();
#endif
}


inline void throw_error(const char* file, int line, const char* str) {
	printf("%s", str);
	printf("Error in %s on line %i\n", file, line);
#ifndef __CUDA_ARCH__
	fflush(stdout);
	ALWAYS_ASSERT(false);
#else
	__trap();
#endif
}


#define FREAD(a,b,c,d) __safe_fread(a,b,c,d,__LINE__,__FILE__)

static void __safe_fread(void* src, size_t size, size_t count, FILE* fp, int line, const char* file) {
	auto read = fread(src, size, count, fp);
	if (read != count) {
		PRINT("Attempt to read %li elements of size %li in %s on line %i failed - only %li elements read.\n", count, size,
				file, line, read);
		abort();
	}
}


#define ASPRINTF(...) if( asprintf(__VA_ARGS__) == 0 ) printf( "asprintf failed\n")
#endif /* SAFE_IO_HPP_ */
