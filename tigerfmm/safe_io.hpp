/*
 * safe_io.hpp
 *
 *  Created on: Jul 18, 2021
 *      Author: dmarce1
 */

#ifndef SAFE_IO_HPP_
#define SAFE_IO_HPP_

#include <stdio.h>


#define PRINT(...) print(__VA_ARGS__)

#define THROW_ERROR(...) throw_error(__FILE__, __LINE__, __VA_ARGS__)

template<class ...Args>
#ifdef __CUDA_ARCH__
__device__
#endif
inline void print(const char* fmt, Args ...args) {
	if( verbose ) {
		printf(fmt, args...);
#ifndef __CUDA_ARCH__
		fflush (stdout);
#endif
	}
}
template<class ...Args>
#ifdef __CUDA_ARCH__
__device__
#endif
inline void throw_error(const char* file, int line, const char* fmt, Args ...args) {
	fprintf(stderr,fmt, args...);
	fprintf(stderr,"Error in %s on line %i\n", file, line);
#ifndef __CUDA_ARCH__
	fflush (stderr);
	abort();
#else
	__trap();
#endif
}


#endif /* SAFE_IO_HPP_ */
