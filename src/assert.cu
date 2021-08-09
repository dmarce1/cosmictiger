#include <cosmictiger/assert.hpp>
#include <cosmictiger/safe_io.hpp>

#ifdef __CUDA_ARCH__
__device__ void cosmictiger_cuda_assert( const char* cond_str, bool cond, const char* filename, int line) {
		if (!cond) {
			PRINT( "Condition \"%s\" failed in %s on line %i in CUDA_ARCH\n", cond_str, filename, line);
			__trap();
		}

}
#endif
