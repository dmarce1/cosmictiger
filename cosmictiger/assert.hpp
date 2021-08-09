
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
