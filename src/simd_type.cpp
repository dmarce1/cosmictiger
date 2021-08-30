
#include <stdio.h>


int main() {
#ifdef __AVX512F__
	printf( "Using AVX512\n");
#elif defined(__AVX2__)
	printf( "Using AVX2\n");
#elif defined(__AVX__)
	printf( "Using AVX\n");
#else
	printf( "Error! No AVX!\n");
#endif
}
