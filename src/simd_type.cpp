
#include <stdio.h>


int main() {
#if defined(__AVX2__)
	printf( "Using AVX2\n");
#elif defined(__AVX__)
	printf( "Using AVX\n");
#else
	printf( "Error! No AVX!\n");
#endif
}
