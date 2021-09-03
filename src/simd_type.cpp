
#include <stdio.h>


int main() {
#if defined(__AVX2__)
	printf( "Using AVX2");
#elif defined(__AVX__)
	printf( "Using AVX");
#else
	printf( "Error! No AVX!");
#endif
}
