/*****************
 *
 * THIS (slightly modifed) CODE TAKEN FROM
 * https://wagonhelm.github.io/articles/2018-03/detecting-cuda-capability-with-cmake
 *
 */


#include <stdio.h>

int main(int argc, char **argv){
    cudaDeviceProp dP;
    float min_cc = 3.0;

    int rc = cudaGetDeviceProperties(&dP, 0);
    if(rc != cudaSuccess) {
        cudaError_t error = cudaGetLastError();
        printf("CUDA error: %s", cudaGetErrorString(error));
        return rc; /* Failure */
    }
  /*  if( dP.major >= 8 ) {
    	dP.major = 7;
    	dP.minor = 5;
    }*/
    if((dP.major+(dP.minor/10)) < min_cc) {
        printf("Min Compute Capability of %2.1f required:  %d.%d found\n Not Building CUDA Code", min_cc, dP.major, dP.minor);
        return 1; /* Failure */
    } else {
        printf("%d%d", dP.major, dP.minor);
        return 0; /* Success */
    }
}

