#include <mpi.h>


int main( int argc, char **argv)
{
    void *v;
    int flag;
    int vval;
    int rank, size;

    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &size );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    MPI_Comm_get_attr( MPI_COMM_WORLD, MPI_TAG_UB, &v, &flag );
    if (!flag) {
        fprintf( stderr, "Could not get TAG_UB\n" );fflush(stderr);
    }
    else {
        vval = *(int*)v;
        fprintf( stderr, "value (%d) for TAG_UB\n", vval );fflush(stderr);
    }

  
    MPI_Finalize( );
    return 0;
}

