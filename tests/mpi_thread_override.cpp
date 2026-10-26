/* CosmicTiger - MPI startup regression; BSL-1.0. */

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <mpi.h>

namespace detail {

int reportedThreadLevel(int actual) {
    char const* level = std::getenv("COSMICTIGER_TEST_MPI_THREAD_LEVEL");
    if (level == nullptr) {
        return actual;
    }
    int rank = -1;
    PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
    char const* targetRank = std::getenv("COSMICTIGER_TEST_MPI_THREAD_RANK");
    if (targetRank != nullptr && rank != std::atoi(targetRank)) {
        return actual;
    }
    int requested = MPI_THREAD_MULTIPLE;
    if (std::strcmp(level, "MPI_THREAD_SINGLE") == 0) {
        requested = MPI_THREAD_SINGLE;
    } else if (std::strcmp(level, "MPI_THREAD_FUNNELED") == 0) {
        requested = MPI_THREAD_FUNNELED;
    } else if (std::strcmp(level, "MPI_THREAD_SERIALIZED") == 0) {
        requested = MPI_THREAD_SERIALIZED;
    }
    // Never claim stronger support than the real MPI implementation supplies.
    return std::min(actual, requested);
}

}    // namespace detail

extern "C" int MPI_Init_thread(int* argc, char*** argv, int required, int* provided) {
    char const* expectedComponents = std::getenv("COSMICTIGER_TEST_HWLOC_COMPONENTS");
    char const* actualComponents = std::getenv("HWLOC_COMPONENTS");
    if (expectedComponents != nullptr &&
        (actualComponents == nullptr || std::strcmp(actualComponents, expectedComponents) != 0)) {
        std::fprintf(stderr,
            "MPI startup probe expected HWLOC_COMPONENTS=%s before MPI initialization; got %s\n",
            expectedComponents, actualComponents == nullptr ? "(unset)" : actualComponents);
        return MPI_ERR_ARG;
    }
    if (required != MPI_THREAD_MULTIPLE) {
        std::fprintf(stderr, "MPI startup probe expected MPI_THREAD_MULTIPLE\n");
        return MPI_ERR_ARG;
    }
    int const result = PMPI_Init_thread(argc, argv, required, provided);
    if (result == MPI_SUCCESS) {
        int rank = -1;
        PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
        std::fprintf(stderr, "MPI startup probe initialized rank %d\n", rank);
        *provided = detail::reportedThreadLevel(*provided);
    }
    return result;
}

extern "C" int MPI_Query_thread(int* provided) {
    int const result = PMPI_Query_thread(provided);
    if (result == MPI_SUCCESS) {
        *provided = detail::reportedThreadLevel(*provided);
    }
    return result;
}

extern "C" int MPI_Finalize() {
    int rank = -1;
    PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int const result = PMPI_Finalize();
    if (result == MPI_SUCCESS) {
        std::fprintf(stderr, "MPI startup probe finalized rank %d\n", rank);
    }
    return result;
}
