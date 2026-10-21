/* CosmicTiger - direct execution of HPX checks without GoogleTest; BSL-1.0. */

#include <tests/application_test.hpp>

void runApplicationTest(char const*, void (*run)()) {
    run();
}
