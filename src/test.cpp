constexpr bool verbose = true;

#include <tigerfmm/domain.hpp>
#include <tigerfmm/particles.hpp>
#include <tigerfmm/safe_io.hpp>
#include <tigerfmm/test.hpp>
#include <tigerfmm/timer.hpp>


static void domain_test() {
	timer tm;

	tm.start();
	particles_random_init();
	tm.stop();
	PRINT( "particles_random_init: %e s\n", tm.read());
	tm.reset();

	tm.start();
	domains_begin();
	tm.stop();
	PRINT( "domains_begin: %e s\n", tm.read());
	tm.reset();

	tm.start();
	domains_end();
	tm.stop();
	PRINT( "domains_end: %e s\n", tm.read());
	tm.reset();

}


void test(std::string test) {
	if( test == "domain" ) {
		domain_test();
	} else {
		THROW_ERROR( "test %s is not known\n", test.c_str());
	}

}
