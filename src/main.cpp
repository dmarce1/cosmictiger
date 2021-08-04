#include <cosmictiger/driver.hpp>
#include <cosmictiger/ewald_indices.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/particles.hpp>
#include <cosmictiger/simd.hpp>
#include <cosmictiger/test.hpp>
#include <cosmictiger/unordered_set_ts.hpp>

int hpx_main(int argc, char *argv[]) {
	hpx_init();
	ewald_const::init();
	if (process_options(argc, argv)) {
		if (get_options().test != "") {
			test(get_options().test);
		} else {
			driver();
		}
	}
	particles_destroy();
	return hpx::finalize();
}

#ifndef HPX_LITE

int main(int argc, char *argv[]) {
	std::vector<std::string> cfg = {"hpx.commandline.allow_unknown=1"};
	cfg.push_back("hpx.stacks.small_size=262144");
	hpx::init(argc, argv, cfg);
}

#endif
