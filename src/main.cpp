#include <tigerfmm/driver.hpp>
#include <tigerfmm/ewald_indices.hpp>
#include <tigerfmm/hpx.hpp>
#include <tigerfmm/options.hpp>
#include <tigerfmm/particles.hpp>
#include <tigerfmm/simd.hpp>
#include <tigerfmm/test.hpp>
#include <tigerfmm/unordered_set_ts.hpp>

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

int main(int argc, char *argv[]) {
	std::vector<std::string> cfg = {"hpx.commandline.allow_unknown=1"};
	cfg.push_back("hpx.stacks.small_size=262144");
	hpx::init(argc, argv, cfg);
}

