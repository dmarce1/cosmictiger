#include <tigerfmm/fmm_kernels.hpp>
#include <tigerfmm/hpx.hpp>
#include <tigerfmm/options.hpp>
#include <tigerfmm/particles.hpp>
#include <tigerfmm/test.hpp>

int hpx_main(int argc, char *argv[]) {
	hpx_init();
	if(process_options(argc,argv)) {
		if( get_options().test != "") {
			test(get_options().test);
		}
	}
	particles_destroy();
	return hpx::finalize();
}

#ifdef USE_HPX
int main(int argc, char *argv[]) {
	std::vector<std::string> cfg = {"hpx.commandline.allow_unknown=1"};
	cfg.push_back("hpx.stacks.small_size=262144");
	hpx::init(argc, argv, cfg);
}

#endif
