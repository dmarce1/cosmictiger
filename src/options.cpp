constexpr bool verbose = true;

#ifndef USE_HPX
#include <boost/program_options.hpp>
#include <fstream>
#endif

#include <tigerfmm/defs.hpp>
#include <tigerfmm/hpx.hpp>
#include <tigerfmm/options.hpp>
#include <tigerfmm/safe_io.hpp>

#define SHOW( opt ) PRINT( "%s = %e\n",  #opt, (double) opts.opt)
#define SHOW_STRING( opt ) std::cout << std::string( #opt ) << " = " << opts.opt << '\n';

options global_opts;

HPX_PLAIN_ACTION (set_options);

const options& get_options() {
	return global_opts;
}

void set_options(const options& opts) {
	std::vector<hpx::future<void>> futs;
	for (auto c : hpx_children()) {
		futs.push_back(hpx::async<set_options_action>(c, opts));
	}
	global_opts = opts;
	hpx::wait_all(futs.begin(), futs.end());
}

bool process_options(int argc, char *argv[]) {
	options opts;
#ifdef USE_HPX
	namespace po = hpx::program_options;
#else
	namespace po = boost::program_options;
#endif
	bool rc;
	po::options_description command_opts("options");

	command_opts.add_options()                                                                       //
	("help", "produce help message")                                                                 //
	("config_file", po::value<std::string>(&(opts.config_file))->default_value(""), "configuration file") //
	("parts_dim", po::value<int>(&(opts.parts_dim))->default_value(128), "nparts^(1/3)") //
	("test", po::value<std::string>(&(opts.test))->default_value(""), "name of test to run") //
			;

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, command_opts), vm);
	po::notify(vm);
	if (vm.count("help")) {
		std::cout << command_opts << "\n";
		rc = false;
	} else {
		if (!opts.config_file.empty()) {
			std::ifstream cfg_fs { vm["config_file"].as<std::string>() };
			if (cfg_fs) {
				po::store(po::parse_config_file(cfg_fs, command_opts), vm);
				rc = true;
			} else {
				PRINT("Configuration file %s not found!\n", opts.config_file.c_str());
				return false;
			}
		} else {
			rc = true;
		}
	}

	if (rc) {
		po::notify(vm);
	}

	opts.hsoft = 1.0 / opts.parts_dim / 25.0;

	PRINT("Simulation Options\n");
	SHOW_STRING(config_file);
	SHOW(hsoft);
	SHOW(parts_dim);
	SHOW_STRING(test);

	set_options(opts);

	return rc;
}

