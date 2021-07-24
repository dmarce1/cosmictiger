constexpr bool verbose = true;
#include <tigerfmm/constants.hpp>
#include <tigerfmm/defs.hpp>
#include <tigerfmm/hpx.hpp>
#include <tigerfmm/math.hpp>
#include <tigerfmm/options.hpp>
#include <tigerfmm/safe_io.hpp>

#include <boost/program_options.hpp>
#include <fstream>

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

	opts.tree_cache_line_size = 512;
	opts.part_cache_line_size = 32 * 1024;
	opts.save_force = opts.test == "force";
	opts.hsoft = 1.0 / 25.0 / opts.parts_dim;
	opts.eta = 0.2 / sqrt(2);
	opts.hubble = 0.7;
	opts.sigma8 = 0.84;
	opts.code_to_cm = 7.108e26 * opts.parts_dim / 1024.0 / opts.hubble;
	opts.code_to_s = opts.code_to_cm / constants::c;
	opts.code_to_g = 1.989e33;
	opts.omega_m = 0.3;
	double H = constants::H0 * opts.code_to_s;
	const size_t nparts = pow(opts.parts_dim, NDIM);
	const double Neff = 3.086;
	const double Theta = 1.0;
	opts.GM = opts.omega_m * 3.0 * sqr(H * opts.hubble) / (8.0 * M_PI) / nparts;
	double omega_r = 32.0 * M_PI / 3.0 * constants::G * constants::sigma
			* (1 + Neff * (7. / 8.0) * std::pow(4. / 11., 4. / 3.)) * std::pow(constants::H0, -2)
			* std::pow(constants::c, -3) * std::pow(2.73 * Theta, 4) * std::pow(opts.hubble, -2);
	opts.omega_r = omega_r;
	opts.z0 = 49.0;

	PRINT("Simulation Options\n");
	SHOW_STRING(config_file);
	SHOW(hsoft);
	SHOW(parts_dim);
	SHOW_STRING(save_force);
	SHOW_STRING(test);
	SHOW(tree_cache_line_size);

	set_options(opts);

	return rc;
}

