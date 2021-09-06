constexpr bool verbose = true;
#include <cosmictiger/constants.hpp>
#include <cosmictiger/defs.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/safe_io.hpp>

#ifdef HPX_LITE
#include <boost/program_options.hpp>
#endif

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
		futs.push_back(hpx::async < set_options_action > (c, opts));
	}
	global_opts = opts;
	hpx::wait_all(futs.begin(), futs.end());
}

bool process_options(int argc, char *argv[]) {
	options opts;
#ifdef HPX_LITE
	namespace po = boost::program_options;
#else
	namespace po = hpx::program_options;
#endif
	bool rc;
	po::options_description command_opts("options");

	command_opts.add_options()                                                                       //
	("help", "produce help message")                                                                 //
	("config_file", po::value < std::string > (&(opts.config_file))->default_value(""), "configuration file") //
	("check_num", po::value<int>(&(opts.check_num))->default_value(-1), "number of checkpoint file to read") //
#ifdef USE_CUDA
	("cuda", po::value<bool>(&(opts.cuda))->default_value(true), "use CUDA") //
#else
	("cuda", po::value<bool>(&(opts.cuda))->default_value(false), "use CUDA") //
#endif
	("check_freq", po::value<int>(&(opts.check_freq))->default_value(3600), "checkpoint frequency in seconds") //
	("max_iter", po::value<int>(&(opts.max_iter))->default_value(1000000), "maximum number of iterations") //
	("do_lc", po::value<bool>(&(opts.do_lc))->default_value(false), "do lightcone") //
	("do_power", po::value<bool>(&(opts.do_power))->default_value(false), "do mass power spectrum") //
	("do_groups", po::value<bool>(&(opts.do_groups))->default_value(false), "do groups") //
	("do_tracers", po::value<bool>(&(opts.do_tracers))->default_value(false), "output tracer_count number of tracer particles to SILO") //
	("do_sample", po::value<bool>(&(opts.do_sample))->default_value(false), "output the box bounded by (0,sample_dim) in all directions to SILO") //
	("do_views", po::value<bool>(&(opts.do_views))->default_value(false), "instantaneous maps") //
	("twolpt", po::value<bool>(&(opts.twolpt))->default_value(true), "Use 2LPT initial conditions") //
#ifdef USE_CUDA
	("tree_cache_line_size", po::value<int>(&(opts.tree_cache_line_size))->default_value(512), "size of tree cache line") //
	("part_cache_line_size", po::value<int>(&(opts.part_cache_line_size))->default_value(8 * 1024), "size of particle cache line") //
#else
	("tree_cache_line_size", po::value<int>(&(opts.tree_cache_line_size))->default_value(1024), "size of tree cache line") //
	("part_cache_line_size", po::value<int>(&(opts.part_cache_line_size))->default_value(64 * 1024), "size of particle cache line") //
#endif
	("tracer_count", po::value<int>(&(opts.tracer_count))->default_value(1000000), "number of tracer particles") //
	("lc_b", po::value<double>(&(opts.lc_b))->default_value(0.2), "b for lightcone group finder") //
	("view_size", po::value<int>(&(opts.view_size))->default_value(1000), "view healpix Nside") //
	("parts_dim", po::value<int>(&(opts.parts_dim))->default_value(128), "nparts^(1/3)") //
	("z0", po::value<double>(&(opts.z0))->default_value(49.0), "starting redshift") //
	("z1", po::value<double>(&(opts.z1))->default_value(0.0), "ending redshift") //
	("sample_dim", po::value<double>(&(opts.sample_dim))->default_value(0.01), "size of box to output to SILO") //
	("theta", po::value<double>(&(opts.theta))->default_value(0.8), "opening angle for test problems") //
	("test", po::value < std::string > (&(opts.test))->default_value(""), "name of test to run") //
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

	opts.save_force = opts.test == "force";
	opts.hsoft = 1.0 / 25.0 / opts.parts_dim;
	opts.eta = 0.2 / sqrt(2);
	opts.hubble = 0.7;
	opts.sigma8 = 0.84;
	opts.code_to_cm = 7.108e26 * opts.parts_dim / 1024.0 / opts.hubble;
	PRINT("box_size = %e Mpc\n", opts.code_to_cm / constants::mpc_to_cm);
	opts.code_to_s = opts.code_to_cm / constants::c;
	opts.omega_m = 0.3;
	double H = constants::H0 * opts.code_to_s;
	const size_t nparts = pow(opts.parts_dim, NDIM);
	const double Neff = 3.086;
	const double Theta = 1.0;
	opts.GM = opts.omega_m * 3.0 * sqr(H * opts.hubble) / (8.0 * M_PI) / nparts;
	opts.code_to_g = 3.0 * opts.omega_m * sqr(constants::H0 * opts.hubble) / (8.0 * M_PI) / nparts / constants::G * std::pow(opts.code_to_cm, 3);
	double omega_r = 32.0 * M_PI / 3.0 * constants::G * constants::sigma * (1 + Neff * (7. / 8.0) * std::pow(4. / 11., 4. / 3.)) * std::pow(constants::H0, -2)
			* std::pow(constants::c, -3) * std::pow(2.73 * Theta, 4) * std::pow(opts.hubble, -2);
	opts.omega_r = omega_r;
	opts.link_len = 1.0 / opts.parts_dim / 5.0;
	opts.min_group = 20;
	opts.lc_min_group = 20;

	if (opts.parts_dim % 2 == 1) {
		THROW_ERROR("parts_dim must be an even number\n");
	}
	PRINT("Simulation Options\n");
	PRINT("code_to_M_solar = %e\n", opts.code_to_g / 1.98e33);
	PRINT("Box size = %e\n", opts.code_to_cm / constants::mpc_to_cm);
	SHOW_STRING(config_file);
	SHOW(hsoft);
	SHOW(parts_dim);
	SHOW_STRING(save_force);
	SHOW_STRING(test);
	SHOW(tree_cache_line_size);
#ifndef USE_CUDA
	if (opts.cuda) {
		THROW_ERROR("This executable was compiled without CUDA support\n");
	}
#endif

	set_options(opts);

	return rc;
}

