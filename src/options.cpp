/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2021  Dominic C. Marcello

 This program is free software; you can redistribute it and/or
 modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation; either version 2
 of the License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */

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
#include <iostream>
#include <fstream>

#define SHOW( opt ) show(#opt, opts.opt)

void show(const char* name, bool opt) {
	PRINT("%-20s: %c\n", name, opt ? 'T' : 'F');
}

void show(const char* name, int opt) {
	PRINT("%-20s: %i\n", name, opt);
}

void show(const char* name, double opt) {
	PRINT("%-20s: %e\n", name, opt);
}

void show(const char* name, std::string opt) {
	PRINT("%-20s: %e\n", name, opt.c_str());
}

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
	("read_check", po::value<bool>(&(opts.read_check))->default_value(false),
			"read checkpoint from checkpoint.hello and then move checkpoint.hello to checkpoint.goodbye (default = false)") //
#ifdef USE_CUDA
	("cuda", po::value<bool>(&(opts.cuda))->default_value(true), "use CUDA (default=true)") //
#else
	("cuda", po::value<bool>(&(opts.cuda))->default_value(false), "use CUDA (not enabled for this build)") //
#endif
	("check_freq", po::value<int>(&(opts.check_freq))->default_value(3600),
			"time int seconds after startup to dump checkpoint \"checkpoint.hello\" and exit (default=3600)") //
	("max_iter", po::value<int>(&(opts.max_iter))->default_value(1000000), "maximum number of time-steps (default=1000000)") //
	("do_lc", po::value<bool>(&(opts.do_lc))->default_value(false), "do lightcone analysis (default=false)") //
	("do_power", po::value<bool>(&(opts.do_power))->default_value(false), "do mass power spectrum analysis (default=false)") //
	("do_groups", po::value<bool>(&(opts.do_groups))->default_value(false), "do group analysis (default=false)") //
	("do_tracers", po::value<bool>(&(opts.do_tracers))->default_value(false), "output tracer_count number of tracer particles to SILO (default=false)") //
	("tracer_count", po::value<int>(&(opts.tracer_count))->default_value(1000000), "number of tracer particles (default=1000000)") //
	("do_slice", po::value<bool>(&(opts.do_slice))->default_value(false), "output a projection of a slice through the volume (default=false)") //
	("do_views", po::value<bool>(&(opts.do_views))->default_value(false), "output instantaneous healpix maps (default=false)") //
	("use_power_file", po::value<bool>(&(opts.use_power_file))->default_value(true),
			"read initial power spectrum from power.init - must be evenly spaced in log k (default=false)") //
	("twolpt", po::value<bool>(&(opts.twolpt))->default_value(true), "use 2LPT initial conditions (default = true)") //
	("scale_dtlim", po::value<double>(&(opts.scale_dtlim))->default_value(0.02), "maximum change in scale factor time-step limit") //
	("lc_b", po::value<double>(&(opts.lc_b))->default_value(0.2), "linking length for lightcone group finder") //
	("lc_map_size", po::value<int>(&(opts.lc_map_size))->default_value(2048), "Nside for lightcone HEALPix map") //
	("view_size", po::value<int>(&(opts.view_size))->default_value(1024), "view healpix Nside") //
	("slice_res", po::value<int>(&(opts.slice_res))->default_value(4096), "slice resolution") //
	("parts_dim", po::value<int>(&(opts.parts_dim))->default_value(128), "nparts^(1/3)") //
	("nsteps", po::value<int>(&(opts.nsteps))->default_value(64), "Number of super-timesteps") //
	("z0", po::value<double>(&(opts.z0))->default_value(49.0), "starting redshift") //
	("z1", po::value<double>(&(opts.z1))->default_value(0.0), "ending redshift") //
	("theta", po::value<double>(&(opts.theta))->default_value(0.8), "opening angle for test problems") //
	("eta", po::value<double>(&(opts.eta))->default_value(0.2), "time-step criterion (default=0.2)") //
	("test", po::value < std::string > (&(opts.test))->default_value(""), "name of test to run") //
	("omega_b", po::value<double>(&(opts.omega_b))->default_value(0.049389), "") //
	("omega_c", po::value<double>(&(opts.omega_c))->default_value(0.26503), "") //
	("Neff", po::value<double>(&(opts.Neff))->default_value(3.046), "") //
	("Theta", po::value<double>(&(opts.Theta))->default_value(2.7255 / 2.73), "") //
	("Y", po::value<double>(&(opts.Y))->default_value(0.2454006), "") //
	("sigma8", po::value<double>(&(opts.sigma8))->default_value(0.8120), "") //
	("hubble", po::value<double>(&(opts.hubble))->default_value(0.6732), "") //
	("ns", po::value<double>(&(opts.ns))->default_value(0.96605), "spectral index") //

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
#ifdef USE_CUDA
	opts.tree_cache_line_size = 512;
	opts.part_cache_line_size = 8 * 1024; //
#else
	opts.tree_cache_line_size = 1024;
	opts.part_cache_line_size = 64 * 1024;
#endif
	opts.save_force = opts.test == "force";
	opts.hsoft = 1.0 / 50.0 / opts.parts_dim;
	opts.code_to_cm = 7.108e26 * opts.parts_dim / 1024.0 / opts.hubble;
	PRINT("box_size = %e Mpc\n", opts.code_to_cm / constants::mpc_to_cm);
	opts.code_to_s = opts.code_to_cm / constants::c;
	opts.omega_m = opts.omega_b + opts.omega_c;
	opts.slice_size = std::min((20.0 / opts.hubble) / (opts.code_to_cm / constants::mpc_to_cm), 1.0);
	double H = constants::H0 * opts.code_to_s;
	const size_t nparts = pow(opts.parts_dim, NDIM);
	const double Neff = 3.086;
	const double Theta = 1.0;
	opts.GM = opts.omega_m * 3.0 * sqr(H * opts.hubble) / (8.0 * M_PI) / nparts;
	opts.code_to_g = 3.0 * opts.omega_m * sqr(constants::H0 * opts.hubble) / (8.0 * M_PI) / nparts / constants::G * std::pow(opts.code_to_cm, 3);
	double omega_r = 32.0 * M_PI / 3.0 * constants::G * constants::sigma * (1 + Neff * (7. / 8.0) * std::pow(4. / 11., 4. / 3.)) * std::pow(constants::H0, -2)
			* std::pow(constants::c, -3) * std::pow(2.73 * Theta, 4) * std::pow(opts.hubble, -2);
	opts.omega_nu = omega_r * opts.Neff / (8.0 / 7.0 * std::pow(11.0 / 4.0, 4.0 / 3.0) + opts.Neff);
	opts.omega_gam = omega_r - opts.omega_nu;
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

	SHOW(check_freq);
	SHOW(code_to_cm);
	SHOW(code_to_g);
	SHOW(code_to_s);
	SHOW(cuda);
	SHOW(do_lc);
	SHOW(do_groups);
	SHOW(do_power);
	SHOW(do_slice);
	SHOW(do_tracers);
	SHOW(do_views);
	SHOW(eta);
	SHOW(GM);
	SHOW(hsoft);
	SHOW(hubble);
	SHOW(lc_b);
	SHOW(lc_min_group);
	SHOW(lc_map_size);
	SHOW(link_len);
	SHOW(max_iter);
	SHOW(min_group);
	SHOW(omega_b);
	SHOW(omega_c);
	SHOW(omega_gam);
	SHOW(omega_m);
	SHOW(omega_nu);
	SHOW(omega_r);
	SHOW(part_cache_line_size);
	SHOW(parts_dim);
	SHOW(read_check);
	SHOW(save_force);
	SHOW(scale_dtlim);
	SHOW(sigma8);
	SHOW(slice_res);
	SHOW(slice_size);
	SHOW(Theta);
	SHOW(theta);
	SHOW(tracer_count);
	SHOW(tree_cache_line_size);
	SHOW(twolpt);
	SHOW(use_power_file);
	SHOW(view_size);
	SHOW(z0);
	SHOW(z1);

	SHOW(ns);
	SHOW(Y);
	SHOW(Neff);
	SHOW(config_file);
	SHOW(test);
#ifndef USE_CUDA
	if (opts.cuda) {
		THROW_ERROR("This executable was compiled without CUDA support\n");
	}
#endif

	set_options(opts);

	return rc;
}

