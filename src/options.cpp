/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

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

#include <fstream>
#include <iostream>

#include <hpx/modules/program_options.hpp>

#include <cosmictiger/constants.hpp>
#include <cosmictiger/defs.hpp>
#include <cosmictiger/gravity.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/safe_io.hpp>
#include <cosmictiger/tree.hpp>
#include <cosmictiger/view.hpp>

#define SHOW(opt) show(#opt, opts.opt)

void show(char const* name, bool opt) { PRINT("%-20s: %c\n", name, opt ? 'T' : 'F'); }

void show(char const* name, int opt) { PRINT("%-20s: %i\n", name, opt); }

void show(char const* name, double opt) { PRINT("%-20s: %e\n", name, opt); }

void show(char const* name, std::string opt) { PRINT("%-20s: %s\n", name, opt.c_str()); }

options global_opts;

HPX_PLAIN_ACTION(set_options);

options const& get_options() { return global_opts; }

void set_options(options const& opts) {
	std::vector<hpx::future<void>> futs;
	for (auto c : hpx_children()) {
		futs.push_back(hpx::async<set_options_action>(c, opts));
	}
	global_opts = opts;
	hpx::wait_all(futs.begin(), futs.end());
}

bool process_options(int argc, char* argv[]) {
	options opts{};
	namespace po = hpx::program_options;
	bool rc;
	po::options_description command_opts("options");

	command_opts.add_options()																														   //
		("help", "produce help message")																											   //
		("config_file", po::value<std::string>(&(opts.config_file))->default_value(""), "configuration file")										   //
		("lc_dir", po::value<std::string>(&(opts.lc_dir))->default_value("./lightcone"), "directory for lightcone output")							   //
		("gadget4_restart", po::value<std::string>(&(opts.gadget4_restart))->default_value(""), "file to read gadget4 file and output power spectrum") //
		("read_check", po::value<int>(&(opts.read_check))->default_value(-1),
		 "read checkpoint from checkpoint.hello and then move checkpoint.hello to checkpoint.goodbye (default = false)") //
		("use_glass", po::value<bool>(&(opts.use_glass))->default_value(false), "Use glass file for IC")				 //
		("create_glass", po::value<bool>(&(opts.create_glass))->default_value(false), "Create glass file")				 //
		("plummer", po::value<bool>(&(opts.plummer))->default_value(false), "Do Plummer model")							 //
		("plummerR", po::value<double>(&(opts.plummerR))->default_value(1.0e-2), "Plummer radius")						 //
		("close_pack", po::value<bool>(&(opts.close_pack))->default_value(false), "Use close packing for grid")			 //
#ifdef USE_CUDA
		("cuda", po::value<bool>(&(opts.cuda))->default_value(true), "use CUDA (default=true)") //
#else
		("cuda", po::value<bool>(&(opts.cuda))->default_value(false), "use CUDA (not enabled for this build)") //
#endif
		("check_freq", po::value<int>(&(opts.check_freq))->default_value(1000000000),
		 "time int seconds after startup to dump checkpoint \"checkpoint.hello\" and exit (default=3600)")													//
		("max_iter", po::value<int>(&(opts.max_iter))->default_value(1000000), "maximum number of time-steps (default=1000000)")							//
		("do_lc", po::value<bool>(&(opts.do_lc))->default_value(false), "do lightcone analysis (default=false)")											//
		("do_power", po::value<bool>(&(opts.do_power))->default_value(false), "do mass power spectrum analysis (default=false)")							//
		("do_groups", po::value<bool>(&(opts.do_groups))->default_value(false), "do group analysis (default=false)")										//
		("do_tracers", po::value<bool>(&(opts.do_tracers))->default_value(false), "output tracer_count number of tracer particles to SILO (default=false)") //
		("bucket_size", po::value<int>(&(opts.bucket_size))->default_value(128), "bucket size")																//
		("minrung", po::value<int>(&(opts.minrung))->default_value(0), "minimum starting rung")																//
		("tracer_count", po::value<int>(&(opts.tracer_count))->default_value(1000000), "number of tracer particles (default=1000000)")						//
		("do_slice", po::value<bool>(&(opts.do_slice))->default_value(false), "output a projection of a slice through the volume (default=false)")			//
		("do_views", po::value<bool>(&(opts.do_views))->default_value(false), "output instantaneous healpix maps (default=false)")							//
		("use_power_file", po::value<bool>(&(opts.use_power_file))->default_value(true),
		 "read initial power spectrum from power.init - must be evenly spaced in log k (default=false)")								  //
		("twolpt", po::value<bool>(&(opts.twolpt))->default_value(false), "use 2LPT initial conditions (default = true)")				  //
		("lc_b", po::value<double>(&(opts.lc_b))->default_value(0.28), "linking length for lightcone group finder")						  //
		("lc_map_size", po::value<int>(&(opts.lc_map_size))->default_value(-1), "Nside for lightcone HEALPix map")						  //
		("view_size", po::value<int>(&(opts.view_size))->default_value(1024), "view healpix Nside")										  //
		("slice_res", po::value<int>(&(opts.slice_res))->default_value(4096), "slice resolution")										  //
		("parts_dim", po::value<int>(&(opts.parts_dim))->default_value(128), "nparts^(1/3)")											  //
		("nsteps", po::value<int>(&(opts.nsteps))->default_value(64), "Number of super-timesteps")										  //
		("z0", po::value<double>(&(opts.z0))->default_value(49.0), "starting redshift")													  //
		("z1", po::value<double>(&(opts.z1))->default_value(0.0), "ending redshift")													  //
		("theta", po::value<double>(&(opts.theta))->default_value(0.8), "opening angle for test problems")								  //
		("hsoft", po::value<double>(&(opts.hsoft))->default_value(1.0 / 80.0), "dark matter softening in units of interparticle spacing") //
		("eta", po::value<double>(&(opts.eta))->default_value(0.25), "time-step criterion (default=0.141)")								  //
		("test", po::value<std::string>(&(opts.test))->default_value(""), "name of test to run")										  //
		("omega_k", po::value<double>(&(opts.omega_k))->default_value(0.0), "")															  //
		("omega_lam", po::value<double>(&(opts.omega_lam))->default_value(-1.0), "")													  //
		("omega_b", po::value<double>(&(opts.omega_b))->default_value(0.049389), "")													  //
		("omega_c", po::value<double>(&(opts.omega_c))->default_value(0.26503), "")														  //
		("Neff", po::value<double>(&(opts.Neff))->default_value(3.046), "")																  //
		("Theta", po::value<double>(&(opts.Theta))->default_value(2.7255 / 2.73), "")													  //
		("Y0", po::value<double>(&(opts.Y0))->default_value(0.2454006), "")																  //
		("sigma8", po::value<double>(&(opts.sigma8))->default_value(0.8607), "")														  //
		("hubble", po::value<double>(&(opts.hubble))->default_value(0.6732), "")														  //
		("ns", po::value<double>(&(opts.ns))->default_value(0.96605), "spectral index")													  //
		("code_to_g", po::value<double>(&(opts.code_to_g))->default_value(1.e9 / .6732), "mass resolution")								  //

		("hydro.sph", po::value<bool>(&(opts.sph))->default_value(false), "use SPH")										   //
		("hydro.chem", po::value<bool>(&(opts.chem))->default_value(false), "enable gas chemistry")							   //
		("hydro.conduction", po::value<bool>(&(opts.conduction))->default_value(false), "do conduction")					   //
		("hydro.gravity", po::value<bool>(&(opts.gravity))->default_value(true), "do gravity")								   //
		("hydro.diffusion", po::value<bool>(&(opts.diffusion))->default_value(false), "do diffusion")						   //
		("hydro.yreflect", po::value<bool>(&(opts.yreflect))->default_value(false), "Reflecting y for SPH only")			   //
		("hydro.gy", po::value<double>(&(opts.gy))->default_value(0.0), "gravitational acceleration in y direction (for SPH)") //
		("hydro.alpha0", po::value<double>(&(opts.alpha0))->default_value(0.05), "alpha0 viscosity")						   //
		("hydro.alpha1", po::value<double>(&(opts.alpha1))->default_value(1.0), "alpha1 for viscosity")						   //
		("hydro.alphaDecay", po::value<double>(&(opts.alpha_decay))->default_value(0.1), "alpha_decay time for viscosity")	   //
		("hydro.beta", po::value<double>(&(opts.beta))->default_value(2.0), "beta for viscosity")							   //
		("hydro.gamma", po::value<double>(&(opts.gamma))->default_value(5.0 / 3.0), "gamma for when chemistry is off")		   //
		("hydro.gcentral", po::value<double>(&(opts.gcentral))->default_value(0.0), "magnitude of central force")			   //
		("hydro.hcentral", po::value<double>(&(opts.hcentral))->default_value(0.01), "softening length for central force")	   //
		("hydro.viscType", po::value<int>(&(opts.visc_type))->default_value(0), "AV type - 0 = Hu 1 = M&M 2 = constant")	   //
		("hydro.hmin", po::value<double>(&(opts.hmin))->default_value(1.0e-6), "minimum smoothing length")					   //
		("hydro.hmax", po::value<double>(&(opts.hmax))->default_value(1000.0), "maximum smoothing length")					   //
		("hydro.sneighborNumber", po::value<double>(&(opts.sneighbor_number))->default_value(128), "SPH neighbor number")	   //
		("hydro.cfl", po::value<double>(&(opts.cfl))->default_value(0.2), "CFL condition")									   //
		;
	PRINT("Processing options\n");
	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, command_opts), vm);
	po::notify(vm);
	if (vm.count("help")) {
		std::cout << command_opts << "\n";
		rc = false;
	} else {
		if (!opts.config_file.empty()) {
			std::ifstream cfg_fs{vm["config_file"].as<std::string>()};
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
	if (!rc) {
		return false;
	}
	opts.glass = 0;
	opts.stars = false;
	opts.vsoft = false;
	opts.htime = true;
	opts.damping = 0.0;
	opts.sigma8_c = opts.sigma8;
	opts.sph_bucket_size = opts.bucket_size;
	opts.hmin /= opts.parts_dim;
	opts.hmax /= opts.parts_dim;
	bool const hydroTest = opts.test == "sphGravity" || opts.test == "sod" || opts.test == "blast" || opts.test == "helmholtz" || opts.test == "rt" ||
						   opts.test == "disc" || opts.test == "star" || opts.test == "plummer";
	if (hydroTest) {
		opts.sph = true;
		opts.chem = false;
		opts.stars = false;
		opts.conduction = false;
		opts.diffusion = false;
		opts.gravity = opts.test == "star" || opts.test == "plummer";
	}
	if (opts.sph) {
		opts.htime = false;
		if (!opts.cuda && opts.test != "sphGravity") {
			THROW_ERROR("SPH requires the CUDA backend\n");
		}
		if (opts.create_glass || opts.plummer || !opts.gadget4_restart.empty()) {
			THROW_ERROR("SPH cannot use the dark-matter-only glass, Plummer or Gadget loaders\n");
		}
		if (opts.do_lc || opts.do_groups) {
			THROW_ERROR("The lightcone and halo finders require equal-mass dark matter particles\n");
		}
		if (opts.stars && !opts.chem) {
			THROW_ERROR("Star formation requires hydro.chem=true\n");
		}
	} else {
		opts.chem = opts.stars = opts.conduction = opts.diffusion = false;
	}
	opts.hsoft *= self_phi();
	opts.eta /= sqrt(self_phi());
	opts.nparts = sqr(opts.parts_dim) * opts.parts_dim;
	opts.Nfour = opts.parts_dim;
	if (opts.close_pack) {
		opts.nparts *= 2;
		//		opts.Nfour *= 2;
	}
	opts.tree_cache_line_size = 65536 / sizeof(tree_node);
	opts.tree_alloc_line_size = 16 * opts.tree_cache_line_size;
	opts.part_cache_line_size = 131072 / (sizeof(fixed32) * NDIM);
	opts.save_force = opts.test == "force";
	opts.omega_m = opts.omega_b + opts.omega_c;
	opts.dm_mass = opts.sph ? opts.omega_c / opts.omega_m : 1.0;
	opts.sph_mass = opts.sph ? opts.omega_b / opts.omega_m : 0.0;
	opts.rho0_b = opts.nparts * opts.omega_b / opts.omega_m;
	opts.rho0_c = opts.nparts * opts.omega_c / opts.omega_m;

	opts.code_to_g *= constants::M0;
	opts.code_to_cm = pow(opts.code_to_g * (8.0 * M_PI) * opts.nparts * constants::G / (3.0 * opts.omega_m * sqr(constants::H0 * opts.hubble)), 1.0 / 3.0);
	opts.code_to_s = opts.code_to_cm / constants::c;
	double H = constants::H0 * opts.code_to_s;
	opts.GM = opts.omega_m * 3.0 * sqr(H * opts.hubble) / (8.0 * M_PI) / opts.nparts;
	PRINT("box_size = %e Mpc\n", opts.code_to_cm / constants::mpc_to_cm);
	opts.slice_size = std::min((20.0 / opts.hubble) / (opts.code_to_cm / constants::mpc_to_cm), 1.0);
	double const Neff = 3.086;
	double const Theta = 1.0;
	auto const nside1 = pow(M_PI * opts.nparts / 10000.0 / 12.0, 0.5);
	int nside = 1;
	while (nside < nside1 / 2.0) {
		nside *= 2;
	}
	opts.lc_map_size = nside;
	PRINT("lc map size = %e\n", nside1);

	double omega_r = 32.0 * M_PI / 3.0 * constants::G * constants::sigma * (1 + Neff * (7. / 8.0) * std::pow(4. / 11., 4. / 3.)) * std::pow(constants::H0, -2) *
					 std::pow(constants::c, -3) * std::pow(2.73 * Theta, 4) * std::pow(opts.hubble, -2);
	opts.omega_nu = omega_r * opts.Neff / (8.0 / 7.0 * std::pow(11.0 / 4.0, 4.0 / 3.0) + opts.Neff);
	opts.omega_gam = omega_r - opts.omega_nu;
	opts.omega_r = omega_r;
	if (opts.omega_lam < 0.0) {
		opts.omega_lam = 1.0 - opts.omega_m - opts.omega_r;
		opts.omega_k = 0.0;
	} else {
		opts.omega_k = 1.0 - opts.omega_m - opts.omega_r - opts.omega_lam;
	}
	opts.link_len = pow(opts.nparts, -1.0 / NDIM) * 0.28;
	opts.min_group = 20;
	opts.lc_min_group = 20;
	if (opts.create_glass) {
		opts.GM = 1.0 / opts.nparts;
		opts.code_to_cm = constants::mpc_to_cm;
		opts.hubble = 1.0;
	}
	if (opts.parts_dim % 2 == 1) {
		THROW_ERROR("parts_dim must be an even number\n");
	}
	opts.hsoft *= pow(opts.nparts, -1.0 / NDIM);
	if (opts.plummer) {
		opts.hsoft = self_phi() * pow((4.0 / 3.0 * M_PI * pow(opts.plummerR, 3.0)) / opts.nparts, 1.0 / 3.0);
	}
	SHOW(sph);
	SHOW(htime);
	SHOW(chem);
	SHOW(stars);
	PRINT("Simulation Options\n");
	PRINT("code_to_M_solar = %e\n", opts.code_to_g / 1.98e33);
	PRINT("Box size = %e cm/h\n", opts.code_to_cm / constants::mpc_to_cm * opts.hubble);
	if (opts.plummer) {
		opts.GM = 1.0 / opts.nparts;
		opts.theta = 0.4;
	}
#ifdef CHECK_MUTUAL_SORT
	opts.do_groups = true;
#endif

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
	SHOW(Y0);
	SHOW(Neff);
	SHOW(config_file);
	SHOW(test);
#ifndef USE_CUDA
	if (opts.cuda) {
		THROW_ERROR("This executable was compiled without CUDA support\n");
	}
#endif
	set_options(opts);
	view_read_view_file();
	return rc;
}
