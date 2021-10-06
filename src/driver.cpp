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

#include <cosmictiger/constants.hpp>
#include <cosmictiger/cosmology.hpp>
#include <cosmictiger/drift.hpp>
#include <cosmictiger/driver.hpp>
#include <cosmictiger/defs.hpp>
#include <cosmictiger/domain.hpp>
#include <cosmictiger/groups_find.hpp>
#include <cosmictiger/group_tree.hpp>
#include <cosmictiger/groups.hpp>
#include <cosmictiger/kick.hpp>
#include <cosmictiger/kick_workspace.hpp>
#include <cosmictiger/initialize.hpp>
#include <cosmictiger/lightcone.hpp>
#include <cosmictiger/output.hpp>
#include <cosmictiger/particles.hpp>
#include <cosmictiger/power.hpp>
#include <cosmictiger/timer.hpp>
#include <cosmictiger/time.hpp>
#include <cosmictiger/view.hpp>

#include <sys/types.h>
#include <dirent.h>

HPX_PLAIN_ACTION (write_checkpoint);
HPX_PLAIN_ACTION (read_checkpoint);

double domain_time = 0.0;
double sort_time = 0.0;
double kick_time = 0.0;
double drift_time = 0.0;
double flops_per_node = 1e6;
double flops_per_particle = 1e5;
bool used_gpu;

void do_groups(int number, double scale) {
	timer total;
	total.start();
	PRINT("Doing groups\n");
	timer tm;
	tm.start();
	group_tree_create();
	tm.stop();
	PRINT("group_tree_create = %e\n", tm.read());
	tm.reset();
	tm.start();
	particles_groups_init();
	tm.stop();
	PRINT("particles_group_init = %e\n", tm.read());
	tm.reset();
	size_t active;
	int iter = 1;
	do {
		tree_id root_id;
		root_id.proc = 0;
		root_id.index = 0;
		vector<tree_id> checklist;
		checklist.push_back(root_id);
		tm.start();
		active = groups_find(root_id, std::move(checklist), get_options().link_len).get();
		tm.stop();
		PRINT("%i groups_find = %e active = %li\n", iter, tm.read(), active);
		tm.reset();
		particles_inc_group_cache_epoch();
		group_tree_inc_cache_epoch();
		iter++;
	} while (active > 0);
	tm.start();
	group_tree_destroy();
	tm.stop();
	PRINT("tree cleanup = %e\n", tm.read());
	timer reduce;
	reduce.start();
	for (int wave = 0; wave < GROUP_WAVES; wave++) {
		tm.start();
		groups_add_particles(wave, scale);
		tm.stop();
		PRINT("%i groups_add_particles %e\n", wave, tm.read());
		tm.reset();
		tm.start();
		groups_reduce(scale);
		tm.stop();
		PRINT("groups_reduce %e\n", tm.read());
		tm.reset();
	}
	reduce.stop();
	tm.start();
	groups_cull();
	tm.stop();
	PRINT("groups_cull %e\n", tm.read());
	tm.reset();
	tm.start();
	auto ngroups = groups_save(number);
	tm.stop();
	PRINT("groups_save %e\n", tm.read());
	total.stop();
	PRINT("Reduction time = %e\n", reduce.read());
	PRINT("Total time = %e\n", total.read());
	PRINT("Group count = %li of %li candidates\n", ngroups.first, ngroups.second);
	particles_groups_destroy();

}

std::pair<kick_return, tree_create_return> kick_step(int minrung, double scale, double t0, double dt_max, double theta, bool first_call, bool full_eval) {
	timer tm;
	tm.start();
	PRINT("domains_begin\n");
	domains_begin();
	PRINT("domains_end \n");
	domains_end();
	tm.stop();
	domain_time += tm.read();
	tm.reset();
	tm.start();
	tree_create_params tparams(minrung, theta);
	PRINT("Create tree %i %e\n", minrung, theta);
	auto sr = tree_create(tparams);
	const double load_max = sr.node_count * flops_per_node + std::pow(get_options().parts_dim, 3) * flops_per_particle;
	const double load = (sr.active_nodes * flops_per_node + sr.nactive * flops_per_particle) / load_max;
	tm.stop();
	sort_time += tm.read();
	tm.reset();
	tm.start();
	PRINT("nactive = %li\n", sr.nactive);
	kick_params kparams;
	kparams.dt_max = dt_max;
	kparams.node_load = flops_per_node / flops_per_particle;
	kparams.gpu = true;
	used_gpu = kparams.gpu;
	kparams.min_level = tparams.min_level;
	kparams.save_force = get_options().save_force;
	kparams.GM = get_options().GM;
	kparams.eta = get_options().eta;
	kparams.h = get_options().hsoft;
	kparams.a = scale;
	kparams.first_call = first_call;
	kparams.min_rung = minrung;
	kparams.t0 = t0;
	kparams.theta = theta;
	expansion<float> L;
	for (int i = 0; i < EXPANSION_SIZE; i++) {
		L[i] = 0.0f;
	}
	array<fixed32, NDIM> pos;
	for (int dim = 0; dim < NDIM; dim++) {
		pos[dim] = 0.f;
	}
	tree_id root_id;
	root_id.proc = 0;
	root_id.index = 0;
	vector<tree_id> checklist;
	checklist.push_back(root_id);
	PRINT("Do kick\n");
	kick_return kr = kick(kparams, L, pos, root_id, checklist, checklist, nullptr).get();
	tm.stop();
	kick_time += tm.read();
	tree_destroy();
	particles_cache_free();
	kr.nactive = sr.nactive;
	PRINT("kick done\n");
	if (min_rung == 0) {
		flops_per_node = kr.node_flops / sr.active_nodes;
		flops_per_particle = kr.part_flops / kr.nactive;
	}
	kr.load = load;
	return std::make_pair(kr, sr);
}

void do_power_spectrum(int num, double a) {
	const float h = get_options().hubble;
	const float omega_m = get_options().omega_m;
	const double box_size = get_options().code_to_cm / constants::mpc_to_cm;
	const int N = get_options().parts_dim;
	const double D1 = cosmos_growth_factor(omega_m, a) / cosmos_growth_factor(omega_m, 1.0);
	const float factor = pow(box_size, 3) / pow(N, 6) / sqr(D1);
	auto power = power_spectrum_compute();
	std::string filename = "power." + std::to_string(num) + ".txt";
	FILE* fp = fopen(filename.c_str(), "wt");
	if (fp == NULL) {
		THROW_ERROR("Unable to open %s for writing\n", filename.c_str());
	}
	for (int i = 0; i < power.size(); i++) {
		const double k = 2.0 * M_PI * i / box_size;
		const double s = sinc(M_PI * i / N);
		const double invs2 = 1.0 / sqr(s);
		fprintf(fp, "%e %e\n", k / h, power[i] * h * h * h * factor * invs2);
	}
	fclose(fp);
}

void output_time_file() {
	const double a0 = 1.0 / (1.0 + get_options().z0);
	const double tau_max = cosmos_conformal_time(a0, 1.0);
	double a = a0;
	int M = get_options().nsteps;
	int N = 64;
	const double dtau = tau_max / (M * N);
	double tau = cosmos_conformal_time(a0 * 1.0e-6, a0);
	double t = cosmos_time(a0 * 1.0e-6, a0);
	double z;
	FILE* fp = fopen("time.txt", "wt");
	if (fp == NULL) {
		THROW_ERROR("Unable to open time.txt for reading\n");
	}
	fprintf(fp, "%4s %13s %13s %13s %13s\n", "step", "real yrs", "conformal yrs", "scale", "redshift");
	for (int i = 0; i <= M; i++) {
		z = 1.0 / a - 1.0;
		const double c0 = get_options().code_to_s / constants::spyr;
		fprintf(fp, "%4i %13.4e %13.4e %13.4e %13.4e\n", i, t * c0, tau * c0, a, z);
		for (int j = 0; j < N; j++) {
			const double dadt1 = cosmos_dadtau(a);
			const double dadt2 = cosmos_dadtau(a + dadt1 * dtau);
			t += 0.5 * (2.0 * a + dadt1 * dtau) * dtau;
			a += 0.5 * (dadt1 + dadt2) * dtau;
			tau += dtau;
		}
	}
	fclose(fp);
}

void driver() {
	timer total_time;
	total_time.start();
	timer tmr;
	tmr.start();
	driver_params params;

	double a0 = 1.0 / (1.0 + get_options().z0);
	drift_return dr;
	if (get_options().read_check) {
		params = read_checkpoint();
	} else {
		output_time_file();
		initialize(get_options().z0);
		if (get_options().do_tracers) {
			particles_set_tracers();
		}
		domains_rebound();
		params.flops = 0;
		params.tau_max = cosmos_conformal_time(a0, 1.0);
		params.tau = 0.0;
		params.a = a0;
		params.cosmicK = 0.0;
		params.itime = 0;
		params.iter = 0;
		params.runtime = 0.0;
		params.total_processed = 0;
		params.years = cosmos_time(1e-6 * a0, a0) * get_options().code_to_s / constants::spyr;
		dr = drift(a0, 0.0, 0.0, 0.0, 0.0);

	}
	PRINT("ekin0 = %e\n", dr.kin);
	PRINT("tau_max = %e\n", params.tau_max);
	auto& years = params.years;
	auto& a = params.a;
	auto& tau = params.tau;
	auto& tau_max = params.tau_max;
	auto& cosmicK = params.cosmicK;
	auto& esum0 = params.esum0;
	auto& itime = params.itime;
	auto& iter = params.iter;
	auto& total_processed = params.total_processed;
	auto& runtime = params.runtime;
	double t0 = tau_max / get_options().nsteps;
	double pot;
	int this_iter = 0;
	double last_theta = -1.0;
	timer reset;
	reset.start();
	if (get_options().do_lc) {
		lc_init(tau, tau_max);
	}
	double dt;
	const auto check_lc = [&tau,&dt,&tau_max,&a](bool force) {
		if (force || lc_time_to_flush(tau + dt, tau_max)) {
			PRINT("Flushing light cone\n");
			lc_init(tau + dt, tau_max);
			lc_buffer2homes();
			lc_particle_boundaries();
			const double link_len = get_options().lc_b / get_options().parts_dim;
			lc_form_trees(tau + dt, link_len);
			PRINT( "Trees formed\n");
			size_t cnt;
			do {
				lc_particle_boundaries();
				cnt = lc_find_groups();
				PRINT( "%li\n", cnt);
			}while( cnt > 0);
			lc_groups2homes();
			lc_parts2groups(a, link_len);
		}
	};
	while (1.0 / a > get_options().z1 + 1.0) {
		//	do_groups(tau / t0 + 1e-6, a);
		tmr.stop();
		if (tmr.read() > get_options().check_freq) {
			total_time.stop();
			runtime += total_time.read();
			total_time.reset();
			total_time.start();
			write_checkpoint(params);
			tmr.reset();
			//kick_workspace::clear_buffers();
			//	return;
		}
//		PRINT("Next iteration\n");
		tmr.start();
		int minrung = min_rung(itime);
		bool full_eval = minrung == 0;
		if (full_eval) {
			const int number = tau / t0 + 0.001 / t0;
			if (get_options().do_tracers) {
				output_tracers(number);
			}
			if (get_options().do_slice) {
				output_slice(number, years);
			}
			if (get_options().do_views) {
				timer tm;
				tm.start();
				output_view(number, years);
				tm.stop();
				PRINT("View %i took %e \n", number, tm.read());
			}
		}
		double imbalance = domains_get_load_imbalance();
		if (imbalance > MAX_LOAD_IMBALANCE) {
			domains_rebound();
			imbalance = domains_get_load_imbalance();
		}
		double theta;
		const double z = 1.0 / a - 1.0;
		auto opts = get_options();
		if (z > 50.0) {
			theta = 0.4;
			opts.part_cache_line_size = 64 * 1024;
		} else if (z > 20.0) {
			theta = 0.5;
			opts.part_cache_line_size = 64 * 1024;
		} else if (z > 2.0) {
			theta = 0.65;
			opts.part_cache_line_size = 32 * 1024;
		} else {
			theta = 0.8;
			opts.part_cache_line_size = 16 * 1024;
		}
		if (last_theta != theta) {
			set_options(opts);
		}
		last_theta = theta;
		PRINT("Kicking\n");
		const double dt_max = get_options().scale_dtlim  / cosmos_dadt(a);
		auto tmp = kick_step(minrung, a, t0, dt_max, theta, tau == 0.0, full_eval);
		kick_return kr = tmp.first;
		tree_create_return sr = tmp.second;
		PRINT("Done kicking\n");
		if (full_eval) {
			kick_workspace::clear_buffers();
			pot = kr.pot * 0.5 / a;
			if (get_options().do_power) {
				do_power_spectrum(tau / t0 + 0.001 / t0, a);
			}
			if (get_options().do_groups) {
				do_groups(tau / t0 + .001 / t0, a);
			}
		}
		dt = t0 / (1 << kr.max_rung);
		const double dadt1 = a * cosmos_dadt(a);
		const double a1 = a;
		a += dadt1 * dt;
		const double dadt2 = a * cosmos_dadt(a);
		a += 0.5 * (dadt2 - dadt1) * dt;
		const double dyears = 0.5 * (a1 + a) * dt * get_options().code_to_s / constants::spyr;
		const double a2 = 2.0 / (1.0 / a + 1.0 / a1);
		PRINT("%e %e\n", a1, a);
		timer dtm;
		dtm.start();
		PRINT("Drift\n");
		dr = drift(a2, dt, tau, tau + dt, tau_max);
		if (get_options().do_lc) {
			check_lc(false);
		}
		PRINT("Drift done\n");
		dtm.stop();
		drift_time += dtm.read();
		cosmicK += dr.kin * (a - a1);
		const double esum = (a * (pot + dr.kin) + cosmicK);
		if (tau == 0.0) {
			esum0 = esum;
		}
		const double eerr = (esum - esum0) / (a * dr.kin + a * std::abs(pot) + cosmicK);
		FILE* textfp = fopen("progress.txt", "at");
		if (textfp == nullptr) {
			THROW_ERROR("unable to open progress.txt for writing\n");
		}
		if (full_eval) {
			PRINT_BOTH(textfp,
					"\n%10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s\n",
					"runtime", "i", "imbalance", "min depth", "max depth", "Z", "a", "cfl. time", "years", "dt", "pot", "kin", "cosmicK", "pot err", "min rung",
					"max rung", "active pct", "nmapped", "load", "dtime", "stime", "ktime", "dtime", "avg total", "pps", "GFLOPSins", "GFLOPS");
		}
		iter++;
		total_processed += kr.nactive;
		timer remaining_time;
		total_time.stop();
		remaining_time.start();
		runtime += total_time.read();
		double pps = total_processed / runtime;
		const auto total_flops = kr.node_flops + kr.part_flops + sr.flops + dr.flops;
		//	PRINT( "%e %e %e %e\n", kr.node_flops, kr.part_flops, sr.flops, dr.flops);
		params.flops += total_flops;
		double act_pct = 100.0 * kr.nactive / std::pow((double) get_options().parts_dim, (double) NDIM);
		PRINT_BOTH(textfp,
				"%10.3e %10li %10.3e %10i %10i %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e %10li %10li %9.2e%% %10li %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e \n",
				runtime, iter - 1, imbalance, sr.min_depth, sr.max_depth, z, a1, tau / tau_max, years, dt / tau_max, a * pot, a * dr.kin, cosmicK, eerr, minrung,
				kr.max_rung, act_pct, dr.nmapped, kr.load, domain_time, sort_time, kick_time, drift_time, runtime / iter, (double ) kr.nactive / total_time.read(),
				total_flops / total_time.read() / (1024 * 1024 * 1024), params.flops / 1024.0 / 1024.0 / 1024.0 / runtime);
		fclose(textfp);
		total_time.reset();
		remaining_time.stop();
		runtime += remaining_time.read();
		total_time.start();
		//	PRINT( "%e\n", total_time.read() - gravity_long_time - sort_time - kick_time - drift_time - domain_time);
		itime = inc(itime, kr.max_rung);
		domain_time = 0.0;
		sort_time = 0.0;
		kick_time = 0.0;
		drift_time = 0.0;
		tau += dt;
		this_iter++;
		years += dyears;
		if (this_iter > get_options().max_iter) {
			break;
		}
	}
	if (get_options().do_lc) {
		check_lc(true);
		particles_free();
		lc_flush_final();
	}
	kick_workspace::clear_buffers();
}

bool dir_exists(const char *path) {
	DIR* dir = opendir(path);
	if (dir) {
		closedir(dir);
		return true;
	} else {
		return false;
	}
}

void write_checkpoint(driver_params params) {
	if (hpx_rank() == 0) {
		PRINT("Writing checkpoint\n");
		std::string command;
		if (dir_exists("checkpoint.hello")) {
			if (dir_exists("checkpoint.goodbye")) {
				command = "rm -r checkpoint.goodbye\n";
				if (system(command.c_str()) != 0) {
					THROW_ERROR("Unable to execute %s\n", command.c_str());
				}
			}
			command = "mv checkpoint.hello checkpoint.goodbye\n";
			if (system(command.c_str()) != 0) {
				THROW_ERROR("Unable to execute %s\n", command.c_str());
			}
		}
		command = std::string("mkdir -p checkpoint.hello\n");
		if (system(command.c_str()) != 0) {
			THROW_ERROR("Unable to execute : %s\n", command.c_str());
		}
	}
	vector<hpx::future<void>> futs;
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async<write_checkpoint_action>(c, params));
	}
//	futs.push_back(hpx::threads::run_as_os_thread([&]() {
	const std::string fname = std::string("checkpoint.hello/checkpoint.") + std::to_string(hpx_rank()) + std::string(".dat");
	FILE* fp = fopen(fname.c_str(), "wb");
	fwrite(&params, sizeof(driver_params), 1, fp);
	particles_save(fp);
	//	PRINT( "parts saved\n");
	domains_save(fp);
	//PRINT( "domains saved\n");
	if (get_options().do_lc) {
		lc_save(fp);
	}
	//PRINT( "lc_saved\n");
	fclose(fp);
	PRINT("closed\n");
//	}));
	hpx::wait_all(futs.begin(), futs.end());
	if (hpx_rank() == 0) {
		PRINT("Done writing checkpoint\n");
	}
}

driver_params read_checkpoint() {
	driver_params params;
	if (hpx_rank() == 0) {
		PRINT("Reading checkpoint\n");
	}
	vector<hpx::future<driver_params>> futs;
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async<read_checkpoint_action>(c));
	}
	const std::string fname = std::string("checkpoint.hello/checkpoint.") + std::to_string(hpx_rank()) + std::string(".dat");
	FILE* fp = fopen(fname.c_str(), "rb");
	if (fp == nullptr) {
		THROW_ERROR("Unable to open %s for reading.\n", fname.c_str());
	}
	FREAD(&params, sizeof(driver_params), 1, fp);
	particles_load(fp);
	domains_load(fp);
	if (get_options().do_lc) {
		lc_load(fp);
	}
	fclose(fp);
	hpx::wait_all(futs.begin(), futs.end());
	if (hpx_rank() == 0) {
		PRINT("Done reading checkpoint\n");
		std::string command = "rm -r checkpoint.goodbye\n";
		if (system(command.c_str()) != 0) {
		}
	}
	return params;
}
