#include <cosmictiger/constants.hpp>
#include <cosmictiger/drift.hpp>
#include <cosmictiger/driver.hpp>
#include <cosmictiger/defs.hpp>
#include <cosmictiger/domain.hpp>
#include <cosmictiger/kick.hpp>
#include <cosmictiger/initialize.hpp>
#include <cosmictiger/map.hpp>
#include <cosmictiger/particles.hpp>
#include <cosmictiger/timer.hpp>
#include <cosmictiger/time.hpp>

HPX_PLAIN_ACTION(write_checkpoint);
HPX_PLAIN_ACTION(read_checkpoint);

double cosmos_dadtau(double a) {
	const auto H = constants::H0 * get_options().code_to_s * get_options().hubble;
	const auto omega_m = get_options().omega_m;
	const auto omega_r = get_options().omega_r;
	const auto omega_lambda = 1.0 - omega_m - omega_r;
	return H * a * a * std::sqrt(omega_r / (a * a * a * a) + omega_m / (a * a * a) + omega_lambda);
}

double cosmos_age(double a0) {
	double a = a0;
	double t = 0.0;
	while (a < 1.0) {
		const double dadt1 = cosmos_dadtau(a);
		const double dt = (a / dadt1) * 1.e-5;
		const double dadt2 = cosmos_dadtau(a + dadt1 * dt);
		a += 0.5 * (dadt1 + dadt2) * dt;
		t += dt;
	}
	return t;
}

double domain_time = 0.0;
double sort_time = 0.0;
double kick_time = 0.0;
double drift_time = 0.0;
double flops_per_node = 1e6;
double flops_per_particle = 1e5;
bool used_gpu;

kick_return kick_step(int minrung, double scale, double t0, double theta, bool first_call, bool full_eval) {
	timer tm;
	tm.start();
//	PRINT( "Domains\n", minrung, theta);
	domains_begin();
	domains_end();
	tm.stop();
	domain_time += tm.read();
	tm.reset();
	tm.start();
	tree_create_params tparams(minrung, theta);
//	PRINT( "Create tree %i %e\n", minrung, theta);
	auto sr = tree_create(tparams);
	const double load_max = sr.node_count * flops_per_node + particles_size() * flops_per_particle;
	const double load = (sr.active_nodes * flops_per_node + sr.nactive * flops_per_particle) / load_max;
	tm.stop();
	sort_time += tm.read();
	tm.reset();
	tm.start();
//	PRINT( "nactive = %li\n", sr.nactive);
	kick_params kparams;
	kparams.gpu = load > GPU_LOAD_MIN;
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
//	PRINT( "Do kick\n");
	kick_return kr = kick(kparams, L, pos, root_id, checklist, checklist, nullptr).get();
	tm.stop();
	kick_time += tm.read();
	tree_destroy();
	particles_cache_free();
	kr.nactive = sr.nactive;
//	PRINT( "kick done\n");
	if (min_rung == 0) {
		flops_per_node = kr.node_flops / sr.active_nodes;
		flops_per_particle = kr.part_flops / kr.nactive;
	}
	kr.load = load;
	return kr;
}

void driver() {
	driver_params params;
	double a0 = 1.0 / (1.0 + get_options().z0);
	if (get_options().check_num >= 0) {
		params = read_checkpoint(get_options().check_num);
	} else {
		initialize();
		params.flops = 0;
		params.tau_max = cosmos_age(a0);
		params.tau = 0.0;
		params.a = a0;
		params.cosmicK = 0.0;
		params.itime = 0;
		params.iter = 0;
		params.runtime = 0.0;
		params.total_processed = 0;
	}
	auto& a = params.a;
	auto& tau = params.tau;
	auto& tau_max = params.tau_max;
	auto& cosmicK = params.cosmicK;
	auto& esum0 = params.esum0;
	auto& itime = params.itime;
	auto& iter = params.iter;
	auto& total_processed = params.total_processed;
	auto& runtime = params.runtime;
	double t0 = tau_max / 100.0;
	double pot;
	timer tmr;
	tmr.start();
	timer total_time;
	total_time.start();
	int this_iter = 0;
	double last_theta = -1.0;
	timer reset;
	reset.start();
	if (get_options().do_map) {
		map_init(tau_max);
	}
	while (tau < tau_max) {
		reset.stop();
		if (false && reset.read() > 60) {
			PRINT("Cycling device\n");
			reset.reset();
			reset.start();
			cuda_cycle_devices();
			reset.stop();
			PRINT("Took %e s\n", reset.read());
			reset.reset();
		}
		reset.start();
		tmr.stop();
		if (tmr.read() > get_options().check_freq) {
			write_checkpoint(params);
			tmr.reset();
		}
		tmr.start();
		int minrung = min_rung(itime);
		bool full_eval = minrung == 0;
		double theta;
		const double z = 1.0 / a - 1.0;
		if (z > 20.0) {
			theta = 0.5;
		} else if (z > 2.0) {
			theta = 0.65;
		} else {
			theta = 0.8;
		}
		if (last_theta != theta) {
			kick_reset_all_list_sizes();
		}
		last_theta = theta;
		kick_return kr = kick_step(minrung, a, t0, theta, tau == 0.0, full_eval);
		if (full_eval) {
			pot = kr.pot * 0.5 / a;
		}
		double dt = t0 / (1 << kr.max_rung);
		const double dadt1 = cosmos_dadtau(a);
		const double a1 = a;
		a += dadt1 * dt;
		const double dadt2 = cosmos_dadtau(a);
		a += 0.5 * (dadt2 - dadt1) * dt;
		const double a2 = 2.0 / (1.0 / a + 1.0 / a1);
		timer dtm;
		dtm.start();
//		PRINT( "Drift\n");
		drift_return dr = drift(a2, tau, dt);
//		PRINT( "Drift done\n");
		dtm.stop();
		drift_time += dtm.read();
		cosmicK += dr.kin * (a - a1);
		const double esum = (a * (pot + dr.kin) + cosmicK);
		if (tau == 0.0) {
			esum0 = esum;
		}
		const double eerr = (esum - esum0) / (a * dr.kin + a * std::abs(pot) + cosmicK);
		if (full_eval) {
			PRINT("\n%12s %12s %12s %12s %12s %12s %12s %12s %12s %12s %12s %12s %12s %12s %12s %12s %12s %12s %12s %12s\n", "i", "Z", "time", "dt", "pot", "kin",
					"cosmicK", "pot err", "min rung", "max rung", "nactive", "nmapped", "load", "dtime", "stime", "ktime", "dtime", "total", "gpu/cpu", "pps",
					"GFLOPS/s");
		}
		iter++;
		total_processed += kr.nactive;
		total_time.stop();
		runtime += total_time.read();
		double pps = total_processed / runtime;
		const auto total_flops = kr.node_flops + kr.part_flops;
		params.flops += total_flops;
		PRINT("%12i %12.3e %12.3e %12.3e %12.3e %12.3e %12.3e %12.3e %12i %12i %12i %12i %12.3e %12.3e %12.3e %12.3e %12.3e %12.3e %12c %12.3e %12.3e \n",
				iter - 1, z, tau / tau_max, dt / tau_max, a * pot, a * dr.kin, cosmicK, eerr, minrung, kr.max_rung, kr.nactive, dr.nmapped, kr.load, domain_time,
				sort_time, kick_time, drift_time, runtime / iter, used_gpu ? 'G' : 'C', (double ) kr.nactive / total_time.read(),
				params.flops / 1024.0 / 1024.0 / 1024.0 / runtime);
		total_time.reset();
		total_time.start();
		//	PRINT( "%e\n", total_time.read() - gravity_long_time - sort_time - kick_time - drift_time - domain_time);
		itime = inc(itime, kr.max_rung);
		domain_time = 0.0;
		sort_time = 0.0;
		kick_time = 0.0;
		drift_time = 0.0;
		tau += dt;
		this_iter++;
		map_flush(tau);
		if (this_iter > get_options().max_iter) {
			break;
		}
	}
	map_flush(tau);
}

void write_checkpoint(driver_params params) {
	if (hpx_rank() == 0) {
		PRINT("Writing checkpoint\n");
	}
	vector<hpx::future<void>> futs;
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async<write_checkpoint_action>(c, params));
	}
	const std::string command = std::string("mkdir -p checkpoint.") + std::to_string(params.iter) + "\n";
	if (system(command.c_str()) != 0) {
		THROW_ERROR("Unable to execute : %s\n", command.c_str());
	}
	const std::string fname = std::string("checkpoint.") + std::to_string(params.iter) + std::string("/checkpoint.") + std::to_string(params.iter) + "."
			+ std::to_string(hpx_rank()) + std::string(".dat");
	FILE* fp = fopen(fname.c_str(), "wb");
	if (fp == nullptr) {
		THROW_ERROR("Unable to open %s for writing.\n", fname.c_str());
	}
	fwrite(&params, sizeof(driver_params), 1, fp);
	particles_save(fp);
	map_save(fp);
	fclose(fp);
	hpx::wait_all(futs.begin(), futs.end());
	if (hpx_rank() == 0) {
		PRINT("Done writing checkpoint\n");
	}
}

driver_params read_checkpoint(int checknum) {
	driver_params params;
	if (hpx_rank() == 0) {
		PRINT("Reading checkpoint\n");
	}
	vector<hpx::future<driver_params>> futs;
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async<read_checkpoint_action>(c, checknum));
	}
	const std::string fname = std::string("checkpoint.") + std::to_string(checknum) + std::string("/checkpoint.") + std::to_string(checknum) + "."
			+ std::to_string(hpx_rank()) + std::string(".dat");
	FILE* fp = fopen(fname.c_str(), "rb");
	if (fp == nullptr) {
		THROW_ERROR("Unable to open %s for reading.\n", fname.c_str());
	}
	FREAD(&params, sizeof(driver_params), 1, fp);
	particles_load(fp);
	map_load(fp);
	fclose(fp);
	hpx::wait_all(futs.begin(), futs.end());
	if (hpx_rank() == 0) {
		PRINT("Done reading checkpoint\n");
	}
	return params;
}
