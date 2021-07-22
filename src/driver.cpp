#include <tigerfmm/constants.hpp>
#include <tigerfmm/drift.hpp>
#include <tigerfmm/defs.hpp>
#include <tigerfmm/domain.hpp>
#include <tigerfmm/kick.hpp>
#include <tigerfmm/initialize.hpp>
#include <tigerfmm/particles.hpp>
#include <tigerfmm/timer.hpp>
#include <tigerfmm/time.hpp>


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

double gravity_long_time = 0.0;
double domain_time = 0.0;
double chain_time = 0.0;
double kick_time = 0.0;
double drift_time = 0.0;

kick_return kick_step(int minrung, double scale, double t0, double theta, bool first_call, bool full_eval) {


	timer tm;
	tm.start();
	domains_begin();
	PRINT("domains_begin: %e s\n", tm.read());
	domains_end();
	tree_create_params tparams(minrung, theta);
	tree_create(tparams);
	kick_params kparams;
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
	kick_return kr = kick(kparams, L, pos, root_id, checklist, checklist);
	tm.stop();
	kick_time += tm.read();
	tree_destroy();
	return kr;
}

void driver() {
	driver_params params;
	double a0 = 1.0 / (1.0 + get_options().z0);
	if (get_options().check_num >= 0) {
		read_checkpoint(params, get_options().check_num);
	} else {
		initialize();
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
	while (tau < tau_max) {
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
			theta = 0.4;
		} else if (z > 2.0) {
			theta = 0.55;
		} else {
			theta = 0.7;
		}
		if (z < 1.6 && get_options().chain_bw == 1) {
			options opts = get_options();
			opts.chain_bw = 2;
			if (opts.four_o_chain % 2 == 1) {
				opts.four_o_chain = opts.four_o_chain / 2 + 1;
			} else {
				opts.four_o_chain = opts.four_o_chain / 2;
			}
			opts.parts_o_chain = opts.parts_o_four * opts.four_o_chain;
			opts.chain_dim = opts.parts_dim / opts.parts_o_chain;
			set_options(opts);
		}
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
		drift_return dr = drift(a2, dt);
		dtm.stop();
		drift_time += dtm.read();
		cosmicK += dr.kin * (a - a1);
		const double esum = (a * (pot + dr.kin) + cosmicK);
		if (tau == 0.0) {
			esum0 = esum;
		}
		const double eerr = (esum - esum0) / (a * dr.kin + a * std::abs(pot) + cosmicK);
		if (full_eval) {
			PRINT("\n%12s %12s %12s %12s %12s %12s %12s %12s %12s %12s %12s %12s %12s %12s %12s %12s %12s %12s %12s\n", "i", "Z", "time", "dt", "pot", "kin",
					"cosmicK", "pot err", "min rung", "max rung", "nactive", "dtime", "gltime", "chtime", "ktime", "dtime", "total", "pps", "GFLOPS/s");
		}
		iter++;
		total_processed += kr.nactive;
		total_time.stop();
		runtime += total_time.read();
		double pps = total_processed / runtime;
		PRINT("%12i %12.3e %12.3e %12.3e %12.3e %12.3e %12.3e %12.3e %12i %12i %12i %12.3e %12.3e %12.3e %12.3e %12.3e %12.3e %12.3e %12.3e \n", iter - 1, z,
				tau / tau_max, dt / tau_max, a * pot, a * dr.kin, cosmicK, eerr, minrung, kr.max_rung, kr.nactive, domain_time, gravity_long_time, chain_time,
				kick_time, drift_time, runtime / iter, (double ) kr.nactive / total_time.read(), kr.flops / 1024.0 / 1024.0 / 1024.0 / kick_time);
		total_time.reset();
		total_time.start();
		//	PRINT( "%e\n", total_time.read() - gravity_long_time - chain_time - kick_time - drift_time - domain_time);
		itime = inc(itime, kr.max_rung);

		gravity_long_time = 0.0;
		domain_time = 0.0;
		chain_time = 0.0;
		kick_time = 0.0;
		drift_time = 0.0;
		tau += dt;
		this_iter++;
		if (this_iter > get_options().max_iter) {
			break;
		}
	}

}
