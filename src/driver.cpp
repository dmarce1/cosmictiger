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

#define  SMOOTHLEN_BUFFER 0.21
#define SCALE_DT 0.05

#include <cosmictiger/all_tree.hpp>
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
#include <cosmictiger/sph_particles.hpp>
#include <cosmictiger/power.hpp>
#include <cosmictiger/timer.hpp>
#include <cosmictiger/time.hpp>
#include <cosmictiger/view.hpp>
#include <cosmictiger/sph_tree.hpp>
#include <cosmictiger/sph.hpp>
#include <cosmictiger/chemistry.hpp>
#include <cosmictiger/stars.hpp>
#include <cosmictiger/profiler.hpp>

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
	profiler_enter("FUNCTION");

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
	auto ngroups = groups_save(number, scale, cosmos_time(1.0e-6 * scale, scale) * get_options().code_to_s / constants::spyr);
	tm.stop();
	PRINT("groups_save %e\n", tm.read());
	total.stop();
	PRINT("Reduction time = %e\n", reduce.read());
	PRINT("Total time = %e\n", total.read());
	PRINT("Group count = %li of %li candidates\n", ngroups.first, ngroups.second);
	particles_groups_destroy();
	profiler_exit();
}

sph_tree_create_return sph_step1(int minrung, double scale, double tau, double t0, int phase, double adot, int max_rung, int iter, double dt, double* eheat,
		bool verbose) {
	const bool stars = get_options().stars;
	const bool diff = get_options().diffusion;
	const bool chem = get_options().chem;
	verbose = true;
	*eheat = 0.0;
	if (verbose)
		PRINT("Doing SPH step with minrung = %i\n", minrung);
	sph_tree_create_params tparams;

	timer tm;
	timer total_tm;
	total_tm.start();
	tparams.min_rung = minrung;
	tparams.h_wt = (1.0 + SMOOTHLEN_BUFFER);
	tree_id root_id;
	root_id.proc = 0;
	root_id.index = 0;
	sph_tree_create_return sr;
	vector<tree_id> checklist;
	checklist.push_back(root_id);
	sph_tree_neighbor_params tnparams;

	tnparams.h_wt = (1.0 + SMOOTHLEN_BUFFER);
	tnparams.min_rung = minrung;

	tm.start();
	if (verbose)
		PRINT("starting sph_tree_create = %e\n", tm.read());
	profiler_enter("sph_tree_create");
	sr = sph_tree_create(tparams);
	profiler_exit();
	tm.stop();
	if (verbose)
		PRINT("sph_tree_create time = %e %i\n", tm.read(), sr.nactive);
	tm.reset();
	return sr;

}

sph_run_return sph_step2(int minrung, double scale, double tau, double t0, int phase, double adot, int max_rung, int iter, double dt, double* eheat,
		bool verbose) {
	const bool stars = get_options().stars;
	const bool diff = get_options().diffusion;
	const bool chem = get_options().chem;
	const bool conduction = get_options().conduction;
	float dtinv_cfl;
	float dtinv_visc;
	float dtinv_diff;
	float dtinv_cond;
	float dtinv_acc;
	float dtinv_divv;
	float dtinv_omega;
	verbose = true;
	double flops;
	*eheat = 0.0;
	if (verbose)
		PRINT("Doing SPH step with minrung = %i\n", minrung);
	sph_particles_apply_updates(minrung, 0, t0, tau);

	sph_run_params sparams;
	if (adot != 0.0) {
		sparams.max_dt = SCALE_DT * scale / fabs(adot);
	}
	sparams.adot = adot;
	sparams.tau = tau;
	sparams.tzero = tau == 0.0;
	sparams.max_rung = max_rung;
	sparams.a = scale;
	sparams.t0 = t0;
	sparams.min_rung = minrung;
	bool cont;
	sph_run_return kr;
	sparams.set = SPH_SET_ACTIVE;
	sparams.phase = phase;
	const bool glass = get_options().glass;
	sph_tree_neighbor_params tnparams;
	tnparams.h_wt = (1.0 + SMOOTHLEN_BUFFER);
	tnparams.min_rung = minrung;
	tree_id root_id;
	root_id.proc = 0;
	root_id.index = 0;
	vector<tree_id> checklist;
	checklist.push_back(root_id);

	profiler_enter("sph_tree_neighbor:SPH_TREE_NEIGHBOR_NEIGHBORS");
	tnparams.seti = SPH_INTERACTIONS_I;
	tnparams.run_type = SPH_TREE_NEIGHBOR_NEIGHBORS;
	sph_tree_neighbor(tnparams, root_id, checklist).get();
	profiler_exit();

	sparams.phase = 0;
	int doneiters = 0;
	sph_particles_reset_converged();
	do {
		sparams.set = SPH_SET_ACTIVE;
		sparams.run_type = SPH_RUN_PREHYDRO1;
		timer tm;
		tm.start();
		kr = sph_run(sparams, true);
		tm.stop();
		if (verbose)
			PRINT("sph_run(SPH_RUN_PREHYDRO1): tm = %e min_h = %e max_h = %e\n", tm.read(), kr.hmin, kr.hmax);
		tm.reset();
		cont = kr.rc;
		tnparams.h_wt = (1.0 + SMOOTHLEN_BUFFER);
		tnparams.run_type = SPH_TREE_NEIGHBOR_BOXES;
		tnparams.seto = cont ? SPH_SET_ACTIVE : SPH_SET_ALL;
		tnparams.seti = cont ? SPH_SET_ALL : SPH_SET_ALL;
//			tnparams.set = SPH_SET_ACTIVE;
		tm.start();
		profiler_enter("sph_tree_neighbor:SPH_TREE_NEIGHBOR_NEIGHBORS");
		sph_tree_neighbor(tnparams, root_id, vector<tree_id>()).get();
		profiler_exit();
		tm.stop();
		tm.reset();
		tm.start();
		tnparams.seti = cont ? SPH_INTERACTIONS_I : SPH_INTERACTIONS_IJ;
		tnparams.run_type = SPH_TREE_NEIGHBOR_NEIGHBORS;
		profiler_enter("sph_tree_neighbor:SPH_TREE_NEIGHBOR_BOXES");
		sph_tree_neighbor(tnparams, root_id, checklist).get();
		profiler_exit();
		tm.stop();
		tm.reset();
		kr = sph_run_return();
	} while (cont);
	sph_particles_reset_converged();
	do {
		sparams.set = SPH_SET_ACTIVE;
		sparams.run_type = SPH_RUN_PREHYDRO2;
		timer tm;
		tm.start();
		kr = sph_run(sparams, true);
		tm.stop();
		if (verbose)
			PRINT("sph_run(SPH_RUN_PREHYDRO2): tm = %e min_h = %e max_h = %e\n", tm.read(), kr.hmin, kr.hmax);
		tm.reset();
		cont = kr.rc;
		tnparams.h_wt = cont ? (1.0 + SMOOTHLEN_BUFFER) : 1.001;
		tnparams.run_type = SPH_TREE_NEIGHBOR_BOXES;
		tnparams.seto = SPH_SET_ALL;
		tnparams.seti = SPH_SET_ALL;
//			tnparams.set = SPH_SET_ACTIVE;
		tm.start();
		profiler_enter("sph_tree_neighbor:SPH_TREE_NEIGHBOR_NEIGHBORS");
		sph_tree_neighbor(tnparams, root_id, vector<tree_id>()).get();
		profiler_exit();
		tm.stop();
		tm.reset();
		tm.start();
		tnparams.seti = SPH_INTERACTIONS_IJ;
		tnparams.run_type = SPH_TREE_NEIGHBOR_NEIGHBORS;
		profiler_enter("sph_tree_neighbor:SPH_TREE_NEIGHBOR_BOXES");
		sph_tree_neighbor(tnparams, root_id, checklist).get();
		profiler_exit();
		tm.stop();
		tm.reset();
		kr = sph_run_return();
	} while (cont);
	timer tm;

	sparams.phase = 0;
	sparams.run_type = SPH_RUN_HYDRO;
	tm.reset();
	tm.start();
	kr = sph_run(sparams, true);
	tm.stop();
	max_rung = kr.max_rung;
	if (verbose)
		PRINT("sph_run(SPH_RUN_HYDRO): tm = %e\n", tm.read());
	tm.reset();

	sph_particles_apply_updates(minrung, 1, t0, tau);

	if (stars && minrung <= 1) {
		stars_statistics(scale);
		if (stars_find(scale, dt, minrung, iter, t0)) {

			tm.start();
			if (verbose)
				PRINT("starting sph_tree_create = %e\n", tm.read());
			profiler_enter("sph_tree_create");
			sph_tree_create_params tparams;
			tparams.min_rung = minrung;
			tparams.h_wt = (1.0 + SMOOTHLEN_BUFFER);
			tree_id root_id;
			root_id.proc = 0;
			root_id.index = 0;
			sph_tree_create_return sr;
			vector<tree_id> checklist;
			checklist.push_back(root_id);
			sr = sph_tree_create(tparams);
			profiler_exit();
			tm.stop();
			if (verbose)
				PRINT("sph_tree_create time = %e %i\n", tm.read(), sr.nactive);
			tm.reset();

			profiler_enter("sph_tree_neighbor:SPH_TREE_NEIGHBOR_NEIGHBORS");
			tnparams.seti = SPH_INTERACTIONS_I;
			tnparams.run_type = SPH_TREE_NEIGHBOR_NEIGHBORS;
			sph_tree_neighbor(tnparams, root_id, checklist).get();
			profiler_exit();

			sparams.phase = 0;
			int doneiters = 0;
			sph_particles_reset_converged();
			do {
				sparams.set = SPH_SET_ACTIVE;
				sparams.run_type = SPH_RUN_PREHYDRO1;
				timer tm;
				tm.start();
				kr = sph_run(sparams, true);
				tm.stop();
				if (verbose)
					PRINT("sph_run(SPH_RUN_PREHYDRO1): tm = %e min_h = %e max_h = %e\n", tm.read(), kr.hmin, kr.hmax);
				tm.reset();
				cont = kr.rc;
				tnparams.h_wt = (1.0 + SMOOTHLEN_BUFFER);
				tnparams.run_type = SPH_TREE_NEIGHBOR_BOXES;
				tnparams.seto = cont ? SPH_SET_ACTIVE : SPH_SET_ALL;
				tnparams.seti = cont ? SPH_SET_ALL : SPH_SET_ALL;
				//			tnparams.set = SPH_SET_ACTIVE;
				tm.start();
				profiler_enter("sph_tree_neighbor:SPH_TREE_NEIGHBOR_NEIGHBORS");
				sph_tree_neighbor(tnparams, root_id, vector<tree_id>()).get();
				profiler_exit();
				tm.stop();
				tm.reset();
				tm.start();
				tnparams.seti = cont ? SPH_INTERACTIONS_I : SPH_INTERACTIONS_IJ;
				tnparams.run_type = SPH_TREE_NEIGHBOR_NEIGHBORS;
				profiler_enter("sph_tree_neighbor:SPH_TREE_NEIGHBOR_BOXES");
				sph_tree_neighbor(tnparams, root_id, checklist).get();
				profiler_exit();
				tm.stop();
				tm.reset();
				kr = sph_run_return();
			} while (cont);
			sph_particles_reset_converged();
			do {
				sparams.set = SPH_SET_ACTIVE;
				sparams.run_type = SPH_RUN_PREHYDRO2;
				timer tm;
				tm.start();
				kr = sph_run(sparams, true);
				tm.stop();
				if (verbose)
					PRINT("sph_run(SPH_RUN_PREHYDRO2): tm = %e min_h = %e max_h = %e\n", tm.read(), kr.hmin, kr.hmax);
				tm.reset();
				cont = kr.rc;
				tnparams.h_wt = cont ? (1.0 + SMOOTHLEN_BUFFER) : 1.001;
				tnparams.run_type = SPH_TREE_NEIGHBOR_BOXES;
				tnparams.seto = SPH_SET_ALL;
				tnparams.seti = SPH_SET_ALL;
				//			tnparams.set = SPH_SET_ACTIVE;
				tm.start();
				profiler_enter("sph_tree_neighbor:SPH_TREE_NEIGHBOR_NEIGHBORS");
				sph_tree_neighbor(tnparams, root_id, vector<tree_id>()).get();
				profiler_exit();
				tm.stop();
				tm.reset();
				tm.start();
				tnparams.seti = SPH_INTERACTIONS_IJ;
				tnparams.run_type = SPH_TREE_NEIGHBOR_NEIGHBORS;
				profiler_enter("sph_tree_neighbor:SPH_TREE_NEIGHBOR_BOXES");
				sph_tree_neighbor(tnparams, root_id, checklist).get();
				profiler_exit();
				tm.stop();
				tm.reset();
				kr = sph_run_return();
			} while (cont);

		}
		PRINT("-----------------------------------------------------------------------------------------------------------------------\n");
	}

	sparams.phase = 1;
	sparams.run_type = SPH_RUN_HYDRO;
	tm.reset();
	tm.start();
	kr = sph_run(sparams, true);
	dtinv_cfl = kr.dtinv_cfl;
	dtinv_divv = kr.dtinv_divv;
	dtinv_visc = kr.dtinv_visc;
	dtinv_diff = kr.dtinv_diff;
	dtinv_cond = kr.dtinv_cond;
	dtinv_acc = kr.dtinv_acc;
	tm.stop();
	max_rung = kr.max_rung;
	if (verbose)
		PRINT("sph_run(SPH_RUN_HYDRO): tm = %e max_vsig = %e max_rung = %i, %i\n", tm.read(), kr.max_vsig, kr.max_rung_hydro, kr.max_rung_grav);
	tm.reset();

	tnparams.h_wt = 1.001;
	tnparams.run_type = SPH_TREE_NEIGHBOR_BOXES;
	tnparams.seti = SPH_SET_ALL;
	tnparams.seto = SPH_SET_ACTIVE;
	tm.start();
	profiler_enter("sph_tree_neighbor:SPH_TREE_NEIGHBOR_NEIGHBORS");
	sph_tree_neighbor(tnparams, root_id, vector<tree_id>()).get();
	profiler_exit();
	tm.stop();
	tm.reset();
	tm.start();
	tnparams.seti = SPH_INTERACTIONS_I;
	tnparams.run_type = SPH_TREE_NEIGHBOR_NEIGHBORS;
	profiler_enter("sph_tree_neighbor:SPH_TREE_NEIGHBOR_BOXES");
	sph_tree_neighbor(tnparams, root_id, checklist).get();
	profiler_exit();
	tm.stop();
	tm.reset();

	bool rc = true;
	while (rc) {
		sparams.run_type = SPH_RUN_RUNGS;
		tm.start();
		rc = sph_run(sparams, true).rc;
		//	max_rung = kr.max_rung;
		tm.stop();
		if (verbose)
			PRINT("sph_run(SPH_RUN_RUNGS): tm = %e \n", tm.read());
		tm.reset();
	}
	sph_particles_apply_updates(minrung, 2, t0, tau);
	if (verbose)
		PRINT("Completing SPH step with max_rungs = %i, %i\n", kr.max_rung_hydro, kr.max_rung_grav);
	sph_particles_cache_free1();

	if (chem) {
		PRINT("Doing chemistry step\n");
		timer tm;
		tm.start();
		*eheat = chemistry_do_step(scale, minrung, t0, cosmos_dadt(scale), -1).first;
		tm.stop();
		PRINT("Took %e s\n", tm.read());
	}
#ifdef IMPLICIT_CONDUCTION
	if (conduction) {

		tnparams.h_wt = 1.001;
		tnparams.run_type = SPH_TREE_NEIGHBOR_BOXES;
		tnparams.seti = SPH_SET_ALL;
		tnparams.seto = SPH_SET_ALL;
		tm.start();
		profiler_enter("sph_tree_neighbor:SPH_TREE_NEIGHBOR_NEIGHBORS");
		sph_tree_neighbor(tnparams, root_id, vector<tree_id>()).get();
		profiler_exit();
		tm.stop();
		tm.reset();
		tm.start();
		tnparams.seti = SPH_INTERACTIONS_IJ;
		tnparams.run_type = SPH_TREE_NEIGHBOR_NEIGHBORS;
		profiler_enter("sph_tree_neighbor:SPH_TREE_NEIGHBOR_BOXES");
		sph_tree_neighbor(tnparams, root_id, checklist).get();
		profiler_exit();
		tm.stop();
		tm.reset();

		timer dtm;
		dtm.start();

		sph_particles_reset_converged();

		sparams.run_type = SPH_RUN_COND_INIT;
		tm.reset();
		tm.start();
		sph_run(sparams, true);
		tm.stop();
		if (verbose)
			PRINT("sph_run(SPH_RUN_COND_INIT): tm = %e \n", tm.read());
		tm.reset();
		cond_update_return err;
		do {
			sparams.run_type = SPH_RUN_CONDUCTION;
			tm.reset();
			tm.start();
			sph_run(sparams, true);
			tm.stop();
			err = sph_apply_conduction_update(minrung);
			if (verbose)
				PRINT("sph_run(SPH_RUN_CONDUCTION): tm = %e err_max = %e err_rms = %e\n", tm.read(), err.err_max, err.err_rms);
			tm.reset();
			sph_particles_cache_free_entr();
		} while (err.err_max > SPH_DIFFUSION_TOLER1 || err.err_rms > SPH_DIFFUSION_TOLER2);
		dtm.stop();
		PRINT("Conduction took %e seconds total\n", dtm.read());
	}
#endif

	sph_tree_destroy(true);
	sph_particles_cache_free2();
	sph_particles_reset_semiactive();
//	PRINT( "%i\n", max_rung);

	float dtinv_max = std::max(dtinv_cfl, dtinv_visc);
	dtinv_max = std::max(dtinv_max, dtinv_diff);
	dtinv_max = std::max(dtinv_max, dtinv_cond);
	dtinv_max = std::max(dtinv_max, dtinv_acc);
	dtinv_max = std::max(dtinv_max, dtinv_divv);
	dtinv_max = std::max(dtinv_max, dtinv_omega);
	FILE* fp = fopen("timestep.txt", "at");
	fprintf(fp, "%e %i %e %e %e %e %e %e %e ", tau, max_rung, dtinv_cfl, dtinv_visc, dtinv_diff, dtinv_cond, dtinv_acc, dtinv_divv, dtinv_omega);
	if (dtinv_max == dtinv_cfl) {
		PRINT("CFL");
		fprintf(fp, "%s\n", "CFL");
	} else if (dtinv_max == dtinv_visc) {
		PRINT("VISCOSITY");
		fprintf(fp, "%s\n", "VISCOSITY");
	} else if (dtinv_max == dtinv_diff) {
		PRINT("DIFFUSION");
		fprintf(fp, "%s\n", "DIFFUSION");
	} else if (dtinv_max == dtinv_cond) {
		PRINT("CONDUCTION");
		fprintf(fp, "%s\n", "CONDUCTION");
	} else if (dtinv_max == dtinv_acc) {
		PRINT("ACCELERATION");
		fprintf(fp, "%s\n", "ACCELERATION");
	} else if (dtinv_max == dtinv_divv) {
		PRINT("VELOCITY DIVERGENCE");
		fprintf(fp, "%s\n", "DIVV");
	} else if (dtinv_max == dtinv_omega) {
		PRINT("OMEGA");
		fprintf(fp, "%s\n", "OMEGA");
	} else {
		ALWAYS_ASSERT(false);
	}
	PRINT(" LIMITED TIMESTEP\n");
	fclose(fp);
	return kr;

}

std::pair<kick_return, tree_create_return> kick_step(int minrung, double scale, double dadt, double t0, double theta, bool first_call, bool full_eval) {
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
	const bool sph = get_options().sph;
	PRINT("Finding max smoothlen");
//	const float h = sph ? std::max((float) sph_particles_max_smooth_len(), (float) get_options().hsoft) : get_options().hsoft;
//ALWAYS_ASSERT(sph_particles_max_smooth_len() != INFINITY);
	tree_create_params tparams(minrung, theta, 0.f);
	PRINT("Create tree %i %e\n", minrung, theta);
	profiler_enter("tree_create");
	auto sr = tree_create(tparams);
	profiler_exit();

	const bool vsoft = get_options().vsoft;
	if (vsoft) {
		all_tree_softlens(minrung);
	}

	PRINT("gravity nactive = %i\n", sr.nactive);
	const double load_max = sr.node_count * flops_per_node + std::pow(get_options().parts_dim, 3) * flops_per_particle;
	const double load = (sr.active_nodes * flops_per_node + sr.nactive * flops_per_particle) / load_max;
	tm.stop();
	sort_time += tm.read();
	tm.reset();
	tm.start();
//	PRINT("nactive = %li\n", sr.nactive);
	kick_params kparams;
	if (dadt != 0.0) {
		kparams.max_dt = SCALE_DT * scale / fabs(dadt);
	}
	kparams.glass = get_options().glass;
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
	profiler_enter("kick");
	kick_return kr = kick(kparams, L, pos, root_id, checklist, checklist, nullptr).get();
	profiler_exit();
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
	profiler_enter("FUNCTION");
	PRINT("Computing power spectrum\n");
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
	profiler_exit();
}

void output_time_file() {
	profiler_enter("FUNCTION");
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
	profiler_exit();
}

void driver() {
	const bool static sph = get_options().sph;
	timer total_time;
	total_time.start();
	timer tmr;
	tmr.start();
	driver_params params;
	const bool stars = get_options().stars;
	double a0 = 1.0 / (1.0 + get_options().z0);
	drift_return dr;
	const int glass = get_options().glass;
	if (get_options().read_check) {
		params = read_checkpoint();
	} else {
		output_time_file();
		if (glass) {
//			PRINT("Glass not implemented\n");
			//		abort();
			initialize_glass();
		} else {
			initialize(get_options().z0);
		}
		if (get_options().do_tracers) {
			particles_set_tracers();
		}
		domains_rebound();
		params.eheat = 0;
		params.step = 0;
		params.flops = 0;
		params.tau_max = cosmos_conformal_time(a0, 1.0);
		PRINT("TAU_MAX = %e\n", params.tau_max);
		params.tau = 0.0;
		params.a = a0;
		params.max_rung = 0;
		params.cosmicK = 0.0;
		params.itime = 0;
		params.iter = 0;
		params.runtime = 0.0;
		params.total_processed = 0;
		params.years = cosmos_time(1e-6 * a0, a0) * get_options().code_to_s / constants::spyr;
//		write_checkpoint(params);
		dr = drift(a0, 0.0, 0.0, 0.0, 0.0);
		PRINT("Initial etherm = %e\n", dr.therm);

	}
	PRINT("ekin0 = %e\n", dr.kin);
	PRINT("tau_max = %e\n", params.tau_max);
	auto& years = params.years;
	int& max_rung = params.max_rung;
	auto& a = params.a;
	auto& tau = params.tau;
	auto& tau_max = params.tau_max;
	auto& cosmicK = params.cosmicK;
	auto& esum0 = params.esum0;
	auto& itime = params.itime;
	auto& iter = params.iter;
	auto& eheat = params.eheat;
	int& step = params.step;
	auto& total_processed = params.total_processed;
	auto& runtime = params.runtime;
	int nsteps = get_options().nsteps;
	double z0 = get_options().z0;
	double dloga = log(z0 + 1.0) / nsteps;
	double pot;
	int this_iter = 0;
	double last_theta = -1.0;
	timer reset;
	reset.start();
	if (get_options().do_lc) {
		lc_init(tau, tau_max);
	}
	double dt;
	int jiter = 0;
	const auto check_lc = [&tau,&dt,&tau_max,&a](bool force) {
		if (force || lc_time_to_flush(tau + dt, tau_max)) {
			timer tm;
			tm.start();
			PRINT("Flushing light cone\n");
			kick_workspace::clear_buffers();
			tree_destroy(true);
			lc_init(tau + dt, tau_max);
			lc_buffer2homes();
			lc_particle_boundaries1();
			const double link_len = get_options().lc_b / get_options().parts_dim;
			lc_form_trees(tau + dt, link_len);
			PRINT( "Trees formed\n");
			size_t cnt;
			do {
				timer tm;
				tm.start();
				lc_particle_boundaries2();
				tm.stop();
				PRINT( "%e ", tm.read());
				tm.reset();
				tm.start();
				cnt = lc_find_groups();
				tm.stop();
				PRINT( "%e %li\n", cnt, tm.read());
			}while( cnt > 0);
			lc_groups2homes();
			lc_parts2groups(a, link_len);
			tm.stop();
			PRINT( "Light cone flush took %e seconds\n", tm.read());
		}
	};
	const float hsoft0 = get_options().hsoft;
	for (;; step++) {
		double t0 = tau_max / get_options().nsteps;
		do {
//			profiler_enter("main driver");
			tmr.stop();
			if (tmr.read() > get_options().check_freq) {
				total_time.stop();
				runtime += total_time.read();
				total_time.reset();
				total_time.start();
				write_checkpoint(params);
				tmr.reset();
			}
			tmr.start();
			int minrung = min_rung(itime);
			bool full_eval = minrung == 0;
			if (full_eval) {
				const int number = step;
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
			opts.hsoft = hsoft0;			// / a;
			if (!glass) {
				if (z > 50.0) {
					theta = 0.4;
				} else if (z > 20.0) {
					theta = 0.5;
				} else if (z > 2.0) {
					theta = 0.65;
				} else {
					theta = 0.8;
				}
			} else {
				theta = 0.4;
			}

			///		if (last_theta != theta) {
			set_options(opts);
////			}
			last_theta = theta;
			PRINT("Kicking\n");
			const bool chem = get_options().chem;
			double heating;
			if (sph && !glass) {
				sph_step1(minrung, a, tau, t0, 1, a * cosmos_dadt(a), max_rung, iter, dt, &heating);
			}
			auto tmp = kick_step(minrung, a, a * cosmos_dadt(a), t0, theta, tau == 0.0, full_eval);
			kick_return kr = tmp.first;
			int max_rung0 = max_rung;
			max_rung = kr.max_rung;
			PRINT("GRAVITY max_rung = %i\n", kr.max_rung);
			if (sph & !glass) {
				max_rung = std::max(max_rung, sph_step2(minrung, a, tau, t0, 1, a * cosmos_dadt(a), max_rung, iter, dt, &heating).max_rung);
				eheat -= a * heating;
			}
			if (full_eval) {
				view_output_views((tau + 1e-6 * t0) / t0, a);
			}
			tree_create_return sr = tmp.second;
			PRINT("Done kicking\n");
			if (full_eval) {
				kick_workspace::clear_buffers();
				tree_destroy(true);
				pot = kr.pot * 0.5 / a;
				if (get_options().do_power) {
					do_power_spectrum(step, a);
				}
#ifndef CHECK_MUTUAL_SORT
				if (get_options().do_groups) {
					do_groups(step, a);
				}
#endif
			}
			dt = t0 / (1 << max_rung);
			const double dadt1 = a * cosmos_dadt(a);
			const double a1 = a;
			a += dadt1 * dt;
			const double dadt2 = a * cosmos_dadt(a);
			a += 0.5 * (dadt2 - dadt1) * dt;
			const double dyears = 0.5 * (a1 + a) * dt * get_options().code_to_s / constants::spyr;
			const double a2 = 2.0 / (1.0 / a + 1.0 / a1);
//			PRINT("%e %e\n", a1, a);
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
			const double total_kinetic = dr.kin + dr.therm;
			cosmicK += (total_kinetic) * (a - a1);
			const double esum = (a * (pot + total_kinetic) + cosmicK + eheat);
			if (tau == 0.0) {
				esum0 = esum;
			}
			const double eerr = (esum - esum0) / (a * total_kinetic + a * std::abs(pot) + cosmicK + eheat);
			FILE* textfp = fopen("progress.txt", "at");
			if (textfp == nullptr) {
				THROW_ERROR("unable to open progress.txt for writing\n");
			}
			if (full_eval) {
				PRINT_BOTH(textfp,
						"\n%10s %6s %10s %4s %4s %4s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %4s %4s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s\n",
						"runtime", "i", "imbalance", "mind", "maxd", "ed", "ppnode", "appanode", "Z", "a", "timestep", "years", "vol", "pot", "kin", "therm",
						"cosmicK", "pot err", "minr", "maxr", "active", "nmapped", "load", "dotime", "stime", "ktime", "drtime", "avg total", "pps", "GFLOPSins",
						"GFLOPS");
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
			const double nparts = std::pow((double) get_options().parts_dim, (double) NDIM);
			double act_pct = kr.nactive / nparts;
			const double parts_per_node = nparts / sr.leaf_nodes;
			const double active_parts_per_active_node = (double) kr.nactive / (double) sr.active_leaf_nodes;
			const double effective_depth = std::log(sr.leaf_nodes) / std::log(2);
			if (full_eval) {
				FILE* fp = fopen("energy.txt", "at");
				if (fp == NULL) {
					THROW_ERROR("Unable to open energy.txt\n");
				}
				fprintf(fp, "%i %e %e %e %e %e %e %e %e %e\n", step, years, 1.0 / a - 1.0, a, a * pot, a * dr.kin, a * dr.therm, eheat, cosmicK, eerr);
				fclose(fp);
			}
			PRINT_BOTH(textfp,
					"%10.3e %6li %10.3e %4i %4i %4.1f %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e %4li %4li %9.2e %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e \n",
					runtime, iter - 1, imbalance, sr.min_depth, sr.max_depth, effective_depth, parts_per_node, active_parts_per_active_node, z, a1, tau / t0, years,
					dr.vol, a * pot, a * dr.kin, a * dr.therm, cosmicK, eerr, minrung, max_rung, act_pct, (double ) dr.nmapped, kr.load, domain_time, sort_time,
					kick_time, drift_time, runtime / iter, (double ) kr.nactive / total_time.read(), total_flops / total_time.read() / (1024 * 1024 * 1024),
					params.flops / 1024.0 / 1024.0 / 1024.0 / runtime);
			fclose(textfp);
			total_time.reset();
			remaining_time.stop();
			runtime += remaining_time.read();
			total_time.start();
			//	PRINT( "%e\n", total_time.read() - gravity_long_time - sort_time - kick_time - drift_time - domain_time);
//			PRINT("%llx\n", itime);
			PRINT("itime inc %i\n", max_rung);
			itime = inc(itime, max_rung);
			domain_time = 0.0;
			sort_time = 0.0;
			kick_time = 0.0;
			drift_time = 0.0;
			tau += dt;
			this_iter++;
			years += dyears;
			if (1.0 / a < get_options().z1 + 1.0) {
				PRINT("Reached minimum redshift, exiting...\n");
				break;
			} else if (this_iter > get_options().max_iter) {
				PRINT("Reached maximum iteration, exiting...\n");
				break;
			}
//			profiler_exit();
//			profiler_enter("main driver");
			profiler_output();
		} while (itime != 0);
		if (1.0 / a < get_options().z1 + 1.0) {
			break;
		}
		if (jiter > 50) {
			break;
		}
	}
	if (glass) {
		if (glass == 1) {
			particles_save_glass("glass_dm.bin");
		} else {
			particles_save_glass("glass_sph.bin");
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
	profiler_enter("FUNCTION");
	params.step--;
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
		futs.push_back(hpx::async<write_checkpoint_action>(HPX_PRIORITY_HI, c, params));
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
//	PRINT("closed\n");
//	}));
	hpx::wait_all(futs.begin(), futs.end());
	if (hpx_rank() == 0) {
		PRINT("Done writing checkpoint\n");
	}
	profiler_exit();
}

driver_params read_checkpoint() {
	driver_params params;
	if (hpx_rank() == 0) {
		PRINT("Reading checkpoint\n");
	}
	vector<hpx::future<driver_params>> futs;
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async<read_checkpoint_action>(HPX_PRIORITY_HI, c));
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
	}
	return params;
}
