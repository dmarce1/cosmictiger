constexpr bool verbose = true;

#include <tigerfmm/analytic.hpp>
#include <tigerfmm/domain.hpp>
#include <tigerfmm/gravity.hpp>
#include <tigerfmm/initialize.hpp>
#include <tigerfmm/kick.hpp>
#include <tigerfmm/particles.hpp>
#include <tigerfmm/safe_io.hpp>
#include <tigerfmm/test.hpp>
#include <tigerfmm/timer.hpp>
#include <tigerfmm/tree.hpp>

//0.7, 0.8
//0.55, 0.65
//0.4, 0.5

constexpr double theta = 0.5;

static void domain_test() {
	timer tm;

	tm.start();
	particles_random_init();
	tm.stop();
	PRINT("particles_random_init: %e s\n", tm.read());
	tm.reset();

	tm.start();
	domains_begin();
	tm.stop();
	PRINT("domains_begin: %e s\n", tm.read());
	tm.reset();

	tm.start();
	domains_end();
	tm.stop();
	PRINT("domains_end: %e s\n", tm.read());
	tm.reset();

}

static void tree_test() {
	timer tm;

	tm.start();
	particles_random_init();
	tm.stop();
	PRINT("particles_random_init: %e s\n", tm.read());
	tm.reset();

	tm.start();
	domains_begin();
	tm.stop();
	PRINT("domains_begin: %e s\n", tm.read());
	tm.reset();

	tm.start();
	domains_end();
	tm.stop();
	PRINT("domains_end: %e s\n", tm.read());
	tm.reset();

	tm.start();
	tree_create_params tparams(0, 0.7);
	tree_create(tparams);
	tm.stop();
	PRINT("tree_create: %e s\n", tm.read());
	tm.reset();

	tm.start();
	tree_destroy();
	tm.stop();
	PRINT("tree_destroy: %e s\n", tm.read());
	tm.reset();

	tm.start();
	tree_create(tparams);
	tm.stop();
	PRINT("tree_create: %e s\n", tm.read());
	tm.reset();

}

static void kick_test() {
	timer tm;

	tm.start();
	particles_random_init();
//	/initialize();
	tm.stop();
	PRINT("initialize: %e s\n", tm.read());
	tm.reset();

	tm.start();
	domains_begin();
	tm.stop();
	PRINT("domains_begin: %e s\n", tm.read());
	tm.reset();

	tm.start();
	domains_end();
	tm.stop();
	PRINT("domains_end: %e s\n", tm.read());
	tm.reset();

	tm.start();
	tree_create_params tparams(0, theta);
	tree_create(tparams);
	tm.stop();
	PRINT("tree_create: %e s\n", tm.read());
	tm.reset();

	tm.start();
	kick_params kparams;
	kparams.save_force = get_options().save_force;
	kparams.GM = get_options().GM;
	kparams.h = get_options().hsoft;
	kparams.eta = get_options().eta;
	kparams.a = 1.0;
	kparams.first_call = true;
	kparams.min_rung = 0;
	kparams.t0 = 1.0;
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
	auto kr = kick(kparams, L, pos, root_id, checklist, checklist, nullptr).get();
	tm.stop();
	PRINT("tree_kick: %e s\n", tm.read());
	PRINT("GFLOPS/s = %e\n", kr.flops / 1024 / 1024 / 1024 / tm.read());
	tm.reset();

	tm.start();
	tree_destroy();
	tm.stop();
	PRINT("tree_destroy: %e s\n", tm.read());
	tm.reset();

}

static void force_test() {
	timer tm;

	tm.start();
	particles_random_init();
//	initialize();
	tm.stop();
	PRINT("particles_random_init: %e s\n", tm.read());
	tm.reset();

	tm.start();
	domains_begin();
	tm.stop();
	PRINT("domains_begin: %e s\n", tm.read());
	tm.reset();

	tm.start();
	domains_end();
	tm.stop();
	PRINT("domains_end: %e s\n", tm.read());
	tm.reset();

	tm.start();
	tree_create_params tparams(0, theta);
	tree_create(tparams);
	tm.stop();
	PRINT("tree_create: %e s\n", tm.read());
	tm.reset();

	tm.start();
	kick_params kparams;
	kparams.save_force = get_options().save_force;
	kparams.GM = get_options().GM;
	kparams.h = get_options().hsoft;
	kparams.eta = get_options().eta;
	kparams.a = 1.0;
	kparams.first_call = true;
	kparams.min_rung = 0;
	kparams.t0 = 1.0;
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
	auto kr = kick(kparams, L, pos, root_id, checklist, checklist, nullptr).get();
	tm.stop();
	PRINT("tree_kick: %e s\n", tm.read());
	tm.reset();

	tm.start();
	tree_destroy();
	tm.stop();
	PRINT("tree_destroy: %e s\n", tm.read());
	tm.reset();

	tm.start();
	analytic_compare(100);
	tm.stop();
	PRINT("analytic_compare: %e s\n", tm.read());
	tm.reset();

}

void test(std::string test) {
	if (test == "domain") {
		domain_test();
	} else if (test == "force") {
		force_test();
	} else if (test == "kick") {
		kick_test();
	} else if (test == "tree") {
		tree_test();
	} else {
		THROW_ERROR("test %s is not known\n", test.c_str());
	}

}
