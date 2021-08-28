constexpr bool verbose = true;

#include <fenv.h>

#include <cosmictiger/analytic.hpp>
#include <cosmictiger/domain.hpp>
#include <cosmictiger/drift.hpp>
#include <cosmictiger/fft.hpp>
#include <cosmictiger/gravity.hpp>
#include <cosmictiger/initialize.hpp>
#include <cosmictiger/kick.hpp>
#include <cosmictiger/kick_workspace.hpp>
#include <cosmictiger/particles.hpp>
#include <cosmictiger/rockstar.hpp>
#include <cosmictiger/safe_io.hpp>
#include <cosmictiger/test.hpp>
#include <cosmictiger/timer.hpp>
#include <cosmictiger/tree.hpp>

//0.7, 0.8
//0.55, 0.65
//0.4, 0.5

double rand1() {
	return ((double) rand() + 0.5) / (double) RAND_MAX;
}

static void fft1_test() {
	const int N = 100;
	range<int64_t> box;
	for (int dim = 0; dim < NDIM; dim++) {
		box.begin[dim] = 0;
		box.end[dim] = N;
	}
	vector<float> A(box.volume());
	for (int i = 0; i < box.volume(); i++) {
		A[i] = 0.1 + (double) rand() / RAND_MAX;
	}
	fft3d_init(N);
	fft3d_accumulate_real(box, A);
	fft3d_execute();
	fft3d_inv_execute();
	auto B = fft3d_read_real(box);
	fft3d_destroy();
	for (int i = 0; i < box.volume(); i++) {
		const double err = std::abs(A[i] - B[i]) / A[i];
		if (err > 1.0e-5) {
			PRINT("%e %e %e\n", err, A[i], B[i]);
		}
	}
}

static void fft2_test() {
	const int N = 4;
	range<int64_t> box;
	for (int dim = 0; dim < NDIM; dim++) {
		box.begin[dim] = 0;
		box.end[dim] = N;
	}
	vector<float> A(box.volume());
	array<int64_t, NDIM> I;
	for (I[0] = 0; I[0] < N; I[0]++) {
		for (I[1] = 0; I[1] < N; I[1]++) {
			for (I[2] = 0; I[2] < N; I[2]++) {
				A[box.index(I)] = cos(M_PI * I[2]);
			}
		}
	}
	fft3d_init(N);
	fft3d_accumulate_real(box, A);
	fft3d_execute();
	box.end[ZDIM] = N / 2 + 1;
	auto B = fft3d_read_complex(box);
	fft3d_destroy();
	for (I[0] = 0; I[0] < N; I[0]++) {
		for (I[1] = 0; I[1] < N; I[1]++) {
			for (I[2] = 0; I[2] < N / 2 + 1; I[2]++) {
				PRINT("%i %i %i %e %e\n", I[0], I[1], I[2], B[box.index(I)].real(), B[box.index(I)].imag());
			}
		}
	}
}

static void rockstar_test() {
	feenableexcept (FE_DIVBYZERO);
	feenableexcept (FE_OVERFLOW);
	feenableexcept (FE_INVALID);
	constexpr int NPARTS = 10000;
	constexpr int NHALOS = 2;
	vector<particle_data> parts;
	std::array<std::array<double, NDIM>, NHALOS> x;
	x[0][0] = x[0][1] = x[1][0] = x[1][1] = 0.5;
	x[0][2] = 0.6;
	x[1][2] = 0.5;
	std::array<double, NHALOS> rad = { 0.1, 0.1 };
	for (int j = 0; j < NHALOS; j++) {
		for (int i = 0; i < NPARTS; i++) {
			double s, rnd;
			double X, Y, Z;
			double vx, vy, vz;
			do {
				X = rand1();
				Y = rand1();
				Z = rand1();
				const double q2 = sqr(X - x[j][XDIM], Y - x[j][YDIM], Z - x[j][ZDIM]) / sqr(rad[j]);
				const double q = std::sqrt(q2);
				s = std::pow(1 + q2, -2.5);
				rnd = rand1();
			} while (s < rnd);
			vx = rand1();
			vy = rand1();
			vz = rand1();
			particle_data part;
			part.x[XDIM] = X;
			part.x[YDIM] = Y;
			part.x[ZDIM] = Z;
			part.v[XDIM] = vx;
			part.v[YDIM] = vy;
			part.v[ZDIM] = vz;
			parts.push_back(part);
		}
	}
	auto halos = rockstar_seed_halos(parts);
	for (int i = 0; i < halos.size(); i++) {
		PRINT("%i : %e %e %e : %e %e %e : %i\n", i, halos[i].x[XDIM], halos[i].x[YDIM], halos[i].x[ZDIM], halos[i].v[XDIM], halos[i].v[YDIM, halos[i].v[ZDIM]],
				halos[i].parts.size());
	}
}

static void domain_test() {
	timer tm;

	tm.start();
	particles_random_init();
	tm.stop();
	PRINT("particles_random_init: %e s\n", tm.read());
	tm.reset();

	tm.start();
	domains_rebound();
	tm.stop();
	PRINT("domains_rebound: %e s\n", tm.read());
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
	domains_rebound();
	tm.stop();
	PRINT("domains_rebound: %e s\n", tm.read());
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
//	initialize();
	tm.stop();
	PRINT("initialize: %e s\n", tm.read());
	tm.reset();

	tm.start();
	domains_rebound();
	tm.stop();
	PRINT("domains_rebound: %e s\n", tm.read());
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

	timer total_time;
	for (int pass = 0; pass < 4; pass++) {
		if (pass == 1) {
			total_time.start();
		}
		tm.reset();
		tm.start();
		tree_create_params tparams(0, get_options().theta);
		tree_create(tparams);
		tm.stop();
		PRINT("tree_create: %e s\n", tm.read());
		tm.reset();
		tm.start();
		kick_params kparams;
		kparams.gpu = true;
		kparams.node_load = 10;
		kparams.min_level = tparams.min_level;
		kparams.save_force = get_options().save_force;
		kparams.GM = get_options().GM;
		kparams.h = get_options().hsoft;
		kparams.eta = get_options().eta;
		kparams.a = 1.0;
		kparams.first_call = true;
		kparams.min_rung = 0;
		kparams.t0 = 1.0;
		kparams.theta = get_options().theta;
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
		drift(1.0, 0.0, 0.0);
		tm.stop();
		PRINT("drift: %e s\n", tm.read());
	}
	total_time.stop();
	PRINT("avg time per step = %e\n", total_time.read() / 3);
	FILE* fp = fopen("results.txt", "at");
	fprintf(fp, "%i %e\n", hpx_size(), total_time.read() / 3.0);
	fclose(fp);
	kick_workspace::clear_buffers();
}

static void force_test() {
	timer tm;

	tm.start();
//	particles_random_init();
	initialize();
	tm.stop();
	PRINT("particles_random_init: %e s\n", tm.read());
	tm.reset();

	tm.start();
	domains_rebound();
	tm.stop();
	PRINT("domains_rebound: %e s\n", tm.read());
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
	tree_create_params tparams(0, get_options().theta);
	tree_create(tparams);
	tm.stop();
	PRINT("tree_create: %e s\n", tm.read());
	tm.reset();

	tm.start();
	kick_params kparams;
	kparams.node_load = 10;
	kparams.gpu = true;
	kparams.min_level = tparams.min_level;
	kparams.save_force = get_options().save_force;
	kparams.GM = get_options().GM;
	kparams.h = get_options().hsoft;
	kparams.eta = get_options().eta;
	kparams.a = 1.0;
	kparams.first_call = true;
	kparams.min_rung = 0;
	kparams.t0 = 1.0;
	kparams.theta = get_options().theta;
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

	kick_workspace::clear_buffers();

}

void test(std::string test) {
	if (test == "domain") {
		domain_test();
	} else if (test == "fft1") {
		fft1_test();
	} else if (test == "fft2") {
		fft2_test();
	} else if (test == "force") {
		force_test();
	} else if (test == "kick") {
		kick_test();
	} else if (test == "rockstar") {
		rockstar_test();
	} else if (test == "tree") {
		tree_test();
	} else {
		THROW_ERROR("test %s is not known\n", test.c_str());
	}

}
