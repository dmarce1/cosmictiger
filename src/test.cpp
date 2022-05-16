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
#include <cosmictiger/bh.hpp>

//0.7, 0.8
//0.55, 0.65
//0.4, 0.5
/*
 float rand_normal() {
 float x = 2.0 * rand1() - 1.0;
 float y = 2.0 * rand1() - 1.0;
 auto normal = expc(cmplx(0, 1) * float(M_PI) * y) * sqrtf(-2.0 * logf(fabsf(x)));
 return normal.real();
 }
 */
static void rockstar_test() {
	feenableexcept (FE_DIVBYZERO);
	feenableexcept (FE_INVALID);
	feenableexcept (FE_OVERFLOW);

	const int N = 1000000;
	const int M = 2;
	vector<rockstar_particle> parts;
	array<array<float, 2 * NDIM>, M> X;
	for (int i = 0; i < M; i++) {
		for (int dim = 0; dim < 2 * NDIM; dim++) {
			X[i][dim] = 2.0 * rand1() - 1.0;
		}
	}
	for (int dim = 0; dim < 2 * NDIM; dim++) {
		X[0][dim] = 0.0;
		X[1][dim] = 0.0;
	}
	X[1][0] = 0.5;
	X[1][NDIM] = 0.0000005;
	X[0][0] = -.5;
	X[0][NDIM] = -0.0000005;
	for (int n = 0; n < M; n++) {
		for (int i = n * N / M; i < (n + 1) * N / M; i++) {
			rockstar_particle part;
			//		PRINT( "%e\n", r );
			part.x = rand_normal() * 0.001 + X[n][XDIM];
			part.y = rand_normal() * 0.001 + X[n][YDIM];
			part.z = rand_normal() * 0.001 + X[n][ZDIM];
			float r = sqrt(sqr(part.x - X[n][XDIM], part.y - X[n][YDIM], part.z - X[n][ZDIM]));
			float v0 = 0.75 * sqrt(get_options().GM / r * N / M) / sqrt(3);
			part.vx = rand_normal() * v0 + X[n][NDIM + XDIM];
			part.vy = rand_normal() * v0 + X[n][NDIM + YDIM];
			part.vz = rand_normal() * v0 + X[n][NDIM + ZDIM];
			parts.push_back(part);
		}
	}
	for (int i = (M - 1) * N / M; i < N; i++) {
		rockstar_particle part;
		part.x = 2.0 * rand1() - 1.0;
		part.y = 2.0 * rand1() - 1.0;
		part.z = 2.0 * rand1() - 1.0;
		part.vx = 2.0 * rand1() - 1.0;
		part.vy = 2.0 * rand1() - 1.0;
		part.vz = 2.0 * rand1() - 1.0;
		//	parts.push_back(part);

	}
	rockstar_find_subgroups(parts);
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
	tree_create_params tparams(0, 0.7, get_options().hsoft);
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

	timer total_time;
	double total_flops = 0.0;
	tm.start();
	initialize(get_options().z0);
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
	tm.reset();
	tm.start();
	tree_create_params tparams(0, get_options().theta, get_options().hsoft);
	auto sr = tree_create(tparams);
	total_flops += sr.flops;
	tm.stop();
	PRINT("tree_create: %e s\n", tm.read());
	tm.reset();
	tm.start();
	tree_destroy();
	tm.stop();
	PRINT("tree_destroy: %e s\n", tm.read());
	total_time.start();
	tm.reset();
	tm.start();
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
	total_flops += kr.part_flops + kr.node_flops;
	tm.stop();
	PRINT("tree_kick: %e s\n", tm.read());
	tm.reset();
	tm.start();
	tree_destroy();
	tm.stop();
	PRINT("tree_destroy: %e s\n", tm.read());
	tm.reset();
	tm.start();
	auto dr = drift(1.0, 0.0, 0.0, 0.0, 0.0);
	total_flops += dr.flops;
	tm.stop();
	PRINT("drift: %e s\n", tm.read());
	total_time.stop();
	PRINT("avg time per step = %e\n", total_time.read());
	const double flops_measure = total_flops / total_time.read() / hpx_size();
	PRINT("FLOPS / s / locality = %e\n", flops_measure);
	FILE* fp = fopen("results.txt", "at");
	fprintf(fp, "%i %e %e\n", hpx_size(), total_time.read(), flops_measure);
	fclose(fp);
	kick_workspace::clear_buffers();
}

static void force_test() {
	timer tm;
	timer tm_main;
	constexpr int NITER = 10;
	tm_main.start();
	for (int iter = 0; iter < NITER; iter++) {
		tm_main.stop();
		tm.start();
		particles_random_init();
		tm.stop();
		tm_main.start();
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

		particles_sort_by_rung(0);

		tm.start();
		tree_create_params tparams(0, get_options().theta, get_options().hsoft);
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
	}
	tm_main.stop();
	tm.start();
	analytic_compare(100);
	tm.stop();
	PRINT("analytic_compare: %e s\n", tm.read());
	tm.reset();

	kick_workspace::clear_buffers();
	PRINT("AVERAGE TIME = %e\n", tm_main.read() / NITER);

}

void bh_test() {
	int N = 100000;
	int M = 10;
	vector<array<float, NDIM>> x(N);
	vector<array<float, NDIM>> y(M);
	for (int i = 0; i < N; i++) {
		for (int dim = 0; dim < NDIM; dim++) {
			x[i][dim] = rand1();
		}
	}
	for (int i = 0; i < M; i++) {
		for (int dim = 0; dim < NDIM; dim++) {
			y[i][dim] = rand1();
		}
	}
	auto x0 = x;
//	auto phi0 = direct_evaluate(x0);
	x0 = x;
	timer tm1, tm2;
	tm1.start();
	auto phi1 = bh_evaluate_points(y, x0, false);
	tm1.stop();
	x0 = x;
	tm2.start();
	PRINT("Doing gpu\n");
	auto phi2 = bh_evaluate_points(y, x0, true);
	tm2.stop();
	PRINT("%e %e\n", tm1.read(), tm2.read());
	double err_tot = 0.0;
	for (int i = 0; i < phi1.size(); i++) {
		double err = fabs((phi1[i] - phi2[i]) / phi1[i]);
		PRINT("%e %e %e\n", phi1[i], phi2[i], err);
		err_tot += err;
	}
	PRINT("%e\n", err_tot / phi1.size());

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
	} else if (test == "rockstar") {
		rockstar_test();
	} else if (test == "kick") {
		kick_test();
	} else if (test == "tree") {
		tree_test();
	} else if (test == "bh") {
		bh_test();
	} else {
		THROW_ERROR("test %s is not known\n", test.c_str());
	}

}
