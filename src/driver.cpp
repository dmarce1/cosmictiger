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
#define SCALE_DT 0.01

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
#include <cosmictiger/profiler.hpp>
#include <cosmictiger/flops.hpp>
#include <cosmictiger/gravity.hpp>

#include <cosmictiger/fft.hpp>

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

struct timing {
	double nparts;
	double time;
	timing() {
		nparts = time = 0.0;
	}
};

#define MIN_BUCKET 64
#define MAX_BUCKET 192

vector<timing> timings;

double nfw_density(double r) {
	double rmax = 10.0;
	return 1.0 / r / sqr(1.0 + r) / (4.0 * M_PI * (-1.0 + 1.0 / (1.0 + rmax) + log(1.0 + rmax)));
}

double sample_density_distribution(const std::function<double(double)>& f) {
	double target = rand1();
	double menc = 0.0;
	double r = 0.0;
	double dr = 0.001;
	double I, I0;
	I = 0.0;
	while (menc < target) {
		menc += 0.5 * I * dr;
		I0 = 0.5 * I;
		r += dr;
		I = f(r) * 4.0 * M_PI * sqr(r);
		I0 += 0.5 * I;
		menc += 0.5 * I * dr;
	}
	double dr2 = (menc - target) / I0;
	r -= dr2;
	return r;
}
/*
 void distribution_init(const std::function<float(float, float, float)>& pdf) {
 const int N = get_options().Nfour;
 range<int64_t> box;
 for (int dim = 0; dim < NDIM; dim++) {
 box.begin[dim] = 0;
 box.end[dim] = N;
 }
 size_t sz = box.volume();
 vector<float> rho_x(sz);
 array<int64_t, NDIM> I;
 array<float, NDIM> x;
 for (I[0] = 0; I[0] < N; I[0]++) {
 for (I[1] = 0; I[1] < N; I[1]++) {
 for (I[2] = 0; I[2] < N; I[2]++) {
 for (int dim = 0; dim < NDIM; dim++) {
 x[dim] = (float) I[dim] / N;
 }
 rho_x[box.index(I)] = pdf(x[XDIM], x[YDIM], x[ZDIM]);
 }
 }
 }
 float rho0 = 0.0;
 for (I[0] = 0; I[0] < N; I[0]++) {
 for (I[1] = 0; I[1] < N; I[1]++) {
 for (I[2] = 0; I[2] < N; I[2]++) {
 rho0 += rho_x[box.index(I)] / sz;
 }
 }
 }
 for (I[0] = 0; I[0] < N; I[0]++) {
 for (I[1] = 0; I[1] < N; I[1]++) {
 for (I[2] = 0; I[2] < N; I[2]++) {
 rho_x[box.index(I)] = (rho_x[box.index(I)] - rho0) / rho0;
 }
 }
 }
 fft3d_init(N);
 fft3d_accumulate_real(box, rho_x);
 fft3d_execute();
 array<float, NDIM> k;
 const auto fbox = fft3d_complex_range();
 auto rho_k = fft3d_read_complex(fbox);
 particles_resize(N * N * N);
 for (int i = 0; i < particles_size(); i++) {
 particles_rung(i) = 0;
 }
 for (int dim1 = 0; dim1 < NDIM; dim1++) {
 auto dphi_dx_k = rho_k;
 for (I[0] = fbox.begin[XDIM]; I[0] < fbox.end[XDIM]; I[0]++) {
 for (I[1] = fbox.begin[YDIM]; I[1] < fbox.end[YDIM]; I[1]++) {
 for (I[2] = fbox.begin[ZDIM]; I[2] < fbox.end[ZDIM]; I[2]++) {
 float k2 = 0.0f;
 for (int dim2 = 0; dim2 < NDIM; dim2++) {
 k[dim2] = 2.0 * M_PI * (I[dim2] < N / 2 ? I[dim2] : I[dim2] - N) / N;
 k2 += sqr(k[dim2]);
 }
 const auto iii = fbox.index(I);
 if (k2 > 0.0) {
 dphi_dx_k[iii] = dphi_dx_k[iii] * cmplx(0.0, 1.0) * (k[dim1] / k2);
 } else {
 dphi_dx_k[iii] = 0.0f;
 }
 }
 }
 }
 fft3d_destroy();

 fft3d_init(N);
 fft3d_accumulate_complex(fbox, dphi_dx_k);
 fft3d_inv_execute();

 auto dphi_dx_x = fft3d_read_real(box);
 for (I[0] = box.begin[XDIM]; I[0] < box.end[XDIM]; I[0]++) {
 for (I[1] = box.begin[YDIM]; I[1] < box.end[YDIM]; I[1]++) {
 for (I[2] = box.begin[ZDIM]; I[2] < box.end[ZDIM]; I[2]++) {
 const int iii = box.index(I);
 const float dx = dphi_dx_x[iii];
 particles_pos(dim1, iii) = ((float) I[dim1] / N + dx / N);
 particles_vel(dim1, iii) = 0.0;

 }
 }
 }

 }
 fft3d_destroy();

 }
 */
void plummer_init(double r0) {
	const auto nparts = get_options().nparts;
	particles_resize(nparts);
	const auto pdf = [](double x) {
		if( x > 0.0 ) {
			return sqr(x)*pow(1.0 + sqr(x), -2.5);
		} else {
			return pow(1.0 + sqr(1.0/x), -2.5)/sqr(sqr(x));
		}
	};

	/*	distribution_init([](float x, float y, float z) {
	 float r = sqrt(sqr(x - 0.51, y - 0.51, z - 0.51));
	 r *= 4.0;
	 return 1.0 / r / sqr(1+r);
	 return 1000*pow(1.0+sqr(r),-2.5);
	 });
	 return;*/

	double x0 = 0.5;
	double y0 = 0.5;
	double z0 = 0.5;
	PRINT( "Initializing Plummer\n");
	for (part_int i = 0; i < nparts; i +=2) {
		double r, p;
//		r = sample_density_distribution(nfw_density);
		r = sample_density_distribution([](double r) {
			return (3.0/(4.0*M_PI))*pow(1.0+sqr(r),-2.5);
		});
		r *= get_options().plummerR;
		double nx, ny, nz, n2;
		do {
			nx = 2.0 * rand1() - 1.0;
			ny = 2.0 * rand1() - 1.0;
			nz = 2.0 * rand1() - 1.0;
			n2 = sqr(nx, ny, nz);
		} while (n2 == 0.0 || n2 > 1.0);
		double ninv = 1.0 / sqrt(n2);
		nx *= ninv;
		ny *= ninv;
		nz *= ninv;
		double x = x0 + r * nx;
		double y = y0 + r * ny;
		double z = z0 + r * nz;
		//double e = abs();
		//	double v = sqrt(-2.0 * log(rand1())) * abs(cos(2.0 * M_PI * rand1()));
		double v = sqrt(get_options().GM * nparts / sqrt(sqr(get_options().plummerR) + sqr(r)) / 2);
		double v0 = sqrt(get_options().GM * nparts * 0.25);
		do {
			nx = 2.0 * rand1() - 1.0;
			ny = 2.0 * rand1() - 1.0;
			nz = 2.0 * rand1() - 1.0;
			n2 = sqr(nx, ny, nz);
		} while (n2 == 0.0 || n2 > 1.0);
		ninv = 1.0 / sqrt(n2);
		nx *= ninv;
		ny *= ninv;
		nz *= ninv;
		double vx = v * nx;
		double vy = v * ny;
		double vz = v * nz;
		particles_pos(XDIM, i) = x;
		particles_pos(YDIM, i) = y;
		particles_pos(ZDIM, i) = z;
		particles_vel(XDIM, i) = vx;
		particles_vel(YDIM, i) = vy;
		particles_vel(ZDIM, i) = vz;
		particles_pos(XDIM, i + 1) = 2.0 * x0 - x;
		particles_pos(YDIM, i + 1) = 2.0 * y0 - y;
		particles_pos(ZDIM, i + 1) = 2.0 * z0 - z;
		particles_vel(XDIM, i + 1) = -vx;
		particles_vel(YDIM, i + 1) = -vy;
		particles_vel(ZDIM, i + 1) = -vz;
		particles_rung(i + 1) = 0;
		particles_rung(i) = 0;
		/*if (i % 4 == 0) {
			particles_pos(XDIM, i) += 0.025;
			particles_vel(XDIM, i) -= 50.0 * v0;
			particles_pos(XDIM, i + 1) += 0.025;
			particles_vel(XDIM, i + 1) -= 50.0 * v0;
		} else {
			particles_pos(XDIM, i) -= 0.025;
			particles_vel(XDIM, i) += 50.0 * v0;
			particles_pos(XDIM, i + 1) -= 0.025;
			particles_vel(XDIM, i + 1) += 50.0 * v0;
		}*/

	}
	PRINT( "Done Initializing Plummer\n");

}

vector<double> read_checkpoint_list() {
	vector<double> chkpts;
	FILE* fp = fopen("checkpoint.txt", "rt");
	if (fp != nullptr) {
		float z;
		PRINT("Checkpoints will be written at z = \n");
		while (fscanf(fp, "%f\n", &z) == 1) {
			PRINT("%f\n", z);
			chkpts.push_back(z);
		}

		fclose(fp);
	}
	return chkpts;
}

static void flush_timings(bool reset) {
	const auto theta = get_options().theta;
	char* fname;
	asprintf(&fname, "buckets.%.2f.txt", theta);
	FILE* fp = fopen(fname, "wt");
	free(fname);
	for (int i = 0; i < timings.size(); i++) {
		if (timings[i].nparts > 0.0) {
			fprintf(fp, "%i %e\n", i, timings[i].nparts / timings[i].time);
		}
	}
	if (reset) {
		timings = decltype(timings)();
	}
	fclose(fp);
}

void do_groups(int number, double scale) {
	profiler_enter(__FUNCTION__);

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

std::pair<kick_return, tree_create_return> kick_step_hierarchical(int& minrung, int max_rung, double scale, double adot, double tau, double t0, double theta,
		energies_t* energies, int minrung0, bool do_phi, hpx::future<void>* lc_fut, bool nocrop, double drag) {
	profiler_enter(__FUNCTION__);
	timer tm;
	kick_return kr;
	tree_create_return sr;
//	minrung = std::max(minrung, 1);
//	max_rung = std::max(max_rung,minrung);
	//PRINT( "%i %i %i\n", minrung, max_rung, minrung0);
	vector<int> levels(std::max(max_rung - minrung, 0) + 1);
	int k = 0;
	for (int i = max_rung; i > minrung; i--) {
		levels[k++] = i;
	}
	levels[k++] = minrung;
	bool ascending = true;
	bool top;
	bool clip_top = false;
//	PRINT("climbing kick ladder\n");
	timer total_time;
	total_time.start();
	double parts_processed = 0.0;
	for (int li = 0; li < levels.size(); li++) {

		if (levels[li] == minrung) {
			ascending = false;
			top = true;
		} else {
			top = false;
		}
		if (ascending) {
//			PRINT("ASCENDING  rung %i\n", levels[li]);
		} else if (top) {
//			PRINT("AT TOP     rung %i\n", levels[li]);
		} else if (!ascending) {
//			PRINT("DESCENDING rung %i\n", levels[li]);
		}

		if (!ascending && !top) {
			particles_push_rungs();
		}

		if (ascending || top) {
			tm.reset();
			tm.start();
			domains_begin(levels[li]);
			domains_end();
			tm.stop();
//			PRINT("Domains took %e\n", tm.read());
		}

		if (!ascending) {
			tm.reset();
			tm.start();
			particles_sort_by_rung(levels[li]);
			tm.stop();
//			PRINT("Rung sort took %e\n", tm.read());
		}
		auto rng = particles_current_range();
		parts_processed += rng.second - rng.first;
		if (top && minrung == minrung0) {
			auto counts = particles_rung_counts();
			if (counts.size() > minrung0 + 1) {
				const auto total = get_options().nparts;
//				PRINT("Rungs\n");
				for (int i = 0; i < counts.size(); i++) {
//					PRINT("%i %li %f %%\n", i, counts[i], 100.0 * counts[i] / total);
				}
				size_t fast = 0;
				size_t slow = counts[minrung0];
				for (int i = minrung0 + 1; i < counts.size(); i++) {
					fast += counts[i];
				}
				if (3 * fast > slow) {
					clip_top = true && !nocrop;
//					PRINT("------------------------------------\n");
//					PRINT("Setting minimum level to %i\n", minrung + 1);
//					PRINT("------------------------------------\n");
				}
			}
		}

		/*		if (ascending || top) {
		 tm.reset();
		 tm.start();
		 const float dt = 0.5 * rung_dt[levels[li]] * t0;
		 drift(scale, dt, tau, tau + dt, t0, levels[li]);
		 tm.stop();
		 PRINT("drift = %e\n", tm.read());
		 }*/
		tree_create_params tparams(levels[li], theta, 0.f);
		tm.reset();
		tm.start();
		auto this_sr = tree_create(tparams);
		tm.stop();
//		PRINT("Tree create took %e\n", tm.read());
		if (top) {
			sr = this_sr;
		}
		kick_params kparams;
		if (clip_top && levels[li] == minrung0 + 1) {
			kparams.top = true;
		} else {
			kparams.top = top;
		}
		kparams.sign = get_options().create_glass ? -1.0 : 1.0;
		kparams.ascending = ascending || top;
		if (clip_top && top) {
			kparams.descending = false;
		} else {
			kparams.descending = !ascending || top;
		}
		kparams.vel_init = get_options().plummer;
		kparams.node_load = flops_per_node / flops_per_particle;
		kparams.gpu = true;
		kparams.min_level = tparams.min_level;
		kparams.save_force = get_options().save_force;
		kparams.GM = get_options().GM;
		kparams.eta = get_options().eta;
		kparams.h = get_options().hsoft;
		kparams.a = scale;
		kparams.first_call = tau == 0.0;
		kparams.min_rung = levels[li];
		kparams.do_phi = do_phi && top;
		kparams.t0 = t0;
		kparams.max_dt = SCALE_DT / adot;
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
		tm.reset();
		tm.start();
		if (top) {
			reset_gravity_counters();
			set_gravity_counter_use(true);
		}
		kick_return this_kr = kick(kparams, L, pos, root_id, checklist, checklist, nullptr);
		if (top) {
			set_gravity_counter_use(false);
		}
		energies->tckin += this_kr.dkin;
		tm.stop();
//		PRINT("Kick took %e\n", tm.read());
		if (clip_top && top) {
			particles_set_minrung(minrung0 + 1);
		}
		if (top) {
			kr = this_kr;
			energies->pot = kr.pot / scale;
			energies->kin = kr.kin / sqr(scale);
			energies->xmom = kr.xmom / scale;
			energies->ymom = kr.ymom / scale;
			energies->zmom = kr.zmom / scale;
			energies->nmom = kr.nmom / scale;
		}
		if (clip_top && top) {
			levels.push_back(minrung0 + 1);
		} else if (!ascending || top) {
			max_rung = this_kr.max_rung;
			kr.max_rung = max_rung;
			if (max_rung > levels[li]) {
				levels.push_back(levels[li] + 1);
			}
		}
		tm.stop();
		if ((!ascending || top) && !(clip_top && top)) {
			tm.reset();
			tm.start();
			const double dt = rung_dt[levels[li]] * t0;
			if (lc_fut) {
				lc_fut->get();
				*lc_fut = hpx::make_ready_future();
			}
			double a = 1.0 / cosmos_ainv(adot, scale, dt);
			if (get_options().plummer) {
				a = 1.0;
			}
//			PRINT( "%e %e %e\n", a, a0, a1);
			drift(a, dt, tau, tau + dt, get_options().nsteps * t0, levels[li], drag);
			tm.stop();
			//PRINT("Drift took %e\n", tm.read());
		}
		tree_reset();
		if (ascending && !top) {
			particles_pop_rungs();
		}
	}
//	PRINT("done climbing kick ladder\n");
	if (clip_top) {
		minrung++;
	}
	total_time.stop();
	int bucket_size = get_options().bucket_size;
	if (timings.size() <= bucket_size) {
		timings.resize(bucket_size + 1);
	}
	timings[bucket_size].nparts += parts_processed;
	timings[bucket_size].time += total_time.read();
	profiler_exit();
	return std::make_pair(kr, sr);

}

double do_power_spectrum(int num, double a) {
	const auto soft_filter = [](double kh) {
		if( kh > 0.01 ) {
			return -15.0 * (3.0 * kh * cos(kh) + ((sqr(kh)-3.0)*sin(kh))) * pow(kh,-5.0);
		} else {
			return 1.0 - sqr(kh) / 14.0 + sqr(sqr(kh)) / 504.0;
		}
	};
	profiler_enter(__FUNCTION__);
	const int M = 8;
	PRINT("Computing power spectrum\n");
	const double h = get_options().hubble;
	const auto glass = get_options().create_glass;
	const double omega_m = get_options().omega_m;
	const double box_size = get_options().code_to_cm / constants::mpc_to_cm;
	const double N3 = pow((double) get_options().Nfour, 3);
	const double D1 = cosmos_growth_factor(omega_m, a) / cosmos_growth_factor(omega_m, 1.0);
	double factor = pow(box_size, 3) / N3 / (glass ? 1.0 : (N3 * sqr(D1)));
	if (get_options().close_pack) {
		factor *= 0.25;
	}
	double maxpow = 0.0;
	std::string filename = "power." + std::to_string(num) + ".txt";
	FILE* fp = fopen(filename.c_str(), "wt");
	if (fp == NULL) {
		THROW_ERROR("Unable to open %s for writing\n", filename.c_str());
	}
	double kmin = 0.0;
	for (int Mfactor = 1; Mfactor <= 1; Mfactor *= M) {
		auto power0 = power_spectrum_compute(Mfactor);
		double kmax = power0.k.back() * 1.000001;
		//	if (Mfactor != M * M * M) {
		//		kmax = sqrt(power0.k.front() * kmax * M);
		//	}
		const double c1 = h * h * h * factor;
		const double kshot = 2.0 * M_PI * get_options().parts_dim / box_size;
		for (int i = 0; i < power0.P.size(); i++) {
			const double k = power0.k[i];
			if (k >= kmin && k <= kmax) {
				const double shot_noise = power0.k[std::min(i + 1, (int) power0.P.size() - 1)] > kshot ? 1.0 * factor * N3 : 0.0;
				fprintf(fp, "%e %e %e %e %e\n", k / h, (power0.P[i] - power0.Perr[i]) * c1, power0.P[i] * c1, (power0.P[i] + power0.Perr[i]) * c1,
						h * h * h * shot_noise);
			}
		}
		kmin = kmax;
	}
	fclose(fp);
	profiler_exit();
	return maxpow;
}

void output_time_file() {
	profiler_enter(__FUNCTION__);
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

void save_glass() {
	const int N = get_options().parts_dim;
	FILE* fp = fopen("glass.bin", "wb");
	fwrite(&N, sizeof(int), 1, fp);
	fwrite(&particles_pos(XDIM, 0), sizeof(fixed32), particles_size(), fp);
	fwrite(&particles_pos(YDIM, 0), sizeof(fixed32), particles_size(), fp);
	fwrite(&particles_pos(ZDIM, 0), sizeof(fixed32), particles_size(), fp);
	fclose(fp);
}

void driver() {
	PRINT( "Entering main driver\n");
	timer total_time;
	total_time.start();
	timer tmr;
	tmr.start();
	driver_params params;
	timer buckets50;
	timer buckets20;
	timer buckets2;
	hpx::future<void> lc_fut = hpx::make_ready_future();
	int buckets[11] = { 80, 112, 112, 128, 128, 160, 176, 184, 184, 184, 192 };
	double a0 = 1.0 / (1.0 + get_options().z0);
	if (get_options().read_check != -1) {
		params = read_checkpoint();
	} else {
//		output_time_file();
		if (get_options().create_glass) {
			if (hpx_size() != 1) {
				THROW_ERROR("Create glass can only be done with a single locality\n");
			}
			const size_t nparts = get_options().nparts;
			const size_t b = hpx_rank() * nparts / hpx_size();
			const size_t e = (hpx_rank() + 1) * nparts / hpx_size();
			particles_resize(e - b);
			for (part_int i = 0; i < e - b; i++) {
				const double x = rand1();
				const double y = rand1();
				const double z = rand1();
				particles_pos(XDIM, i) = x;
				particles_pos(YDIM, i) = y;
				particles_pos(ZDIM, i) = z;
				particles_vel(XDIM, i) = 0.0;
				particles_vel(YDIM, i) = 0.0;
				particles_vel(ZDIM, i) = 0.0;
				particles_rung(0) = 0;
			}
			params.tau_max = 1.0;
			params.a = 1.0;
			params.adot = 0.0;
		} else {
			if (get_options().plummer) {
				params.tau_max = 1.0e-4;
				params.a = 1.0;
				params.adot = 0.0;
				plummer_init(1e-3);
			} else {
				params.a = a0;
				params.adot = cosmos_dadt(a0);
				params.tau_max = cosmos_conformal_time(a0, 1.0);
				initialize(get_options().z0);
			}
		}
		if (get_options().do_tracers) {
			particles_set_tracers();
		}
		params.step = 0;
		params.flops = 0;
		params.bucket_size = 80;
		PRINT("TAU_MAX = %e\n", params.tau_max);
		params.tau = 0.0;
		params.max_rung = 0;
		params.itime = 0;
		params.iter = 0;
		params.minrung0 = 0;
		params.runtime = 0.0;
		params.total_processed = 0;
		params.years = cosmos_time(1e-6 * a0, a0) * get_options().code_to_s / constants::spyr;
		auto tmp1 = particles_sum_energies();
		params.energies.tckin = tmp1.kin;
//		write_checkpoint(params);
	}
	const auto gadget_file = get_options().gadget4_restart;
	if (gadget_file != "") {
		auto header = particles_read_gadget4(gadget_file);
		const auto nparts = header.npartTotal[1];
		auto opts = get_options();
		opts.code_to_g = header.mass[1] * 1e10 / opts.hubble;
		opts.code_to_g *= constants::M0;
		opts.code_to_cm = pow(opts.code_to_g * (8.0 * M_PI) * nparts * constants::G / (3.0 * opts.omega_m * sqr(constants::H0 * opts.hubble)), 1.0 / 3.0);
		opts.code_to_s = opts.code_to_cm / constants::c;
		double H = constants::H0 * opts.code_to_s;
		opts.GM = opts.omega_m * 3.0 * sqr(H * opts.hubble) / (8.0 * M_PI) / nparts;
		set_options(opts);
		params.a = 1.0 / (header.redshift + 1.0);
		params.adot = cosmos_dadt(a0);
		params.tau_max = cosmos_conformal_time(params.a, 1.0);
		params.years = cosmos_time(1e-6 * a0, params.a) * get_options().code_to_s / constants::spyr;
		PRINT("box_size = %e Mpc\n", opts.code_to_cm / constants::mpc_to_cm);
		PRINT("code_to_g = %e\n", opts.code_to_g);
		PRINT("code_to_cm = %e\n", opts.code_to_cm);
		PRINT("code_to_s = %e\n", opts.code_to_s);
	}
	PRINT("tau_max = %e\n", params.tau_max);
	auto& years = params.years;
	int& max_rung = params.max_rung;
	auto& a = params.a;
	auto& adot = params.adot;
	auto& tau = params.tau;
	auto& tau_max = params.tau_max;
	auto& energies = params.energies;
	auto& energy0 = params.energy0;
	auto& bucket_size = params.bucket_size;
	auto& itime = params.itime;
	auto& minrung0 = params.minrung0;
	minrung0 = std::max(minrung0, get_options().minrung);
	auto& iter = params.iter;
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
	if (tau == 0.0) {
		particles_set_minrung(minrung0);
	}
	const auto check_lc = [&tau,&dt,&tau_max,&a,&iter](bool force) {
		//	profiler_enter("light cone");
			if (force || lc_time_to_flush(tau, tau_max)) {
				timer tm;
				PRINT("Flushing light cone\n");
				tm.start();
				lc_buffer2homes();
				tm.stop();
				PRINT( "lc_buffer2homes %e\n", tm.read());
				tm.reset();
				tm.start();
				lc_particle_boundaries1();
				tm.stop();
				PRINT( "lc_particles_boundaries1 %e\n", tm.read());
				tm.reset();
				tm.start();
				const double link_len = get_options().lc_b * pow( get_options().nparts, -NDIM);
				lc_form_trees(tau, link_len);
				tm.stop();
				PRINT( "lc_form_trees %e\n", tm.read());
				tm.reset();
				tm.start();
				size_t cnt;
				lc_find_neighbors();
				PRINT( "neighbors found\n");
				tm.stop();
				PRINT( "lc_neighbors_found %e\n", tm.read());
				timer tm2;
				tm2.start();
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
				tm2.stop();
				PRINT( "loop time = %e\n", tm2.read());
				tm.start();
				lc_groups2homes();
				tm.stop();
				PRINT( "lc_groups2homes %e\n", tm.read());
				tm.reset();
				tm.start();
				lc_parts2groups(a, link_len, iter);
				tm.stop();
				PRINT( "lc_parts2groups %e\n", tm.read());
				tm.reset();
				tm.start();
				lc_init(tau, tau_max);
				tm.stop();
				PRINT( "lc_init %e\n", tm.read());
				tm.reset();
			}
			//	profiler_exit();
		};
	auto checkpointlist = read_checkpoint_list();
	const double hsoft0 = get_options().hsoft;
	bool do_check = false;
	double checkz;
	buckets50.start();
	energies_t energies0;
	double glass_error = 1.0;
	for (;; step++) {

//		PRINT("STEP  = %i\n", step);
		double t0 = tau_max / get_options().nsteps;
		do {
			timer step_tm;
			step_tm.start();
			profiler_enter(__FUNCTION__);
			tmr.stop();
			if (tmr.read() > get_options().check_freq || do_check) {
				total_time.stop();
				runtime += total_time.read();
				total_time.reset();
				total_time.start();
				PRINT("WRITING CHECKPOINT FOR Z = %e\n", checkz);
				write_checkpoint(params);
				tmr.reset();
				if (do_check) {
					char* cmd;
					asprintf(&cmd, "mv checkpoint.%i checkpoint.z.%.1f\n", params.iter, checkz);
					system(cmd);
					free(cmd);
				}
				do_check = false;
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
				if (full_eval) {
					view_output_views((tau + 1e-6 * t0) / t0, a);
				}
				if (get_options().do_views) {
					timer tm;
					tm.start();
					output_view(number, years);
					tm.stop();
//					PRINT("View %i took %e \n", number, tm.read());
				}
			}
			bool rebound = false;
			if (tau == 0.0) {
				rebound = true;
			} else if (minrung == 0) {
//				PRINT("Checking imbalance\n");
				double imbalance = domains_get_load_imbalance();
				rebound = (imbalance > MAX_LOAD_IMBALANCE);
			}
			if (rebound) {
				PRINT("Doing rebound\n");
				domains_rebound();
				domains_begin(0);
				domains_end();
				PRINT("Rebound done\n");
			}
			double theta;
			const double z = 1.0 / a - 1.0;
			auto opts = get_options();
			opts.hsoft = hsoft0;			// / a;
			if (get_options().create_glass) {
				theta = 0.4;
			} else {
				/*				if (z > 20.0) {
				 theta = 0.41;
				 } else if (z > 2.0) {
				 theta = 0.51;
				 } else {
				 theta = 0.61;
				 }*/
				if (z > 20.0) {
					theta = 0.41;
				} else if (z > 2.0) {
					theta = 0.51;
				} else {
					theta = 0.61;
				}
			}
			const auto ts = 100 * tau / t0 / get_options().nsteps;
			if (ts <= 10.0) {
				bucket_size = 112;
			} else if (ts < 55.0) {
				bucket_size = 112 + (ts - 10.0) / 45.0 * (184 - 112);
			} else {
				bucket_size = 184;
			}
			//bucket_size = 128;
			opts.bucket_size = bucket_size;
			opts.theta = theta;
			set_options(opts);
			last_theta = theta;
			std::pair<kick_return, tree_create_return> tmp;
			bool nocrop = false;
			if (int(tau / t0) % 8 == 0 && full_eval) {
				nocrop = true;
				minrung0 = std::max(minrung0 - 1, get_options().minrung);
			}
			int this_minrung = std::max(minrung, minrung0);
			int om = this_minrung;
//				PRINT("MINRUNG0 = %i\n", minrung0);
//			PRINT( "Doing kick\n");
			reset_flops();
			if (full_eval) {
//				kick_workspace::clear_buffers();
//				tree_destroy(true);
				if (get_options().do_power) {
					do_power_spectrum(step, a);
				}
#ifndef CHECK_MUTUAL_SORT
				if (get_options().do_groups) {
					do_groups(step, a);
				}
#endif
			}
			if (get_options().use_glass && minrung == minrung0) {
				//			particles_displace(rand1(), rand1(), rand1());
			}
			tmp = kick_step_hierarchical(om, max_rung, a, adot, tau, t0, theta, &energies, minrung0, full_eval, &lc_fut, nocrop, sqrt(glass_error));
			if (tau == 0.0) {
				energies0 = energies;
			}
			const double flops = flops_per_second();
			reset_flops();
			if (om != this_minrung) {
				minrung0++;
			}
			kick_return kr = tmp.first;
			int max_rung0 = max_rung;
			max_rung = kr.max_rung;
//			PRINT("GRAVITY max_rung = %i\n", kr.max_rung);
			if (minrung <= 0) {
				double energy = a * (energies.kin + energies.pot) + energies.cosmic;
				if (tau == 0) {
					energy0 = 0.0;
					energies.cosmic = -energy;
					energy = 0.0;
				}
				const double norm = (energies.kin + fabs(energies.pot)) + energies.cosmic;
				const double err = (energy - energy0) / norm;
				if (tau > 0.0) {
					glass_error = fabs(energies0.pot - a * energies.pot) / (a * energies.pot);
				}
				FILE* fp = fopen("energy.txt", "at");
				fprintf(fp, "%e %e %e %e %e %e %e %e %e %e\n", tau / t0, a, energies.xmom / energies.nmom, energies.ymom / energies.nmom,
						energies.zmom / energies.nmom, a * energies.pot, a * energies.kin, energies.cosmic, err, glass_error);
				fclose(fp);
				energies0 = energies;
				energies0.pot *= a;
			}
			dt = t0 / (1 << max_rung);
			tree_create_return sr = tmp.second;
//			PRINT("Done kicking\n");
			double adotdot;
			double a0 = a;
			if (!get_options().create_glass && !get_options().plummer) {
				cosmos_update(adotdot, adot, a, dt);
			}
			energies.cosmic += (a - a0) * energies.tckin / (a0 * a0);
			const double dyears = 0.5 * (a0 + a) * dt * get_options().code_to_s / constants::spyr;
			const auto z0 = 1.0 / a0 - 1.0;
			const auto z1 = 1.0 / a - 1.0;
			for (auto& z : checkpointlist) {
				if ((z - z0) * (z - z1) < 0.0) {
					do_check = true;
					checkz = z;
				}
			}

//			PRINT("%e %e\n", a1, a);
			timer dtm;
			dtm.start();
			dtm.stop();
			drift_time += dtm.read();
			FILE* textfp = fopen("progress.txt", "at");
			if (textfp == nullptr) {
				THROW_ERROR("unable to open progress.txt for writing\n");
			}
			iter++;
			timer remaining_time;
			total_time.stop();
			remaining_time.start();
			runtime += total_time.read();
			double pps = total_processed / runtime;
			//	PRINT( "%e %e %e %e\n", kr.node_flops, kr.part_flops, sr.flops, dr.flops);
			const double nparts = get_options().nparts;
			double close, direct;
			get_gravity_counters(close, direct);
			direct += close;
			if (full_eval) {
				PRINT_BOTH(textfp, "\n%10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s\n", "runtime", "i", "z", "a", "adot", "timestep",
						"years", "mnr", "mxr", "bs", "Tflops", "time", "close", "direct");
			}
			step_tm.stop();
			PRINT_BOTH(textfp, "%10.3e %10i %10.3e %10.3e %10.3e %10.3e %10.3e %10i %10i %10i %10.3e %10.3e %10.3e %10.3e \n", runtime, iter - 1, z, a, adot,
					tau / t0, years, minrung, max_rung, bucket_size, flops * 1e-12, step_tm.read(), close , direct);
			fclose(textfp);
			total_time.reset();
			remaining_time.stop();
			runtime += remaining_time.read();
			total_time.start();
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
			profiler_exit();
			profiler_output();
			profiler_enter(__FUNCTION__);
			jiter++;
			if (jiter > 100) {
				//			abort();
			}
			if (get_options().do_lc && min_rung(itime) <= minrung0) {
				check_lc(false);
			}
			lc_fut = hpx::make_ready_future();

		} while (itime != 0);
		if (1.0 / a < get_options().z1 + 1.0) {
			break;
		} else if (get_options().create_glass && std::abs(glass_error) < 1.0e-5) {
			save_glass();
			break;
		}
	}
	buckets2.stop();
	FILE* fp = fopen("buckets.txt", "at");
	fprintf(fp, "%i %e %e %e\n", get_options().bucket_size, buckets50.read(), buckets20.read(), buckets2.read());
	fclose(fp);
	if (get_options().do_lc) {
		check_lc(true);
		particles_free();
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
	profiler_enter(__FUNCTION__);
//	params.step--;
	if (hpx_rank() == 0) {
		PRINT("Writing checkpoint\n");
		std::string command;
		command = std::string("mkdir -p checkpoint.") + std::to_string(params.iter);
		if (system(command.c_str()) != 0) {
			THROW_ERROR("Unable to execute : %s\n", command.c_str());
		}
	}
	vector<hpx::future<void>> futs;
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async<write_checkpoint_action>(c, params));
	}
	const std::string fname = std::string("checkpoint.") + std::to_string(params.iter) + std::string("/checkpoint.") + std::to_string(hpx_rank())
			+ std::string(".dat");
	FILE* fp = fopen(fname.c_str(), "wb");
	fwrite(&params, sizeof(driver_params), 1, fp);
	particles_save(fp);
	domains_save(fp);
	if (get_options().do_lc) {
		lc_save(fp);
	}
	fclose(fp);
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
		futs.push_back(hpx::async<read_checkpoint_action>(c));
	}
	const int iter = get_options().read_check;
	const std::string fname = std::string("checkpoint.") + std::to_string(iter) + std::string("/checkpoint.") + std::to_string(hpx_rank()) + std::string(".dat");
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
