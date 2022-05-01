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
#include <cosmictiger/chemistry.hpp>
#include <cosmictiger/containers.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/constants.hpp>
#include <cosmictiger/sph_particles.hpp>
#include <cosmictiger/hpx.hpp>
#include <fenv.h>
#include <cosmictiger/timer.hpp>

#define NRATES 20
#define NCOOL 15

const float Ktoev = float(1. / 11604.45);

HPX_PLAIN_ACTION (chemistry_do_step);

pair<double> chemistry_do_step(float a, int minrung, float t0, float adot, int dir) {

	profiler_enter(__FUNCTION__);
	timer tm;
	tm.start();
	vector<hpx::future<pair<double>>>futs;
	if (!get_options().chem) {
		pair<double> rc;
		rc.first = rc.second = 0.0;
		return rc;
	}
	for (auto& c : hpx_children()) {
		futs.push_back(hpx::async<chemistry_do_step_action>(c, a, minrung, t0, adot, dir));
	}
	double echange = 0.0;
	int nthreads = hpx_hardware_concurrency();
	for (int proc = 0; proc < nthreads; proc++) {
		futs.push_back(hpx::async([proc, nthreads, a, minrung,t0,dir,adot]() {
			const part_int b = (size_t) proc * sph_particles_size() / nthreads;
			const part_int e = (size_t) (proc+1) * sph_particles_size() / nthreads;
			vector<chem_attribs> chems;
			const float mass = get_options().sph_mass;
			const float stars = get_options().stars;
			const int N = get_options().sneighbor_number;
			for( part_int i = b; i < e; i++) {
				int rung1 = sph_particles_rung(i);
				//	int rung2 = sph_particles_oldrung(i);
				if( rung1 >= minrung && !sph_particles_isstar(i)) {
					chem_attribs chem;
					float T = sph_particles_temperature(i,a);
					sph_particles_normalize_fracs(i);
					const float factor = 1.0f / (1.0f - sph_particles_Z(i));
					chem.H = sph_particles_H(i) * factor;
					chem.He = sph_particles_He0(i) * factor;
					chem.Hp = sph_particles_Hp(i) * factor;
					chem.Hn = sph_particles_Hn(i) * factor;
					chem.H2 = sph_particles_H2(i) * factor;
					chem.Hep = sph_particles_Hep(i) * factor;
					chem.Hepp = sph_particles_Hepp(i) * factor;
					chem.eint = sph_particles_eint(i);
					if( chem.He < 0.0 && chem.He > -5e-7) {
						chem.Hep += chem.He;
						chem.He = 0.0;
					}
					if( chem.Hepp < 0.0 && chem.Hepp > -5e-7) {
						chem.Hep += chem.Hepp;
						chem.Hepp = 0.0;
					}
					ALWAYS_ASSERT(chem.Hepp >= 0.f);
					ALWAYS_ASSERT(chem.H2 >= 0.f);
					ALWAYS_ASSERT(chem.Hp >= 0.f);
					ALWAYS_ASSERT(chem.Hn >= 0.f);
					ALWAYS_ASSERT(chem.Hep >= 0.f);
//					if( stars ) {
//						chem.cold_mass = sph_particles_cold_mass(i);
//					} else {
						chem.cold_mass = 0.f;
//					}
					if( chem.cold_mass > -1e-4 && chem.cold_mass < 0.0) {
						chem.cold_mass = 0.0;
					} else if( chem.cold_mass < 0.0) {
						PRINT( "%e\n", chem.cold_mass);
						ALWAYS_ASSERT(chem.cold_mass >=0.0);
					}
					double dt = (rung_dt[rung1]) * t0;
					chem.rho = sph_particles_rho(i);
					if( stars ) {

					}
					chem.dt = dt;
					if( T > 5e8) {
						PRINT( "T-------------> %e\n", T);
						if( T > TMAX) {
							int k = sph_particles_dm_index(i);
							float vx = particles_vel(XDIM,k);
							float vy = particles_vel(YDIM,k);
							float vz = particles_vel(ZDIM,k);
							PRINT( "CHEMISTRY OUT OF RANGE %e %e %e  %e  %e  %e \n", chem.rho, chem.eint, sph_particles_smooth_len(i), vx, vy, vz);
							PRINT( "%e %e %e %e %e %e %e\n", chem.He, chem.Hp, chem.Hn, chem.H2, chem.Hep, chem.Hepp, sph_particles_Z(i));
						}
					}
					if(T < 0.f) {
						PRINT( "NEGATIVE T !\n", T);
						abort();
					}
					chems.push_back(chem);
				}
			}
			double flops;
			flops = cuda_chemistry_step(chems, a);
			int j = 0;
			double echange = 0.0;
			const double sph_mass = get_options().sph_mass;
			for( part_int i = b; i < e; i++) {
				int rung1 = sph_particles_rung(i);
				if( rung1 >= minrung && !sph_particles_isstar(i)) {
					chem_attribs chem = chems[j++];
//					double cv = 1.5 + 0.5* chem.H2 / (1. - .75 * (chem.He+chem.Hep+chem.Hepp) - 0.5 * chem.H2);
//					double gamma = 1. + 1. / cv;
					double gamma = get_options().gamma;
				//		PRINT( "%e\n", gamma);
					const float factor = 1.0f - sph_particles_Z(i);
					sph_particles_H(i) = chem.H * factor;
					sph_particles_He0(i) = chem.He * factor;
					sph_particles_Hp(i) = chem.Hp * factor;
					sph_particles_Hn(i) = chem.Hn * factor;
					sph_particles_H2(i) = chem.H2 * factor;
					sph_particles_Hep(i) = chem.Hep * factor;
					sph_particles_Hepp(i) = chem.Hepp * factor;
					sph_particles_normalize_fracs(i);
					const float e0 = sph_particles_eint(i);
					const float fh = 1.f - chem.cold_mass;
					const float rho = sph_particles_rho(i);
					sph_particles_rec2(i).A = chem.eint * (gamma - 1.0) / powf(fh*rho,gamma-1.0);
					const float this_change = (sph_particles_eint(i) - e0) * sph_mass / sqr(a);
		//			PRINT( "%e\n", this_change );
					echange += this_change;
					ALWAYS_ASSERT(fh > 0.0);
					ALWAYS_ASSERT( sph_particles_entr(i)>0.0);
					if(stars) {
						ALWAYS_ASSERT(chem.cold_mass >=0.0);
//						sph_particles_cold_mass(i) = chem.cold_mass;
					}
				}
			}
			pair<double> rc;
			rc.first = echange;
			rc.second = flops;
			return rc;

		}));
	}
	double flops = 0.0;
	for (auto& f : futs) {
		auto tmp = f.get();
		echange += tmp.first;
		flops += tmp.second;
	}
	tm.stop();
	if (hpx_rank() == 0) {
		PRINT("CHEM GFLOPS = %e eheat = %e\n", flops / 1024 / 1024 / 1024 / tm.read(), echange);
	}
	profiler_exit();
	pair<double> rc;
	rc.first = echange;
	rc.second = flops;
	return rc;
}
