/*
 * grid_scf.cpp
 *
 *  Created on: Oct 17, 2015
 *      Author: dmarce1
 */

#include "grid.hpp"
#include "node_server.hpp"
#include "lane_emden.hpp"
#include "node_client.hpp"
#include "options.hpp"
#include "eos.hpp"
#include "util.hpp"
#include "physcon.hpp"
#include <mutex>
#include "profiler.hpp"
#include <stdio.h>
extern options opts;

// w0 = speed of convergence. Adjust lower if nan
const real w0 = 1.0 / 4.0;

namespace scf_options {

static real async1 = -0.0e-2;
static real async2 = -0.0e-2;
static bool equal_struct_eos = true; // If true, EOS of accretor will be set to that of donor
static real M1 = 0.6;// Mass of primary
static real M2 = 0.3;// Mass of sfecondaries
static real nc1 = 2.5;// Primary core polytropic index
static real nc2 = 1.5;// Secondary core polytropic index
static real ne1 = 1.5;// Primary envelope polytropic index // Ignored if equal_struct_eos=true
static real ne2 = 1.5;// Secondary envelope polytropic index
static real mu1 = 1.0;// Primary ratio of molecular weights // Ignored if equal_struct_eos=true
static real mu2 = 1.0;// Primary ratio of molecular weights
static real a = 1.00;// approx. orbital sep
static real core_frac1 = 0.9;// Desired core fraction of primary // Ignored if equal_struct_eos=true
static real core_frac2 = 0.9;// Desired core fraction of secondary - IGNORED FOR CONTACT binaries
static real fill1 = 1.0;// 1d Roche fill factor for primary (ignored if contact fill is > 0.0) //  - IGNORED FOR CONTACT binaries  // Ignored if equal_struct_eos=true
static real fill2 = 1.0;// 1d Roche fill factor for secondary (ignored if contact fill is > 0.0) // - IGNORED FOR CONTACT binaries
static real contact_fill = 0.00; //  Degree of contact - IGNORED FOR NON-CONTACT binaries // SET to ZERO for equal_struct_eos=true


#define READ_LINE(s) 		\
	else if( cmp(ptr,#s) ) { \
		s = read_float(ptr); \
		if( hpx::get_locality_id() == 0 ) printf( #s "= %e\n", double(s)); \
	}

void read_option_file() {
	FILE* fp = fopen("scf.init", "rt");
	if (fp != NULL  ) {
		if( hpx::get_locality_id() == 0 ) printf( "SCF option file found\n" );
		const auto cmp = [](char* ptr, const char* str) {
			return strncmp(ptr,str,strlen(str))==0;
		};
		const auto read_float = [](char* ptr) {
			while( *ptr != '\0' && *ptr != '=') {
				++ptr;
			}
			if( *ptr == '=') {
				++ptr;
			}
			return atof(ptr);
		};
		while (!feof(fp)) {
			char buffer[1024];
			if (fgets(buffer, 1023, fp) != NULL) {
				char* ptr = buffer;
				while (isspace(*ptr) && *ptr != '\0') {
					++ptr;
				}
				if( isspace(*ptr)) {
					++ptr;
				}
				if (false) {
				}
				READ_LINE(equal_struct_eos)
				READ_LINE(contact_fill)
				READ_LINE(core_frac1)
				READ_LINE(core_frac2)
				READ_LINE(async1)
				READ_LINE(async2)
				READ_LINE(fill1)
				READ_LINE(fill2)
				READ_LINE(nc1)
				READ_LINE(nc2)
				READ_LINE(ne1)
				READ_LINE(ne2)
				READ_LINE(mu1)
				READ_LINE(mu2)
				READ_LINE(M1)
				READ_LINE(M2)
				READ_LINE(a)
				else if( strlen(ptr) ){
					if( hpx::get_locality_id() == 0 )  printf( "unknown SCF option - %s\n", buffer);
				}
			}
		}
		fclose(fp);
	} else {
		if( hpx::get_locality_id() == 0 )  printf( "SCF option file \"scf.init\" not found - using defaults\n" );
	}

}


}
//0.5=.313
//0.6 .305

future<void> node_client::rho_move(real x) const {
	return hpx::async<typename node_server::rho_move_action>(get_unmanaged_gid(), x);
}

void node_server::rho_move(real x) {
	std::array<future<void>, NCHILD> futs;
	if (is_refined) {
        integer index = 0;
		for (auto& child : children) {
			futs[index++] = child.rho_move(x);
		}
	}
	grid_ptr->rho_move(x);
	all_hydro_bounds();
	if( is_refined ) {
		for( auto& f : futs ) {
			GET(f);
		}
//		wait_all_and_propagate_exceptions(futs);
	}
}

typedef typename node_server::scf_update_action scf_update_action_type;
HPX_REGISTER_ACTION (scf_update_action_type);

typedef typename node_server::rho_mult_action rho_mult_action_type;
HPX_REGISTER_ACTION (rho_mult_action_type);

future<void> node_client::rho_mult(real f0, real f1) const {
	return hpx::async<typename node_server::rho_mult_action>(get_unmanaged_gid(), f0, f1);
}

future<real> node_client::scf_update(real com, real omega, real c1, real c2, real c1_x, real c2_x, real l1_x, struct_eos e1, struct_eos e2) const {
	return hpx::async<typename node_server::scf_update_action>(get_unmanaged_gid(), com, omega, c1, c2, c1_x, c2_x, l1_x, e1, e2);
}

void node_server::rho_mult(real f0, real f1) {
	std::array<future<void>, NCHILD> futs;
	if (is_refined) {
        integer index = 0;
		for (auto& child : children) {
			futs[index++] = child.rho_mult(f0, f1);
		}
	}
	grid_ptr->rho_mult(f0, f1);
	all_hydro_bounds();
	if( is_refined ) {
		for( auto& f : futs ) {
			GET(f);
		}

//		wait_all_and_propagate_exceptions(futs);
	}
}

real node_server::scf_update(real com, real omega, real c1, real c2, real c1_x, real c2_x, real l1_x, struct_eos e1, struct_eos e2) {
	grid::set_omega(omega);
	std::array<future<real>, NCHILD> futs;
	real res;
	if (is_refined) {
		integer index = 0;
		for (auto& child : children) {
			futs[index++] = child.scf_update(com, omega, c1, c2, c1_x, c2_x, l1_x, e1, e2);
		}
		res = ZERO;
	} else {
		res = grid_ptr->scf_update(com, omega, c1, c2, c1_x, c2_x, l1_x, e1, e2);
	}
	all_hydro_bounds();
	if (is_refined) {
        res = std::accumulate(
            futs.begin(), futs.end(), res,
            [](real res, future<real> & f)
            {
                return res + f.get();
            });
	}
	current_time += 1.0e-100;
	return res;
}

struct scf_parameters {
	real R1;
	real R2;
	real omega;
	real G;
	real q;
	std::shared_ptr<struct_eos> struct_eos1;
	std::shared_ptr<struct_eos> struct_eos2;
	real l1_x;
	real c1_x;
	real c2_x;
	scf_parameters() {
		if (scf_options::equal_struct_eos) {
			scf_options::contact_fill = 0.0;
		}
		const real M1 = scf_options::M1;
		const real M2 = scf_options::M2;
		const real fill1 = scf_options::fill1;
		const real contact = scf_options::contact_fill;
		const real a = scf_options::a;
		G = 1.0;
		const real c = 4.0 * M_PI / 3.0;
		q = M2 / M1;
		c1_x = -a * M2 / (M1 + M2);
		c2_x = +a * M1 / (M1 + M2);
		l1_x = a * (0.5 - 0.227 * log10(q)) + c1_x;
		omega = std::sqrt((G * (M1 + M2)) / (a * a * a));
		const real fill2 = scf_options::fill2;
		const real V1 = find_V(M1 / M2) * cube(a);
		const real V2 = find_V(M2 / M1) * cube(a);
		R1 = std::pow(V1 / c, 1.0 / 3.0) * std::pow(fill1,5);
		R2 = std::pow(V2 / c, 1.0 / 3.0) * std::pow(fill2,5);
		if (opts.eos == WD) {
		//	printf( "!\n");
			struct_eos2 = std::make_shared < struct_eos > (scf_options::M2, R2);
			struct_eos1 = std::make_shared < struct_eos > (scf_options::M1, *struct_eos2);
		} else {
			if (scf_options::equal_struct_eos) {
				struct_eos2 = std::make_shared < struct_eos > (scf_options::M2, R2, scf_options::nc2, scf_options::ne2, scf_options::core_frac2, scf_options::mu2);
				struct_eos1 = std::make_shared < struct_eos > (scf_options::M1, scf_options::nc1, *struct_eos2);
			} else {
				struct_eos1 = std::make_shared < struct_eos > (scf_options::M1, R1, scf_options::nc1, scf_options::ne1, scf_options::core_frac1, scf_options::mu1);
				if (contact > 0.0) {

					/* TODO: Something is missing here */

				} else {
					struct_eos2 = std::make_shared < struct_eos > (scf_options::M2, R2, scf_options::nc2, scf_options::ne2, scf_options::core_frac2, scf_options::mu2);
				}
			}
		}
	//	printf( "R1 R2 %e %e\n", R1, R2);
	}
};

//0.15=0.77
//0.30=0.41
//0.33=0.35
static scf_parameters& initial_params() {
	static scf_parameters a;
	return a;
}

real grid::scf_update(real com, real omega, real c1, real c2, real c1_x, real c2_x, real l1_x, struct_eos struct_eos_1, struct_eos struct_eos_2) {
	PROF_BEGIN;
	if (omega <= 0.0) {
		printf("OMEGA <= 0.0\n");
		abort();
	}
	real rho_int = 10.0 * rho_floor;
	rho_int = std::sqrt(rho_int * rho_floor);
	for (integer i = H_BW; i != H_NX - H_BW; ++i) {
		for (integer j = H_BW; j != H_NX - H_BW; ++j) {
			for (integer k = H_BW; k != H_NX - H_BW; ++k) {
				const integer D = -H_BW;
				const integer iiih = hindex(i, j, k);
				const integer iiig = gindex(i + D, j + D, k + D);
				const real x = X[XDIM][iiih];
				const real y = X[YDIM][iiih];
				const real z = X[ZDIM][iiih];
				const real R = std::sqrt(std::pow(x - com, 2) + y * y);
				real rho = U[rho_i][iiih];
				real phi_eff = G[iiig][phi_i] - 0.5 * std::pow(omega * R, 2);
				const real fx = G[iiig][gx_i] + (x - com) * std::pow(omega, 2);
				const real fy = G[iiig][gy_i] + y * std::pow(omega, 2);
				const real fz = G[iiig][gz_i];

				bool is_donor_side;
				real g;
				real g1 = (x - c1_x) * fx + y * fy + z * fz;
				real g2 = (x - c2_x) * fx + y * fy + z * fz;
				if (x >= l1_x /*+ 10.0*dx*/) {
					is_donor_side = true;
					g = g2;
				} else if (x <= l1_x /*- 10.0*dx*/) {
					g = g1;
					is_donor_side = false;
				} /*else {
				 if( g1 < g2 ) {
				 is_donor_side = false;
				 g = g1;
				 } else {
				 is_donor_side = true;
				 g = g2;
				 }
				 }*/
				real C = is_donor_side ? c2 : c1;
				//			real x0 = is_donor_side ? c2_x : c1_x;
				auto this_struct_eos = is_donor_side ? struct_eos_2 : struct_eos_1;
				real cx, ti_omega; //, Rc;
				if (!is_donor_side) {
					cx = c1_x;
					ti_omega = scf_options::async1 * omega;
				} else {
					cx = c2_x;
					ti_omega = scf_options::async2 * omega;
				}
				//	Rc = std::sqrt( x*x + cx*cx - 2.0*x*cx + y*y );
				phi_eff -= 0.5 * ti_omega * ti_omega * R * R;
				phi_eff -= omega * ti_omega * R * R;
				phi_eff += (omega + ti_omega) * ti_omega * cx * x;
				real new_rho, eint;
				const auto smallest = 1.0e-20;
				if (g <= 0.0) {
					ASSERT_NONAN(phi_eff);
					ASSERT_NONAN(C);
					new_rho = std::max(this_struct_eos.enthalpy_to_density(std::max(C - phi_eff, 0.0)), rho_floor);
				} else {
					new_rho = rho_floor;
				}
				ASSERT_NONAN(new_rho);
				rho = std::max((1.0 - w0) * rho + w0 * new_rho, rho_floor);
				if( opts.eos == WD ) {
					eint = ztwd_energy(rho);
				} else {
					eint = std::max(0.0, this_struct_eos.pressure(rho) / (fgamma - 1.0));
				}
				if (new_rho < rho_int) {
					rho = rho_floor;
				}
				U[rho_i][iiih] = rho;
				const real rho0 = rho - rho_floor;
				if( opts.eos == WD ) {
					U[spc_ac_i][iiih] = (is_donor_side ? 0.0 : rho0);
					U[spc_dc_i][iiih] = (is_donor_side ? rho0 : 0.0);
					U[spc_ae_i][iiih] = 0.0;
					U[spc_de_i][iiih] = 0.0;
				} else {
					U[spc_ac_i][iiih] = rho > this_struct_eos.dE() ? (is_donor_side ? 0.0 : rho0) : 0.0;
					U[spc_dc_i][iiih] = rho > this_struct_eos.dE() ? (is_donor_side ? rho0 : 0.0) : 0.0;
					U[spc_ae_i][iiih] = rho <= this_struct_eos.dE() ? (is_donor_side ? 0.0 : rho0) : 0.0;
					U[spc_de_i][iiih] = rho <= this_struct_eos.dE() ? (is_donor_side ? rho0 : 0.0) : 0.0;
				}
				real sx, sy;
				U[spc_vac_i][iiih] = rho_floor;
				sx = -omega * y * rho;
				sy = +omega * (x - com) * rho;
				sx += -ti_omega * y * rho;
				sy += +ti_omega * (x - cx) * rho;
				U[sz_i][iiih] = 0.0;
				if (rho == rho_floor) {
					sx = sy = 0.0;
					eint = -0.5 * rho_floor * G[iiig][phi_i];
					if (opts.eos == WD) {
						eint -= 3.0 * ztwd_pressure(rho);
					}
					eint = std::max(eint,0.0);
					eint /= 3.0 * (fgamma - 1.0);
					if( opts.eos == WD) {
						eint += ztwd_energy(rho);
					}
		//			eint = 0.0;
				}
				real etherm = eint;
				if (opts.eos == WD) {
					etherm -= ztwd_energy(rho);
					etherm = std::max(0.0, etherm);
				}

				U[sx_i][iiih] = sx;
				U[sy_i][iiih] = sy;
				U[tau_i][iiih] = std::pow(etherm, 1.0 / fgamma);
				U[egas_i][iiih] = eint + (sx*sx+sy*sy) / 2.0 / rho;
				U[zx_i][iiih] = 0.0;
				U[zy_i][iiih] = 0.0;
				U[zz_i][iiih] = dx * dx * omega * rho / 6.0;
			}
		}
	}
	if( opts.radiation) {
		rad_grid_ptr->initialize_erad(U[rho_i], U[tau_i]);
	}
	PROF_END;
	return 0.0;
}

real interpolate(real x1, real x2, real x3, real x4, real y1, real y2, real y3, real y4, real x) {
	x1 -= x2;
	x3 -= x2;
	x4 -= x2;
	x -= x2;

	real a, b, c, d;

	a = y2;

	b = (x3 * x4) / (x1 * (x1 - x3) * (x1 - x4)) * y1;
	b += -(1.0 / x1 + (x3 + x4) / (x3 * x4)) * y2;
	b += (x1 * x4) / ((x1 - x3) * x3 * (x4 - x3)) * y3;
	b += (x1 * x3) / ((x1 - x4) * x4 * (x3 - x4)) * y4;

	c = -(x3 + x4) / (x1 * (x1 - x3) * (x1 - x4)) * y1;
	c += (x1 + x3 + x4) / (x1 * x3 * x4) * y2;
	c += (x1 + x4) / (x3 * (x1 - x3) * (x3 - x4)) * y3;
	c += (x3 + x1) / (x4 * (x1 - x4) * (x4 - x3)) * y4;

	d = y1 / (x1 * (x1 - x3) * (x1 - x4));
	d -= y2 / (x1 * x3 * x4);
	d += y3 / (x3 * (x3 - x1) * (x3 - x4));
	d += y4 / ((x1 - x4) * (x3 - x4) * x4);

	return a + b * x + c * x * x + d * x * x * x;

}

void node_server::run_scf(std::string const& data_dir) {
	solve_gravity(false,false);
	real omega = initial_params().omega;
	real jorb0;
//	printf( "Starting SCF\n");
	grid::set_omega(omega);
	printf( "Starting SCF\n");
	real l1_phi = 0.0, l2_phi, l3_phi;
	for (integer i = 0; i != 100; ++i) {
//		profiler_output(stdout);
        char buffer[33];    // 21 bytes for int (max) + some leeway
        sprintf(buffer, "X.scf.%i.silo", int(i));
		auto& params = initial_params();
		//	set_omega_and_pivot();
		auto diags = diagnostics();
		if (i % 10 == 0 ) {
            if (!opts.disable_output) {
			    output_all(buffer, i);
			}
		}
		real f0 = scf_options::M1 / (diags.m[0]);
		real f1 = scf_options::M2 / (diags.m[1]);
		real f = (scf_options::M1 + scf_options::M2) / (diags.m[0] + diags.m[1]);
		f = (f + 1.0)/2.0;
	///	printf( "%e %e \n", f0, f1);
		rho_mult(f0, f1);
		diags = diagnostics();
		rho_move(diags.grid_com[0] / 2.0);
		real iorb = diags.z_mom_orb;
		real is1 = diags.z_moment[0];
		real is2 = diags.z_moment[1];
	//	iorb -= is1 + is2;
		real M1 = diags.m[0];
		real M2 = diags.m[1];
		real j1 = is1 * omega * (1.0 + scf_options::async1);
		real j2 = is2 * omega * (1.0 + scf_options::async2);
		real jorb = iorb * omega;
		if (i == 0) {
			jorb0 = jorb;
		}
		real spin_ratio = (j1 + j2) / (jorb);
		real this_m = (diags.m[0] + diags.m[1]);
		solve_gravity(false,false);

#ifdef FIND_AXIS_V2
		auto axis = find_axis();
#else
		auto axis = grid_ptr->find_axis();
#endif
		auto loc = line_of_centers(axis);

		real l1_x, c1_x, c2_x; //, l2_x, l3_x;

		real com = axis.second[0];
		real new_omega;
		new_omega = jorb0 / iorb;
		omega = new_omega;
		std::pair < real, real > rho1_max;
		std::pair < real, real > rho2_max;
		std::pair < real, real > l1_phi_pair;
		std::pair < real, real > l2_phi_pair;
		std::pair < real, real > l3_phi_pair;
		real phi_1, phi_2;
		line_of_centers_analyze(loc, omega, rho1_max, rho2_max, l1_phi_pair, l2_phi_pair, l3_phi_pair, phi_1, phi_2);
		real rho1, rho2;
		if (rho1_max.first > rho2_max.first) {
			std::swap(phi_1, phi_2);
			std::swap(rho1_max, rho2_max);
		}
		c1_x = diags.com[0][XDIM];
		c2_x = diags.com[1][XDIM];
		rho1 = diags.rho_max[0];
		rho2 = diags.rho_max[1];
		l1_x = l1_phi_pair.first;
		l1_phi = l1_phi_pair.second;
		l2_phi = l2_phi_pair.second;
		l3_phi = l3_phi_pair.second;

		//	printf( "++++++++++++++++++++%e %e %e %e \n", rho1, rho2, c1_x, c2_x);
		params.struct_eos2->set_d0(rho2 * f1);
		if (scf_options::equal_struct_eos) {
			//	printf( "%e %e \n", rho1, rho1*f0);
			params.struct_eos1->set_d0_using_struct_eos(rho1 * f0, *(params.struct_eos2));
		} else {
			params.struct_eos1->set_d0(rho1 * f0);
		}

		real h_1 = params.struct_eos1->h0();
		real h_2 = params.struct_eos2->h0();

		real c_1, c_2;
		if (scf_options::equal_struct_eos) {
			const real alo2 = 1.0 - scf_options::fill2;
			const real ahi2 = scf_options::fill2;
			c_2 = phi_2 * alo2 + ahi2 * l1_phi;
			c_1 = params.struct_eos1->h0() + phi_1;
		} else {
			if (scf_options::contact_fill > 0.0) {
				const real alo = 1.0 - scf_options::contact_fill;
				const real ahi = scf_options::contact_fill;
				c_1 = c_2 = l1_phi * alo + ahi * std::min(l3_phi, l2_phi);
			} else {
				const real alo1 = 1.0 - scf_options::fill1;
				const real ahi1 = scf_options::fill1;
				const real alo2 = 1.0 - scf_options::fill2;
				const real ahi2 = scf_options::fill2;
				c_1 = phi_1 * alo1 + ahi1 * l1_phi;
				c_2 = phi_2 * alo2 + ahi2 * l1_phi;
			}
		}
//		printf( "%e %e %e %e %e %e %e\n", l1_phi, l2_phi, l3_phi, rho1_max.first, rho1_max.second, rho2_max.first, rho2_max.second);
//		printf( "c_2, phi_2, %e %e\n", c_1, phi_1);
		if (!scf_options::equal_struct_eos) {
			params.struct_eos1->set_h0(c_1 - phi_1);
		}
		params.struct_eos2->set_h0(c_2 - phi_2);
	//	printf( "---------\n");
		auto e1 = params.struct_eos1;
		auto e2 = params.struct_eos2;

		real core_frac_1 = diags.grid_sum[spc_ac_i] / M1;
		real core_frac_2 = diags.grid_sum[spc_dc_i] / M2;
		const real virial = diags.virial;
		real e1f;
		if (opts.eos != WD) {
			if (!scf_options::equal_struct_eos) {
				e1f = e1->get_frac();
				if (core_frac_1 == 0.0) {
					e1f = 0.5 + 0.5 * e1f;
				} else {
					e1f = (1.0 - w0) * e1f + w0 * std::pow(e1f, scf_options::core_frac1 / core_frac_1);
				}
				e1->set_frac(e1f);
			}
			real e2f = e2->get_frac();
			if (scf_options::contact_fill <= 0.0) {
				if (core_frac_2 == 0.0) {
					e2f = 0.5 + 0.5 * e2f;
				} else {
					e2f = (1.0 - w0) * e2f + w0 * std::pow(e2f, scf_options::core_frac2 / core_frac_2);
				}
				if( !scf_options::equal_struct_eos) {
					e2->set_frac(e2f);
				}
			} else {
				e2->set_entropy(e1->s0());
			}
			e1f = e1->get_frac();
			e2f = e2->get_frac();
		}
		real amin, jmin, mu;
		mu = M1 * M2 / (M1 + M2);
		amin = std::sqrt(3.0 * (is1 + is2) / mu);

		const real r0 = std::pow(diags.stellar_vol[0]/(1.3333333333*3.14159), 1.0/3.0);
		const real r1 = std::pow(diags.stellar_vol[1]/(1.3333333333*3.14159), 1.0/3.0);
		const real fi0 = diags.stellar_vol[0]/diags.roche_vol[0];
		const real fi1 = diags.stellar_vol[1]/diags.roche_vol[1];

		jmin = std::sqrt((M1 + M2)) * (mu * std::pow(amin, 0.5) + (is1 + is2) * std::pow(amin, -1.5));
		if (i % 5 == 0) {
			printf("   %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s\n", "rho1", "rho2", "M1", "M2", "is1", "is2",
				"omega", "virial", "core_frac_1", "core_frac_2", "jorb", "jmin", "amin", "jtot", "com", "spin_ratio", "iorb", "R1", "R2", "fill1", "fill2");
        }
		lprintf((opts.data_dir + "log.txt").c_str(), "%i %13e %13e %13e %13e %13e %13e %13e %13e %13e %13e %13e %13e %13e %13e  %13e %13e %13e %13e %13e %13e %13e\n", i, rho1, rho2, M1, M2,is1, is2,
			omega, virial, core_frac_1, core_frac_2, jorb, jmin, amin, j1 + j2 + jorb, com, spin_ratio, iorb, r0, r1, fi0, fi1 );
        if (i % 10 == 0) {
			regrid(me.get_unmanaged_gid(), omega, -1, false);
		}
        else {
            grid::set_omega(omega);
        }
		if( opts.eos == WD ) {
			set_AB(e2->A, e2->B());
		}
//		printf( "%e %e\n", grid::get_A(), grid::get_B());
		const real dx = axis.second[0];
	//	printf( "%e %e %e\n", rho1_max.first, rho2_max.first, l1_x);
		scf_update(com, omega, c_1, c_2, rho1_max.first, rho2_max.first, l1_x, *e1, *e2);
		solve_gravity(false,false);

	}
	if (opts.radiation) {
		if (opts.eos == WD) {
			set_cgs();
			all_hydro_bounds();
			grid_ptr->rad_init();
		}
	}
}

std::vector<real> scf_binary(real x, real y, real z, real dx) {

	{
		static std::once_flag flag;
		std::call_once(flag, [](){
			scf_options::read_option_file();
		});
	}
	//printf( "!\n");

	const real fgamma = grid::get_fgamma();
	std::vector < real > u(NF, real(0));
	static auto& params = initial_params();
	std::shared_ptr<struct_eos> this_struct_eos;
	real rho, r, ei;
	if (x < params.l1_x) {
		this_struct_eos = params.struct_eos1;
	} else {
		this_struct_eos = params.struct_eos2;
	}
	rho = 0;
	const real R0 = this_struct_eos->get_R0();
	int M = 4;
//	printf( "%e %e %i\n", dx, R0, M);
	int nsamp = 0;
	for (double x0 = x - dx / 2.0 + dx / 2.0 / M; x0 < x + dx / 2.0;
			x0 += dx / M) {
		for (double y0 = y - dx / 2.0 + dx / 2.0 / M; y0 < y + dx / 2.0;
				y0 += dx / M) {
			for (double z0 = z - dx / 2.0 + dx / 2.0 / M; z0 < z + dx / 2.0;
					z0 += dx / M) {
				++nsamp;
				if (x < params.l1_x) {
					r = std::sqrt(std::pow(x0 - params.c1_x, 2) + y0 * y0 + z0 * z0);
				} else {
					r = std::sqrt(std::pow(x0 - params.c2_x, 2) + y0 * y0 + z0 * z0);
				}
				if (r <= R0) {
					rho += this_struct_eos->density_at(r, dx);
				}
			}
		}
	}
//	grid::set_AB(this_struct_eos->A, this_struct_eos->B());
	rho = std::max(rho / nsamp, rho_floor);
	if( opts.eos == WD ) {
		ei = this_struct_eos->energy(rho);
	} else {
		ei = this_struct_eos->pressure(rho) / (fgamma - 1.0);
	}
	u[rho_i] = rho;
	if( opts.eos == WD ) {
		u[spc_ac_i] = x > params.l1_x ? 0.0 : rho;
		u[spc_dc_i] = x > params.l1_x ? rho : 0.0;
		u[spc_ae_i] = 0.0;
		u[spc_de_i] = 0.0;
	} else {
		u[spc_ac_i] = rho > this_struct_eos->dE() ? (x > params.l1_x ? 0.0 : rho) : 0.0;
		u[spc_dc_i] = rho > this_struct_eos->dE() ? (x > params.l1_x ? rho : 0.0) : 0.0;
		u[spc_ae_i] = rho <= this_struct_eos->dE() ? (x > params.l1_x ? 0.0 : rho) : 0.0;
		u[spc_de_i] = rho <= this_struct_eos->dE() ? (x > params.l1_x ? rho : 0.0) : 0.0;
	}
	u[egas_i] = ei + 0.5 * rho * (x * x + y * y) * params.omega * params.omega;
	u[sx_i] = -y * params.omega * rho;
	u[sy_i] = +x * params.omega * rho;
	u[sz_i] = 0.0;
	if( opts.eos != WD ) {
		u[tau_i] = std::pow(ei, 1.0 / fgamma);
	} else {
		u[tau_i] = std::pow(1.0e-15, 1.0 / fgamma);
	}
	return u;
}
