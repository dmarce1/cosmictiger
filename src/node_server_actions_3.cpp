#include "node_server.hpp"
#include "node_client.hpp"
#include "future.hpp"
#include "options.hpp"
#include "util.hpp"

#include <hpx/include/run_as.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/util.hpp>
#include <hpx/lcos/broadcast.hpp>

extern options opts;

typedef node_server::send_gravity_boundary_action send_gravity_boundary_action_type;
HPX_REGISTER_ACTION (send_gravity_boundary_action_type);

void node_client::send_gravity_boundary(gravity_boundary_type&& data, const geo::direction& dir, bool monopole, std::size_t cycle) const {
	hpx::apply<typename node_server::send_gravity_boundary_action>(get_unmanaged_gid(), std::move(data), dir, monopole, cycle);
}

void node_server::recv_gravity_boundary(gravity_boundary_type&& bdata, const geo::direction& dir, bool monopole, std::size_t cycle) {
	neighbor_gravity_type tmp;
	tmp.data = std::move(bdata);
	tmp.is_monopole = monopole;
	tmp.direction = dir;
	neighbor_gravity_channels[dir].set_value(std::move(tmp), cycle);
}

typedef node_server::send_gravity_expansions_action send_gravity_expansions_action_type;
HPX_REGISTER_ACTION(send_gravity_expansions_action_type);

void node_server::recv_gravity_expansions(expansion_pass_type&& v) {
    parent_gravity_channel.set_value(std::move(v));
}

void node_client::send_gravity_expansions(expansion_pass_type&& data) const {
    hpx::apply<typename node_server::send_gravity_expansions_action>(get_unmanaged_gid(),
        std::move(data));
}

typedef node_server::send_gravity_multipoles_action send_gravity_multipoles_action_type;
HPX_REGISTER_ACTION(send_gravity_multipoles_action_type);

void node_client::send_gravity_multipoles(multipole_pass_type&& data,
    const geo::octant& ci) const {
    hpx::apply<typename node_server::send_gravity_multipoles_action>(get_unmanaged_gid(),
        std::move(data), ci);
}

void node_server::recv_gravity_multipoles(multipole_pass_type&& v,
    const geo::octant& ci) {
    child_gravity_channels[ci].set_value(std::move(v));
}

typedef node_server::send_hydro_boundary_action send_hydro_boundary_action_type;
HPX_REGISTER_ACTION(send_hydro_boundary_action_type);

void node_client::send_hydro_boundary(std::vector<real>&& data,
    const geo::direction& dir, std::size_t cycle) const {
    hpx::apply<typename node_server::send_hydro_boundary_action>(get_unmanaged_gid(),
        std::move(data), dir, cycle);
}

void node_server::recv_hydro_boundary(std::vector<real>&& bdata,
    const geo::direction& dir, std::size_t cycle) {
    sibling_hydro_type tmp;
    tmp.data = std::move(bdata);
    tmp.direction = dir;
    sibling_hydro_channels[dir].set_value(std::move(tmp),cycle);
}


typedef node_server::send_flux_check_action send_flux_check_action_type;
HPX_REGISTER_ACTION(send_flux_check_action_type);

void node_client::send_flux_check(std::vector<real>&& data,
    const geo::direction& dir, std::size_t cycle) const {
    hpx::apply<typename node_server::send_flux_check_action>(get_unmanaged_gid(),
        std::move(data), dir, cycle);
}

void node_server::recv_flux_check(std::vector<real>&& bdata,
    const geo::direction& dir, std::size_t cycle) {
	sibling_hydro_type tmp;
    tmp.data = std::move(bdata);
    tmp.direction = dir;
    sibling_hydro_channels[dir].set_value(std::move(tmp),cycle);
}

typedef node_server::send_hydro_children_action send_hydro_children_action_type;
HPX_REGISTER_ACTION(send_hydro_children_action_type);

void node_server::recv_hydro_children(std::vector<real>&& data, const geo::octant& ci, std::size_t cycle) {
    child_hydro_channels[ci].set_value(std::move(data), cycle);
}

void node_client::send_hydro_children(std::vector<real>&& data,
    const geo::octant& ci, std::size_t cycle) const {
    hpx::apply<typename node_server::send_hydro_children_action>(get_unmanaged_gid(),
        std::move(data), ci, cycle);
}

typedef node_server::send_hydro_flux_correct_action send_hydro_flux_correct_action_type;
HPX_REGISTER_ACTION(send_hydro_flux_correct_action_type);

void node_client::send_hydro_flux_correct(std::vector<real>&& data,
    const geo::face& face,
    const geo::octant& ci) const {
    hpx::apply<typename node_server::send_hydro_flux_correct_action>(get_unmanaged_gid(),
        std::move(data), face, ci);
}

void node_server::recv_hydro_flux_correct(std::vector<real>&& data, const geo::face& face,
    const geo::octant& ci) {
    const geo::quadrant index(ci, face.get_dimension());
    if( face >= nieces.size() ) {
		for (integer i = 0; i != 100; ++i) {
			printf( "NIECE OVERFLOW\n");
		}
		abort();
	}
    if (nieces[face]!=1) {
		for (integer i = 0; i != 100; ++i) {
			printf("Big bad flux error  %c %i\n", is_refined ? 'R' : 'N', int(nieces[face]));
		}
		abort();
	}


    niece_hydro_channels[face][index].set_value(std::move(data));
}

typedef node_server::line_of_centers_action line_of_centers_action_type;
HPX_REGISTER_ACTION(line_of_centers_action_type);

hpx::future<line_of_centers_t> node_client::line_of_centers(
    const std::pair<space_vector, space_vector>& line) const {
    return hpx::async<typename node_server::line_of_centers_action>(get_unmanaged_gid(), line);
}

void output_line_of_centers(FILE* fp, const line_of_centers_t& loc) {
    for (integer i = 0; i != loc.line.size(); ++i) {
        fprintf(fp, "%e ", loc.line[i].first);
        for (integer j = 0; j != NF + NGF; ++j) {
            fprintf(fp, "%e ", loc.line[i].second[j]);
        }
        fprintf(fp, "\n");
    }
}

line_of_centers_t node_server::line_of_centers(
    const std::pair<space_vector, space_vector>& line) const {
    line_of_centers_t return_line;
    if (is_refined) {
        std::array<hpx::future<line_of_centers_t>, NCHILD> futs;
        for (integer ci = 0; ci != NCHILD; ++ci) {
            futs[ci] = children[ci].line_of_centers(line);
        }
        std::map<real, std::vector<real>> map;
        for (auto&& fut : futs) {
            auto tmp = fut.get();
            return_line.core1 += tmp.core1;
            return_line.core2 += tmp.core2;
            for( integer d = 0; d != NDIM; ++d) {
                return_line.core1_s[d] += tmp.core1_s[d];
                return_line.core2_s[d] += tmp.core2_s[d];
            }
             for (integer ii = 0; ii != tmp.line.size(); ++ii) {
                if (map.find(tmp.line[ii].first) == map.end()) {
                    map.emplace(std::move(tmp.line[ii]));
                }
            }
        }
        return_line.line.resize(map.size());
        std::move(map.begin(), map.end(), return_line.line.begin());
    } else {
        return_line = grid_ptr->line_of_centers(line);
    }

    return return_line;
}

real line_of_centers_analyze(const line_of_centers_t& loc, std::pair<space_vector,space_vector> axis,
    std::pair<real, real>& rho1_max, std::pair<real, real>& rho2_max,
    std::pair<real, real>& l1_phi, std::pair<real, real>& l2_phi,
    std::pair<real, real>& l3_phi, real& rho1_phi, real& rho2_phi) {
	real omega;
	const auto n = axis.first;

/*
	FILE* fp = fopen( "line.dat", "wt");
    for (integer i = 0; i != loc.size(); ++i) {
    	const real x = loc[i].first;
        const real rho = loc[i].second[rho_i];
        const real pot = loc[i].second[pot_i];
        const real phi_eff = pot / ASSERT_POSITIVE(rho) - 0.5 * x * x * omega * omega;
    	fprintf( fp, "%e %e %e\n", x, rho, phi_eff);
    }
    fclose(fp);
*/
    for (auto& l : loc.line) {
        ASSERT_NONAN(l.first);
        for (integer f = 0; f != NF + NGF; ++f) {
            ASSERT_NONAN(l.second[f]);
        }
    }

    rho1_max.second = rho2_max.second = 0.0;
    integer rho1_maxi, rho2_maxi;
    space_vector v1, v2;
    ///	printf( "loc.lineSIZE %i\n", loc.line.size());
    for (integer i = 0; i != loc.line.size(); ++i) {
		const real x = loc.line[i].first;
		const real rho = loc.line[i].second[rho_i];
		const real pot = loc.line[i].second[pot_i];
		const real core1 = loc.line[i].second[spc_ac_i] / rho;
		const real core2 = loc.line[i].second[spc_dc_i] / rho;
		if( std::max(core1,core2) > 0.5) {
			if (core1 > core2) {
				if (rho1_max.second < rho) {
					rho1_max.second = rho;
					rho1_max.first = x;
					rho1_maxi = i;
				}
			} else {
				if (rho2_max.second < rho) {
					rho2_max.second = rho;
					rho2_max.first = x;
					rho2_maxi = i;
				}
			}
		}
	}

    for( integer d = 0; d != NDIM; ++d) {
		v1[d] = loc.core1_s[d] / loc.core1;
		v2[d] = loc.core2_s[d] / loc.core2;
    }
    v1[ZDIM] = v2[ZDIM] = 0.0;
    real v1dn = 0.0, v2dn = 0.0;
    for( integer d = 0; d != NDIM; ++d) {
       	v1dn += n[d] * v1[d];
       	v2dn += n[d] * v2[d];
    }
    for( integer d = 0; d != NDIM; ++d) {
    	v1[d] -= n[d] * v1dn;
    	v2[d] -= n[d] * v2dn;
    	v1[d] -= v2[d];
    }
    real v1norm = 0.0;
    for( integer d = 0; d != NDIM; ++d) {
       	v1norm += v1[d] * v1[d];
    }
    v1norm = std::sqrt(v1norm);

    omega = v1norm / std::abs(rho1_max.first - rho2_max.first);
  //  omega = loc.jcores / loc.icores;
   //
    printf( "OMEGA = %e\n", omega);
    real pot, phi, rho, phi_eff, x;
    pot = loc.line[rho1_maxi].second[pot_i];
    rho = loc.line[rho1_maxi].second[rho_i];
    x = rho1_max.first;
	phi_eff = pot / ASSERT_POSITIVE(rho)
			- 0.5 * x * x * omega * omega;
	rho2_phi = phi_eff;

    pot = loc.line[rho2_maxi].second[pot_i];
    rho = loc.line[rho2_maxi].second[rho_i];
    x = rho2_max.first;
	phi_eff = pot / ASSERT_POSITIVE(rho)
			- 0.5 * x * x * omega * omega;
	rho2_phi = phi_eff;

    l1_phi.second = -std::numeric_limits < real > ::max();
    l2_phi.second = -std::numeric_limits < real > ::max();
    l3_phi.second = -std::numeric_limits < real > ::max();
    for (integer i = 0; i != loc.line.size(); ++i) {
        const real x = loc.line[i].first;
        const real rho = loc.line[i].second[rho_i];
        const real pot = loc.line[i].second[pot_i];
        real phi_eff = pot / ASSERT_POSITIVE(rho) - 0.5 * x * x * omega * omega;
        if (x > std::min(rho1_max.first, rho2_max.first)
            && x < std::max(rho1_max.first, rho2_max.first)) {
            if (phi_eff > l1_phi.second) {
                l1_phi.second = phi_eff;
                l1_phi.first = x;
            }
        } else if (std::abs(x) > std::abs(rho2_max.first) && x * rho2_max.first > 0.0) {
            if (phi_eff > l2_phi.second) {
                l2_phi.second = phi_eff;
                l2_phi.first = x;
            }
        } else if (std::abs(x) > std::abs(rho1_max.first)) {
            if (phi_eff > l3_phi.second) {
                l3_phi.second = phi_eff;
                l3_phi.first = x;
            }
        }
    }
    return omega;
}

void node_server::start_run(bool scf, integer ngrids)
{
    timings_.times_[timings::time_regrid] = 0.0;
    timings::scope ts(timings_, timings::time_total);
    integer output_cnt;

    if (!hydro_on) {
        if (!opts.disable_output) {
            save_to_file("X.chk", opts.data_dir);
        }
        diagnostics();
        return;
    }
    if (scf) {
        run_scf(opts.data_dir);
        set_pivot();
        printf("Adjusting velocities:\n");
        auto diag = diagnostics();
        space_vector dv;
        dv[XDIM] = -diag.grid_sum[sx_i] / diag.grid_sum[rho_i];
        dv[YDIM] = -diag.grid_sum[sy_i] / diag.grid_sum[rho_i];
        dv[ZDIM] = -diag.grid_sum[sz_i] / diag.grid_sum[rho_i];
        this->velocity_inc(dv);
        if (!opts.disable_output) {
            save_to_file("scf.chk", opts.data_dir);
        }
    }
#ifdef RADIATION
    if( opts.eos == WD && opts.problem == STAR) {
    	printf( "Initialized radiation and cgs\n");
    	set_cgs();
    	erad_init();
    }
#endif

    auto fut_ptr = me.get_ptr();
    node_server* root_ptr = fut_ptr.get();
    if( opts.output_only ) {
   // 	diagnostics(0.0);
    	diagnostics();
 //       printf("doing silo out...\n");
  	output_cnt = root_ptr->get_rotation_count() / opts.output_dt;
        std::string fname = "X." + std::to_string(int(output_cnt));
        output(opts.data_dir, fname, output_cnt, false);
    //	for( real rhoc = 1.0e-10; rhoc < 1.0e+5; rhoc *= 10.0) {
    //		printf( "%e\n", rhoc);
    	//	diagnostics(rhoc);
    //	}
    	return;
    }

    printf("Starting...\n");
    solve_gravity(false);
    ngrids = regrid(me.get_gid(), grid::get_omega(), -1,  false);

    real output_dt = opts.output_dt;

    printf("OMEGA = %e, output_dt = %e\n", grid::get_omega(), output_dt);
    real& t = current_time;
    integer step_num = 0;

    printf( "1\n");

    output_cnt = root_ptr->get_rotation_count() / output_dt;

    profiler_output(stdout);

    real bench_start, bench_stop;

    printf( "2\n");

    while (current_time < opts.stop_time) {
        if (step_num > opts.stop_step)
            break;
        printf( "3\n");

        auto time_start = std::chrono::high_resolution_clock::now();
        if (!opts.disable_output && root_ptr->get_rotation_count() / output_dt >= output_cnt) {
            //	if (step_num != 0) {

            std::string fname = "X." + std::to_string(int(output_cnt)) + ".chk";
            save_to_file(fname, opts.data_dir);
            printf("doing silo out...\n");

            fname = "X." + std::to_string(int(output_cnt));
            output(opts.data_dir, fname, output_cnt, false);

            //	SYSTEM(std::string("cp *.dat ./dat_back/\n"));
            //	}
            ++output_cnt;

        }
        printf( "4\n");
        if (step_num == 0) {
            bench_start = hpx::util::high_resolution_clock::now() / 1e9;
        }

        real dt = 0;
        printf( "5\n");
        integer next_step = (std::min)(step_num + refinement_freq(), opts.stop_step + 1);
        real omega_dot = 0.0, omega = 0.0, theta = 0.0, theta_dot = 0.0;

        if ((opts.problem == DWD) && (step_num % refinement_freq() == 0)) {
            printf("dwd step...\n");
            auto dt = step(next_step - step_num).get();
            auto diags = diagnostics();
            omega = grid::get_omega();

            const real dx = diags.com[1][XDIM] - diags.com[0][XDIM];
            const real dy = diags.com[1][YDIM] - diags.com[0][YDIM];
            const real dx_dot = diags.com_dot[1][XDIM]
                - diags.com_dot[0][XDIM];
            const real dy_dot = diags.com_dot[1][YDIM]
                - diags.com_dot[0][YDIM];
            theta = atan2(dy, dx);
            omega = grid::get_omega();
            printf( "Old Omega = %e\n", omega );
           if( opts.vomega ) {
            	theta_dot = (dy_dot * dx - dx_dot * dy) / (dx * dx + dy * dy) - omega;
            	const real w0 = grid::get_omega() * 10.0;
            	const real theta_dot_dot = (2.0 * w0 * theta_dot + w0 * w0 * theta);
            	omega_dot = theta_dot_dot;
            	omega += omega_dot * dt;
            }
            printf( "New Omega = %e\n", omega );
 //            omega_dot += theta_dot_dot*dt;
//             grid::set_omega(omega);          // now done during check_for_refinement below
        }
        else {
            printf("normal step...\n");
            dt = step(next_step - step_num).get();
            omega = grid::get_omega();
        }

        double time_elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(
            std::chrono::high_resolution_clock::now() - time_start).count();

        // run output on separate thread
        if (!opts.disable_output)
        {
            hpx::threads::run_as_os_thread([=]()
            {
                FILE* fp = fopen( (opts.data_dir + "step.dat").c_str(), "at");
                fprintf(fp, "%i %e %e %e %e %e %e %e %e %i\n",
                    int(next_step - 1), double(t), double(dt), time_elapsed, rotational_time,
                    theta, theta_dot, omega, omega_dot, int(ngrids));
                fclose(fp);
            });     // do not wait for it to finish
        }

        hpx::threads::run_as_os_thread([=]()
        {
            printf("%i %e %e %e %e %e %e %e %e\n", int(next_step - 1), double(t), double(dt),
                time_elapsed, rotational_time, theta, theta_dot, omega, omega_dot);
        });     // do not wait for output to finish

//		t += dt;
        step_num = next_step;

        if (step_num % refinement_freq() == 0) {
        	real new_floor = opts.refinement_floor;
			if (opts.ngrids > 0) {
				new_floor *= std::pow( real(ngrids) / real(opts.ngrids), 2);
				printf("Old refinement floor = %e\n", opts.refinement_floor);
				printf("New refinement floor = %e\n", new_floor);
			}

            ngrids = regrid(me.get_gid(), omega, new_floor, false);

            // run output on separate thread
            auto need_break = hpx::threads::run_as_os_thread([&]()
            {
                if (!opts.disable_output) {
                    FILE* fp = fopen((opts.data_dir + "profile.txt").c_str(), "wt");
                    profiler_output(fp);
                    fclose(fp);
                }

                //		set_omega_and_pivot();
                bench_stop = hpx::util::high_resolution_clock::now() / 1e9;
                if (scf || opts.bench) {
                    printf("Total time = %e s\n", double(bench_stop - bench_start));
                    if (!opts.disable_output) {
                        FILE* fp = fopen((opts.data_dir + "bench.dat").c_str(), "at");
                        fprintf(fp, "%i %e\n", int(options::all_localities.size()),
                            double(bench_stop - bench_start));
                        fclose(fp);
                    }
                    return true;
                }
                return false;
            });
            if (need_break.get())
                break;
        }
        //		set_omega_and_pivot();
        if (scf) {
            bench_stop = hpx::util::high_resolution_clock::now() / 1e9;
            printf("Total time = %e s\n", double(bench_stop - bench_start));
            //	FILE* fp = fopen( "bench.dat", "at" );
            //	fprintf( fp, "%i %e\n", int(options::all_localities.size()), double(bench_stop - bench_start));
            //	fclose(fp);
            break;
        }
    }
    bench_stop = hpx::util::high_resolution_clock::now() / 1e9;
    {
        timings::scope ts(timings_, timings::time_compare_analytic);
        compare_analytic();
        if (!opts.disable_output)
            output(opts.data_dir, "final", output_cnt, true);
    }

    if(opts.bench && !opts.disable_output) {
        hpx::threads::run_as_os_thread([&]()
        {
            std::string fname;
            if (output_cnt > 0)
                fname = opts.data_dir + "X." + std::to_string(int(output_cnt) - 1) + ".chk";
            else
                fname = opts.data_dir + "X.0.chk";

            file_copy(fname.c_str(), (opts.data_dir + "restart.chk").c_str());
            FILE* fp = fopen( (opts.data_dir + "scaling.dat").c_str(), "at");
            const auto nproc = options::all_localities.size();
            fprintf( fp, "%i %e\n", int(nproc), float(bench_stop - bench_start));
            fclose( fp );
        }).get();
    }
}

typedef node_server::step_action step_action_type;
HPX_REGISTER_ACTION(step_action_type);

hpx::future<real> node_client::step(integer steps) const {
    return hpx::async<typename node_server::step_action>(get_unmanaged_gid(), steps);
}

void node_server::refined_step() {
#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
    static hpx::util::itt::string_handle sh("node_server::refined_step");
    hpx::util::itt::task t(hpx::get_thread_itt_domain(), sh);
#endif

    timings::scope ts(timings_, timings::time_computation);
    const real dx = TWO * grid::get_scaling_factor() / real(INX << my_location.level());
    real cfl0 = cfl;

    real a = std::numeric_limits<real>::min();

    all_hydro_bounds();
    local_timestep_channels[NCHILD].set_value(std::numeric_limits<real>::max());
    auto dt_fut = global_timestep_channel.get_future();

#ifdef RADIATION
    dt_ = dt_fut.get();
    compute_radiation(dt_/2.0);
    all_hydro_bounds();
#endif

    for (integer rk = 0; rk < NRK; ++rk) {

        compute_fmm(DRHODT, false);

        compute_fmm(RHO, true);
        all_hydro_bounds();

    }
#ifdef RADIATION
    compute_radiation(dt_/2.0);
    all_hydro_bounds();
#else
    dt_ = dt_fut.get();
#endif

    update();
}

hpx::future<void> node_server::nonrefined_step() {
#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
    static hpx::util::itt::string_handle sh("node_server::nonrefined_step");
    hpx::util::itt::task t(hpx::get_thread_itt_domain(), sh);
#endif

    timings::scope ts(timings_, timings::time_computation);

    real cfl0 = cfl;
    dt_ = ZERO;

    all_hydro_bounds();

    grid_ptr->store();
    hpx::future<void> fut = hpx::make_ready_future();

    hpx::shared_future<real> dt_fut = global_timestep_channel.get_future();

    for (integer rk = 0; rk < NRK; ++rk) {

        fut = fut.then(hpx::launch::async(hpx::threads::thread_priority_boost),
            hpx::util::annotated_function(
                [rk, cfl0, this, dt_fut](hpx::future<void> f)
                {
                    f.get();        // propagate exceptions

                    grid_ptr->reconstruct();
                    real a = grid_ptr->compute_fluxes();
#ifdef RADIATION
                    if( rk == 0 ) {
                        const real dx = TWO * grid::get_scaling_factor() /
                            real(INX << my_location.level());
                        dt_ = cfl0 * dx / a;
                        local_timestep_channels[NCHILD].set_value(dt_);
                        dt_ = dt_fut.get();
                        compute_radiation(dt_/2.0);
                        all_hydro_bounds();
                        grid_ptr->reconstruct();
                        grid_ptr->compute_fluxes();
                    }

                    hpx::future<void> fut_flux = exchange_flux_corrections();
#else
                    hpx::future<void> fut_flux = exchange_flux_corrections();

                    if (rk == 0) {
                        const real dx = TWO * grid::get_scaling_factor() /
                            real(INX << my_location.level());
                        dt_ = cfl0 * dx / a;
                        local_timestep_channels[NCHILD].set_value(dt_);
                    }
#endif

                    fut_flux.then(
                        hpx::launch::async(hpx::threads::thread_priority_boost),
                        hpx::util::annotated_function(
                            [rk, this, dt_fut](hpx::future<void> f)
                            {
                                f.get();        // propagate exceptions
#ifdef OCTOTIGER_FLUX_CHECK
                                auto fcheck = check_flux_consistency();
#endif

                                grid_ptr->compute_sources(current_time);
                                grid_ptr->compute_dudt();

                                compute_fmm(DRHODT, false);

                                if (rk == 0) {
                                    dt_ = dt_fut.get();
                                }
                                grid_ptr->next_u(rk, current_time, dt_);

                                compute_fmm(RHO, true);
#ifdef OCTOTIGER_FLUX_CHECK
                                fcheck.get();
#endif
                                all_hydro_bounds();
#ifdef RADIATION
                                if(rk == NRK - 1) {
                      //          	all_hydro_bounds();
                                	compute_radiation(dt_/2.0);
                                	all_hydro_bounds();
                                }
#endif
                            }, "node_server::nonrefined_step::compute_fmm"
                        )).get();
                }, "node_server::nonrefined_step::compute_fluxes"
            )
        );
    }

    return fut.then(hpx::launch::sync,
        [this](hpx::future<void>&& f)
        {
            f.get(); // propagate exceptions...
            update();
        }
    );
}

void node_server::update()
{
    grid_ptr->dual_energy_update();
    current_time += dt_;
    if (grid::get_omega() != 0.0)
    {
        rotational_time += grid::get_omega() * dt_;
    }
    else
    {
        rotational_time = current_time;
    }
}

hpx::future<real> node_server::local_step(integer steps) {
	hpx::future<real> fut = hpx::make_ready_future(0.0);
	for (integer i = 0; i != steps; ++i) {
		fut =
				fut.then(
						hpx::launch::async(hpx::threads::thread_priority_boost),
						[this, i, steps](hpx::future<void> fut) -> real
						{
							fut.get();
							auto time_start = std::chrono::high_resolution_clock::now();
							auto next_dt = timestep_driver_descend();

							if (is_refined)
							{
								refined_step();
							}
							else
							{
								nonrefined_step().get();
							}

							if (my_location.level() == 0)
							{
								double time_elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(
										std::chrono::high_resolution_clock::now() - time_start).count();

								hpx::threads::run_as_os_thread([=]()
										{
											printf("%i %e %e %e %e\n", int(step_num), double(current_time), double(dt_),
													time_elapsed, rotational_time);
										});  // do not wait for output to finish
							}
							++step_num;
							next_dt.get();
						});
	}
	return fut.then([this](hpx::future<void> f) {
		f.get();
		return dt_;
	});
}

hpx::future<real> node_server::step(integer steps) {
    grid_ptr->set_coordinates();

    std::array<hpx::future<void>, NCHILD> child_futs;
    if (is_refined)
    {
        for (integer ci = 0; ci != NCHILD; ++ci) {
            child_futs[ci] = children[ci].step(steps);
        }
    }

    hpx::future<real> fut = local_step(steps);

    if (is_refined)
    {
        return hpx::dataflow(hpx::launch::sync,
            [this](hpx::future<real> dt_fut, hpx::future<std::array<hpx::future<void>, NCHILD>>&& f)
            {
                auto fi = f.get(); // propagate exceptions
                for( auto& f : fi ) {
                	f.get();
                }
                return dt_fut;
            },
            std::move(fut),
            hpx::when_all(std::move(child_futs))
        );
    }

    return fut;
}


typedef node_server::timestep_driver_ascend_action timestep_driver_ascend_action_type;
HPX_REGISTER_ACTION(timestep_driver_ascend_action_type);

void node_client::timestep_driver_ascend(real dt) const {
    hpx::apply<typename node_server::timestep_driver_ascend_action>(get_unmanaged_gid(), dt);
}

void node_server::timestep_driver_ascend(real dt) {
    global_timestep_channel.set_value(dt);
    if (is_refined) {
        for(auto& child: children) {
            child.timestep_driver_ascend(dt);
        }
    }
}

typedef node_server::set_local_timestep_action set_local_timestep_action_type;
HPX_REGISTER_ACTION(set_local_timestep_action_type);

void node_client::set_local_timestep(integer idx, real dt) const {
    hpx::apply<typename node_server::set_local_timestep_action>(get_unmanaged_gid(), idx, dt);
}

void node_server::set_local_timestep(integer idx, real dt)
{
    local_timestep_channels[idx].set_value(dt);
}

hpx::future<void> node_server::timestep_driver_descend() {
    if (is_refined) {
        std::array<hpx::future<real>, NCHILD+1> futs;
        integer index = 0;
        for(auto& local_timestep: local_timestep_channels)
        {
            futs[index++] = local_timestep.get_future();
        }

        return hpx::dataflow(hpx::launch::sync,
            hpx::util::annotated_function(
                [this](std::array<hpx::future<real>, NCHILD+1> dts_fut)
                {
                    auto dts = hpx::util::unwrapped(dts_fut);
                    real dt = *std::min_element(dts.begin(), dts.end());

                    if (my_location.level() == 0)
                    {
                        timestep_driver_ascend(dt);
                    }
                    else
                    {
                        parent.set_local_timestep(my_location.get_child_index(), dt);
                    }

                    return;
                },
                "node_server::timestep_driver_descend"),
            futs);
    } else {
        return local_timestep_channels[NCHILD].get_future().then(hpx::launch::sync,
            [this](hpx::future<real>&& f)
            {
                real dt = f.get();
                parent.set_local_timestep(my_location.get_child_index(), dt);
                return;
            });
    }
}

typedef node_server::velocity_inc_action velocity_inc_action_type;
HPX_REGISTER_ACTION(velocity_inc_action_type);

hpx::future<void> node_client::velocity_inc(const space_vector& dv) const {
    return hpx::async<typename node_server::velocity_inc_action>(get_gid(), dv);
}

void node_server::velocity_inc(const space_vector& dv) {
    if (is_refined) {
        std::array<hpx::future<void>, NCHILD> futs;
        integer index = 0;
        for (auto& child : children) {
            futs[index++] = child.velocity_inc(dv);
        }
        wait_all_and_propagate_exceptions(futs);
    } else {
        grid_ptr->velocity_inc(dv);
    }
}

