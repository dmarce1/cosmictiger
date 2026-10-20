/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

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

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cosmictiger/analysis/lightcone.hpp>
#include <cosmictiger/analysis/rockstar.hpp>
#include <cosmictiger/core/driver.hpp>
#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/memused.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/core/options_metadata.hpp>
#include <cosmictiger/core/unordered_set_ts.hpp>
#include <cosmictiger/fmm/ewald_indices.hpp>
#include <cosmictiger/fmm/fmm_kernels.hpp>
#include <cosmictiger/fmm/fmm_order.hpp>
#include <cosmictiger/fmm/fmm_types.hpp>
#include <cosmictiger/gravity/gravity.hpp>
#include <cosmictiger/gravity/tree.hpp>
#include <cosmictiger/io/checkpoint.hpp>
#include <cosmictiger/io/output.hpp>
#include <cosmictiger/math/fp16.hpp>
#include <cosmictiger/math/simd.hpp>
#include <cosmictiger/particles/particles.hpp>
#include <cosmictiger/sph/sph.hpp>
#include <tests/test.hpp>

int hpx_main(int argc, char* argv[]) {
    /*	simd_double8 x;
     for( x[0] = 0.0; x[0] < 0.87; x[0] += 0.01 ) {
     PRINT( "%e %e\n", -x[0], (erfnearzero(-x)[0] - erf(-x[0]))/erf(-x[0]));
     }*/

    std::atomic<int> i;
    PRINT("%i\n", sizeof(RockstarRecord));
    for (double q = 0.0; q < 1.0; q += 0.01) {
        //		PRINT( "%e %e %e\n",q, sph_Wh3(q,1.0),sph_dWh3dq(q,1.0));
    }
    if (!i.is_lock_free()) {
        PRINT("std::atomic<int> is not lock free!\n");
        PRINT("std::atomic<int> must be lock_free for CosmicTiger to run properly.\n");
        PRINT("Exiting...\n");
        return hpx::finalize();
    } else {
        PRINT("std::atomic<int> is lock free!\n");
    }
    PRINT("tree_node size = %i\n", sizeof(TreeNode));
    hpxInit();
    bool daemonStarted = false;
    try {
        if (processOptions(argc, argv)) {
            startMemuseDaemon();
            daemonStarted = true;
            FILE* fp = fopen(outputPath("soft.txt").c_str(), "wt");
            if (fp == nullptr) {
                THROW_ERROR("Unable to write soft.txt\n");
            }
            for (double r = 0.0; r < 1.0; r += 0.01) {
                double f, phi;
                gravitySofteningKernel(f, phi, r * r, 1.0, 1.0, 1.0, true);
                fprintf(fp, "%e %e %e\n", r, phi, f);
            }
            fclose(fp);

            // Help, invalid options and the metadata-only checkpoint test need no GPU.
            if (getOptions().run.test == "checkpointState" && !getOptions().hydro.enabled) {
                checkpointStateTest();
                stopMemuseDaemon();
                daemonStarted = false;
                return hpx::finalize();
            }
#ifdef USE_CUDA
            if (getOptions().compute.cuda) {
                hpxInitCuda();
                healpixInit();
            }
#endif
            EwaldConst::init();
            // Debug dump of the Ewald Green's function along x for the runtime order.
            fmmDispatch([](auto orderTag) {
                constexpr int kOrder = decltype(orderTag)::value;
                FmmExpansion<SimdFloat, kOrder> d;
                Array<SimdFloat, NDIM> xValue;
                FILE* fp = fopen(outputPath("ewald.txt").c_str(), "wt");
                if (fp == nullptr) {
                    THROW_ERROR("Unable to write ewald.txt\n");
                }
                xValue[0] = xValue[1] = xValue[2] = SimdFloat(0.0f);
                for (double x = 0.00; x < 0.5; x += 0.001) {
                    xValue[0] = simdBroadcast<SimdFloat>(x);
                    d = SimdFloat(0.0f);
                    ewaldGreensFunction(d, xValue);
                    fprintf(fp, "%e ", x);
                    for (int n = 0; n < kOrder; n++) {
                        for (int m = 0; m < kOrder - n; m++) {
                            for (int l = 0; l < kOrder - n - m; l++) {
                                if (l > 1 && !(l == 2 && n == 0 && m == 0)) {
                                    continue;
                                }
                                fprintf(fp, "%e ",
                                    double(static_cast<FmmExpansion<SimdFloat, kOrder> const&>(d)(
                                        n, m, l)[0]));
                            }
                        }
                    }
                    fprintf(fp, "\n");
                }
                fclose(fp);
            });
            if (getOptions().run.test != "") {
                test(getOptions().run.test);
            } else {
                driver();
            }
        }
        particlesDestroy();
    } catch (std::exception const& error) {
        PRINT("CosmicTiger failed: %s\n", error.what());
        if (daemonStarted) {
            stopMemuseDaemon();
        }
        hpx::finalize();
        return EXIT_FAILURE;
    } catch (...) {
        PRINT("CosmicTiger failed with an unknown exception\n");
        if (daemonStarted) {
            stopMemuseDaemon();
        }
        hpx::finalize();
        return EXIT_FAILURE;
    }
    if (daemonStarted) {
        stopMemuseDaemon();
    }
    return hpx::finalize();
}

int main(int argc, char* argv[]) {
    recordCommandLineArguments(argc, argv);
    PRINT("STARTING MAIN\n");
    std::vector<std::string> cfg = {"hpx.commandline.allow_unknown=1"};
    cfg.push_back("hpx.stacks.small_size=524288");
#ifdef HPX_EARLY
    return hpx::init(argc, argv, cfg);
#else
    hpx::init_params initParams;
    initParams.cfg = std::move(cfg);
    return hpx::init(argc, argv, initParams);
#endif
}
