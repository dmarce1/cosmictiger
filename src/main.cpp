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
#include <string>
#include <utility>
#include <vector>

#include <cosmictiger/driver.hpp>
#include <cosmictiger/ewald_indices.hpp>
#include <cosmictiger/fmm_kernels.hpp>
#include <cosmictiger/fp16.hpp>
#include <cosmictiger/gravity.hpp>
#include <cosmictiger/healpix.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/memused.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/options_metadata.hpp>
#include <cosmictiger/particles.hpp>
#include <cosmictiger/rockstar.hpp>
#include <cosmictiger/simd.hpp>
#include <cosmictiger/sph.hpp>
#include <cosmictiger/test.hpp>
#include <cosmictiger/tree.hpp>
#include <cosmictiger/unordered_set_ts.hpp>

int hpx_main(int argc, char* argv[]) {
    /*	simd_double8 x;
     for( x[0] = 0.0; x[0] < 0.87; x[0] += 0.01 ) {
     PRINT( "%e %e\n", -x[0], (erfnearzero(-x)[0] - erf(-x[0]))/erf(-x[0]));
     }*/

    Expansion<SimdFloat> d;
    Array<SimdFloat, NDIM> xValue;

    std::atomic<int> i;
    PRINT("%i\n", sizeof(RockstarRecord));
    for (double q = 0.0; q < 1.0; q += 0.01) {
        //		PRINT( "%e %e %e\n",q, sph_Wh3(q,1.0),sph_dWh3dq(q,1.0));
    }
    FILE* fp = fopen("soft.txt", "wt");
    for (double r = 0.0; r < 1.0; r += 0.01) {
        double f, phi;
        gravitySofteningKernel(f, phi, r * r, 1.0, 1.0, 1.0, true);
        fprintf(fp, "%e %e %e\n", r, phi, f);
    }
    fclose(fp);
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
    PRINT("Starting mem use daemon\n");
    startMemuseDaemon();
    try {
        if (processOptions(argc, argv)) {
            // Help, invalid options and the metadata-only checkpoint test need no GPU.
            if (getOptions().test == "checkpointState" && !getOptions().sph) {
                checkpointStateTest();
                stopMemuseDaemon();
                return hpx::finalize();
            }
            hpxInitCuda();
            healpixInit();
            EwaldConst::init();
            FILE* fp = fopen("ewald.txt", "wt");
            xValue[0] = xValue[1] = xValue[2] = SimdFloat(0.0f);
            for (double x = 0.00; x < 0.5; x += 0.001) {
                xValue[0] = simdBroadcast<SimdFloat>(x);
                d = SimdFloat(0.0f);
                ewaldGreensFunction(d, xValue);
                fprintf(fp, "%e ", x);
                for (int n = 0; n < LORDER; n++) {
                    for (int m = 0; m < LORDER - n; m++) {
                        for (int l = 0; l < LORDER - n - m; l++) {
                            if (l > 1 && !(l == 2 && n == 0 && m == 0)) {
                                continue;
                            }
                            fprintf(fp, "%e ", double(d(n, m, l)[0]));
                        }
                    }
                }
                fprintf(fp, "\n");
            }
            fclose(fp);
            if (getOptions().test != "") {
                test(getOptions().test);
            } else {
                driver();
            }
        }
        particlesDestroy();
    } catch (std::exception const& error) {
        PRINT("CosmicTiger failed: %s\n", error.what());
        stopMemuseDaemon();
        hpx::finalize();
        return EXIT_FAILURE;
    } catch (...) {
        PRINT("CosmicTiger failed with an unknown exception\n");
        stopMemuseDaemon();
        hpx::finalize();
        return EXIT_FAILURE;
    }
    stopMemuseDaemon();
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
