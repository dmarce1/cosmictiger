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

#include <fstream>
#include <iostream>

#include <hpx/modules/program_options.hpp>

#include <cosmictiger/constants.hpp>
#include <cosmictiger/defs.hpp>
#include <cosmictiger/gravity.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/safe_io.hpp>
#include <cosmictiger/tree.hpp>
#include <cosmictiger/view.hpp>

#define SHOW(label, opt) show(label, opts.opt)

void show(char const* name, bool opt) {
    PRINT("%-20s: %c\n", name, opt ? 'T' : 'F');
}

void show(char const* name, int opt) {
    PRINT("%-20s: %i\n", name, opt);
}

void show(char const* name, double opt) {
    PRINT("%-20s: %e\n", name, opt);
}

void show(char const* name, std::string opt) {
    PRINT("%-20s: %s\n", name, opt.c_str());
}

Options globalOpts;

HPX_PLAIN_ACTION(setOptions, SetOptionsAction);

Options const& getOptions() {
    return globalOpts;
}

void setOptions(Options const& opts) {
    std::vector<hpx::future<void>> futs;
    for (auto c : hpxChildren()) {
        futs.push_back(hpx::async<SetOptionsAction>(c, opts));
    }
    globalOpts = opts;
    hpx::wait_all(futs.begin(), futs.end());
}

bool processOptions(int argc, char* argv[]) {
    Options opts{};
    namespace po = hpx::program_options;
    bool rc;
    po::options_description commandOpts("options");

    commandOpts.add_options()               //
        ("help", "produce help message")    //
        ("config_file", po::value<std::string>(&(opts.configFile))->default_value(""),
            "configuration file")    //
        ("lc_dir", po::value<std::string>(&(opts.lcDir))->default_value("./lightcone"),
            "directory for lightcone output")    //
        ("gadget4_restart", po::value<std::string>(&(opts.gadget4Restart))->default_value(""),
            "file to read gadget4 file and output power spectrum")    //
        ("read_check", po::value<int>(&(opts.readCheck))->default_value(-1),
            "read checkpoint from checkpoint.hello and then move checkpoint.hello to "
            "checkpoint.goodbye (default = false)")    //
        ("use_glass", po::value<bool>(&(opts.useGlass))->default_value(false),
            "Use glass file for IC")    //
        ("create_glass", po::value<bool>(&(opts.createGlass))->default_value(false),
            "Create glass file")    //
        ("plummer", po::value<bool>(&(opts.plummer))->default_value(false),
            "Do Plummer model")    //
        ("plummerR", po::value<double>(&(opts.plummerR))->default_value(1.0e-2),
            "Plummer radius")    //
        ("close_pack", po::value<bool>(&(opts.closePack))->default_value(false),
            "Use close packing for grid")    //
#ifdef USE_CUDA
        ("cuda", po::value<bool>(&(opts.cuda))->default_value(true),
            "use CUDA (default=true)")    //
#else
        ("cuda", po::value<bool>(&(opts.cuda))->default_value(false),
            "use CUDA (not enabled for this build)")    //
#endif
        ("check_freq", po::value<int>(&(opts.checkFreq))->default_value(1000000000),
            "time int seconds after startup to dump checkpoint \"checkpoint.hello\" and exit "
            "(default=3600)")    //
        ("max_iter", po::value<int>(&(opts.maxIter))->default_value(1000000),
            "maximum number of time-steps (default=1000000)")    //
        ("do_lc", po::value<bool>(&(opts.doLc))->default_value(false),
            "do lightcone analysis (default=false)")    //
        ("do_power", po::value<bool>(&(opts.doPower))->default_value(false),
            "do mass power spectrum analysis (default=false)")    //
        ("do_groups", po::value<bool>(&(opts.doGroups))->default_value(false),
            "do group analysis (default=false)")    //
        ("do_tracers", po::value<bool>(&(opts.doTracers))->default_value(false),
            "output tracer_count number of tracer particles to SILO (default=false)")             //
        ("bucket_size", po::value<int>(&(opts.bucketSize))->default_value(128), "bucket size")    //
        ("minrung", po::value<int>(&(opts.minrung))->default_value(0),
            "minimum starting rung")    //
        ("tracer_count", po::value<int>(&(opts.tracerCount))->default_value(1000000),
            "number of tracer particles (default=1000000)")    //
        ("do_slice", po::value<bool>(&(opts.doSlice))->default_value(false),
            "output a projection of a slice through the volume (default=false)")    //
        ("do_views", po::value<bool>(&(opts.doViews))->default_value(false),
            "output instantaneous healpix maps (default=false)")    //
        ("use_power_file", po::value<bool>(&(opts.usePowerFile))->default_value(true),
            "read initial power spectrum from power.init - must be evenly spaced in log k "
            "(default=false)")    //
        ("twolpt", po::value<bool>(&(opts.twolpt))->default_value(false),
            "use 2LPT initial conditions (default = true)")    //
        ("lc_b", po::value<double>(&(opts.lcB))->default_value(0.28),
            "linking length for lightcone group finder")    //
        ("lc_map_size", po::value<int>(&(opts.lcMapSize))->default_value(-1),
            "Nside for lightcone HEALPix map")    //
        ("view_size", po::value<int>(&(opts.viewSize))->default_value(1024),
            "view healpix Nside")    //
        ("slice_res", po::value<int>(&(opts.sliceRes))->default_value(4096),
            "slice resolution")                                                                //
        ("parts_dim", po::value<int>(&(opts.partsDim))->default_value(128), "nparts^(1/3)")    //
        ("nsteps", po::value<int>(&(opts.numberSteps))->default_value(64),
            "Number of super-timesteps")                                                   //
        ("z0", po::value<double>(&(opts.z0))->default_value(49.0), "starting redshift")    //
        ("z1", po::value<double>(&(opts.z1))->default_value(0.0), "ending redshift")       //
        ("theta", po::value<double>(&(opts.theta))->default_value(0.8),
            "opening angle for test problems")    //
        ("hsoft", po::value<double>(&(opts.hsoft))->default_value(1.0 / 80.0),
            "dark matter softening in units of interparticle spacing")    //
        ("eta", po::value<double>(&(opts.eta))->default_value(0.25),
            "time-step criterion (default=0.141)")    //
        ("test", po::value<std::string>(&(opts.test))->default_value(""),
            "name of test to run")                                                          //
        ("omega_k", po::value<double>(&(opts.omegaK))->default_value(0.0), "")              //
        ("omega_lam", po::value<double>(&(opts.omegaLam))->default_value(-1.0), "")         //
        ("omega_b", po::value<double>(&(opts.omegaB))->default_value(0.049389), "")         //
        ("omega_c", po::value<double>(&(opts.omegaC))->default_value(0.26503), "")          //
        ("Neff", po::value<double>(&(opts.neff))->default_value(3.046), "")                 //
        ("Theta", po::value<double>(&(opts.thetaCmb))->default_value(2.7255 / 2.73), "")    //
        ("Y0", po::value<double>(&(opts.y0))->default_value(0.2454006), "")                 //
        ("sigma8", po::value<double>(&(opts.sigma8))->default_value(0.8607), "")            //
        ("hubble", po::value<double>(&(opts.hubble))->default_value(0.6732), "")            //
        ("ns", po::value<double>(&(opts.ns))->default_value(0.96605), "spectral index")     //
        ("code_to_g", po::value<double>(&(opts.codeToG))->default_value(1.e9 / .6732),
            "mass resolution")    //

        ("hydro.sph", po::value<bool>(&(opts.sph))->default_value(false), "use SPH")    //
        ("hydro.chem", po::value<bool>(&(opts.chem))->default_value(false),
            "enable gas chemistry")    //
        ("hydro.conduction", po::value<bool>(&(opts.conduction))->default_value(false),
            "do conduction")                                                                      //
        ("hydro.gravity", po::value<bool>(&(opts.gravity))->default_value(true), "do gravity")    //
        ("hydro.diffusion", po::value<bool>(&(opts.diffusion))->default_value(false),
            "do diffusion")    //
        ("hydro.yreflect", po::value<bool>(&(opts.yreflect))->default_value(false),
            "Reflecting y for SPH only")    //
        ("hydro.gy", po::value<double>(&(opts.gy))->default_value(0.0),
            "gravitational acceleration in y direction (for SPH)")    //
        ("hydro.alpha0", po::value<double>(&(opts.alpha0))->default_value(0.05),
            "alpha0 viscosity")    //
        ("hydro.alpha1", po::value<double>(&(opts.alpha1))->default_value(1.0),
            "alpha1 for viscosity")    //
        ("hydro.alphaDecay", po::value<double>(&(opts.alphaDecay))->default_value(0.1),
            "alpha_decay time for viscosity")    //
        ("hydro.beta", po::value<double>(&(opts.beta))->default_value(2.0),
            "beta for viscosity")    //
        ("hydro.gamma", po::value<double>(&(opts.gamma))->default_value(5.0 / 3.0),
            "gamma for when chemistry is off")    //
        ("hydro.gcentral", po::value<double>(&(opts.gcentral))->default_value(0.0),
            "magnitude of central force")    //
        ("hydro.hcentral", po::value<double>(&(opts.hcentral))->default_value(0.01),
            "softening length for central force")    //
        ("hydro.viscType", po::value<int>(&(opts.viscType))->default_value(0),
            "AV type - 0 = Hu 1 = M&M 2 = constant")    //
        ("hydro.hmin", po::value<double>(&(opts.hmin))->default_value(1.0e-6),
            "minimum smoothing length")    //
        ("hydro.hmax", po::value<double>(&(opts.hmax))->default_value(1000.0),
            "maximum smoothing length")    //
        ("hydro.sneighborNumber", po::value<double>(&(opts.sneighborNumber))->default_value(128),
            "SPH neighbor number")                                                            //
        ("hydro.cfl", po::value<double>(&(opts.cfl))->default_value(0.2), "CFL condition")    //
        ;
    PRINT("Processing options\n");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, commandOpts), vm);
    po::notify(vm);
    if (vm.count("help")) {
        std::cout << commandOpts << "\n";
        rc = false;
    } else {
        if (!opts.configFile.empty()) {
            std::ifstream cfgFs{vm["config_file"].as<std::string>()};
            if (cfgFs) {
                po::store(po::parse_config_file(cfgFs, commandOpts), vm);
                rc = true;
            } else {
                PRINT("Configuration file %s not found!\n", opts.configFile.c_str());
                return false;
            }
        } else {
            rc = true;
        }
    }

    if (rc) {
        po::notify(vm);
    }
    if (!rc) {
        return false;
    }
    opts.glass = 0;
    opts.stars = false;
    opts.vsoft = false;
    opts.htime = true;
    opts.damping = 0.0;
    opts.sigma8C = opts.sigma8;
    opts.sphBucketSize = opts.bucketSize;
    opts.hmin /= opts.partsDim;
    opts.hmax /= opts.partsDim;
    bool const hydroTest = opts.test == "sphGravity" || opts.test == "sod" ||
        opts.test == "blast" || opts.test == "helmholtz" || opts.test == "rt" ||
        opts.test == "disc" || opts.test == "star" || opts.test == "plummer";
    if (hydroTest) {
        opts.sph = true;
        opts.chem = false;
        opts.stars = false;
        opts.conduction = false;
        opts.diffusion = false;
        opts.gravity = opts.test == "star" || opts.test == "plummer";
    }
    if (opts.sph) {
        opts.htime = false;
        if (!opts.cuda && opts.test != "sphGravity") {
            THROW_ERROR("SPH requires the CUDA backend\n");
        }
        if (opts.createGlass || opts.plummer || !opts.gadget4Restart.empty()) {
            THROW_ERROR("SPH cannot use the dark-matter-only glass, Plummer or Gadget loaders\n");
        }
        if (opts.doLc || opts.doGroups) {
            THROW_ERROR(
                "The lightcone and halo finders require equal-mass dark matter particles\n");
        }
        if (opts.stars && !opts.chem) {
            THROW_ERROR("Star formation requires hydro.chem=true\n");
        }
    } else {
        opts.chem = opts.stars = opts.conduction = opts.diffusion = false;
    }
    opts.hsoft *= selfPhi();
    opts.eta /= sqrt(selfPhi());
    opts.numberParticles = sqr(opts.partsDim) * opts.partsDim;
    opts.nfour = opts.partsDim;
    if (opts.closePack) {
        opts.numberParticles *= 2;
        //		opts.Nfour *= 2;
    }
    opts.treeCacheLineSize = 65536 / sizeof(TreeNode);
    opts.treeAllocLineSize = 16 * opts.treeCacheLineSize;
    opts.partCacheLineSize = 131072 / (sizeof(Fixed32) * NDIM);
    opts.saveForce = opts.test == "force";
    opts.omegaM = opts.omegaB + opts.omegaC;
    opts.dmMass = opts.sph ? opts.omegaC / opts.omegaM : 1.0;
    opts.sphMass = opts.sph ? opts.omegaB / opts.omegaM : 0.0;
    opts.rho0B = opts.numberParticles * opts.omegaB / opts.omegaM;
    opts.rho0C = opts.numberParticles * opts.omegaC / opts.omegaM;

    opts.codeToG *= constants::kM0;
    opts.codeToCm = pow(opts.codeToG * (8.0 * M_PI) * opts.numberParticles * constants::kG /
            (3.0 * opts.omegaM * sqr(constants::kH0 * opts.hubble)),
        1.0 / 3.0);
    opts.codeToS = opts.codeToCm / constants::kC;
    double h = constants::kH0 * opts.codeToS;
    opts.gm = opts.omegaM * 3.0 * sqr(h * opts.hubble) / (8.0 * M_PI) / opts.numberParticles;
    PRINT("box_size = %e Mpc\n", opts.codeToCm / constants::kMpcToCm);
    opts.sliceSize = std::min((20.0 / opts.hubble) / (opts.codeToCm / constants::kMpcToCm), 1.0);
    double const neff = 3.086;
    double const thetaValue = 1.0;
    auto const nside1 = pow(M_PI * opts.numberParticles / 10000.0 / 12.0, 0.5);
    int nside = 1;
    while (nside < nside1 / 2.0) {
        nside *= 2;
    }
    opts.lcMapSize = nside;
    PRINT("lc map size = %e\n", nside1);

    double omegaR = 32.0 * M_PI / 3.0 * constants::kG * constants::kSigma *
        (1 + neff * (7. / 8.0) * std::pow(4. / 11., 4. / 3.)) * std::pow(constants::kH0, -2) *
        std::pow(constants::kC, -3) * std::pow(2.73 * thetaValue, 4) * std::pow(opts.hubble, -2);
    opts.omegaNu = omegaR * opts.neff / (8.0 / 7.0 * std::pow(11.0 / 4.0, 4.0 / 3.0) + opts.neff);
    opts.omegaGam = omegaR - opts.omegaNu;
    opts.omegaR = omegaR;
    if (opts.omegaLam < 0.0) {
        opts.omegaLam = 1.0 - opts.omegaM - opts.omegaR;
        opts.omegaK = 0.0;
    } else {
        opts.omegaK = 1.0 - opts.omegaM - opts.omegaR - opts.omegaLam;
    }
    opts.linkLen = pow(opts.numberParticles, -1.0 / NDIM) * 0.28;
    opts.minGroup = 20;
    opts.lcMinGroup = 20;
    if (opts.createGlass) {
        opts.gm = 1.0 / opts.numberParticles;
        opts.codeToCm = constants::kMpcToCm;
        opts.hubble = 1.0;
    }
    if (opts.partsDim % 2 == 1) {
        THROW_ERROR("parts_dim must be an even number\n");
    }
    opts.hsoft *= pow(opts.numberParticles, -1.0 / NDIM);
    if (opts.plummer) {
        opts.hsoft = selfPhi() *
            pow((4.0 / 3.0 * M_PI * pow(opts.plummerR, 3.0)) / opts.numberParticles, 1.0 / 3.0);
    }
    SHOW("sph", sph);
    SHOW("htime", htime);
    SHOW("chem", chem);
    SHOW("stars", stars);
    PRINT("Simulation Options\n");
    PRINT("code_to_M_solar = %e\n", opts.codeToG / 1.98e33);
    PRINT("Box size = %e cm/h\n", opts.codeToCm / constants::kMpcToCm * opts.hubble);
    if (opts.plummer) {
        opts.gm = 1.0 / opts.numberParticles;
        opts.theta = 0.4;
    }
#ifdef CHECK_MUTUAL_SORT
    opts.doGroups = true;
#endif

    SHOW("check_freq", checkFreq);
    SHOW("code_to_cm", codeToCm);
    SHOW("code_to_g", codeToG);
    SHOW("code_to_s", codeToS);
    SHOW("cuda", cuda);
    SHOW("do_lc", doLc);
    SHOW("do_groups", doGroups);
    SHOW("do_power", doPower);
    SHOW("do_slice", doSlice);
    SHOW("do_tracers", doTracers);
    SHOW("do_views", doViews);
    SHOW("eta", eta);
    SHOW("GM", gm);
    SHOW("hsoft", hsoft);
    SHOW("hubble", hubble);
    SHOW("lc_b", lcB);
    SHOW("lc_min_group", lcMinGroup);
    SHOW("lc_map_size", lcMapSize);
    SHOW("link_len", linkLen);
    SHOW("max_iter", maxIter);
    SHOW("min_group", minGroup);
    SHOW("omega_b", omegaB);
    SHOW("omega_c", omegaC);
    SHOW("omega_gam", omegaGam);
    SHOW("omega_m", omegaM);
    SHOW("omega_nu", omegaNu);
    SHOW("omega_r", omegaR);
    SHOW("part_cache_line_size", partCacheLineSize);
    SHOW("parts_dim", partsDim);
    SHOW("read_check", readCheck);
    SHOW("save_force", saveForce);
    SHOW("sigma8", sigma8);
    SHOW("slice_res", sliceRes);
    SHOW("slice_size", sliceSize);
    SHOW("Theta", thetaCmb);
    SHOW("theta", theta);
    SHOW("tracer_count", tracerCount);
    SHOW("tree_cache_line_size", treeCacheLineSize);
    SHOW("twolpt", twolpt);
    SHOW("use_power_file", usePowerFile);
    SHOW("view_size", viewSize);
    SHOW("z0", z0);
    SHOW("z1", z1);

    SHOW("ns", ns);
    SHOW("Y0", y0);
    SHOW("Neff", neff);
    SHOW("config_file", configFile);
    SHOW("test", test);
#ifndef USE_CUDA
    if (opts.cuda) {
        THROW_ERROR("This executable was compiled without CUDA support\n");
    }
#endif
    setOptions(opts);
    viewReadViewFile();
    return rc;
}
