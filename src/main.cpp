/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <format>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#if COSMICTIGER_WITH_GRAVITY
#include <fftw3.h>
#endif
#include <hpx/modules/program_options.hpp>
#include <mpi.h>

#if COSMICTIGER_WITH_GRAVITY
#include <cosmictiger/analysis/lightcone.hpp>
#include <cosmictiger/math/vector3.hpp>
#endif
#include <cosmictiger/core/driver.hpp>
#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/interrupt_profile.hpp>
#include <cosmictiger/core/memused.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/core/options_metadata.hpp>
#include <cosmictiger/core/profiler.hpp>
#include <cosmictiger/core/profiling_runtime.hpp>
#include <cosmictiger/core/stack_trace.hpp>
#include <cosmictiger/core/unordered_set_ts.hpp>
#if COSMICTIGER_WITH_GRAVITY
#include <cosmictiger/cosmology/fftw_threads.hpp>
#endif
#if COSMICTIGER_WITH_GRAVITY
#include <cosmictiger/gravity/fmm_order.hpp>
#include <cosmictiger/gravity/fmm_types.hpp>
#include <cosmictiger/gravity/gravity.hpp>
#include <cosmictiger/gravity/kernels/cartesian_tracefree_operators.hpp>
#include <cosmictiger/gravity/pmmm/solver.hpp>
#include <cosmictiger/gravity/tree.hpp>
#endif
#include <cosmictiger/io/checkpoint.hpp>
#include <cosmictiger/io/output.hpp>
#include <cosmictiger/math/fp16.hpp>
#include <cosmictiger/math/position.hpp>
#include <cosmictiger/math/simd.hpp>
#include <cosmictiger/particles/particles.hpp>
#ifdef COSMICTIGER_BUILD_TESTING
#include <tests/test.hpp>
#endif

namespace detail {

// The coordinator owns the phase stack; print and save its exclusive phase times.
void writeProfile() {
    if (hpxRank() != 0 || !getOptions().run.test.empty()) {
        return;
    }
    std::string const report = profile::profileReport();
    if (report.empty()) {
        return;
    }
    std::cout << std::format("\n{:s}", report.c_str()) << std::flush;
    profile::profileOutput(outputPath("profile.txt"));
}

}    // namespace detail

int hpx_main(int argc, char* argv[]) {
    std::atomic<int> i;
    if (!i.is_lock_free()) {
        std::cout << "std::atomic<int> is not lock free!\n" << std::flush;
        std::cout << "std::atomic<int> must be lock_free for CosmicTiger to run properly.\n"
                  << std::flush;
        std::cout << "Exiting...\n" << std::flush;
        return hpx::finalize();
    }
    hpxInit();
    bool daemonStarted = false;
    try {
        profile::checkForInterrupt();
        if (processOptions(argc, argv)) {
            profile::checkForInterrupt();
            validateCheckpointOutput();
            startMemuseDaemon();
            daemonStarted = true;
#if COSMICTIGER_WITH_GRAVITY
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

#endif

#ifdef COSMICTIGER_BUILD_TESTING
            // The metadata-only checkpoint test needs no GPU.
            if (getOptions().run.test == "checkpointState" && !getOptions().hydro.enabled) {
                test(getOptions().run.test);
                stopMemuseDaemon();
                daemonStarted = false;
                return hpx::finalize();
            }
#endif
#ifdef USE_CUDA
            if (usesCuda(getOptions())) {
                hpxInitCuda();
            }
#endif
            profile::checkForInterrupt();
            profile::checkForInterrupt();
#ifdef COSMICTIGER_BUILD_TESTING
            if (getOptions().run.test != "") {
                test(getOptions().run.test);
            } else {
#endif
                driver();
                detail::writeProfile();
#ifdef COSMICTIGER_BUILD_TESTING
            }
#endif
        }
#if COSMICTIGER_WITH_GRAVITY
        pmmmDestroy();
#endif
        particlesDestroy();
    } catch (profile::Interrupted const&) {
        std::cout << "\n  Interrupted: finishing cleanup and writing profiling reports.\n"
                  << std::flush;
        detail::writeProfile();
        try {
#if COSMICTIGER_WITH_GRAVITY
            pmmmDestroy();
#endif
            particlesDestroy();
        } catch (std::exception const& error) {
            std::cout << std::format("Cleanup after interruption failed: {:s}\n", error.what())
                      << std::flush;
            if (daemonStarted) {
                stopMemuseDaemon();
            }
            hpx::finalize();
            return EXIT_FAILURE;
        } catch (...) {
            std::cout << "Cleanup after interruption failed with an unknown exception\n"
                      << std::flush;
            if (daemonStarted) {
                stopMemuseDaemon();
            }
            hpx::finalize();
            return EXIT_FAILURE;
        }
    } catch (std::exception const& error) {
        std::cout << std::format("CosmicTiger failed: {:s}\n", error.what()) << std::flush;
        try {
#if COSMICTIGER_WITH_GRAVITY
            pmmmDestroy();
#endif
        } catch (std::exception const& cleanupError) {
            std::cout << std::format("PMMM cleanup failed: {:s}\n", cleanupError.what())
                      << std::flush;
        } catch (...) {
            std::cout << "PMMM cleanup failed during exception handling\n" << std::flush;
        }
        if (daemonStarted) {
            stopMemuseDaemon();
        }
        hpx::finalize();
        return EXIT_FAILURE;
    } catch (...) {
        std::cout << "CosmicTiger failed with an unknown exception\n" << std::flush;
        try {
#if COSMICTIGER_WITH_GRAVITY
            pmmmDestroy();
#endif
        } catch (...) {
            std::cout << "PMMM cleanup failed during exception handling\n" << std::flush;
        }
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

namespace detail {

char const* mpiThreadSupportName(int threadSupport) {
    switch (threadSupport) {
    case MPI_THREAD_SINGLE:
        return "MPI_THREAD_SINGLE";
    case MPI_THREAD_FUNNELED:
        return "MPI_THREAD_FUNNELED";
    case MPI_THREAD_SERIALIZED:
        return "MPI_THREAD_SERIALIZED";
    case MPI_THREAD_MULTIPLE:
        return "MPI_THREAD_MULTIPLE";
    default:
        return "an unknown MPI thread support level";
    }
}

void initializeMpi(int& argc, char**& argv, bool& ownsMpi) {
    // MPI and HPX use hwloc for topology discovery. Its OpenGL component can
    // block connecting to an X server even for --help; display discovery is
    // unnecessary here. Preserve an explicitly configured component list.
    if (::setenv("HWLOC_COMPONENTS", "-gl", 0) != 0) {
        throw std::system_error(
            errno, std::generic_category(), "Unable to configure hwloc before MPI startup");
    }
    int finalized = 0;
    if (MPI_Finalized(&finalized) != MPI_SUCCESS || finalized) {
        throw std::runtime_error("Cannot start CosmicTiger after MPI finalization");
    }
    int initialized = 0;
    if (MPI_Initialized(&initialized) != MPI_SUCCESS) {
        throw std::runtime_error("MPI_Initialized failed");
    }
    if (!initialized) {
        int provided = MPI_THREAD_SINGLE;
        if (MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided) != MPI_SUCCESS) {
            throw std::runtime_error("MPI_Init_thread failed while requesting MPI_THREAD_MULTIPLE");
        }
        ownsMpi = true;
    }
    int provided = MPI_THREAD_SINGLE;
    if (MPI_Query_thread(&provided) != MPI_SUCCESS) {
        throw std::runtime_error("MPI_Query_thread failed");
    }
    // Before HPX starts, all MPI calls are on the initializing OS thread.
    // Make every rank reject together if any process has weaker thread support.
    int minimumProvided = MPI_THREAD_SINGLE;
    if (MPI_Allreduce(&provided, &minimumProvided, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD) !=
        MPI_SUCCESS) {
        throw std::runtime_error("MPI_Allreduce failed while checking MPI thread support");
    }
    if (minimumProvided != MPI_THREAD_MULTIPLE) {
        throw std::runtime_error(std::string("CosmicTiger requires MPI_THREAD_MULTIPLE; ") +
            "the lowest thread support across MPI ranks is " +
            mpiThreadSupportName(minimumProvided));
    }
}

#if COSMICTIGER_WITH_GRAVITY
void initializeFftw() {
    // Initialize before HPX starts: cosmology may create serial plans before
    // PMMM initializes FFTW MPI. Thread support must precede both kinds of use.
    int const initialized = fftwf_init_threads() ? 1 : 0;
    int allInitialized = 0;
    if (MPI_Allreduce(&initialized, &allInitialized, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD) !=
        MPI_SUCCESS) {
        throw std::runtime_error("MPI_Allreduce failed while checking FFTW thread support");
    }
    if (!allInitialized) {
        throw std::runtime_error("Unable to initialize FFTW thread support on every MPI rank");
    }
}
#endif

// Reject a position ABI mismatch before HPX/MPI initializes. Keep unknown
// application/HPX options for their normal parsers after this check.
void validatePositionType(int argc, char* argv[]) {
    namespace po = hpx::program_options;
    po::options_description selection;
    selection.add_options()("help", "help")("configuration", po::value<std::string>(), "INI file")(
        "compute.positionType", po::value<std::string>(), "position storage");
    po::variables_map values;
    po::store(
        po::command_line_parser(argc, argv)
            .options(selection)
            .style(po::command_line_style::default_style & ~po::command_line_style::allow_guessing)
            .allow_unregistered()
            .run(),
        values);
    if (!values.count("help") && values.count("configuration")) {
        auto const filename = values["configuration"].as<std::string>();
        if (!filename.empty()) {
            std::ifstream stream(filename);
            if (!stream) {
                throw std::runtime_error("Cannot open configuration file " + filename);
            }
            // As in processOptions, CLI values are stored first and take precedence.
            po::store(po::parse_config_file(stream, selection, true), values);
        }
    }
    std::string const requested = values.count("compute.positionType") ?
        values["compute.positionType"].as<std::string>() :
        (kPositionBits == 64 ? "fixed64" : "fixed32");
    if (requested != "fixed32" && requested != "fixed64") {
        if (values.count("help")) {
            return;
        }
        throw std::invalid_argument("compute.positionType must be fixed32 or fixed64");
    }
    if ((requested == "fixed64") == (kPositionBits == 64)) {
        return;
    }
    std::string const compiled = kPositionBits == 64 ? "fixed64" : "fixed32";
    throw std::invalid_argument("this build uses " + compiled +
        " positions; configure with -DCOSMICTIGER_POSITION_TYPE=" + requested + " to use " +
        requested);
}

}    // namespace detail

int main(int argc, char* argv[]) {
    StackTraceHandler stackTraceHandler;
    profile::InterruptReportHandler interruptReportHandler;
    bool ownsMpi = false;
    int result = EXIT_FAILURE;
    try {
        profile::configureRuntimeProfiling();
        detail::validatePositionType(argc, argv);
        recordCommandLineArguments(argc, argv);
        detail::initializeMpi(argc, argv, ownsMpi);
#if COSMICTIGER_WITH_GRAVITY
        detail::initializeFftw();
        hpx::register_startup_function(fftw_threads::initialize);
#endif
        // Startup callbacks also run on worker localities that do not enter hpx_main.
        hpx::register_startup_function(
            [&stackTraceHandler]() { stackTraceHandler.install(hpx::get_locality_id()); });
        hpx::register_startup_function(
            [&interruptReportHandler]() { interruptReportHandler.install(); });
        std::vector<std::string> cfg = {"hpx.commandline.allow_unknown=1"};
        cfg.push_back("hpx.stacks.small_size=524288");
#ifdef HPX_EARLY
        result = hpx::init(argc, argv, cfg);
#else
        hpx::init_params initParams;
        initParams.cfg = std::move(cfg);
        result = hpx::init(argc, argv, initParams);
#endif
    } catch (std::exception const& error) {
        std::fprintf(stderr, "CosmicTiger failed: %s\n", error.what());
    } catch (...) {
        std::fprintf(stderr, "CosmicTiger failed with an unknown startup exception\n");
    }
    // HPX leaves externally initialized MPI to its owner. Finalize on this
    // same OS thread only after HPX has stopped, including startup failures.
    if (ownsMpi && MPI_Finalize() != MPI_SUCCESS) {
        std::fprintf(stderr, "CosmicTiger failed: MPI_Finalize failed\n");
        result = EXIT_FAILURE;
    }
    return result;
}
