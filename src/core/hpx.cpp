/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2021  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <cstdio>
#include <string>

#if COSMICTIGER_ENABLE_ROCKSTAR
#include <cosmictiger/analysis/lightcone.hpp>
#endif
#include <cosmictiger/core/containers.hpp>
#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/terminal.hpp>
#include <cosmictiger/cuda/cuda.hpp>

static int rank;
static int numberOfRanks;
static Vector<hpx::id_type> localities;
static Vector<hpx::id_type> children;
static int ppnode;
static int localRank;
static bool hyper;

HPX_PLAIN_ACTION(hpxInit, HpxInitAction);
HPX_PLAIN_ACTION(hpxInitCuda, HpxInitCudaAction);

static void detectProcessesPerNode();
static bool detectHyperthreading();

/// Yield the current HPX task so other queued work can run.
void hpxYield() {
    hpx::this_thread::yield();
}

/// Estimate the worker count per process after accounting for SMT and colocated ranks.
int hpxHardwareConcurrency() {
    static bool firstCall = true;
    int threads = hpx::thread::hardware_concurrency();
    if (hyper) {
        threads /= 2 * ppnode;
    } else {
        threads /= ppnode;
    }
    if (threads == 0) {
        threads = 1;
    }
    if (firstCall) {
        terminal::detailPrint("Detected a hardware concurrency of %i\n", threads);
        firstCall = false;
    }
    return threads;
}

/// Discover locality IDs, build the broadcast tree, and determine node-local rank
/// placement.
void hpxInit() {
    rank = hpx::get_locality_id();
    auto tmp = hpx::find_all_localities();
    localities.insert(localities.end(), tmp.begin(), tmp.end());
    numberOfRanks = localities.size();
    int base = (rank + 1) << 1;
    int const index1 = base - 1;
    int const index2 = base;
    if (index1 < numberOfRanks) {
        children.push_back(localities[index1]);
    }
    if (index2 < numberOfRanks) {
        children.push_back(localities[index2]);
    }

    Vector<hpx::future<void>> futs;
    for (auto c : hpxChildren()) {
        futs.push_back(hpx::async<HpxInitAction>(c));
    }
    hpx::wait_all(futs.begin(), futs.end());
    detectProcessesPerNode();
    hyper = detectHyperthreading();
}

/// Initialize each locality's CUDA backend and optional HEALPix device lookup tables.
void hpxInitCuda() {
#ifdef USE_CUDA
    cudaInit();
#if COSMICTIGER_ENABLE_ROCKSTAR
    healpixInit();
#endif
#endif
    Vector<hpx::future<void>> futures;
    for (auto const& child : hpxChildren()) {
        futures.push_back(hpx::async<HpxInitCudaAction>(child));
    }
    hpx::wait_all(futures.begin(), futures.end());
    for (auto& future : futures) {
        future.get();
    }
}

/// Return this locality's global rank.
int hpxRank() {
    return rank;
}

/// Return this locality's rank among processes on the same host.
int hpxLocalRank() {
    return localRank;
}

/// Return the total number of HPX localities.
int hpxSize() {
    return numberOfRanks;
}

/// Return the global locality-ID list.
Vector<hpx::id_type> const& hpxLocalities() {
    return localities;
}

/// Return this rank's children in the binary collective tree.
Vector<hpx::id_type> const& hpxChildren() {
    return children;
}

/// Read Linux's SMT activity flag, returning false if it cannot be read.
static bool detectHyperthreading() {
    FILE* fp = fopen("/sys/devices/system/cpu/smt/active", "rt");
    if (fp != nullptr) {
        char c;
        size_t const charsRead = fread(&c, sizeof(c), 1, fp);
        fclose(fp);
        if (charsRead != 1) {
            return false;
        }
        bool const rc = c == '1';
        if (rc) {
            terminal::detailPrint("Hyperthreading detected\n");
        } else {
            terminal::detailPrint("Hyperthreading NOT detected\n");
        }
        return rc;
    } else {
        return false;
    }
}

/// Return the operating system hostname used to group colocated ranks.
static std::string getHostname() {
    char buffer[1024];
    gethostname(buffer, 1023);
    std::string name = buffer;
    return name;
}

HPX_PLAIN_ACTION(getHostname, GetHostnameAction);

/// Query locality hostnames to determine this process's node-local rank and sharing count.
static void detectProcessesPerNode() {
    Vector<hpx::future<std::string>> futs;
    for (int i = 0; i < localities.size(); i++) {
        futs.push_back(hpx::async<GetHostnameAction>(localities[i]));
    }

    std::string const ownHostname = getHostname();
    ppnode = 0;
    localRank = 0;
    for (int i = 0; i < futs.size(); ++i) {
        if (futs[i].get() == ownHostname) {
            ++ppnode;
            if (i < rank) {
                ++localRank;
            }
        }
    }
    ALWAYS_ASSERT(ppnode > 0);
    terminal::detailPrint("Detected %i processes per node\n", ppnode);
}
