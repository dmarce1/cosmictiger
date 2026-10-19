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

#include <set>
#include <string>

#include <cosmictiger/core/containers.hpp>
#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/cuda/cuda.hpp>

static int rank;
static int numberOfRanks;
static Vector<hpx::id_type> localities;
static Vector<hpx::id_type> children;
static int ppnode;
static bool hyper;

HPX_PLAIN_ACTION(hpxInit, HpxInitAction);
HPX_PLAIN_ACTION(hpxInitCuda, HpxInitCudaAction);

static void detectProcessesPerNode();
static bool detectHyperthreading();

void hpxYield() {
    hpx::this_thread::yield();
}

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
        PRINT("Detected a hardware concurrency of %i\n", threads);
        firstCall = false;
    }
    return threads;
}

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

void hpxInitCuda() {
#ifdef USE_CUDA
    cudaInit();
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

int hpxRank() {
    return rank;
}

int hpxSize() {
    return numberOfRanks;
}

Vector<hpx::id_type> const& hpxLocalities() {
    return localities;
}

Vector<hpx::id_type> const& hpxChildren() {
    return children;
}

static bool detectHyperthreading() {
    FILE* fp = fopen("/sys/devices/system/cpu/smt/active", "rt");
    if (fp != nullptr) {
        char c;
        fread(&c, sizeof(char), 1, fp);
        auto rc = c == '1';
        if (rc) {
            PRINT("Hyperthreading detected\n");
        } else {
            PRINT("Hyperthreading NOT detected\n");
        }
        return rc;
    } else {
        return false;
    }
}

static std::string getHostname() {
    char buffer[1024];
    gethostname(buffer, 1023);
    std::string name = buffer;
    return name;
}

HPX_PLAIN_ACTION(getHostname, GetHostnameAction);

static void detectProcessesPerNode() {
    Vector<hpx::future<std::string>> futs;
    for (int i = 0; i < localities.size(); i++) {
        futs.push_back(hpx::async<GetHostnameAction>(localities[i]));
    }

    std::set<std::string> names;
    for (auto& f : futs) {
        auto const name = f.get();
        if (names.find(name) == names.end()) {
            names.insert(name);
        }
    }
    ppnode = localities.size() / names.size();
    PRINT("Detected %i processes per node\n", ppnode);
}
