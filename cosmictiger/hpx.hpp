
#include <cosmictiger/containers.hpp>

#ifndef __CUDACC__

#include <hpx/hpx.hpp>
#include <hpx/parallel/algorithms/sort.hpp>
#include <hpx/parallel/algorithms/copy.hpp>
#include <hpx/serialization/unordered_map.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/hpx_finalize.hpp>

#if (HPX_VERSION_FULL < ((1<<16) | (6<<8)))
#define HPX_EARLY
#else
#define HPX_LATE
#endif

#ifdef HPX_EARLY
#define PAR_EXECUTION_POLICY hpx::parallel::execution::par(hpx::parallel::execution::task)
#define HPX_PRIORITY_BOOST hpx::launch::async(hpx::threads::thread_priority_boost)
#define hpx_copy hpx::parallel::copy
#else
#define PAR_EXECUTION_POLICY hpx::execution::par(hpx::execution::task)
#define HPX_PRIORITY_BOOST hpx::launch::async(thread_priority::priority)
#define hpx_copy hpx::copy
#endif


const vector<hpx::id_type>& hpx_localities();
const vector<hpx::id_type>& hpx_children();
void hpx_init();

using mutex_type = hpx::lcos::local::mutex;
using spinlock_type = hpx::lcos::local::spinlock;
using shared_mutex_type = hpx::lcos::local::shared_mutex;

#endif


int hpx_rank();
int hpx_size();
void hpx_yield();
int hpx_hardware_concurrency();
