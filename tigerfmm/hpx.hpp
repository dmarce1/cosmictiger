
#include <tigerfmm/containers.hpp>

#ifndef __CUDACC__

#include <hpx/hpx.hpp>
#include <hpx/parallel/algorithms/sort.hpp>
#include <hpx/parallel/algorithms/copy.hpp>

const vector<hpx::id_type>& hpx_localities();
const vector<hpx::id_type>& hpx_children();
void hpx_init();

using mutex_type = hpx::lcos::local::mutex;

#endif


int hpx_rank();
int hpx_size();

