
#include <tigerfmm/containers.hpp>

#include <hpx/hpx.hpp>
#include <hpx/parallel/algorithms/sort.hpp>
#include <hpx/parallel/algorithms/copy.hpp>

void hpx_init();
int hpx_rank();
int hpx_size();
const vector<hpx::id_type>& hpx_localities();
const vector<hpx::id_type>& hpx_children();


using mutex_type = hpx::lcos::local::mutex;
