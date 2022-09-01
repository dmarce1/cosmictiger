#pragma once
#include <cosmictiger/defs.hpp>
#ifdef TREEPM


#include <cosmictiger/particles.hpp>
#include <cosmictiger/device_vector.hpp>

void treepm_compute_gravity(int Nres, bool do_phi);
void treepm_create_chainmesh(int Nres);
void treepm_restore_parts_size();
device_vector<float> treepm_compute_density_local(int Nres, const device_vector<pair<part_int>>& chain_mesh, range<int64_t> int_box, range<int64_t> chain_box, range<int64_t> rho_box);


#endif
