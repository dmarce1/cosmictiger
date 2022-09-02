#pragma once
#include <cosmictiger/defs.hpp>
#ifdef TREEPM


#include <cosmictiger/particles.hpp>
#include <cosmictiger/device_vector.hpp>

void treepm_compute_gravity(int Nres, bool do_phi);
void treepm_create_chainmesh(int Nres);
void treepm_cleanup();
device_vector<float> treepm_compute_density_local(int Nres, const device_vector<pair<part_int>>& chain_mesh, range<int64_t> int_box, range<int64_t> chain_box, range<int64_t> rho_box);
void treepm_allocate_fields(int Nres_);
CUDA_EXPORT float treepm_get_field(int dim, float x, float y, float z);
void treepm_set_field(int dim, const vector<float>& values);
void treepm_free_fields();
range<int64_t> treepm_get_fourier_box(int Nres);

#endif
