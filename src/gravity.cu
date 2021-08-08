#include <cosmictiger/cuda_reduce.hpp>
#include <cosmictiger/gravity.hpp>

__device__
int cuda_gravity_cc(const cuda_kick_data& data, expansion<float>& Lacc, const tree_node& self, const fixedcapvec<int, MULTLIST_SIZE>& multlist,
		gravity_cc_type type, bool do_phi) {
	int flops = 0;
	const int &tid = threadIdx.x;
	const auto& tree_nodes = data.tree_nodes;
	if (multlist.size()) {
		expansion<float> L;
		expansion<float> D;
		for (int i = 0; i < EXPANSION_SIZE; i++) {
			L[i] = 0.0f;
		}
		for (int i = tid; i < multlist.size(); i += WARP_SIZE) {
			const tree_node& other = tree_nodes[multlist[i]];
			const multipole<float>& M = other.multi;
			array<float, NDIM> dx;
			for (int dim = 0; dim < NDIM; dim++) {
				dx[dim] = distance(self.pos[dim], other.pos[dim]);
			}
			flops += 3;
			if (type == GRAVITY_CC_DIRECT) {
				flops += greens_function(D, dx);
			} else {
				flops += ewald_greens_function(D, dx);
			}
			flops += M2L(L, M, D, do_phi);
		}
		for (int i = 0; i < EXPANSION_SIZE; i++) {
			shared_reduce_add(L[i]);
		}
		for (int i = tid; i < EXPANSION_SIZE; i += WARP_SIZE) {
			Lacc[i] += L[i];
		}
		__syncwarp();
	}
	shared_reduce_add(flops);
	return flops;
}

__device__
int cuda_gravity_cp(const cuda_kick_data& data, expansion<float>& Lacc, const tree_node& self, const fixedcapvec<int, PARTLIST_SIZE>& partlist, bool do_phi) {
	int flops = 0;
	__shared__
	extern int shmem_ptr[];
	cuda_kick_shmem &shmem = *(cuda_kick_shmem*) shmem_ptr;
	const auto* main_src_x = data.x;
	const auto* main_src_y = data.y;
	const auto* main_src_z = data.z;
	auto& src_x = shmem.src.x;
	auto& src_y = shmem.src.y;
	auto& src_z = shmem.src.z;
	const auto* tree_nodes = data.tree_nodes;
	const int &tid = threadIdx.x;
	if (partlist.size()) {
		int part_index;
		expansion<float> L;
		for (int j = 0; j < EXPANSION_SIZE; j++) {
			L[j] = 0.0;
		}
		int i = 0;
		auto these_parts = tree_nodes[partlist[0]].part_range;
		const auto partsz = partlist.size();
		while (i < partsz) {
			part_index = 0;
			while (part_index < KICK_PP_MAX && i < partsz) {
				while (i + 1 < partsz) {
					const auto other_tree_parts = tree_nodes[partlist[i + 1]].part_range;
					if (these_parts.second == other_tree_parts.first) {
						these_parts.second = other_tree_parts.second;
						i++;
					} else {
						break;
					}
				}
				const part_int imin = these_parts.first;
				const part_int imax = min(these_parts.first + (KICK_PP_MAX - part_index), these_parts.second);
				const int sz = imax - imin;
				for (int j = tid; j < sz; j += WARP_SIZE) {
					const int i1 = part_index + j;
					const part_int i2 = j + imin;
					assert(i2 >= 0);
					assert(i2 < data.source_size);
					src_x[i1] = main_src_x[i2];
					src_y[i1] = main_src_y[i2];
					src_z[i1] = main_src_z[i2];
				}
				__syncwarp();
				these_parts.first += sz;
				part_index += sz;
				if (these_parts.first == these_parts.second) {
					i++;
					if (i < partsz) {
						these_parts = tree_nodes[partlist[i]].part_range;
					}
				}
			}
			__syncwarp();
			for (int j = tid; j < part_index; j += warpSize) {
				array<float, NDIM> dx;
				dx[XDIM] = distance(self.pos[XDIM], src_x[j]);
				dx[YDIM] = distance(self.pos[YDIM], src_y[j]);
				dx[ZDIM] = distance(self.pos[ZDIM], src_z[j]);
				flops += 3;
				expansion<float> D;
				flops += greens_function(D, dx);
				for (int k = 0; k < EXPANSION_SIZE; k++) {
					L[k] += D[k];
				}
				flops += EXPANSION_SIZE;
			}
		}
		for (int k = 0; k < EXPANSION_SIZE; k++) {
			shared_reduce_add(L[k]);
		}
		for (int i = tid; i < EXPANSION_SIZE; i += WARP_SIZE) {
			Lacc[i] += L[i];
		}

		__syncwarp();
	}
	shared_reduce_add(flops);
	return flops;

}

__device__
int cuda_gravity_pc(const cuda_kick_data& data, const tree_node&, const fixedcapvec<int, MULTLIST_SIZE>& multlist, int nactive, bool do_phi) {
	int flops = 0;
	const int &tid = threadIdx.x;
	__shared__
	extern int shmem_ptr[];
	cuda_kick_shmem &shmem = *(cuda_kick_shmem*) shmem_ptr;
	auto &gx = shmem.gx;
	auto &gy = shmem.gy;
	auto &gz = shmem.gz;
	auto &phi = shmem.phi;
	const auto& sink_x = shmem.sink_x;
	const auto& sink_y = shmem.sink_y;
	const auto& sink_z = shmem.sink_z;
	const auto* tree_nodes = data.tree_nodes;
	if (multlist.size()) {
		for (int n = 0; n < multlist.size(); n += WARP_SIZE) {
			const int maxi = min(multlist.size(), n + WARP_SIZE) - n;
			for (int i = 0; i < maxi; i++) {
				const auto& other = tree_nodes[multlist[n + i]];
				float* dest = (float*) &shmem.m[i];
				float* src = (float*) other.get_multipole_ptr();
				constexpr int size = sizeof(multipole_pos) / sizeof(float);
				for (int j = tid; j < size; j += WARP_SIZE) {
					dest[j] = src[j];
				}
			}
			__syncwarp();
			int kmid;
			if ((nactive % WARP_SIZE) < MIN_KICK_PC_WARP) {
				kmid = nactive - (nactive % WARP_SIZE);
			} else {
				kmid = nactive;
			}
			for (int k = tid; k < kmid; k += WARP_SIZE) {
				expansion2<float> L;
				L(0, 0, 0) = L(1, 0, 0) = L(0, 1, 0) = L(0, 0, 1) = 0.0f;
				for (int j = 0; j < maxi; j++) {
					array<float, NDIM> dx;
					const auto& pos = shmem.m[j].pos;
					const auto& M = shmem.m[j].m;
					dx[XDIM] = distance(sink_x[k], pos[XDIM]);
					dx[YDIM] = distance(sink_y[k], pos[YDIM]);
					dx[ZDIM] = distance(sink_z[k], pos[ZDIM]);
					flops += 3;
					expansion<float> D;
					flops += greens_function(D, dx);
					flops += M2L(L, M, D, do_phi);
				}
				gx[k] -= L(1, 0, 0);
				gy[k] -= L(0, 1, 0);
				gz[k] -= L(0, 0, 1);
				phi[k] += L(0, 0, 0);
			}
			__syncwarp();
			for (int k = kmid; k < nactive; k++) {
				expansion2<float> L;
				L(0, 0, 0) = L(1, 0, 0) = L(0, 1, 0) = L(0, 0, 1) = 0.0f;
				for (int j = tid; j < maxi; j += WARP_SIZE) {
					array<float, NDIM> dx;
					const auto& pos = shmem.m[j].pos;
					const auto& M = shmem.m[j].m;
					dx[XDIM] = distance(sink_x[k], pos[XDIM]);
					dx[YDIM] = distance(sink_y[k], pos[YDIM]);
					dx[ZDIM] = distance(sink_z[k], pos[ZDIM]);
					flops += 3;
					expansion<float> D;
					flops += greens_function(D, dx);
					flops += M2L(L, M, D, do_phi);
				}
				shared_reduce_add(L(0, 0, 0));
				shared_reduce_add(L(1, 0, 0));
				shared_reduce_add(L(0, 1, 0));
				shared_reduce_add(L(0, 0, 1));
				if (tid == 0) {
					gx[k] -= L(1, 0, 0);
					gy[k] -= L(0, 1, 0);
					gz[k] -= L(0, 0, 1);
					phi[k] += L(0, 0, 0);
				}
			}
		}
		__syncwarp();
	}
	shared_reduce_add(flops);
	__syncwarp();
	return flops;

}

__device__
int cuda_gravity_pp(const cuda_kick_data& data, const tree_node& self, const fixedcapvec<int, PARTLIST_SIZE>& partlist, int nactive, float h, bool do_phi) {
	const int &tid = threadIdx.x;
	__shared__
	extern int shmem_ptr[];
	cuda_kick_shmem &shmem = *(cuda_kick_shmem*) shmem_ptr;
	auto &gx = shmem.gx;
	auto &gy = shmem.gy;
	auto &gz = shmem.gz;
	auto &phi = shmem.phi;
	const auto& sink_x = shmem.sink_x;
	const auto& sink_y = shmem.sink_y;
	const auto& sink_z = shmem.sink_z;
	const auto* main_src_x = data.x;
	const auto* main_src_y = data.y;
	const auto* main_src_z = data.z;
	auto& src_x = shmem.src.x;
	auto& src_y = shmem.src.y;
	auto& src_z = shmem.src.z;
	const auto* tree_nodes = data.tree_nodes;
	const float h2 = h * h;
	const float hinv = 1.f / h;
	const float h3inv = hinv * hinv * hinv;
	int part_index;
	int nnear = 0;
	int nfar = 0;
	if (partlist.size()) {
		int i = 0;
		auto these_parts = tree_nodes[partlist[0]].part_range;
		const auto partsz = partlist.size();
		while (i < partsz) {
			part_index = 0;
			while (part_index < KICK_PP_MAX && i < partsz) {
				while (i + 1 < partsz) {
					const auto other_tree_parts = tree_nodes[partlist[i + 1]].part_range;
					if (these_parts.second == other_tree_parts.first) {
						these_parts.second = other_tree_parts.second;
						i++;
					} else {
						break;
					}
				}
				const part_int imin = these_parts.first;
				const part_int imax = min(these_parts.first + (KICK_PP_MAX - part_index), these_parts.second);
				const int sz = imax - imin;
				for (int j = tid; j < sz; j += WARP_SIZE) {
					const int i1 = part_index + j;
					const part_int i2 = j + imin;
					assert(i2 >= 0);
					assert(i2 < data.source_size);
					src_x[i1] = main_src_x[i2];
					src_y[i1] = main_src_y[i2];
					src_z[i1] = main_src_z[i2];
				}
				__syncwarp();
				these_parts.first += sz;
				part_index += sz;
				if (these_parts.first == these_parts.second) {
					i++;
					if (i < partsz) {
						these_parts = tree_nodes[partlist[i]].part_range;
					}
				}
			}
			float fx;
			float fy;
			float fz;
			float pot;
			float dx0;
			float dx1;
			float dx2;
			float r3inv;
			float r1inv;
			__syncwarp();
			int kmid;
			if ((nactive % WARP_SIZE) < MIN_KICK_WARP) {
				kmid = nactive - (nactive % WARP_SIZE);
			} else {
				kmid = nactive;
			}
			for (int k = tid; k < kmid; k += WARP_SIZE) {
				fx = 0.f;
				fy = 0.f;
				fz = 0.f;
				pot = 0.f;
				for (int j = 0; j < part_index; j++) {
					dx0 = distance(sink_x[k], src_x[j]); // 1
					dx1 = distance(sink_y[k], src_y[j]); // 1
					dx2 = distance(sink_z[k], src_z[j]); // 1
					const auto r2 = sqr(dx0, dx1, dx2);  // 5
					if (r2 >= h2) {                      // 1
						r1inv = rsqrt(r2);                // 4
						r3inv = r1inv * r1inv * r1inv;    // 2
						nnear++;
					} else {
						const float r1oh1 = sqrtf(r2) * hinv; // 5
						const float r2oh2 = r1oh1 * r1oh1;    // 1
						r3inv = +15.0f / 8.0f;
						r1inv = -5.0f / 16.0f;
						r3inv = fmaf(r3inv, r2oh2, -21.0f / 4.0f);  // 2
						r1inv = fmaf(r1inv, r2oh2, 21.0f / 16.0f);  // 2
						r3inv = fmaf(r3inv, r2oh2, +35.0f / 8.0f);  // 2
						r1inv = fmaf(r1inv, r2oh2, -35.0f / 16.0f); // 2
						r3inv *= h3inv;                             // 1
						r1inv = fmaf(r1inv, r2oh2, 35.0f / 16.0f);  // 2
						r1inv *= hinv;                              // 1
						nfar++;
					}
					fx = fmaf(dx0, r3inv, fx);                     // 2
					fy = fmaf(dx1, r3inv, fy);                     // 2
					fz = fmaf(dx2, r3inv, fz);                     // 2
					pot -= r1inv;                                  // 1
				}
				gx[k] -= fx;
				gy[k] -= fy;
				gz[k] -= fz;
				phi[k] += pot;
			}
			__syncwarp();
			for (int k = kmid; k < nactive; k++) {
				fx = 0.f;
				fy = 0.f;
				fz = 0.f;
				pot = 0.f;
				for (int j = tid; j < part_index; j += WARP_SIZE) {
					dx0 = distance(sink_x[k], src_x[j]);
					dx1 = distance(sink_y[k], src_y[j]);
					dx2 = distance(sink_z[k], src_z[j]);
					const auto r2 = sqr(dx0, dx1, dx2);
					if (r2 >= h2) {
						r1inv = rsqrt(r2);
						r3inv = r1inv * r1inv * r1inv;
						nnear++;
					} else {
						const float r1oh1 = sqrtf(r2) * hinv;
						const float r2oh2 = r1oh1 * r1oh1;
						r3inv = +15.0f / 8.0f;
						r1inv = -5.0f / 16.0f;
						r3inv = fmaf(r3inv, r2oh2, -21.0f / 4.0f);
						r1inv = fmaf(r1inv, r2oh2, 21.0f / 16.0f);
						r3inv = fmaf(r3inv, r2oh2, +35.0f / 8.0f);
						r1inv = fmaf(r1inv, r2oh2, -35.0f / 16.0f);
						r3inv *= h3inv;
						r1inv = fmaf(r1inv, r2oh2, 35.0f / 16.0f);
						r1inv *= hinv;
						nfar++;
					}
					fx = fmaf(dx0, r3inv, fx);
					fy = fmaf(dx1, r3inv, fy);
					fz = fmaf(dx2, r3inv, fz);
					pot -= r1inv;
				}
				shared_reduce_add(fx);
				shared_reduce_add(fy);
				shared_reduce_add(fz);
				shared_reduce_add(pot);
				if (tid == 0) {
					gx[k] -= fx;
					gy[k] -= fy;
					gz[k] -= fz;
					phi[k] += pot;
				}
			}
		}
		__syncwarp();
	}
	shared_reduce_add(nnear);
	shared_reduce_add(nfar);
	__syncwarp();
	return nfar * 22 + 37 * nnear;

}
