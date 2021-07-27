#include <tigerfmm/cuda_reduce.hpp>
#include <tigerfmm/gravity.hpp>

__device__
int cuda_gravity_cc(const cuda_kick_data& data, expansion<float>& Lacc, const tree_node& self, gravity_cc_type type, bool do_phi) {
	int flops = 0;
	const int &tid = threadIdx.x;
	__shared__
	extern int shmem_ptr[];
	cuda_kick_shmem &shmem = *(cuda_kick_shmem*) shmem_ptr;
	const auto& tree_nodes = data.tree_nodes;
	const auto& multlist = shmem.multlist;
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
			if (type == GRAVITY_CC_DIRECT) {
				greens_function(D, dx);
			} else {
				ewald_greens_function(D, dx);
			}
			M2L(L, M, D, do_phi);
		}
		for (int i = 0; i < EXPANSION_SIZE; i++) {
			shared_reduce_add(L[i]);
		}
		for (int i = tid; i < EXPANSION_SIZE; i += WARP_SIZE) {
			Lacc[i] += L[i];
		}
		__syncwarp();
	}
	return flops;
}

__device__
int cuda_gravity_cp(const cuda_kick_data& data, expansion<float>& Lacc, const tree_node& self, bool do_phi) {
	int flops = 0;
	__shared__
	extern int shmem_ptr[];
	cuda_kick_shmem &shmem = *(cuda_kick_shmem*) shmem_ptr;
	const auto& partlist = shmem.partlist;
	const auto* main_src_x = data.x;
	const auto* main_src_y = data.y;
	const auto* main_src_z = data.z;
	auto& src_x = shmem.src_x;
	auto& src_y = shmem.src_y;
	auto& src_z = shmem.src_z;
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
				const int imin = these_parts.first;
				const int imax = min(these_parts.first + (KICK_PP_MAX - part_index), these_parts.second);
				const int sz = imax - imin;
				for (int j = tid; j < sz; j += WARP_SIZE) {
					src_x[part_index + j] = main_src_x[j + imin];
					src_y[part_index + j] = main_src_y[j + imin];
					src_z[part_index + j] = main_src_z[j + imin];
				}
				these_parts.first += sz;
				part_index += sz;
				if (these_parts.first == these_parts.second) {
					i++;
					if (i < partsz) {
						these_parts = tree_nodes[partlist[i]].part_range;
					}
				}
			}
			for (int j = tid; j < part_index; j += warpSize) {
				array<float, NDIM> dx;
				dx[XDIM] = distance(self.pos[XDIM], src_x[j]);
				dx[YDIM] = distance(self.pos[YDIM], src_y[j]);
				dx[ZDIM] = distance(self.pos[ZDIM], src_z[j]);
				expansion<float> D;
				greens_function(D, dx);
				for (int k = 0; k < EXPANSION_SIZE; k++) {
					L[k] += D[k];
				}
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
	return flops;

}

__device__
int cuda_gravity_pc(const cuda_kick_data& data, const tree_node&, int nactive, bool do_phi) {
	int flops = 0;
	const int &tid = threadIdx.x;
	__shared__
	extern int shmem_ptr[];
	cuda_kick_shmem &shmem = *(cuda_kick_shmem*) shmem_ptr;
	const auto& multlist = shmem.multlist;
	auto &gx = shmem.gx;
	auto &gy = shmem.gy;
	auto &gz = shmem.gz;
	auto &phi = shmem.phi;
	const auto& sink_x = shmem.sink_x;
	const auto& sink_y = shmem.sink_y;
	const auto& sink_z = shmem.sink_z;
	const auto* tree_nodes = data.tree_nodes;
	if (multlist.size()) {
		__syncwarp();
		int kmid;
		if ((nactive % WARP_SIZE) < MIN_KICK_WARP) {
			kmid = nactive - (nactive % WARP_SIZE);
		} else {
			kmid = nactive;
		}
		for (int k = tid; k < kmid; k += WARP_SIZE) {
			expansion2<float> L;
			L(0, 0, 0) = L(1, 0, 0) = L(0, 1, 0) = L(0, 0, 1) = 0.0f;
			for (int j = 0; j < multlist.size(); j++) {
				array<float, NDIM> dx;
				const auto& other = tree_nodes[multlist[j]];
				dx[XDIM] = distance(sink_x[k], other.pos[XDIM]);
				dx[YDIM] = distance(sink_y[k], other.pos[YDIM]);
				dx[ZDIM] = distance(sink_z[k], other.pos[ZDIM]);
				const auto& M = other.multi;
				expansion<float> D;
				greens_function(D, dx);
				M2L(L, M, D, do_phi);
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
			for (int j = tid; j < multlist.size(); j += WARP_SIZE) {
				array<float, NDIM> dx;
				const auto& other = tree_nodes[multlist[j]];
				dx[XDIM] = distance(sink_x[k], other.pos[XDIM]);
				dx[YDIM] = distance(sink_y[k], other.pos[YDIM]);
				dx[ZDIM] = distance(sink_z[k], other.pos[ZDIM]);
				const auto& M = other.multi;
				expansion<float> D;
				greens_function(D, dx);
				M2L(L, M, D, do_phi);
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
	return flops;

}

__device__
int cuda_gravity_pp(const cuda_kick_data& data, const tree_node& self, int nactive, float h, bool do_phi) {
	int flops = 0;
	const int &tid = threadIdx.x;
	__shared__
	extern int shmem_ptr[];
	cuda_kick_shmem &shmem = *(cuda_kick_shmem*) shmem_ptr;
	const auto& partlist = shmem.partlist;
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
	auto& src_x = shmem.src_x;
	auto& src_y = shmem.src_y;
	auto& src_z = shmem.src_z;
	const auto* tree_nodes = data.tree_nodes;
	const float h2 = h * h;
	const float hinv = 1.f / h;
	const float h3inv = hinv * hinv * hinv;
	int part_index;
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
				const int imin = these_parts.first;
				const int imax = min(these_parts.first + (KICK_PP_MAX - part_index), these_parts.second);
				const int sz = imax - imin;
				for (int j = tid; j < sz; j += WARP_SIZE) {
					src_x[part_index + j] = main_src_x[j + imin];
					src_y[part_index + j] = main_src_y[j + imin];
					src_z[part_index + j] = main_src_z[j + imin];
				}
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
					dx0 = distance(sink_x[k], src_x[j]);
					dx1 = distance(sink_y[k], src_y[j]);
					dx2 = distance(sink_z[k], src_z[j]);
					const auto r2 = fmaf(dx0, dx0, fmaf(dx1, dx1, sqr(dx2)));
					if (r2 >= h2) {
						r1inv = rsqrt(r2);
						r3inv = r1inv * r1inv * r1inv;
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
					}
					fx = fmaf(dx0, r3inv, fx);
					fy = fmaf(dx1, r3inv, fy);
					fz = fmaf(dx2, r3inv, fz);
					pot -= r1inv;
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
					const auto r2 = fmaf(dx0, dx0, fmaf(dx1, dx1, sqr(dx2)));
					if (r2 >= h2) {
						r1inv = rsqrt(r2);
						r3inv = r1inv * r1inv * r1inv;
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
	}
	__syncwarp();
	return flops;

}
