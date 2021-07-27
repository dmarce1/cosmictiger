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
int cuda_gravity_cp(const cuda_kick_data&, expansion<float>&, const tree_node&, bool do_phi) {
	int flops = 0;
	return flops;

}

__device__
int cuda_gravity_pc(const cuda_kick_data& data, const tree_node&, int, bool) {
	int flops = 0;
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
					for (int dim = 0; dim < NDIM; dim++) {
						src_x[part_index + j] = main_src_x[j + imin];
						src_y[part_index + j] = main_src_y[j + imin];
						src_z[part_index + j] = main_src_z[j + imin];
					}
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
