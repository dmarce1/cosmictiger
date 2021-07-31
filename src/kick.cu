#include <tigerfmm/cuda.hpp>
#include <tigerfmm/cuda_reduce.hpp>
#include <tigerfmm/defs.hpp>
#include <tigerfmm/fixedcapvec.hpp>
#include <tigerfmm/fmm_kernels.hpp>
#include <tigerfmm/gravity.hpp>
#include <tigerfmm/kick.hpp>
#include <tigerfmm/particles.hpp>
#include <tigerfmm/timer.hpp>

#include <atomic>

/*__managed__ int node_count;
 __managed__ double total_time;
 __managed__ double tree_time;
 __managed__ double gravity_time;
 static __managed__ double kick_time;*/


struct cuda_lists_type {
	fixedcapvec<int, NEXTLIST_SIZE> nextlist;
	fixedcapvec<int, LEAFLIST_SIZE> leaflist;
	fixedcapvec<int, PARTLIST_SIZE> partlist;
	fixedcapvec<int, MULTLIST_SIZE> multlist;
	fixedcapvec<expansion<float>, CUDA_MAX_DEPTH> L;
	fixedcapvec<int, CUDA_MAX_DEPTH> phase;
	fixedcapvec<int, CUDA_MAX_DEPTH> self;
	fixedcapvec<kick_return, CUDA_MAX_DEPTH> returns;
	fixedcapvec<array<fixed32,NDIM>, CUDA_MAX_DEPTH> Lpos;
	stack_vector<int, DCHECKS_SIZE, CUDA_MAX_DEPTH> dchecks;
	stack_vector<int, ECHECKS_SIZE, CUDA_MAX_DEPTH> echecks;
};

__device__ int max_depth = 0;
__device__ int max_nextlist = 0;
__device__ int max_partlist = 0;
__device__ int max_leaflist = 0;
__device__ int max_multlist = 0;
__device__ int max_echecks = 0;
__device__ int max_dchecks = 0;

__global__ void kick_reset_list_sizes_kernel() {
	if (threadIdx.x == 0) {
		max_depth = 0;
		max_nextlist = 0;
		max_partlist = 0;
		max_leaflist = 0;
		max_multlist = 0;
		max_echecks = 0;
		max_dchecks = 0;
	}
}

void kick_reset_list_sizes() {
	kick_reset_list_sizes_kernel<<<1,1>>>();
	CUDA_CHECK(cudaDeviceSynchronize());
}

static __constant__ float rung_dt[MAX_RUNG] = { 1.0 / (1 << 0), 1.0 / (1 << 1), 1.0 / (1 << 2), 1.0 / (1 << 3), 1.0 / (1 << 4), 1.0 / (1 << 5), 1.0 / (1 << 6),
		1.0 / (1 << 7), 1.0 / (1 << 8), 1.0 / (1 << 9), 1.0 / (1 << 10), 1.0 / (1 << 11), 1.0 / (1 << 12), 1.0 / (1 << 13), 1.0 / (1 << 14), 1.0 / (1 << 15), 1.0
				/ (1 << 16), 1.0 / (1 << 17), 1.0 / (1 << 18), 1.0 / (1 << 19), 1.0 / (1 << 20), 1.0 / (1 << 21), 1.0 / (1 << 22), 1.0 / (1 << 23), 1.0 / (1 << 24),
		1.0 / (1 << 25), 1.0 / (1 << 26), 1.0 / (1 << 27), 1.0 / (1 << 28), 1.0 / (1 << 29), 1.0 / (1 << 30), 1.0 / (1 << 31) };

struct cuda_kick_params {
	array<fixed32, NDIM> Lpos;
	expansion<float> L;
	int self;
	int* dchecks;
	int* echecks;
	int dcount;
	int ecount;
	kick_return* kreturn;
};

__device__ int __noinline__ do_kick(kick_return& return_, kick_params params, const cuda_kick_data& data, const expansion<float>& L, int nactive,
		const tree_node& self) {
//	auto tm = clock64();
	const int& tid = threadIdx.x;
	extern __shared__ int shmem_ptr[];
	cuda_kick_shmem& shmem = *(cuda_kick_shmem*) shmem_ptr;

	auto* all_phi = data.pot;
	auto* all_gx = data.gx;
	auto* all_gy = data.gy;
	auto* all_gz = data.gz;
	auto* write_rungs = data.rungs;
	auto* vel_x = data.vx;
	auto* vel_y = data.vy;
	auto* vel_z = data.vz;
	auto& phi = shmem.phi;
	auto& gx = shmem.gx;
	auto& gy = shmem.gy;
	auto& gz = shmem.gz;
	auto& active_indexes = shmem.active;
	const auto& sink_x = shmem.sink_x;
	const auto& sink_y = shmem.sink_y;
	const auto& sink_z = shmem.sink_z;
	const auto& read_rungs = shmem.rungs;
	const float log2ft0 = log2f(params.t0);
	const float tfactor = params.eta * sqrtf(params.a * params.h);
	int max_rung = 0;
	expansion2<float> L2;
	float vx;
	float vy;
	float vz;
	float dt;
	float g2;
	float phi_tot = 0.0f;
	float fx_tot = 0.f;
	float fy_tot = 0.f;
	float fz_tot = 0.f;
	float fnorm_tot = 0.f;
	int rung;
	array<float, NDIM> dx;
	int snki;
	for (int i = tid; i < nactive; i += WARP_SIZE) {
		snki = active_indexes[i];
		assert(snki >= 0);
		assert(snki < data.sink_size);
		dx[XDIM] = distance(sink_x[i], self.pos[XDIM]);
		dx[YDIM] = distance(sink_y[i], self.pos[YDIM]);
		dx[ZDIM] = distance(sink_z[i], self.pos[ZDIM]);
		L2 = L2P(L, dx, params.min_rung == 0);
		phi[i] += L2(0, 0, 0);
		gx[i] -= L2(1, 0, 0);
		gy[i] -= L2(0, 1, 0);
		gz[i] -= L2(0, 0, 1);
		phi[i] *= params.GM;
		gx[i] *= params.GM;
		gy[i] *= params.GM;
		gz[i] *= params.GM;
		if (params.save_force) {
			all_gx[snki] = gx[i];
			all_gy[snki] = gy[i];
			all_gz[snki] = gz[i];
			all_phi[snki] = phi[i];
		}
		vx = vel_x[snki];
		vy = vel_y[snki];
		vz = vel_z[snki];
		rung = read_rungs[i];
		dt = 0.5f * rung_dt[rung] * params.t0;
		if (!params.first_call) {
			vx = fmaf(gx[i], dt, vx);
			vy = fmaf(gy[i], dt, vy);
			vz = fmaf(gz[i], dt, vz);
		}
		g2 = sqr(gx[i], gy[i], gz[i]);
		dt = fminf(tfactor * rsqrt(sqrtf(g2)), params.t0);
		rung = max((int) ceilf(log2ft0 - log2f(dt)), max(rung - 1, params.min_rung));
		max_rung = max(rung, max_rung);
		if (rung < 0 || rung >= MAX_RUNG) {
			PRINT("Rung out of range %i\n", rung);
		}
		assert(rung >= 0);
		assert(rung < MAX_RUNG);
		dt = 0.5f * rung_dt[rung] * params.t0;
		vx = fmaf(gx[i], dt, vx);
		vy = fmaf(gy[i], dt, vy);
		vz = fmaf(gz[i], dt, vz);
		write_rungs[snki] = rung;
		vel_x[snki] = vx;
		vel_y[snki] = vy;
		vel_z[snki] = vz;
		phi_tot += phi[i];
		fx_tot += gx[i];
		fy_tot += gy[i];
		fz_tot += gz[i];
		fnorm_tot += g2;
	}
	shared_reduce_add(phi_tot);
	shared_reduce_add(fx_tot);
	shared_reduce_add(fy_tot);
	shared_reduce_add(fz_tot);
	shared_reduce_add(fnorm_tot);
	shared_reduce_max(max_rung);
	if (tid == 0) {
		return_.max_rung = max(return_.max_rung, max_rung);
		return_.pot += phi_tot;
		return_.fx += fx_tot;
		return_.fy += fy_tot;
		return_.fz += fz_tot;
		return_.fnorm += fnorm_tot;
	}
//	atomicAdd(&kick_time, (double) clock64() - tm);
	return max_rung;
}

__global__ void cuda_kick_kernel(kick_params global_params, cuda_kick_data data, cuda_lists_type* lists, cuda_kick_params* params, int item_count, int* next_item, int ntrees) {
//	auto tm1 = clock64();
	const int& tid = threadIdx.x;
	const int& bid = blockIdx.x;
	extern __shared__ int shmem_ptr[];
	cuda_kick_shmem& shmem = *(cuda_kick_shmem*) shmem_ptr;
	auto& L = lists[bid].L;
	auto& phase = lists[bid].phase;
	auto& Lpos = lists[bid].Lpos;
	auto& self_index = lists[bid].self;
	auto& returns = lists[bid].returns;
	auto& dchecks = lists[bid].dchecks;
	auto& echecks = lists[bid].echecks;
	auto& nextlist = lists[bid].nextlist;
	auto& multlist = lists[bid].multlist;
	auto& partlist = lists[bid].partlist;
	auto& leaflist = lists[bid].leaflist;
	auto& activei = shmem.active;
	auto& sink_x = shmem.sink_x;
	auto& sink_y = shmem.sink_y;
	auto& sink_z = shmem.sink_z;
	auto& rungs = shmem.rungs;
	auto& phi = shmem.phi;
	auto& gx = shmem.gx;
	auto& gy = shmem.gy;
	auto& gz = shmem.gz;
	const float& h = global_params.h;
	const float hinv = 1.f / h;
	const float thetainv = 1.f / global_params.theta;
	const int min_rung = global_params.min_rung;
	const float sink_bias = 1.5;
	auto* tree_nodes = data.tree_nodes;
	auto* all_rungs = data.rungs;
	auto* src_x = data.x;
	auto* src_y = data.y;
	auto* src_z = data.z;
	L.initialize();
	dchecks.initialize();
	echecks.initialize();
	nextlist.initialize();
	multlist.initialize();
	leaflist.initialize();
	nextlist.initialize();
	phase.initialize();
	Lpos.initialize();
	returns.initialize();
	self_index.initialize();
	L.resize(max_depth);
	int index;
	if (tid == 0) {
		index = atomicAdd(next_item, 1);
	}
	index = __shfl_sync(0xFFFFFFFF, index, 0);
	while (index < item_count) {
		L.resize(0);
		dchecks.resize(0);
		echecks.resize(0);
		phase.resize(0);
		self_index.resize(0);
		Lpos.resize(0);
		returns.push_back(kick_return());
		L.push_back(params[index].L);
		dchecks.resize(params[index].dcount);
		echecks.resize(params[index].ecount);
		for (int i = tid; i < params[index].dcount; i += WARP_SIZE) {
			dchecks[i] = params[index].dchecks[i];
		}
		for (int i = tid; i < params[index].ecount; i += WARP_SIZE) {
			echecks[i] = params[index].echecks[i];
		}
		phase.push_back(0);
		self_index.push_back(params[index].self);
		Lpos.push_back(params[index].Lpos);
		__syncwarp();
		int depth = 0;
		int maxi;
		while (depth >= 0) {
//			auto tm2 = clock64();
//			node_count++;
			assert(Lpos.size() == depth + 1);
			assert(self_index.size() == depth + 1);
			assert(phase.size() == depth + 1);
			const auto& self = tree_nodes[self_index.back()];

			switch (phase.back()) {

			case 0: {
				// shift expansion
				array<float, NDIM> dx;
				for (int dim = 0; dim < NDIM; dim++) {
					dx[dim] = distance(self.pos[dim], Lpos.back()[dim]);
				}
				auto this_L = L2L_cuda(L.back(), dx, global_params.min_rung == 0);
				if (tid == 0) {
					L.back() = this_L;
				}
				// Do ewald walk
				nextlist.resize(0);
				multlist.resize(0);
				maxi = round_up(echecks.size(), WARP_SIZE);
				for (int i = tid; i < maxi; i += WARP_SIZE) {
					bool mult = false;
					bool next = false;
					if (i < echecks.size()) {
						const tree_node& other = tree_nodes[echecks[i]];
						for (int dim = 0; dim < NDIM; dim++) {
							dx[dim] = distance(self.pos[dim], other.pos[dim]);
						}
						float R2 = sqr(dx[XDIM], dx[YDIM], dx[ZDIM]);
						R2 = fmaxf(R2, EWALD_DIST2);
						const float r2 = sqr((sink_bias * self.radius + other.radius) * thetainv + h); // 5
						mult = R2 > r2;
						next = !mult;
					}
					int l;
					int total;
					int start;
					l = mult;
					compute_indices(l, total);
					start = multlist.size();
					multlist.resize(start + total);
					if (mult) {
						multlist[l + start] = echecks[i];
					}
					l = next;
					compute_indices(l, total);
					start = nextlist.size();
					nextlist.resize(start + total);
					if (next) {
						nextlist[l + start] = echecks[i];
					}
				}
				__syncwarp();
				echecks.resize(NCHILD * nextlist.size());
				for (int i = tid; i < nextlist.size(); i += WARP_SIZE) {
					//	PRINT( "nextlist = %i\n", nextlist[i]);
					const auto children = tree_nodes[nextlist[i]].children;
					assert(children[LEFT].index!=-1);
					assert(children[RIGHT].index!=-1);
					echecks[NCHILD * i + LEFT] = children[LEFT].index;
					echecks[NCHILD * i + RIGHT] = children[RIGHT].index;
				}
				__syncwarp();

//				auto tm = clock64();
				cuda_gravity_cc(data, L.back(), self, multlist, GRAVITY_CC_EWALD, min_rung == 0);
//				atomicAdd(&gravity_time, (double) clock64() - tm);

				// Direct walk
				nextlist.resize(0);
				partlist.resize(0);
				leaflist.resize(0);
				multlist.resize(0);
				do {
					maxi = round_up(dchecks.size(), WARP_SIZE);
					for (int i = tid; i < maxi; i += WARP_SIZE) {

						bool mult = false;
						bool next = false;
						bool leaf = false;
						bool part = false;
						if (i < dchecks.size()) {
							const tree_node& other = tree_nodes[dchecks[i]];
							for (int dim = 0; dim < NDIM; dim++) {
								dx[dim] = distance(self.pos[dim], other.pos[dim]);
							}
							const float R2 = sqr(dx[XDIM], dx[YDIM], dx[ZDIM]);
							const bool far1 = R2 > sqr((sink_bias * self.radius + other.radius) * thetainv + h);     // 5
							const bool far2 = R2 > sqr(sink_bias * self.radius * thetainv + other.radius + h);       // 5
							//				PRINT("%e %e\n", R2, sqr((sink_bias * self.radius + other.radius) * thetainv + h));
							mult = far1;                                                                  // 4
							part = !mult && (far2 && other.source_leaf && (self.part_range.second - self.part_range.first) > MIN_CP_PARTS);
							leaf = !mult && !part && other.source_leaf;
							next = !mult && !part && !leaf;
						}
						int l;
						int total;
						int start;
						l = mult;
						compute_indices(l, total);
						start = multlist.size();
						multlist.resize(start + total);
						if (mult) {
							multlist[l + start] = dchecks[i];
						}
						l = next;
						compute_indices(l, total);
						start = nextlist.size();
						nextlist.resize(start + total);
						if (next) {
							nextlist[l + start] = dchecks[i];
						}
						l = part;
						compute_indices(l, total);
						start = partlist.size();
						partlist.resize(start + total);
						if (part) {
							partlist[l + start] = dchecks[i];
						}
						l = leaf;
						compute_indices(l, total);
						start = leaflist.size();
						leaflist.resize(start + total);
						if (leaf) {
							leaflist[l + start] = dchecks[i];
						}
					}
					__syncwarp();
					dchecks.resize(NCHILD * nextlist.size());
					for (int i = tid; i < nextlist.size(); i += WARP_SIZE) {
						assert(index >= 0);
						assert(index < ntrees);
						const auto& node = tree_nodes[nextlist[i]];
						const auto& children = node.children;
						dchecks[NCHILD * i + LEFT] = children[LEFT].index;
						dchecks[NCHILD * i + RIGHT] = children[RIGHT].index;
					}
					nextlist.resize(0);
					__syncwarp();

				} while (dchecks.size() && self.sink_leaf);
//				tm = clock64();
				cuda_gravity_cc(data, L.back(), self, multlist, GRAVITY_CC_DIRECT, min_rung == 0);
				cuda_gravity_cp(data, L.back(), self, partlist, min_rung == 0);
//				atomicAdd(&gravity_time, (double) clock64() - tm);

				if (self.sink_leaf) {
					int nactive = 0;
					const int begin = self.sink_part_range.first;
					const int end = self.sink_part_range.second;
					const int src_begin = self.part_range.first;
					maxi = round_up(end - begin, WARP_SIZE);
					for (int i = tid; i < maxi; i += WARP_SIZE) {
						bool active = false;
						char rung;
						if (i < end - begin) {
							assert(begin + i < data.sink_size);
							rung = all_rungs[begin + i];
							active = rung >= min_rung;
						}
						int l;
						int total;
						l = active;
						compute_indices(l, total);
						l += nactive;
						if (active) {
							const int srci = src_begin + i;
							assert(begin + i < data.sink_size);
							activei[l] = begin + i;
							rungs[l] = rung;
							sink_x[l] = src_x[srci];
							sink_y[l] = src_y[srci];
							sink_z[l] = src_z[srci];
						}
						nactive += total;
						__syncwarp();
					}
					__syncwarp();
					partlist.resize(0);
					multlist.resize(0);
					maxi = round_up((int) leaflist.size(), WARP_SIZE);
					for (int i = tid; i < maxi; i += WARP_SIZE) {
						bool pc = false;
						bool pp = false;
						if (i < leaflist.size()) {
							const tree_node& other = tree_nodes[leaflist[i]];
							const int begin = other.part_range.first;
							const int end = other.part_range.second;
							bool far;
							if (end - begin < MIN_PC_PARTS) {
								far = false;
							} else {
								far = true;
								for (int j = 0; j < nactive; j++) {
									const float dx = distance(sink_x[j], other.pos[XDIM]);
									const float dy = distance(sink_y[j], other.pos[YDIM]);
									const float dz = distance(sink_z[j], other.pos[ZDIM]);
									const float R2 = sqr(dx, dy, dz);
									far = R2 > sqr(other.radius * thetainv + h);
									if (!far) {
										break;
									}
								}
							}
							pp = !far;
							pc = far;
						}
						int total;
						int l = pp;
						compute_indices(l, total);
						l += partlist.size();
						partlist.resize(partlist.size() + total);
						if (pp) {
							partlist[l] = leaflist[i];
						}
						l = pc;
						compute_indices(l, total);
						l += multlist.size();
						multlist.resize(multlist.size() + total);
						if (pc) {
							multlist[l] = leaflist[i];
						}
					}
					__syncwarp();
					for (int i = tid; i < nactive; i += WARP_SIZE) {
						phi[i] = -SELF_PHI * hinv;
						gx[i] = gy[i] = gz[i] = 0.f;
					}
					__syncwarp();
//					tm = clock64();
					cuda_gravity_pc(data, self, multlist, nactive, min_rung == 0);
					cuda_gravity_pp(data, self, partlist, nactive, h, min_rung == 0);
//					atomicAdd(&gravity_time, (double) clock64() - tm);
					__syncwarp();
					do_kick(returns.back(), global_params, data, L.back(), nactive, self);
					phase.pop_back();
					self_index.pop_back();
					Lpos.pop_back();
					depth--;
				} else {
					const int start = dchecks.size();
					dchecks.resize(start + leaflist.size());
					for (int i = tid; i < leaflist.size(); i += WARP_SIZE) {
						dchecks[start + i] = leaflist[i];
					}
					__syncwarp();
					const int active_left = tree_nodes[self.children[LEFT].index].nactive;
					const int active_right = tree_nodes[self.children[RIGHT].index].nactive;
					Lpos.push_back(self.pos);
					returns.push_back(kick_return());
					if (active_left && active_right) {
						const tree_id child = self.children[LEFT];
						const auto l = L.back();
						L.push_back(l);
						dchecks.push_top();
						echecks.push_top();
						phase.back() += 1;
						phase.push_back(0);
						self_index.push_back(child.index);
					} else {
						const tree_id child = active_left ? self.children[LEFT] : self.children[RIGHT];
						phase.back() += 2;
						phase.push_back(0);
						self_index.push_back(child.index);
					}
					depth++;
				}

			}
				break;
			case 1: {
				L.pop_back();
				Lpos.push_back(self.pos);
				dchecks.pop_top();
				echecks.pop_top();
				phase.back() += 1;
				phase.push_back(0);
				const tree_id child = self.children[RIGHT];
				assert(child.proc == data.rank);
				self_index.push_back(child.index);
				const auto this_return = returns.back();
				returns.pop_back();
				if (tid == 0) {
					returns.back() += this_return;
				}
				returns.push_back(kick_return());
				depth++;
			}
				break;
			case 2: {
				self_index.pop_back();
				phase.pop_back();
				Lpos.pop_back();
				const auto this_return = returns.back();
				returns.pop_back();
				if (tid == 0) {
					returns.back() += this_return;
				}
				depth--;
			}
				break;
			}
//			atomicAdd(&tree_time, (double) clock64() - tm2);
		}

		if (tid == 0) {
			*(params[index].kreturn) = returns.back();
		}
		if (tid == 0) {
			index = atomicAdd(next_item, 1);
		}
		index = __shfl_sync(0xFFFFFFFF, index, 0);
		returns.pop_back();
	}
	assert(returns.size() == 1);
	assert(L.size() == 1);
	assert(Lpos.size() == 0);
	assert(phase.size() == 0);
	assert(self_index.size() == 0);
//	atomicAdd(&total_time, ((double) (clock64() - tm1)));
}

vector<kick_return> cuda_execute_kicks(kick_params kparams, fixed32* dev_x, fixed32* dev_y, fixed32* dev_z, tree_node* dev_tree_nodes,
		vector<kick_workitem> workitems, cudaStream_t stream, int part_count, int ntrees, std::atomic<int>& outer_lock) {
	timer tm;
//	PRINT("shmem size = %i\n", sizeof(cuda_kick_shmem));
	tm.start();
	int* current_index;
	int zero = 0;
//	kick_time = total_time = tree_time = gravity_time = 0.0f;
//	node_count = 0;
	CUDA_CHECK(cudaMalloc(&current_index, sizeof(int)));
	CUDA_CHECK(cudaMemcpyAsync(current_index, &zero, sizeof(int), cudaMemcpyHostToDevice, stream));
	vector<kick_return> returns(workitems.size());
	vector<cuda_kick_params> kick_params(workitems.size());
	vector<int> dchecks;
	vector<int> echecks;
	int* dev_dchecks;
	int* dev_echecks;
	kick_return* dev_returns;
	cuda_kick_params* dev_kick_params;
	CUDA_CHECK(cudaMalloc(&dev_kick_params, sizeof(cuda_kick_params) * kick_params.size()));
	CUDA_CHECK(cudaMalloc(&dev_returns, sizeof(kick_return) * returns.size()));

	vector<int> dindices(workitems.size() + 1);
	vector<int> eindices(workitems.size() + 1);

	int dcount = 0;
	int ecount = 0;
	for (int i = 0; i < workitems.size(); i++) {
		//	PRINT( "%i\n", workitems[i].echecklist.size());
		dcount += workitems[i].dchecklist.size();
		ecount += workitems[i].echecklist.size();
	}
	dchecks.reserve(dcount);
	echecks.reserve(ecount);
	dcount = 0;
	ecount = 0;
	for (int i = 0; i < workitems.size(); i++) {
		dindices[i] = dcount;
		eindices[i] = ecount;
		for (int j = 0; j < workitems[i].dchecklist.size(); j++) {
			dchecks.push_back(workitems[i].dchecklist[j].index);
			dcount++;
		}
		for (int j = 0; j < workitems[i].echecklist.size(); j++) {
			echecks.push_back(workitems[i].echecklist[j].index);
			ecount++;
		}
	}
	dindices[workitems.size()] = dcount;
	eindices[workitems.size()] = ecount;
	CUDA_CHECK(cudaMalloc(&dev_dchecks, sizeof(int) * dchecks.size()));
	CUDA_CHECK(cudaMalloc(&dev_echecks, sizeof(int) * echecks.size()));
	CUDA_CHECK(cudaMemcpyAsync(dev_dchecks, dchecks.data(), sizeof(int) * dchecks.size(), cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(dev_echecks, echecks.data(), sizeof(int) * echecks.size(), cudaMemcpyHostToDevice, stream));
	tm.stop();
	cuda_kick_data data;
	data.source_size = part_count;
	data.tree_size = ntrees;
	data.sink_size = particles_size();
	data.x = dev_x;
	data.y = dev_y;
	data.z = dev_z;
	data.tree_nodes = dev_tree_nodes;
	data.vx = &particles_vel(XDIM, 0);
	data.vy = &particles_vel(YDIM, 0);
	data.vz = &particles_vel(ZDIM, 0);
	data.rungs = &particles_rung(0);
	data.rank = hpx_rank();
	if (kparams.save_force) {
		data.gx = &particles_gforce(XDIM, 0);
		data.gy = &particles_gforce(YDIM, 0);
		data.gz = &particles_gforce(ZDIM, 0);
		data.pot = &particles_pot(0);
	} else {
		data.gx = data.gy = data.gz = data.pot = nullptr;
	}

	for (int i = 0; i < workitems.size(); i++) {
		cuda_kick_params params;
		params.Lpos = workitems[i].pos;
		params.L = workitems[i].L;
		params.self = workitems[i].self.index;
		params.dchecks = dev_dchecks + dindices[i];
		params.echecks = dev_echecks + eindices[i];
		params.dcount = dindices[i + 1] - dindices[i];
		params.ecount = eindices[i + 1] - eindices[i];
		params.kreturn = dev_returns + i;
		kick_params[i] = std::move(params);
	}
	CUDA_CHECK(cudaMemcpyAsync(dev_kick_params, kick_params.data(), sizeof(cuda_kick_params) * kick_params.size(), cudaMemcpyHostToDevice, stream));
	int nblocks = kick_block_count();
	nblocks = std::min(nblocks, (int) workitems.size());
	CUDA_CHECK(cudaStreamSynchronize(stream));
	static std::atomic<int> cnt(0);
	while (cnt++ != 0) {
		cnt--;
		hpx_yield();
	}
	outer_lock--;
	tm.reset();
	tm.start();
	cuda_lists_type* dev_lists;
	CUDA_CHECK(cudaMalloc(&dev_lists, sizeof(cuda_lists_type) * nblocks));
	cuda_kick_kernel<<<nblocks, WARP_SIZE, sizeof(cuda_kick_shmem), stream>>>(kparams, data,dev_lists, dev_kick_params, kick_params.size(), current_index, ntrees);
//	PRINT("One done\n");
	CUDA_CHECK(cudaMemcpyAsync(returns.data(), dev_returns, sizeof(kick_return) * returns.size(), cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaStreamSynchronize(stream));
	tm.stop();
//	PRINT( "%i %e\n", nblocks, tm.read());
//	PRINT("%i nodes traversed\n", node_count);
	CUDA_CHECK(cudaFree(dev_dchecks));
	CUDA_CHECK(cudaFree(dev_echecks));
	CUDA_CHECK(cudaFree(dev_returns));
	CUDA_CHECK(cudaFree(dev_lists));
	CUDA_CHECK(cudaFree(dev_kick_params));
	CUDA_CHECK(cudaFree(current_index));
	cnt--;
//	PRINT("%i %i %i %i %i %i %i\n", max_depth, max_dchecks, max_echecks, max_nextlist, max_leaflist, max_partlist, max_multlist);
//	PRINT("%i %e %e %e\n", nblocks, tree_time / total_time, gravity_time / total_time,  kick_time / total_time);
	return returns;
}

int kick_block_count() {
	int nblocks;
	CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nblocks, (const void*) cuda_kick_kernel, WARP_SIZE, sizeof(cuda_kick_shmem)));
//	PRINT("Occupancy is %i shmem size = %li\n", nblocks, sizeof(cuda_kick_shmem));
	nblocks *= cuda_smp_count();
	return nblocks;

}
