#include <tigerfmm/fmm_kernels.hpp>
#include <tigerfmm/gravity.hpp>
#include <tigerfmm/math.hpp>
#include <tigerfmm/safe_io.hpp>
#include <tigerfmm/timer.hpp>
#include <tigerfmm/tree.hpp>


void gravity_cc(expansion<float>& L, const vector<tree_id>& list, tree_id self, gravity_cc_type type, bool do_phi) {
	if (list.size()) {
		static const simd_float _2float(fixed2float);
		vector<const tree_node*> tree_ptrs(list.size());
		const tree_node* self_ptr = tree_get_node(self);
		const int nsink = self_ptr->nparts();
		const int nsource = round_up((int) list.size(), SIMD_FLOAT_SIZE) / SIMD_FLOAT_SIZE;
		for (int i = 0; i < list.size(); i++) {
			tree_ptrs[i] = tree_get_node(list[i]);
		}
		static thread_local vector<multipole<simd_float>> M;
		static thread_local vector<array<simd_int, NDIM>> Y;
		M.resize(nsource);
		Y.resize(nsource);
		int count = 0;
		for (int i = 0; i < tree_ptrs.size(); i++) {
			const int k = i / SIMD_FLOAT_SIZE;
			const int l = i % SIMD_FLOAT_SIZE;
			const auto& m = tree_ptrs[i]->multi;
			const auto& y = tree_ptrs[i]->pos;
			for (int j = 0; j < MULTIPOLE_SIZE; j++) {
				M[k][j][l] = m[j];
			}
			for (int j = 0; j < NDIM; j++) {
				Y[k][j][l] = y[j].raw();
			}
		}
		const int last = (tree_ptrs.size() - 1) / SIMD_FLOAT_SIZE;
		for (int i = tree_ptrs.size(); i < nsource * SIMD_FLOAT_SIZE; i++) {
			const int k = i / SIMD_FLOAT_SIZE;
			const int l = i % SIMD_FLOAT_SIZE;
			for (int j = 0; j < MULTIPOLE_SIZE; j++) {
				M[k][j][l] = 0.0;
			}
			for (int j = 0; j < NDIM; j++) {
				Y[k][j][l] = Y[last][j][l];
			}
		}
		array<simd_int, NDIM> X;
		for (int dim = 0; dim < NDIM; dim++) {
			X[dim] = self_ptr->pos[dim].raw();
		}
		expansion<simd_float> L0;
		L0 = simd_float(0.0f);
		for (int j = 0; j < nsource; j++) {
			array<simd_float, NDIM> dx;
			for (int dim = 0; dim < NDIM; dim++) {
				dx[dim] = (X[dim] - Y[j][dim]) * _2float;
			}
			expansion<simd_float> D;
			if (type == GRAVITY_CC_DIRECT) {
				greens_function(D, dx);
			} else {
				ewald_greens_function(D, dx);
			}
			M2L(L0, M[j], D, do_phi);
		}
		for (int i = 0; i < EXPANSION_SIZE; i++) {
			L[i] += L0[i].sum();
		}
	}
}

void gravity_cp(expansion<float>& L, const vector<tree_id>& list, tree_id self, bool do_phi) {
	constexpr int chunk_size = 32;
	if (list.size()) {
		static const simd_float _2float(fixed2float);
		const simd_float h(get_options().hsoft);
		const simd_float h2 = h * h;
		const simd_float one(1.0);
		const simd_float tiny(1.0e-20);
		const simd_float hinv = simd_float(1) / h;
		const simd_float hinv3 = hinv * hinv * hinv;
		const tree_node* self_ptr = tree_get_node(self);
		const int nsink = self_ptr->nparts();
		for (int li = 0; li < list.size(); li += chunk_size) {
			array<const tree_node*, chunk_size> tree_ptrs;
			int nsource = 0;
			const int maxi = std::min((int) list.size(), li + chunk_size) - li;
			for (int i = 0; i < maxi; i++) {
				tree_ptrs[i] = tree_get_node(list[i + li]);
				nsource += tree_ptrs[i]->nparts();
			}
			nsource = round_up(nsource, SIMD_FLOAT_SIZE);
			static thread_local vector<fixed32> srcx;
			static thread_local vector<fixed32> srcy;
			static thread_local vector<fixed32> srcz;
			static thread_local vector<float> masks;
			srcx.resize(nsource);
			srcy.resize(nsource);
			srcz.resize(nsource);
			masks.resize(nsource);
			int count = 0;
			for (int i = 0; i < maxi; i++) {
				particles_global_read_pos(tree_ptrs[i]->global_part_range(), srcx, srcy, srcz, count);
				count += tree_ptrs[i]->nparts();
			}
			for (int i = 0; i < count; i++) {
				masks[i] = 1.0;
			}
			for (int i = count; i < nsource; i++) {
				masks[i] = 0.0;
			}
			const auto range = self_ptr->part_range;
			array<simd_int, NDIM> X;
			array<simd_int, NDIM> Y;
			for (int dim = 0; dim < NDIM; dim++) {
				X[dim] = self_ptr->pos[dim].raw();
			}
			expansion<simd_float> L0;
			L0 = simd_float(0.0f);
			for (int j = 0; j < nsource; j += SIMD_FLOAT_SIZE) {
				const int k = j / SIMD_FLOAT_SIZE;
				Y[XDIM] = *((simd_int*) srcx.data() + k);
				Y[YDIM] = *((simd_int*) srcy.data() + k);
				Y[ZDIM] = *((simd_int*) srcz.data() + k);
				simd_float mask = *((simd_float*) masks.data() + k);
				array<simd_float, NDIM> dx;
				for (int dim = 0; dim < NDIM; dim++) {
					dx[dim] = (X[dim] - Y[dim]) * _2float;
				}
				expansion<simd_float> D;
				greens_function(D, dx);
				for (int l = 0; l < EXPANSION_SIZE; l++) {
					L0[l] += mask * D[l];
				}
			}
			for (int i = 0; i < EXPANSION_SIZE; i++) {
				L[i] += L0[i].sum();
			}
		}
	}

}


void gravity_pc(force_vectors& f, int min_rung, tree_id self, const vector<tree_id>& list) {
	if (list.size()) {
		static const simd_float _2float(fixed2float);
		vector<const tree_node*> tree_ptrs(list.size());
		const tree_node* self_ptr = tree_get_node(self);
		const int nsink = self_ptr->nparts();
		const int nsource = round_up((int) list.size(), SIMD_FLOAT_SIZE) / SIMD_FLOAT_SIZE;
		for (int i = 0; i < list.size(); i++) {
			tree_ptrs[i] = tree_get_node(list[i]);
		}
		static thread_local vector<multipole<simd_float>> M;
		static thread_local vector<array<simd_int, NDIM>> Y;
		M.resize(nsource);
		Y.resize(nsource);
		int count = 0;
		for (int i = 0; i < tree_ptrs.size(); i++) {
			const int k = i / SIMD_FLOAT_SIZE;
			const int l = i % SIMD_FLOAT_SIZE;
			const auto& m = tree_ptrs[i]->multi;
			const auto& y = tree_ptrs[i]->pos;
			for (int j = 0; j < MULTIPOLE_SIZE; j++) {
				M[k][j][l] = m[j];
			}
			for (int j = 0; j < NDIM; j++) {
				Y[k][j][l] = y[j].raw();
			}
		}
		const int last = (tree_ptrs.size() - 1) / SIMD_FLOAT_SIZE;
		for (int i = tree_ptrs.size(); i < nsource * SIMD_FLOAT_SIZE; i++) {
			const int k = i / SIMD_FLOAT_SIZE;
			const int l = i % SIMD_FLOAT_SIZE;
			for (int j = 0; j < MULTIPOLE_SIZE; j++) {
				M[k][j][l] = 0.0;
			}
			for (int j = 0; j < NDIM; j++) {
				Y[k][j][l] = Y[last][j][l];
			}
		}
		const auto range = self_ptr->part_range;
		array<simd_int, NDIM> X;
		for (int i = range.first; i < range.second; i++) {
			if (particles_rung(i) >= min_rung) {
				expansion2<simd_float> L;
				L(0, 0, 0) = simd_float(0.0f);
				L(1, 0, 0) = simd_float(0.0f);
				L(0, 1, 0) = simd_float(0.0f);
				L(0, 0, 1) = simd_float(0.0f);
				for (int dim = 0; dim < NDIM; dim++) {
					X[dim] = particles_pos(dim, i).raw();
				}
				for (int j = 0; j < nsource; j++) {
					array<simd_float, NDIM> dx;
					for (int dim = 0; dim < NDIM; dim++) {
						dx[dim] = (X[dim] - Y[j][dim]) * _2float;
					}
					expansion<simd_float> D;
					greens_function(D, dx);
					M2L(L, M[j], D, min_rung == 0);
				}
				const int j = i - range.first;
				f.gx[j] -= L(1, 0, 0).sum();
				f.gy[j] -= L(0, 1, 0).sum();
				f.gz[j] -= L(0, 0, 1).sum();
				f.phi[j] += L(0, 0, 0).sum();
			}
		}
	}

}

void gravity_pp(force_vectors& f, int min_rung, tree_id self, const vector<tree_id>& list) {
	timer tm;
	tm.start();
	constexpr int chunk_size = 32;
	int flops = 0;
	if (list.size()) {
		static const simd_float _2float(fixed2float);
		const simd_float h(get_options().hsoft);
		const simd_float h2 = h * h;
		const simd_float one(1.0);
		const simd_float tiny(1.0e-20);
		const simd_float hinv = simd_float(1) / h;
		const simd_float hinv3 = hinv * hinv * hinv;
		const tree_node* self_ptr = tree_get_node(self);
		const int nsink = self_ptr->nparts();
		for (int li = 0; li < list.size(); li += chunk_size) {
			array<const tree_node*, chunk_size> tree_ptrs;
			int nsource = 0;
			const int maxi = std::min((int) list.size(), li + chunk_size) - li;
			for (int i = li; i < li + maxi; i++) {
				tree_ptrs[i - li] = tree_get_node(list[i]);
				nsource += tree_ptrs[i - li]->nparts();
			}
			nsource = round_up(nsource, SIMD_FLOAT_SIZE);
			static thread_local vector<fixed32> srcx;
			static thread_local vector<fixed32> srcy;
			static thread_local vector<fixed32> srcz;
			static thread_local vector<float> masks;
			srcx.resize(nsource);
			srcy.resize(nsource);
			srcz.resize(nsource);
			masks.resize(nsource);
			int count = 0;
			for (int i = 0; i < maxi; i++) {
				particles_global_read_pos(tree_ptrs[i]->global_part_range(), srcx, srcy, srcz, count);
				count += tree_ptrs[i]->nparts();
			}
			for (int i = 0; i < count; i++) {
				masks[i] = 1.0;
			}
			for (int i = count; i < nsource; i++) {
				masks[i] = 0.0;
			}
			const auto range = self_ptr->part_range;
			array<simd_int, NDIM> X;
			array<simd_int, NDIM> Y;
			for (int i = range.first; i < range.second; i++) {
				if (particles_rung(i) >= min_rung) {
					simd_float gx(0.0);
					simd_float gy(0.0);
					simd_float gz(0.0);
					simd_float phi(0.0);
					for (int dim = 0; dim < NDIM; dim++) {
						X[dim] = particles_pos(dim, i).raw();
					}
					for (int j = 0; j < nsource; j += SIMD_FLOAT_SIZE) {
						const int k = j / SIMD_FLOAT_SIZE;
						Y[XDIM] = *((simd_int*) srcx.data() + k);
						Y[YDIM] = *((simd_int*) srcy.data() + k);
						Y[ZDIM] = *((simd_int*) srcz.data() + k);
						simd_float mask = *((simd_float*) masks.data() + k);
						array<simd_float, NDIM> dx;
						for (int dim = 0; dim < NDIM; dim++) {
							dx[dim] = (X[dim] - Y[dim]) * _2float;                                              // 3
						}
						const simd_float r2 = max(sqr(dx[XDIM], dx[YDIM], dx[ZDIM]), tiny);                    // 5
						const simd_float far_flag = r2 > h2;
						simd_float rinv1, rinv3;
						if (far_flag.sum() == SIMD_FLOAT_SIZE) {                                               // 4
							rinv1 = mask * rsqrt(r2);                                                           // 5
							rinv3 = rinv1 * rinv1 * rinv1;                                                      // 2
						} else {
							const simd_float rinv1_far = mask * simd_float(1) / sqrt(r2);
							const simd_float rinv3_far = rinv1_far * rinv1_far * rinv1_far;
							const simd_float r1overh1 = sqrt(r2) * hinv;
							const simd_float r2oh2 = r1overh1 * r1overh1;
							simd_float rinv3_near = +15.0f / 8.0f;
							rinv3_near = fma(rinv3_near, r2oh2, simd_float(-21.0f / 4.0f));
							rinv3_near = fma(rinv3_near, r2oh2, simd_float(+35.0f / 8.0f));
							rinv3_near *= hinv3;
							simd_float rinv1_near = -5.0f / 16.0f;
							rinv1_near = fma(rinv1_near, r2oh2, simd_float(21.0f / 16.0f));
							rinv1_near = fma(rinv1_near, r2oh2, simd_float(-35.0f / 16.0f));
							rinv1_near = fma(rinv1_near, r2oh2, simd_float(35.0f / 16.0f));
							rinv1_near *= hinv;
							rinv1 = far_flag * rinv1_far + (simd_float(1) - far_flag) * rinv1_near * mask;
							rinv3 = far_flag * rinv3_far + (simd_float(1) - far_flag) * rinv3_near * mask;
						}
						gx = fma(rinv3, dx[XDIM], gx);																			// 2
						gy = fma(rinv3, dx[YDIM], gy);																			// 2
						gz = fma(rinv3, dx[ZDIM], gz);																			// 2
						phi -= rinv1;																									// 1
					}
					const int j = i - range.first;
					f.gx[j] += gx.sum();
					f.gy[j] += gy.sum();
					f.gz[j] += gz.sum();
					f.phi[j] += phi.sum();
				}
			}
		}
	}

}
