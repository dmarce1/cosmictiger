#include <tigerfmm/gravity.hpp>
#include <tigerfmm/math.hpp>
#include <tigerfmm/safe_io.hpp>
#include <tigerfmm/tree.hpp>

void gravity_cc_ewald(const vector<tree_id>&) {

}

void gravity_cc(const vector<tree_id>&) {

}

void gravity_cp(const vector<tree_id>&) {

}

void gravity_pc(force_vectors& f, int min_rung, tree_id self, const vector<tree_id>&) {

}

void gravity_pp(force_vectors& f, int min_rung, tree_id self, const vector<tree_id>& list) {
	if (list.size()) {
		static const simd_float _2float(fixed2float);
		const simd_float h(get_options().hsoft);
		const simd_float h2 = h * h;
		const simd_float one(1.0);
		const simd_float tiny(1.0e-20);
		const simd_float hinv = simd_float(1) / h;
		const simd_float hinv3 = hinv * hinv * hinv;
		vector<const tree_node*> tree_ptrs(list.size());
		const tree_node* self_ptr = tree_get_node(self);
		const int nsink = self_ptr->nparts();
		int nsource = 0;
		for (int i = 0; i < list.size(); i++) {
			tree_ptrs[i] = tree_get_node(list[i]);
			nsource += tree_ptrs[i]->nparts();
		}
		nsource = round_up(nsource, SIMD_FLOAT_SIZE);
		vector<fixed32> srcx(nsource);
		vector<fixed32> srcy(nsource);
		vector<fixed32> srcz(nsource);
		vector<float> masks(nsource);
		int count = 0;
		for (int i = 0; i < tree_ptrs.size(); i++) {
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
			simd_float gx(0.0);
			simd_float gy(0.0);
			simd_float gz(0.0);
			simd_float phi(0.0);
			if (particles_rung(i) >= min_rung) {
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
						dx[dim] = (X[dim] - Y[dim]) * _2float;
					}
					const simd_float r2 = max(sqr(dx[XDIM], dx[YDIM], dx[ZDIM]), tiny);                   // 6
					const simd_float far_flag = r2 > h2;                                                                    // 1
					simd_float rinv1, rinv3;
					if (far_flag.sum() == SIMD_FLOAT_SIZE) {                                                             // 7
						rinv1 = mask * rsqrt(r2);                                      // 1 + FLOP_DIV + FLOP_SQRT
						rinv3 = rinv1 * rinv1 * rinv1;                                                     // 2
					} else {
						const simd_float rinv1_far = mask * simd_float(1) / sqrt(r2);                 // 1 + FLOP_DIV + FLOP_SQRT
						const simd_float rinv3_far = rinv1_far * rinv1_far * rinv1_far;                                      // 2
						const simd_float r1overh1 = sqrt(r2) * hinv;                                             // FLOP_SQRT + 1
						const simd_float r2oh2 = r1overh1 * r1overh1;                                                     // 1
						simd_float rinv3_near = +15.0f / 8.0f;
						rinv3_near = fma(rinv3_near, r2oh2, simd_float(-21.0f / 4.0f));
						rinv3_near = fma(rinv3_near, r2oh2, simd_float(+35.0f / 8.0f));
						rinv3_near *= hinv3;
						simd_float rinv1_near = -5.0f / 16.0f;
						rinv1_near = fma(rinv1_near, r2oh2, simd_float(21.0f / 16.0f));
						rinv1_near = fma(rinv1_near, r2oh2, simd_float(-35.0f / 16.0f));
						rinv1_near = fma(rinv1_near, r2oh2, simd_float(35.0f / 16.0f));
						rinv1_near *= hinv;
						rinv1 = far_flag * rinv1_far + (simd_float(1) - far_flag) * rinv1_near * mask;                  // 4
						rinv3 = far_flag * rinv3_far + (simd_float(1) - far_flag) * rinv3_near * mask;                  // 4
					}
					gx = fma(rinv3, dx[XDIM], gx);                                                                // 2
					gy = fma(rinv3, dx[YDIM], gy);                                                                // 2
					gz = fma(rinv3, dx[ZDIM], gz);                                                                // 2
					phi -= rinv1;                                                                                           // 1
				}
			}
			const int j = i - range.first;
			f.gx[j] += gx.sum();
			f.gy[j] += gy.sum();
			f.gz[j] += gz.sum();
			f.phi[j] += phi.sum();
		}
	}
}
