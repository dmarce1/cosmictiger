#include <cosmictiger/assert.hpp>
#include <cosmictiger/bh.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/gravity.hpp>
#include <cosmictiger/simd.hpp>

#include <fenv.h>

int bh_sort(vector<array<float, NDIM>>& parts, vector<int>& sort_order, int begin, int end, float xm, int xdim) {
	int lo = begin;
	int hi = end;
	float xmid(xm);
	while (lo < hi) {
		if (parts[lo][xdim] >= xmid) {
			while (lo != hi) {
				hi--;
				if (parts[hi][xdim] < xmid) {
					std::swap(sort_order[hi], sort_order[lo]);
					auto tmp = parts[hi];
					parts[hi] = parts[lo];
					parts[lo] = tmp;
					break;
				}
			}
		}
		lo++;
	}
	return hi;

}

void bh_create_tree(vector<bh_tree_node>& nodes, vector<int>& sink_buckets, int self, vector<array<float, NDIM>>& parts, vector<int>& sort_order,
		range<float> box, int begin, int end, int bucket_size, int depth = 0) {
	/*feenableexcept (FE_DIVBYZERO);
	feenableexcept (FE_INVALID);
	feenableexcept (FE_OVERFLOW);*/
	auto* node_ptr = &nodes[self];
	array<float, NDIM> com;
	float radius;
	if (end - begin <= bucket_size) {
		node_ptr->parts.first = begin;
		node_ptr->parts.second = end;
		sink_buckets.push_back(self);
		node_ptr->mass = end - begin;
		for (int dim = 0; dim < NDIM; dim++) {
			com[dim] = 0.0;
		}
		if (node_ptr->mass > 0.0) {
			for (int i = begin; i < end; i++) {
				for (int dim = 0; dim < NDIM; dim++) {
					com[dim] += parts[i][dim];
				}
			}
			const float massinv = 1.0f / node_ptr->mass;
			for (int dim = 0; dim < NDIM; dim++) {
				com[dim] *= massinv;
			}
		}
		radius = 0.0;
		for (int i = begin; i < end; i++) {
			float r2 = 0.0;
			for (int dim = 0; dim < NDIM; dim++) {
				r2 += sqr(com[dim] - parts[i][dim]);
			}
			radius = std::max(radius, r2);
		}
		radius = std::sqrt(radius);
		node_ptr->radius = radius;
		node_ptr->pos = com;
	} else {
		const int xdim = box.longest_dim();
		const float xmid = (box.begin[xdim] + box.end[xdim]) * 0.5;
		const int mid = bh_sort(parts, sort_order, begin, end, xmid, xdim);
		auto boxl = box;
		auto boxr = box;
		boxl.end[xdim] = boxr.begin[xdim] = xmid;
		node_ptr->children[LEFT] = nodes.size();
		node_ptr->children[RIGHT] = nodes.size() + 1;
		nodes.resize(nodes.size() + NCHILD);
		node_ptr = &nodes[self];
		bh_create_tree(nodes, sink_buckets, node_ptr->children[LEFT], parts, sort_order, boxl, begin, mid, bucket_size, depth + 1);
		node_ptr = &nodes[self];
		bh_create_tree(nodes, sink_buckets, node_ptr->children[RIGHT], parts, sort_order, boxr, mid, end, bucket_size, depth + 1);
		node_ptr = &nodes[self];
		const auto& childl = nodes[node_ptr->children[LEFT]];
		const auto& childr = nodes[node_ptr->children[RIGHT]];
		node_ptr->mass = childl.mass + childr.mass;
		if (childl.mass == 0.0) {
			for (int dim = 0; dim < NDIM; dim++) {
				com[dim] = childr.pos[dim];
			}
			radius = childr.radius;
		} else if (childr.mass == 0.0) {
			for (int dim = 0; dim < NDIM; dim++) {
				com[dim] = childl.pos[dim];
			}
			radius = childl.radius;
		} else {
			const float massinv = 1.0f / node_ptr->mass;
			for (int dim = 0; dim < NDIM; dim++) {
				com[dim] = (childl.mass * childl.pos[dim] + childr.mass * childr.pos[dim]) * massinv;
			}
			float r2l = 0.0;
			float r2r = 0.0;
			float r2 = 0.0;
			for (int dim = 0; dim < NDIM; dim++) {
				r2l += sqr(com[dim] - childl.pos[dim]);
				r2r += sqr(com[dim] - childr.pos[dim]);
			}
			r2 = std::max(r2, (float) sqr(box.begin[XDIM] - com[XDIM], box.begin[YDIM] - com[YDIM], box.begin[ZDIM] - com[ZDIM]));
			r2 = std::max(r2, (float) sqr(box.begin[XDIM] - com[XDIM], box.begin[YDIM] - com[YDIM], box.end[ZDIM] - com[ZDIM]));
			r2 = std::max(r2, (float) sqr(box.begin[XDIM] - com[XDIM], box.end[YDIM] - com[YDIM], box.begin[ZDIM] - com[ZDIM]));
			r2 = std::max(r2, (float) sqr(box.begin[XDIM] - com[XDIM], box.end[YDIM] - com[YDIM], box.end[ZDIM] - com[ZDIM]));
			r2 = std::max(r2, (float) sqr(box.end[XDIM] - com[XDIM], box.begin[YDIM] - com[YDIM], box.begin[ZDIM] - com[ZDIM]));
			r2 = std::max(r2, (float) sqr(box.end[XDIM] - com[XDIM], box.begin[YDIM] - com[YDIM], box.end[ZDIM] - com[ZDIM]));
			r2 = std::max(r2, (float) sqr(box.end[XDIM] - com[XDIM], box.end[YDIM] - com[YDIM], box.begin[ZDIM] - com[ZDIM]));
			r2 = std::max(r2, (float) sqr(box.end[XDIM] - com[XDIM], box.end[YDIM] - com[YDIM], box.end[ZDIM] - com[ZDIM]));
			radius = std::sqrt(r2);
		}
		node_ptr->radius = radius;
		node_ptr->pos = com;

	}
}

void bh_tree_evaluate(const vector<bh_tree_node>& nodes, vector<int>& sink_buckets, vector<float>& phi, vector<array<float, NDIM>>& parts, float theta) {
	const float h = get_options().hsoft;
	const float GM = get_options().GM;
	const float hinv = 1.0 / h;
	const float h2inv = 1.0 / (h * h);
	const float h2 = h * h;
	const simd_float tiny(1e-20);
	const int nthreads = std::max(std::min((int) parts.size() / 512, 8 * (int) hpx::threads::hardware_concurrency()), 1);
	vector<hpx::future<void>> futs;
	for (int proc = 0; proc < nthreads; proc++) {
		futs.push_back(hpx::async([proc,nthreads,&sink_buckets,&nodes,&phi,&parts,h,GM,hinv,h2inv,h2,theta,tiny]() {
			for (int i = proc; i < sink_buckets.size(); i+=nthreads) {
				const auto& sink_bucket = sink_buckets[i];
				vector<int> checklist(1, 0);
				vector<int> nextlist;
				vector<bh_source> sourcelist;
				const auto myparts = nodes[sink_bucket].parts;
				const auto mypos = nodes[sink_bucket].pos;
				const auto myradius = nodes[sink_bucket].radius;
				for (int i = myparts.first; i < myparts.second; i++) {
					phi[i] = -SELF_PHI * hinv;
				}
				const float thetainv = 1.0f / theta;
				while (checklist.size()) {
					nextlist.resize(0);
					for (int ci = 0; ci < checklist.size(); ci++) {
						const auto& node = nodes[checklist[ci]];
						const float dx = mypos[XDIM] - node.pos[XDIM];
						const float dy = mypos[YDIM] - node.pos[YDIM];
						const float dz = mypos[ZDIM] - node.pos[ZDIM];
						const float r2 = sqr(dx, dy, dz);
						if (r2 > sqr(thetainv * (node.radius + myradius))) {
							bh_source source;
							source.x = node.pos;
							source.m = node.mass;
							sourcelist.push_back(source);
						} else if( node.children[LEFT] == -1) {
							for (int i = node.parts.first; i < node.parts.second; i++) {
								bh_source source;
								source.x = parts[i];
								source.m = 1.0f;
								sourcelist.push_back(source);
							}
						} else {
							nextlist.push_back(node.children[LEFT]);
							nextlist.push_back(node.children[RIGHT]);
						}
					}
					std::swap(nextlist, checklist);
				}
				const int maxi = round_up((int) sourcelist.size(), SIMD_FLOAT_SIZE);
				while(sourcelist.size() < maxi) {
					bh_source src;
					src.m = 0.0f;
					for( int dim = 0; dim < NDIM; dim++) {
						src.x[dim] = 0.f;
					}
					sourcelist.push_back(src);
				}
				for (int i = myparts.first; i < myparts.second; i++) {
					array<simd_float,NDIM> X;
					for( int dim = 0; dim < NDIM; dim++) {
						X[dim] = parts[i][dim];
					}
					for (int j = 0; j < sourcelist.size(); j += SIMD_FLOAT_SIZE) {
						array<simd_float,NDIM> Y;
						simd_float M;
						for( int k = 0; k < SIMD_FLOAT_SIZE; k++) {
							const auto& src = sourcelist[j+k];
							for( int dim = 0; dim < NDIM; dim++) {
								Y[dim][k] = src.x[dim];
							}
							M[k] = src.m;
						}
						array<simd_float, NDIM> dx;
						for (int dim = 0; dim < NDIM; dim++) {
							dx[dim] = X[dim] - Y[dim];                                 // 3
			}
			const simd_float r2 = max(sqr(dx[XDIM], dx[YDIM], dx[ZDIM]), tiny);                 // 5
			const simd_float far_flag = r2 > h2;// 1
			simd_float rinv1;
			if (far_flag.sum() == SIMD_FLOAT_SIZE) {                                            // 7/8
				rinv1 = rsqrt(r2);// 5
			} else {
				const simd_float r = sqrt(r2);                                                    // 4
				const simd_float rinv1_far = simd_float(1) / r;// 5
				const simd_float r1overh1 = r * hinv;// 1
				const simd_float r2oh2 = r1overh1 * r1overh1;// 1
				simd_float rinv1_near = -5.0f / 16.0f;
				rinv1_near = fma(rinv1_near, r2oh2, simd_float(21.0f / 16.0f));// 2
				rinv1_near = fma(rinv1_near, r2oh2, simd_float(-35.0f / 16.0f));// 2
				rinv1_near = fma(rinv1_near, r2oh2, simd_float(35.0f / 16.0f));// 2
				rinv1_near *= hinv;// 1
				const auto near_flag = (simd_float(1) - far_flag);// 1
				rinv1 = far_flag * rinv1_far + near_flag * rinv1_near;// 4
			}
			phi[i] -= (M * rinv1).sum();																									// 1
		}
		phi[i] *= GM;
	}
}}));
	}
	hpx::wait_all(futs.begin(), futs.end());
}

vector<float> direct_evaluate(const vector<array<float, NDIM>>& x) {
	const float h = get_options().hsoft;
	const float GM = get_options().GM;
	const float hinv = 1.0 / h;
	const float h2inv = 1.0 / (h * h);
	const float h2 = h * h;
	vector<float> phi(x.size(), 0.0);
	for (int i = 0; i < x.size(); i++) {
		for (int j = i + 1; j < x.size(); j++) {
			const float dx = x[i][XDIM] - x[j][XDIM];
			const float dy = x[i][YDIM] - x[j][YDIM];
			const float dz = x[i][ZDIM] - x[j][ZDIM];
			const float r2 = sqr(dx, dy, dz);
			float pot;
			if (r2 > h2) {
				pot = 1.0f / sqrtf(r2);
			} else {
				const float q2 = r2 * h2inv;
				pot = -5.0f / 16.0f;
				pot = fmaf(pot, q2, 21.0f / 16.0f);
				pot = fmaf(pot, q2, -35.0f / 16.0f);
				pot = fmaf(pot, q2, 35.0f / 16.0f);
				pot *= hinv;
			}
			phi[i] -= GM * pot;
			phi[j] -= GM * pot;
		}
	}
	return phi;
}

vector<float> bh_evaluate_potential(const vector<array<fixed32, NDIM>>& x_fixed) {
	ALWAYS_ASSERT(x_fixed.size() > 1);
//	const bool gpu = x_fixed.size() >= BH_CUDA_MIN;
	const bool gpu = false;
	vector<array<float, NDIM>> x(x_fixed.size());
	array<fixed32, NDIM> x0 = x_fixed[0];
	for (int i = 0; i < x_fixed.size(); i++) {
		array<float, NDIM> this_x;
		for (int dim = 0; dim < NDIM; dim++) {
			this_x[dim] = distance(x_fixed[i][dim], x0[dim]);
		}
		x[i] = this_x;
	}
	auto xcopy = x;
	vector<bh_tree_node> nodes(1);
	range<float> box;
	for (int dim = 0; dim < NDIM; dim++) {
		box.begin[dim] = std::numeric_limits<float>::max();
		box.end[dim] = -std::numeric_limits<float>::max();
	}
	for (const auto& pos : x) {
		for (int dim = 0; dim < NDIM; dim++) {
			const auto this_x = pos[dim];
			box.begin[dim] = std::min(box.begin[dim], this_x);
			box.end[dim] = std::max(box.end[dim], this_x);
		}
	}
	vector<float> rpot;
	vector<int> sink_buckets;
	vector<int> sort_order;
	sort_order.reserve(x.size());
	for (int i = 0; i < x.size(); i++) {
		sort_order.push_back(i);
	}
	vector<float> pot(x.size());
	bh_create_tree(nodes, sink_buckets, 0, x, sort_order, box, 0, x.size(), gpu ? BH_CUDA_BUCKET_SIZE : BH_CPU_BUCKET_SIZE);
	if (gpu) {
		pot = bh_cuda_tree_evaluate(nodes, sink_buckets, x, 1.0);
	} else {
		bh_tree_evaluate(nodes, sink_buckets, pot, x, 1.0);
	}
	rpot.resize(x.size());
	for (int i = 0; i < sort_order.size(); i++) {
		rpot[sort_order[i]] = pot[i];
	}
/*	auto apot = direct_evaluate(xcopy);
	double err = 0.0;
	if (x.size() > 1024) {
		for (int i = 0; i < rpot.size(); i++) {
			err +=sqr((rpot[i] - apot[i]) / apot[i]);
	//		PRINT("%e %e\n", rpot[i], apot[i]);
		}
		err /= rpot.size();
		err = std::sqrt(err);
		if (rpot.size() > 1024) {
			PRINT("%e %i\n", err, rpot.size());
		}
	}*/
	return rpot;
}
