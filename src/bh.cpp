#include <cosmictiger/assert.hpp>
#include <cosmictiger/bh.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/gravity.hpp>

#include <fenv.h>

int bh_sort(vector<array<float, NDIM>>& parts, int begin, int end, float xm, int xdim) {
	int lo = begin;
	int hi = end;
	float xmid(xm);
	while (lo < hi) {
		if (parts[lo][xdim] >= xmid) {
			while (lo != hi) {
				hi--;
				if (parts[hi][xdim] < xmid) {
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

void bh_create_tree(vector<bh_tree_node>& nodes, int self, vector<array<float, NDIM>>& parts, range<float> box, int begin, int end, int depth = 0) {
	feenableexcept (FE_DIVBYZERO);
	feenableexcept (FE_INVALID);
	feenableexcept (FE_OVERFLOW);
	auto* node_ptr = &nodes[self];
	bool leaf = true;
	for (int i = begin + 1; i < end; i++) {
		if (parts[i] != parts[begin]) {
			leaf = false;
			break;
		}
	}
	if (leaf) {
		node_ptr->count = end - begin;
		if (node_ptr->count) {
			node_ptr->pos = parts[begin];
		} else {
			node_ptr->pos = box.begin;
		}
		node_ptr->radius = 0.0;
	} else {
		const int xdim = box.longest_dim();
		const float xmid = (box.begin[xdim] + box.end[xdim]) * 0.5;
		const int mid = bh_sort(parts, begin, end, xmid, xdim);
		auto boxl = box;
		auto boxr = box;
		boxl.end[xdim] = boxr.begin[xdim] = xmid;
		node_ptr->children[LEFT] = nodes.size();
		node_ptr->children[RIGHT] = nodes.size() + 1;
		nodes.resize(nodes.size() + NCHILD);
		node_ptr = &nodes[self];
		bh_create_tree(nodes, node_ptr->children[LEFT], parts, boxl, begin, mid, depth + 1);
		node_ptr = &nodes[self];
		bh_create_tree(nodes, node_ptr->children[RIGHT], parts, boxr, mid, end, depth + 1);
		node_ptr = &nodes[self];
		const auto& childl = nodes[node_ptr->children[LEFT]];
		const auto& childr = nodes[node_ptr->children[RIGHT]];
		node_ptr->count = childl.count + childr.count;
		array<float, NDIM> com;
		float radius;
		if (childl.count == 0) {
			for (int dim = 0; dim < NDIM; dim++) {
				com[dim] = childr.pos[dim];
			}
			radius = childr.radius;
		} else if (childr.count == 0) {
			for (int dim = 0; dim < NDIM; dim++) {
				com[dim] = childl.pos[dim];
			}
			radius = childl.radius;
		} else {
			const float countinv = 1.0f / node_ptr->count;
			for (int dim = 0; dim < NDIM; dim++) {
				com[dim] = (childl.count * childl.pos[dim] + childr.count * childr.pos[dim]) * countinv;
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

float bh_tree_evaluate(const vector<bh_tree_node>& nodes, const array<float, NDIM>& sink, float theta) {
	const float h = get_options().hsoft;
	const float GM = get_options().GM;
	const float hinv = 1.0 / h;
	const float h2inv = 1.0 / (h * h);
	const float h2 = h * h;
	vector<int> checklist(1, 0);
	vector<int> nextlist;
	float phi = -SELF_PHI * hinv;
	const float thetainv = 1.0f / theta;
	int interactions = 0;
	while (checklist.size()) {
		nextlist.resize(0);
		for (int ci = 0; ci < checklist.size(); ci++) {
			const auto& node = nodes[checklist[ci]];
			if (node.count) {
				const float dx = sink[XDIM] - node.pos[XDIM];
				const float dy = sink[YDIM] - node.pos[YDIM];
				const float dz = sink[ZDIM] - node.pos[ZDIM];
				const float r2 = sqr(dx, dy, dz);
				if (r2 > thetainv * node.radius || node.children[LEFT] == -1) {
					if (r2 > h2) {
						phi -= node.count / sqrtf(r2);
					} else {
						const float q2 = r2 * h2inv;
						float rinv = -5.0f / 16.0f;
						rinv = fmaf(rinv, q2, 21.0f / 16.0f);
						rinv = fmaf(rinv, q2, -35.0f / 16.0f);
						rinv = fmaf(rinv, q2, 35.0f / 16.0f);
						rinv *= hinv;
						phi -= node.count * rinv;
					}
					interactions++;
				} else {
					nextlist.push_back(node.children[LEFT]);
					nextlist.push_back(node.children[RIGHT]);
				}
			}
		}
		std::swap(nextlist, checklist);
	}
	phi *= GM;
//	PRINT( "%i\n", interactions);
	return phi;
}

vector<float> bh_evaluate_potential(const vector<array<fixed32, NDIM>>& x_fixed) {
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
	bh_create_tree(nodes, 0, xcopy, box, 0, xcopy.size());
	if (x.size() >= 512) {
		auto pot = bh_cuda_tree_evaluate(nodes, x, 0.5);
		return pot;
	} else {
		vector<float> pot(x.size());
		for (int i = 0; i < x.size(); i++) {
			pot[i] = bh_tree_evaluate(nodes, x[i], 0.5);
		}
		return pot;
	}
}
