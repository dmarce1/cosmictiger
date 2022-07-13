#include <cosmictiger/compress.hpp>

#include <cassert>

double compressed_particles::compression_ratio() {
	size_t nparts = 0;
	size_t nbytes = 36;
	for (int i = 0; i < blocks.size(); i++) {
		const int n = blocks[i].x[XDIM].size();
		nbytes += 33 + 2 * NDIM * n;
		nparts += n;
	}
	return (double) (24 * nparts) / nbytes;
}

int compressed_particles::size() {
	int count = 0;
	for (int i = 0; i < blocks.size(); i++) {
		count += blocks[i].x[XDIM].size();
	}
	return count;
}

void compressed_block::write(FILE* fp) {
	unsigned char size = x[XDIM].size();
	fwrite(&size, sizeof(unsigned char), 1, fp);
	fwrite(&xc[0], NDIM * sizeof(lc_real ), 1, fp);
	fwrite(&vc[0], NDIM * sizeof(float), 1, fp);
	fwrite(&xmax, sizeof(float), 1, fp);
	fwrite(&vmax, sizeof(float), 1, fp);
	for (int dim = 0; dim < NDIM; dim++) {
		fwrite(x[dim].data(), sizeof(unsigned short), size, fp);
	}
	for (int dim = 0; dim < NDIM; dim++) {
		fwrite(v[dim].data(), sizeof(unsigned short), size, fp);
	}
}

void compressed_block::read(FILE* fp) {
	unsigned char size;
	fread(&size, sizeof(unsigned char), 1, fp);
	for (int dim = 0; dim < NDIM; dim++) {
		x[dim].resize(size);
		v[dim].resize(size);
	}
	fread(&xc[0], NDIM * sizeof(lc_real ), 1, fp);
	fread(&vc[0], NDIM * sizeof(float), 1, fp);
	fread(&xmax, sizeof(float), 1, fp);
	fread(&vmax, sizeof(float), 1, fp);
	for (int dim = 0; dim < NDIM; dim++) {
		fread(x[dim].data(), sizeof(unsigned short), size, fp);
	}
	for (int dim = 0; dim < NDIM; dim++) {
		fread(v[dim].data(), sizeof(unsigned short), size, fp);
	}
}

void compressed_particles::write(FILE* fp) {
	int size = blocks.size();
	fwrite(&group_id, sizeof(lc_group), 1, fp);
	fwrite(&size, sizeof(int), 1, fp);
	fwrite(&xc[0], NDIM * sizeof(lc_real ), 1, fp);
	fwrite(&vc[0], NDIM * sizeof(float), 1, fp);
	fwrite(&xmax, sizeof(float), 1, fp);
	fwrite(&vmax, sizeof(float), 1, fp);
	for (int i = 0; i < blocks.size(); i++) {
		blocks[i].write(fp);
	}
}

void compressed_particles::read(FILE* fp) {
	int size;
	fread(&group_id, sizeof(lc_group), 1, fp);
	fread(&size, sizeof(int), 1, fp);
	blocks.resize(size);
	fread(&xc[0], NDIM * sizeof(lc_real ), 1, fp);
	fread(&vc[0], NDIM * sizeof(float), 1, fp);
	fread(&xmax, sizeof(float), 1, fp);
	fread(&vmax, sizeof(float), 1, fp);
	for (int i = 0; i < blocks.size(); i++) {
		blocks[i].read(fp);
	}
}

vector<compressed_block> compressed_leaves_to_blocks(const vector<pair<int>>& leaves, vector<array<double, 2 * NDIM>>& parts) {
	vector<compressed_block> blocks;
	range<double, NDIM> xbox;
	range<double, NDIM> vbox;
	for (const auto& leaf : leaves) {
		if (leaf.second - leaf.first) {
			array<double, 2 * NDIM> xc;
			for (int dim = 0; dim < NDIM; dim++) {
				xbox.begin[dim] = std::numeric_limits<double>::max();
				xbox.end[dim] = -std::numeric_limits<double>::max();
				vbox.begin[dim] = std::numeric_limits<double>::max();
				vbox.end[dim] = -std::numeric_limits<double>::max();
			}
			for (int i = leaf.first; i < leaf.second; i++) {
				for (int dim = 0; dim < NDIM; dim++) {
					const double x = parts[i][dim];
					const double v = parts[i][NDIM + dim];
					xbox.begin[dim] = std::min(x, xbox.begin[dim]);
					xbox.end[dim] = std::max(x, xbox.end[dim]);
					vbox.begin[dim] = std::min(v, vbox.begin[dim]);
					vbox.end[dim] = std::max(v, vbox.end[dim]);
				}
			}
			for (int dim = 0; dim < NDIM; dim++) {
				xc[dim] = 0.5 * (xbox.begin[dim] + xbox.end[dim]);
				xc[NDIM + dim] = 0.5 * (vbox.begin[dim] + vbox.end[dim]);
			}
			for (int i = leaf.first; i < leaf.second; i++) {
				for (int dim = 0; dim < 2 * NDIM; dim++) {
					parts[i][dim] -= xc[dim];
				}
			}
			double xmax = 0.0, vmax = 0.0;
			for (int dim = 0; dim < NDIM; dim++) {
				xmax = std::max(xmax, 0.5 * (xbox.end[dim] - xbox.begin[dim]));
				vmax = std::max(vmax, 0.5 * (vbox.end[dim] - vbox.begin[dim]));
			}
			compressed_block entry;
			for (int dim = 0; dim < NDIM; dim++) {
				entry.xc[dim] = xc[dim];
				entry.vc[dim] = xc[NDIM + dim];
			}
			xmax *= 1.0 + 1.0 / (1<<16);
			vmax *= 1.0 + 1.0 / (1<<16);
			entry.xmax = xmax;
			entry.vmax = vmax;
			for (int i = leaf.first; i < leaf.second; i++) {
				for (int dim = 0; dim < NDIM; dim++) {
					auto& x = parts[i][dim];
					auto& v = parts[i][NDIM + dim];
					x /= xmax;
					v /= vmax;
					entry.x[dim].push_back(floor((1 << 15) * (1.0 + x)));
					entry.v[dim].push_back(floor((1 << 15) * (1.0 + v)));
				}
			}
			blocks.push_back(entry);
		}
	}
	return blocks;
}

void compressed_tree_create(vector<pair<int>>& leaves, vector<array<double, 2 * NDIM>>& parts, const range<double, 2 * NDIM>& box,
		const pair<int>& part_range) {
	const int nparts = part_range.second - part_range.first;
	if (nparts <= 64) {
		leaves.push_back(part_range);
	} else {
		int xdim = box.longest_dim();
		int lo = part_range.first;
		int hi = part_range.second;
		double xmid = 0.5 * (box.begin[xdim] + box.end[xdim]);
		while (lo < hi) {
			if (parts[lo][xdim] >= xmid) {
				while (lo != hi) {
					hi--;
					if (parts[hi][xdim] < xmid) {
						std::swap(parts[hi], parts[lo]);
						break;
					}
				}
			}
			lo++;
		}
		const int mid = hi;
		auto left_range = part_range;
		auto right_range = part_range;
		auto left_box = box;
		auto right_box = box;
		left_range.second = right_range.first = mid;
		left_box.end[xdim] = right_box.begin[xdim] = xmid;
		compressed_tree_create(leaves, parts, left_box, left_range);
		compressed_tree_create(leaves, parts, right_box, right_range);
	}
}

#ifndef LC2GADGET2

compressed_particles compress_particles(const vector<lc_entry>& inparts, lc_group id) {
	vector<array<double, 2 * NDIM>> particles;
	particles.reserve(inparts.size());
	array<double, NDIM> xc;
	array<double, NDIM> vc;
	for (int dim = 0; dim < NDIM; dim++) {
		xc[dim] = vc[dim] = 0.0;
	}
	for (int i = 0; i < inparts.size(); i++) {
		array<double, 2 * NDIM> x;
		x[XDIM] = inparts[i].x.to_double();
		x[YDIM] = inparts[i].y.to_double();
		x[ZDIM] = inparts[i].z.to_double();
		x[NDIM + XDIM] = inparts[i].vx;
		x[NDIM + YDIM] = inparts[i].vy;
		x[NDIM + ZDIM] = inparts[i].vz;
		for (int dim = 0; dim < NDIM; dim++) {
			xc[dim] += x[dim];
			vc[dim] += x[NDIM + dim];
		}
		particles.push_back(x);
	}
	for (int dim = 0; dim < NDIM; dim++) {
		xc[dim] /= particles.size();
		vc[dim] /= particles.size();
	}
	double xmax = 0.0;
	double vmax = 0.0;
	for (int i = 0; i < particles.size(); i++) {
		for (int dim = 0; dim < NDIM; dim++) {
			auto& x = particles[i][dim];
			auto& v = particles[i][NDIM + dim];
			x -= xc[dim];
			v -= vc[dim];
			xmax = std::max(xmax, fabs(x));
			vmax = std::max(vmax, fabs(v));
		}
	}
	range<double, 2 * NDIM> box;
	for (int dim = 0; dim < 2 * NDIM; dim++) {
		box.begin[dim] = std::numeric_limits<double>::max();
		box.end[dim] = -std::numeric_limits<double>::max();
	}
	for (int i = 0; i < particles.size(); i++) {
		for (int dim = 0; dim < NDIM; dim++) {
			particles[i][dim] /= xmax;
			particles[i][NDIM + dim] /= vmax;
		}
		for (int dim = 0; dim < 2 * NDIM; dim++) {
			const auto& x = particles[i][dim];
			box.begin[dim] = std::min(box.begin[dim], x);
			box.end[dim] = std::max(box.end[dim], x);
		}
	}
	vector<pair<int>> leaves;
	pair<int> part_range;
	part_range.first = 0;
	part_range.second = particles.size();
	compressed_tree_create(leaves, particles, box, part_range);
	compressed_particles arc;
	arc.blocks = compressed_leaves_to_blocks(leaves, particles);
	for (int dim = 0; dim < NDIM; dim++) {
		arc.xc[dim] = xc[dim];
		arc.vc[dim] = vc[dim];
	}
	arc.xmax = xmax;
	arc.vmax = vmax;
	arc.group_id = id;
	return arc;
}

#endif
