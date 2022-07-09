#include <cosmictiger/compress.hpp>
#include <cosmictiger/lightcone.hpp>

void compressed_block::write(FILE* fp) {
	unsigned char size = x.size();
	fwrite(&size, sizeof(unsigned char), 1, fp);
	fwrite(&xc[0], NDIM * sizeof(fixed<int, 31> ), 1, fp);
	fwrite(&vc[0], NDIM * sizeof(float), 1, fp);
	fwrite(&xmax, sizeof(float), 1, fp);
	fwrite(&vmax, sizeof(float), 1, fp);
	for (int dim = 0; dim < NDIM; dim++) {
		fwrite(x[dim].data(), sizeof(signed char), size, fp);
	}
	for (int dim = 0; dim < NDIM; dim++) {
		fwrite(v[dim].data(), sizeof(signed char), size, fp);
	}
}

double compressed_particles::compression_ratio() {
	size_t nparts = 0;
	size_t nbytes = 36;
	for (int i = 0; i < blocks.size(); i++) {
		nbytes += 33;
		nparts += blocks[i].x[XDIM].size();
	}
	return (double) (24 * nparts) / nbytes;
}

void compressed_block::read(FILE* fp) {
	unsigned char size;
	;
	FREAD(&size, sizeof(unsigned char), 1, fp);
	for (int dim = 0; dim < NDIM; dim++) {
		x[dim].resize(size);
		v[dim].resize(size);
	}
	FREAD(&xc[0], NDIM * sizeof(fixed<int, 31> ), 1, fp);
	FREAD(&vc[0], NDIM * sizeof(float), 1, fp);
	FREAD(&xmax, sizeof(float), 1, fp);
	FREAD(&vmax, sizeof(float), 1, fp);
	for (int dim = 0; dim < NDIM; dim++) {
		FREAD(x[dim].data(), sizeof(signed char), size, fp);
	}
	for (int dim = 0; dim < NDIM; dim++) {
		FREAD(v[dim].data(), sizeof(signed char), size, fp);
	}
}

void compressed_particles::write(FILE* fp) {
	int size = blocks.size();
	fwrite(&size, sizeof(unsigned char), 1, fp);
	fwrite(&xc[0], NDIM * sizeof(fixed<int, 31> ), 1, fp);
	fwrite(&vc[0], NDIM * sizeof(float), 1, fp);
	fwrite(&xmax, sizeof(float), 1, fp);
	fwrite(&vmax, sizeof(float), 1, fp);
	for (int i = 0; i < blocks.size(); i++) {
		blocks[i].write(fp);
	}
}

void compressed_particles::read(FILE* fp) {
	int size;
	FREAD(&size, sizeof(unsigned char), 1, fp);
	blocks.resize(size);
	FREAD(&xc[0], NDIM * sizeof(fixed<int, 31> ), 1, fp);
	FREAD(&vc[0], NDIM * sizeof(float), 1, fp);
	FREAD(&xmax, sizeof(float), 1, fp);
	FREAD(&vmax, sizeof(float), 1, fp);
	for (int i = 0; i < blocks.size(); i++) {
		blocks[i].read(fp);
	}
}

vector<compressed_block> compressed_leaves_to_blocks(const vector<pair<int>>& leaves, vector<array<double, 2 * NDIM>>& parts) {
	vector<compressed_block> blocks;
	for (const auto& leaf : leaves) {
		if (leaf.second - leaf.first) {
			array<double, 2 * NDIM> xc;
			for (int dim = 0; dim < 2 * NDIM; dim++) {
				xc[dim] = 0.0;
			}
			for (int i = leaf.first; i < leaf.second; i++) {
				for (int dim = 0; dim < 2 * NDIM; dim++) {
					xc[dim] += parts[i][dim];
				}
			}
			for (int dim = 0; dim < 2 * NDIM; dim++) {
				xc[dim] /= parts.size();
			}
			for (int i = leaf.first; i < leaf.second; i++) {
				for (int dim = 0; dim < 2 * NDIM; dim++) {
					parts[i][dim] -= xc[dim];
				}
			}
			double xmax = 0.0;
			double vmax = 0.0;
			for (int i = leaf.first; i < leaf.second; i++) {
				for (int dim = 0; dim < NDIM; dim++) {
					xmax = std::max(xmax, fabs(parts[i][dim]));
					vmax = std::max(vmax, fabs(parts[i][NDIM + dim]));
				}
			}
			compressed_block entry;
			for (int dim = 0; dim < NDIM; dim++) {
				entry.xc[dim] = xc[dim];
				entry.vc[dim] = xc[NDIM + dim];
			}
			xmax *= 1.0 + 1.0 / 256.0;
			vmax *= 1.0 + 1.0 / 256.0;
			entry.xmax = xmax;
			entry.vmax = vmax;
			for (int i = leaf.first; i < leaf.second; i++) {
				for (int dim = 0; dim < NDIM; dim++) {
					auto& x = parts[i][dim];
					auto& v = parts[i][NDIM + dim];
					x /= xmax;
					v /= vmax;
					entry.x[dim].push_back(floor(128.0 * (1.0 + x)));
					entry.v[dim].push_back(floor(128.0 * (1.0 + v)));
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
	if (nparts <= std::numeric_limits<unsigned char>::max()) {
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

compressed_particles compress_particles(const vector<lc_entry>& inparts) {
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
	for (int dim = 0; dim < NDIM; dim++) {
		box.begin[dim] = 1.0;
		box.end[dim] = -1.0;
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
	vector<array<double, 2 * NDIM>> parts;
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
	return arc;
}

