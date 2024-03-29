#include <cosmictiger/compress.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/constants.hpp>
#include <unordered_map>
#include <fstream>

struct gadget2_header {
	std::uint32_t npart[6]; /*!< npart[1] gives the number of particles in the present file, other particle types are ignored */
	double mass[6]; /*!< mass[1] gives the particle mass */
	double time; /*!< time (=cosmological scale factor) of snapshot */
	double redshift; /*!< redshift of snapshot */
	std::int32_t flag_sfr; /*!< flags whether star formation is used (not available in L-Gadget2) */
	std::int32_t flag_feedback; /*!< flags whether feedback from star formation is included */
	std::uint32_t npartTotal[6]; /*!< npart[1] gives the total number of particles in the run. If this number exceeds 2^32, the npartTotal[2] stores
	 the result of a division of the particle number by 2^32, while npartTotal[1] holds the remainder. */
	std::int32_t flag_cooling; /*!< flags whether radiative cooling is included */
	std::int32_t num_files; /*!< determines the number of files that are used for a snapshot */
	double BoxSize; /*!< Simulation box size (in code units) */
	double Omega0; /*!< matter density */
	double OmegaLambda; /*!< vacuum energy density */
	double HubbleParam; /*!< little 'h' */
	std::int32_t flag_stellarage; /*!< flags whether the age of newly formed stars is recorded and saved */
	std::int32_t flag_metals; /*!< flags whether metal enrichment is included */
	std::int32_t hashtabsize; /*!< gives the size of the hashtable belonging to this snapshot file */
	char fill[84]; /*!< fills to 256 Bytes */
};

struct particles_set {
	array<vector<float>, NDIM> x;
	array<vector<float>, NDIM> v;
	vector<lc_group> g;
	void write(FILE* fp) const;
};

void uncompress_particles(particles_set& parts, const compressed_particles& arc) {
	int total = 0;
	int id = 0;
	for (int i = 0; i < arc.blocks.size(); i++) {
		const auto& block = arc.blocks[i];
		const int nparts = block.x[XDIM].size();
		total += nparts;
		for (int j = 0; j < nparts; j++) {
			for (int dim = 0; dim < NDIM; dim++) {
				double x = (block.x[dim][j] + 0.5) / (1<<15) - 1.0;
				double v = (block.v[dim][j] + 0.5) / (1<<15) - 1.0;
				x *= block.xmax;
				x += block.xc[dim].to_double();
				x *= arc.xmax;
				v *= block.vmax;
				v += block.vc[dim];
				v *= arc.vmax;
				v += arc.vc[dim];
				parts.x[dim].push_back(x);
				parts.v[dim].push_back(v);
			}
			parts.g.push_back(id++);
		}

	}
//	parts.g.resize(total + parts.g.size(), arc.group_id);
}

void particles_set::write(FILE* fp) const {
	int size;
	size = sizeof(float) * x[0].size() * NDIM;
	fwrite(&size, sizeof(int), 1, fp);
	for (int dim = 0; dim < NDIM; dim++) {
		fwrite(x[dim].data(), sizeof(float), x[dim].size(), fp);
	}
	fwrite(&size, sizeof(int), 1, fp);
	fwrite(&size, sizeof(int), 1, fp);
	for (int dim = 0; dim < NDIM; dim++) {
		fwrite(v[dim].data(), sizeof(float), v[dim].size(), fp);
	}
	fwrite(&size, sizeof(int), 1, fp);
	size = sizeof(lc_group) * g.size();
	fwrite(&size, sizeof(int), 1, fp);
	fwrite(g.data(), sizeof(lc_group), g.size(), fp);
	fwrite(&size, sizeof(int), 1, fp);
}

void read_ascii(FILE* fp, vector<std::string>& fields, std::unordered_map<std::string, vector<double>>& data) {
	char line[10000];
	fgets(line, 10000, fp);
	char* ptr = line;
	fields.push_back(std::string());
	ptr++;
	while (*ptr != '\0' && *ptr != '\n') {
		if (*ptr == ' ') {
			ptr++;
			fields.push_back(std::string());
		}
		fields.back() += *ptr;
		ptr++;
	}
	std::string field;
	while (fgets(line, 10000, fp) != 0) {
		ptr = line;
		int fi = 0;
		if (*ptr != '#') {
			while (fi != fields.size()) {
				if (isspace(*ptr) || *ptr == '\0') {
					data[fields[fi]].push_back(atof(field.c_str()));
					field = std::string();
					fi++;
					while (isspace(*ptr) || *ptr == '\0') {
						ptr++;
					}
				} else {
					field += *ptr;
					ptr++;
				}
			}
		}
	}
}

size_t num_lines(std::string file) {
	size_t numLines = 0;
	std::ifstream in(file);
	if (in) {
		std::string unused;
		while (std::getline(in, unused)) {
			++numLines;
		}
	}
	return numLines;
}

#define DATABASE_NAME "lightcone.txt"

int main(int argc, char* argv[]) {
	const char* infile = argv[1];
	FILE* fpin = fopen(infile, "rb");
	if (!fpin) {
		printf("Unable to open %s for reading\n", infile);
		return -1;
	}
	size_t base_id = num_lines(DATABASE_NAME);
	int num_groups;
	fread(&num_groups, sizeof(int), 1, fpin);
	printf("%i FoF groups found.\n", num_groups);
	options* opts = (options*) malloc(sizeof(options));
	fread(opts, sizeof(options), 1, fpin);
	std::unordered_map<std::string, vector<double>> db;
	size_t halo_cnt = 0;
	size_t subhalo_cnt = 0;
	for (int i = 0; i < num_groups; i++) {
		printf("\r%i %li %li %li", i, halo_cnt, subhalo_cnt, halo_cnt + subhalo_cnt);
		fflush (stdout);
		FILE* fpout = fopen("gadget.tmp", "wb");
		if (!fpout) {
			printf("Unable to open gadget.tmp for writing\n");
			return -2;
		}
		gadget2_header header;
		particles_set parts;
		compressed_particles arc;
		arc.read(fpin);
		uncompress_particles(parts, arc);
		int nparts = arc.size();
		const double xscale = opts->code_to_cm * opts->hubble / constants::mpc_to_cm;
		const double vscale = opts->code_to_cm / opts->code_to_s / 100000.0;
		array<double, NDIM> minx;
		for (int dim = 0; dim < NDIM; dim++) {
			minx[dim] = std::numeric_limits<double>::max();
		}
		for (int i = 0; i < parts.x[XDIM].size(); i++) {
			for (int dim = 0; dim < NDIM; dim++) {
				auto& x = parts.x[dim][i];
				x *= xscale;
				parts.v[dim][i] *= vscale;
				minx[dim] = std::min(minx[dim], (double) x);
			}
		}
		double box_size = 0.0;
		for (int i = 0; i < parts.x[XDIM].size(); i++) {
			for (int dim = 0; dim < NDIM; dim++) {
				auto& x = parts.x[dim][i];
				x -= minx[dim];
				box_size = std::max(box_size, (double) x);
			}
		}
		for (int i = 0; i < 6; i++) {
			header.npart[i] = header.npartTotal[i] = 0;
			header.mass[i] = 0.0;
		}
		header.BoxSize = box_size;
		header.npart[0] = 0;
		header.npart[1] = nparts;
		header.npartTotal[1] = nparts;
		header.npartTotal[2] = 0;
		header.mass[1] = opts->code_to_g * opts->hubble / constants::M0;
		header.HubbleParam = opts->hubble;
		header.redshift = 0.0;
		header.time = 1.0;
		header.flag_sfr = 0;
		header.flag_cooling = 0;
		header.flag_feedback = 0;
		header.flag_metals = 0;
		header.flag_stellarage = 0;
		header.num_files = 1;
		header.Omega0 = opts->omega_m;
		header.OmegaLambda = 1.0 - opts->omega_m;
		header.hashtabsize = 0;
		for (int i = 0; i < 84; i++) {
			header.fill[i] = 0;
		}
		int size;
		size = 256;
		fwrite(&size, sizeof(int), 1, fpout);
		fwrite(&header, sizeof(gadget2_header), 1, fpout);
		fwrite(&size, sizeof(int), 1, fpout);
		parts.write(fpout);
		fclose(fpout);
		fpout = fopen("config.cfg", "wt");
		if (!fpout) {
			printf("Unable to open config.cfg for writing\n");
			return -3;
		}
		const double h = opts->hsoft * opts->code_to_cm * opts->hubble / constants::mpc_to_cm / 2.0;
		const double link_len = opts->lc_b;/// opts->parts_dim * opts->code_to_cm * opts->hubble / constants::mpc_to_cm / box_size * pow(nparts, 1.0 / 3.0);
		fprintf(fpout, "FILE_FORMAT = \"GADGET2\"\n");
		fprintf(fpout, "h0 = %f\n", opts->hubble);
		fprintf(fpout, "Om = %f\n", opts->omega_m);
		fprintf(fpout, "Ol = %f\n", 1.0 - opts->omega_m);
		fprintf(fpout, "GADGET_LENGTH_CONVERSION = 1\n");
		fprintf(fpout, "GADGET_MASS_CONVERSION = 1\n");
		fprintf(fpout, "FORCE_RES = %e\n", h);
		//	fprintf(fpout, "LIGHTCONE = 1\n");
			fprintf(fpout, "FOF_LINKING_LENGTH = %e\n", link_len);
		//	fprintf(fpout, "STRICT_SO_MASSES = 1\n");
//		fprintf(fpout, "UNBOUND_THRESHOLD = -10000000.0\n");
//		fprintf(fpout, "AVG_PARTICLE_SPACING = %e\n", box_size / pow(nparts, 1.0 / 3.0));
		fclose(fpout);
		if (system("./rockstar -c config.cfg ./gadget.tmp\n") != 0) {
			printf("Unable to execute rockstar\n");
			return -4;
		}
		FILE* fp = fopen("halos_0.0.ascii", "rt");
		if (!fp) {
			printf("unable to open halos.txt\n");
			return -10;
		}
		vector<std::string> fields;
		std::unordered_map<std::string, vector<double>> data;
		read_ascii(fp, fields, data);
		const int sz = data["x"].size();
		for (int i = 0; i < sz; i++) {
			data["x"][i] += arc.xc[XDIM].to_double();
			data["y"][i] += arc.xc[YDIM].to_double();
			data["z"][i] += arc.xc[ZDIM].to_double();
			data["id"][i] += base_id;
		}
		data["PID"].resize(sz);
		for (int i = 0; i < sz; i++) {
			int bigj = -1;
			double mass = 0.0;
			for (int j = 0; j < sz; j++) {
				if (i != j) {
					const auto rj = data["rvir"][j];
					if (data["rvir"][i] < rj) {
						const auto x = data["x"][i] - data["x"][j];
						const auto y = data["y"][i] - data["y"][j];
						const auto z = data["z"][i] - data["z"][j];
						const auto r = 1000.0 * sqrtf(sqr(x, y, z));
						if (r < rj) {
							const double test_mass = data["mvir"][j];
							if (mass < test_mass) {
								mass = test_mass;
								bigj = j;
							}
						}
					}
				}
			}
			if (bigj != -1) {
				data["PID"][i] = data["id"][bigj];
				subhalo_cnt++;
			} else {
				halo_cnt++;
				data["PID"][i] = -1;
			}
		}
		base_id += sz;
		fclose(fp);
		for (int i = 0; i < fields.size(); i++) {
			db[fields[i]].insert(db[fields[i]].begin(), data[fields[i]].begin(), data[fields[i]].end());
		}
	}
	printf("\n%lli halos found\n", halo_cnt);
	printf("%lli subhalos found\n", subhalo_cnt);
	fclose(fpin);
	return 0;
}
