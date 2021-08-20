#include <cosmictiger/group_entry.hpp>

#include <silo.h>

#include <boost/program_options.hpp>

class group_files {
	FILE* fp;
	int innum;
	int current_num;
	bool eof;
	void open_next() {
		const std::string filename = "./groups." + std::to_string(innum) + "/groups." + std::to_string(innum) + "." + std::to_string(current_num) + ".dat";
		if (current_num > 0) {
			fclose(fp);
		}
		fp = fopen(filename.c_str(), "rb");
		if (fp == NULL) {
			eof = true;
			if (current_num == 0) {
				printf("Unable to open %s\n", filename.c_str());
				abort();
			}
		}
	}
public:
	group_files(int innum_) {
		innum = innum_;
		current_num = 0;
		eof = false;
		open_next();
	}
	bool get_next_entry(group_entry& entry) {
		while (!entry.read(fp)) {
			current_num++;
			open_next();
			if (eof) {
				return false;
			}
		}
		return true;
	}
	~group_files() {
		if (!eof) {
			fclose(fp);
		}
	}
};

struct halo_t {
	double mass;
	double radius;
	double x;
	double y;
	double z;
	double vx;
	double vy;
	double vz;
};

int main(int argc, char* argv[]) {
	std::string outfile;
	int innum;

	namespace po = boost::program_options;
	po::options_description command_opts("options");

	command_opts.add_options()                                                                       //
	("help", "produce help message")                                                                 //
	("in", po::value<int>(&innum)->default_value(-1), "input number") //
	("out", po::value < std::string > (&outfile)->default_value(""), "output file") //
			;
	boost::program_options::variables_map vm;
	po::store(po::parse_command_line(argc, argv, command_opts), vm);
	po::notify(vm);

	if (innum == -1) {
		printf("Input number not specified. Use --in=\n");
		return -1;
	}

	if (outfile == "") {
		printf("Output file not specified. Use --out=\n");
		return -1;
	}

	group_files file(innum);
	group_entry entry;
	vector<halo_t> halos;
	array<double, 2 * NDIM> extents;
	while (file.get_next_entry(entry)) {
		halo_t halo;
		halo.mass = entry.count;
		halo.radius = std::pow(2.0, 1.0 / 3.0) * entry.r50;
		halo.x = entry.com[XDIM];
		halo.y = entry.com[YDIM];
		halo.z = entry.com[ZDIM];
		halo.vx = entry.vel[XDIM];
		halo.vy = entry.vel[YDIM];
		halo.vz = entry.vel[ZDIM];
		halos.push_back(halo);
		extents[XDIM] = std::min(extents[XDIM], halo.x - halo.radius);
		extents[YDIM] = std::min(extents[YDIM], halo.y - halo.radius);
		extents[ZDIM] = std::min(extents[ZDIM], halo.z - halo.radius);
		extents[NDIM + XDIM] = std::max(extents[NDIM + XDIM], halo.x + halo.radius);
		extents[NDIM + YDIM] = std::max(extents[NDIM + YDIM], halo.y + halo.radius);
		extents[NDIM + ZDIM] = std::max(extents[NDIM + ZDIM], halo.z + halo.radius);
	}
	auto db = DBCreate(outfile.c_str(), DB_CLOBBER, DB_LOCAL, NULL, DB_PDB);
	vector<float> coeffs((NDIM + 1) * halos.size());
	for (int i = 0; i < halos.size(); i++) {
		coeffs[(NDIM + 1) * i + 0] = halos[i].x;
		coeffs[(NDIM + 1) * i + 1] = halos[i].y;
		coeffs[(NDIM + 1) * i + 2] = halos[i].z;
		coeffs[(NDIM + 1) * i + 3] = halos[i].radius;
	}
	DBPutCsgmesh(db, "halos", NDIM, halos.size(), vector<int>(halos.size(), DBCSG_SPHERE_PR).data(), NULL, coeffs.data(), coeffs.size(), DB_FLOAT,
			extents.data(), "csg", NULL);

	DBClose(db);
	return 0;
}
