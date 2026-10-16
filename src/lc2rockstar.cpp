#include <fstream>
#include <unordered_map>

#include <cosmictiger/compress.hpp>
#include <cosmictiger/constants.hpp>
#include <cosmictiger/options.hpp>

struct Gadget2Header {
    std::uint32_t npart[6]; /*!< npart[1] gives the number of particles in the present file, other
                               particle types are ignored */
    double mass[6];         /*!< mass[1] gives the particle mass */
    double time;            /*!< time (=cosmological scale factor) of snapshot */
    double redshift;        /*!< redshift of snapshot */
    std::int32_t flagSfr; /*!< flags whether star formation is used (not available in L-Gadget2) */
    std::int32_t flagFeedback;   /*!< flags whether feedback from star formation is included */
    std::uint32_t npartTotal[6]; /*!< npart[1] gives the total number of particles in the run. If
     this number exceeds 2^32, the npartTotal[2] stores the result of a division of the particle
     number by 2^32, while npartTotal[1] holds the remainder. */
    std::int32_t flagCooling;    /*!< flags whether radiative cooling is included */
    std::int32_t numFiles;       /*!< determines the number of files that are used for a snapshot */
    double boxSize;              /*!< Simulation box size (in code units) */
    double omega0;               /*!< matter density */
    double omegaLambda;          /*!< vacuum energy density */
    double hubbleParam;          /*!< little 'h' */
    std::int32_t
        flagStellarage; /*!< flags whether the age of newly formed stars is recorded and saved */
    std::int32_t flagMetals; /*!< flags whether metal enrichment is included */
    std::int32_t
        hashtabsize; /*!< gives the size of the hashtable belonging to this snapshot file */
    char fill[84];   /*!< fills to 256 Bytes */
};

struct ParticlesSet {
    Array<Vector<float>, NDIM> x;
    Array<Vector<float>, NDIM> v;
    Vector<LcGroup> g;
    void write(FILE* fp) const;
};

void uncompressParticles(ParticlesSet& parts, CompressedParticles const& arc) {
    int total = 0;
    int id = 0;
    for (int i = 0; i < arc.blocks.size(); i++) {
        auto const& block = arc.blocks[i];
        int const nparts = block.x[XDIM].size();
        total += nparts;
        for (int j = 0; j < nparts; j++) {
            for (int dim = 0; dim < NDIM; dim++) {
                double x = (block.x[dim][j] + 0.5) / (1 << 15) - 1.0;
                double v = (block.v[dim][j] + 0.5) / (1 << 15) - 1.0;
                x *= block.xmax;
                x += block.xc[dim].toDouble();
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

void ParticlesSet::write(FILE* fp) const {
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
    size = sizeof(LcGroup) * g.size();
    fwrite(&size, sizeof(int), 1, fp);
    fwrite(g.data(), sizeof(LcGroup), g.size(), fp);
    fwrite(&size, sizeof(int), 1, fp);
}

void readAscii(
    FILE* fp, Vector<std::string>& fields, std::unordered_map<std::string, Vector<double>>& data) {
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

size_t numLines(std::string file) {
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
    char const* infile = argv[1];
    FILE* fpin = fopen(infile, "rb");
    if (!fpin) {
        printf("Unable to open %s for reading\n", infile);
        return -1;
    }
    size_t baseId = numLines(DATABASE_NAME);
    int numGroups;
    fread(&numGroups, sizeof(int), 1, fpin);
    printf("%i FoF groups found.\n", numGroups);
    Options* opts = (Options*) malloc(sizeof(Options));
    fread(opts, sizeof(Options), 1, fpin);
    std::unordered_map<std::string, Vector<double>> db;
    size_t haloCnt = 0;
    size_t subhaloCnt = 0;
    for (int i = 0; i < numGroups; i++) {
        printf("\r%i %li %li %li", i, haloCnt, subhaloCnt, haloCnt + subhaloCnt);
        fflush(stdout);
        FILE* fpout = fopen("gadget.tmp", "wb");
        if (!fpout) {
            printf("Unable to open gadget.tmp for writing\n");
            return -2;
        }
        Gadget2Header header;
        ParticlesSet parts;
        CompressedParticles arc;
        arc.read(fpin);
        uncompressParticles(parts, arc);
        int nparts = arc.size();
        double const xscale = opts->codeToCm * opts->hubble / constants::kMpcToCm;
        double const vscale = opts->codeToCm / opts->codeToS / 100000.0;
        Array<double, NDIM> minx;
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
        double boxSize = 0.0;
        for (int i = 0; i < parts.x[XDIM].size(); i++) {
            for (int dim = 0; dim < NDIM; dim++) {
                auto& x = parts.x[dim][i];
                x -= minx[dim];
                boxSize = std::max(boxSize, (double) x);
            }
        }
        for (int i = 0; i < 6; i++) {
            header.npart[i] = header.npartTotal[i] = 0;
            header.mass[i] = 0.0;
        }
        header.boxSize = boxSize;
        header.npart[0] = 0;
        header.npart[1] = nparts;
        header.npartTotal[1] = nparts;
        header.npartTotal[2] = 0;
        header.mass[1] = opts->codeToG * opts->hubble / constants::kM0;
        header.hubbleParam = opts->hubble;
        header.redshift = 0.0;
        header.time = 1.0;
        header.flagSfr = 0;
        header.flagCooling = 0;
        header.flagFeedback = 0;
        header.flagMetals = 0;
        header.flagStellarage = 0;
        header.numFiles = 1;
        header.omega0 = opts->omegaM;
        header.omegaLambda = 1.0 - opts->omegaM;
        header.hashtabsize = 0;
        for (int i = 0; i < 84; i++) {
            header.fill[i] = 0;
        }
        int size;
        size = 256;
        fwrite(&size, sizeof(int), 1, fpout);
        fwrite(&header, sizeof(Gadget2Header), 1, fpout);
        fwrite(&size, sizeof(int), 1, fpout);
        parts.write(fpout);
        fclose(fpout);
        fpout = fopen("config.cfg", "wt");
        if (!fpout) {
            printf("Unable to open config.cfg for writing\n");
            return -3;
        }
        double const h = opts->hsoft * opts->codeToCm * opts->hubble / constants::kMpcToCm / 2.0;
        double const linkLen =
            opts->lcB;    /// opts->parts_dim * opts->code_to_cm * opts->hubble /
                          /// constants::mpc_to_cm / box_size * pow(nparts, 1.0 / 3.0);
        fprintf(fpout, "FILE_FORMAT = \"GADGET2\"\n");
        fprintf(fpout, "h0 = %f\n", opts->hubble);
        fprintf(fpout, "Om = %f\n", opts->omegaM);
        fprintf(fpout, "Ol = %f\n", 1.0 - opts->omegaM);
        fprintf(fpout, "GADGET_LENGTH_CONVERSION = 1\n");
        fprintf(fpout, "GADGET_MASS_CONVERSION = 1\n");
        fprintf(fpout, "FORCE_RES = %e\n", h);
        //	fprintf(fpout, "LIGHTCONE = 1\n");
        fprintf(fpout, "FOF_LINKING_LENGTH = %e\n", linkLen);
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
        Vector<std::string> fields;
        std::unordered_map<std::string, Vector<double>> data;
        readAscii(fp, fields, data);
        int const sz = data["x"].size();
        for (int i = 0; i < sz; i++) {
            data["x"][i] += arc.xc[XDIM].toDouble();
            data["y"][i] += arc.xc[YDIM].toDouble();
            data["z"][i] += arc.xc[ZDIM].toDouble();
            data["id"][i] += baseId;
        }
        data["PID"].resize(sz);
        for (int i = 0; i < sz; i++) {
            int bigj = -1;
            double mass = 0.0;
            for (int j = 0; j < sz; j++) {
                if (i != j) {
                    auto const rj = data["rvir"][j];
                    if (data["rvir"][i] < rj) {
                        auto const x = data["x"][i] - data["x"][j];
                        auto const y = data["y"][i] - data["y"][j];
                        auto const z = data["z"][i] - data["z"][j];
                        auto const r = 1000.0 * sqrtf(sqr(x, y, z));
                        if (r < rj) {
                            double const testMass = data["mvir"][j];
                            if (mass < testMass) {
                                mass = testMass;
                                bigj = j;
                            }
                        }
                    }
                }
            }
            if (bigj != -1) {
                data["PID"][i] = data["id"][bigj];
                subhaloCnt++;
            } else {
                haloCnt++;
                data["PID"][i] = -1;
            }
        }
        baseId += sz;
        fclose(fp);
        for (int i = 0; i < fields.size(); i++) {
            db[fields[i]].insert(
                db[fields[i]].begin(), data[fields[i]].begin(), data[fields[i]].end());
        }
    }
    printf("\n%lli halos found\n", haloCnt);
    printf("%lli subhalos found\n", subhaloCnt);
    fclose(fpin);
    return 0;
}
