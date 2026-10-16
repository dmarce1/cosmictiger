/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2021  Dominic C. Marcello

 This program is free software; you can redistribute it and/or
 modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation; either version 2
 of the License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */

#include <boost/program_options.hpp>

#include <cosmictiger/group_entry.hpp>

class GroupFiles {
    FILE* fp_;
    int innum_;
    int currentNum_;
    bool eof_;
    void openNext() {
        std::string const filename = "./groups." + std::to_string(innum_) + "/groups." +
            std::to_string(innum_) + "." + std::to_string(currentNum_) + ".dat";
        if (currentNum_ > 0) {
            fclose(fp_);
        }
        fp_ = fopen(filename.c_str(), "rb");
        if (fp_ == NULL) {
            eof_ = true;
            if (currentNum_ == 0) {
                printf("Unable to open %s\n", filename.c_str());
                abort();
            }
        }
    }

public:
    GroupFiles(int innumArg) {
        innum_ = innumArg;
        currentNum_ = 0;
        eof_ = false;
        openNext();
    }
    bool getNextEntry(GroupEntry& entry) {
        while (!entry.read(fp_)) {
            currentNum_++;
            openNext();
            if (eof_) {
                return false;
            }
        }
        return true;
    }
    ~GroupFiles() {
        if (!eof_) {
            fclose(fp_);
        }
    }
};

struct HaloT {
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
    int nbins;
    float boxSize;

    namespace po = boost::program_options;
    po::options_description commandOpts("options");

    commandOpts.add_options()                                                                //
        ("help", "produce help message")                                                     //
        ("in", po::value<int>(&innum)->default_value(-1), "input number")                    //
        ("box_size", po::value<float>(&boxSize)->default_value(684.0), "box size in Mpc")    //
        ("nbins", po::value<int>(&nbins)->default_value(1000), "number of log bins")         //
        ;
    boost::program_options::variables_map vm;
    po::store(po::parse_command_line(argc, argv, commandOpts), vm);
    po::notify(vm);

    if (innum == -1) {
        printf("Input number not specified. Use --in=\n");
        return -1;
    }

    GroupFiles file(innum);
    GroupEntry entry;
    Vector<float> masses;
    float massMin = std::numeric_limits<float>::max();
    float massMax = 0.0f;
    while (file.getNextEntry(entry)) {
        HaloT halo;
        masses.push_back(entry.mass);
        massMin = std::min(massMin, entry.mass);
        if (entry.mass > massMax) {
            massMax = entry.mass;
            PRINT("%i %i\n", (int) (massMin / 1.481962e+09 + 0.5),
                (int) (massMax / 1.481962e+09 + 0.5));
        }
    }
    float const logmax = log10(massMax);
    float const logmin = log10(massMin);
    float const dlog = (logmax - logmin) / nbins;
    std::vector<int> bins(nbins, 0);
    for (int i = 0; i < masses.size(); i++) {
        int bin = std::min((int) ((log10(masses[i]) - logmin) / dlog), nbins - 1);
        bins[bin]++;
    }
    for (int i = 0; i < nbins; i++) {
        float logmass = (i + 0.5) * dlog + logmin;
        printf("%e %e\n", logmass, bins[i] / dlog / boxSize / boxSize / boxSize);
    }

    return 0;
}
