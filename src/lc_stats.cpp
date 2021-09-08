#include <stdio.h>
#include <math.h>
#include <string>
#include <vector>

#include <cosmictiger/lc_group_archive.hpp>

int main() {

	int rank = 0;
	int dump_num = 0;

	std::vector<float> masses;

	const auto filename = [&rank,&dump_num]() {
		return std::string("lc_groups/lc_groups.") + std::to_string(rank) + "." + std::to_string(dump_num) + ".dat";
	};

	FILE * fp = fopen(filename().c_str(), "rb");
	while (fp) {
		printf("Reading rank %i dump # %i\n", rank, dump_num);
		lc_group_archive arc;
		while (!feof(fp)) {
			if (fread(&arc, sizeof(lc_group_archive), 1, fp)) {
				masses.push_back(arc.mass);
			}
		}
		dump_num++;
		fclose(fp);
		fp = fopen(filename().c_str(), "rb");
		if (fp == NULL) {
			dump_num = 0;
			rank++;
			fp = fopen(filename().c_str(), "rb");
		}
	}

	float mass_min = 1.0e20;
	float mass_max = 0.0;
	for (int i = 0; i < masses.size(); i++) {
		mass_min = std::min(mass_min, masses[i]);
		mass_max = std::max(mass_max, masses[i]);
	}
	printf("minimum halo mass = %e\n", mass_min);
	printf("maximum halo mass = %e\n", mass_max);
	const int nbins = masses.size() / 1000;
	printf("using %i bins\n", nbins);
	const float logmmax = log10(mass_max);
	const float logmmin = log10(mass_min);
	const float dlogm = (logmmax - logmmin) / nbins;
	std::vector<float> bins(nbins, 0.0f);
	for (int i = 0; i < masses.size(); i++) {
		const int j = (log10(masses[i]) - logmmin) / dlogm;
		bins[j]++;
	}
	fp = fopen("halo_mass.txt", "wt");
	for (int i = 0; i < nbins; i++) {
		float logm = logmmin + dlogm * (i + 0.5);
		fprintf(fp, "%e %e\n", logm, bins[i]);
	}
	fclose(fp);
}
