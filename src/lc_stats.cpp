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

#include <stdio.h>
#include <math.h>
#include <string>
#include <vector>

#include <cosmictiger/lc_group_archive.hpp>

int main() {

	int rank = 0;

	std::vector<float> masses;

	const auto filename = [&rank]() {
		return std::string("lc/lc.") + std::to_string(rank) + ".dat";
	};

	FILE * fp = fopen(filename().c_str(), "rb");
	while (fp) {
		printf("Reading rank %i\n", rank);
		lc_group_archive arc;
		while (!feof(fp)) {
			if (fread(&arc, sizeof(lc_group_archive), 1, fp)) {
				masses.push_back(arc.mass);
			}
		}
		fclose(fp);
		rank++;
		fp = fopen(filename().c_str(), "rb");
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
