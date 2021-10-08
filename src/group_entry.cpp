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


#include <cosmictiger/group_entry.hpp>
#include <cosmictiger/safe_io.hpp>

template<class T>
void fwrite_single(FILE* fp, const T& data) {
	fwrite(&data, sizeof(T), 1, fp);
}

template<class T>
void fread_single(FILE* fp, T& data) {
	FREAD(&data, sizeof(T), 1, fp);
}

void group_entry::write(FILE* fp) {
	fwrite_single(fp, id);
	fwrite_single(fp, com);
	fwrite_single(fp, vel);
	fwrite_single(fp, lang);
	fwrite_single(fp, mass);
	fwrite_single(fp, ekin);
	fwrite_single(fp, epot);
	fwrite_single(fp, r25);
	fwrite_single(fp, r50);
	fwrite_single(fp, r75);
	fwrite_single(fp, r90);
	fwrite_single(fp, rmax);
	fwrite_single(fp, ravg);
	fwrite_single(fp, vxdisp);
	fwrite_single(fp, vydisp);
	fwrite_single(fp, vzdisp);
	fwrite_single(fp, Ixx);
	fwrite_single(fp, Ixy);
	fwrite_single(fp, Ixz);
	fwrite_single(fp, Iyy);
	fwrite_single(fp, Iyz);
	fwrite_single(fp, Izz);
	int parent_count = parents.size();
	fwrite_single(fp, parent_count);
	for (auto i = parents.begin(); i != parents.end(); i++) {
		fwrite_single(fp, *i);
	}
}

bool group_entry::read(FILE* fp) {
	if( fread(&id, sizeof(id), 1, fp) == 0 ) {
		return false;
	}
	fread_single(fp, com);
	fread_single(fp, vel);
	fread_single(fp, lang);
	fread_single(fp, mass);
	fread_single(fp, ekin);
	fread_single(fp, epot);
	fread_single(fp, r25);
	fread_single(fp, r50);
	fread_single(fp, r75);
	fread_single(fp, r90);
	fread_single(fp, rmax);
	fread_single(fp, ravg);
	fread_single(fp, vxdisp);
	fread_single(fp, vydisp);
	fread_single(fp, vzdisp);
	fread_single(fp, Ixx);
	fread_single(fp, Ixy);
	fread_single(fp, Ixz);
	fread_single(fp, Iyy);
	fread_single(fp, Iyz);
	fread_single(fp, Izz);
	int parent_count;
	fread_single(fp, parent_count);
	for (int i = 0; i < parent_count; i++) {
		std::pair<group_int, int> entry;
		fread_single(fp, entry);
		parents[entry.first] = entry.second;
	}
	return true;
}
