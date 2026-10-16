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

template <class T>
void fwriteSingle(FILE* fp, T const& data) {
    fwrite(&data, sizeof(T), 1, fp);
}

template <class T>
void freadSingle(FILE* fp, T& data) {
    FREAD(&data, sizeof(T), 1, fp);
}

void GroupEntry::write(FILE* fp) {
    fwriteSingle(fp, id);
    fwriteSingle(fp, com);
    fwriteSingle(fp, vel);
    fwriteSingle(fp, lang);
    fwriteSingle(fp, mass);
    fwriteSingle(fp, ekin);
    fwriteSingle(fp, epot);
    fwriteSingle(fp, r25);
    fwriteSingle(fp, r50);
    fwriteSingle(fp, r75);
    fwriteSingle(fp, r90);
    fwriteSingle(fp, rmax);
    fwriteSingle(fp, ravg);
    fwriteSingle(fp, vxdisp);
    fwriteSingle(fp, vydisp);
    fwriteSingle(fp, vzdisp);
    fwriteSingle(fp, ixx);
    fwriteSingle(fp, ixy);
    fwriteSingle(fp, ixz);
    fwriteSingle(fp, iyy);
    fwriteSingle(fp, iyz);
    fwriteSingle(fp, izz);
    int parentCount = parents.size();
    fwriteSingle(fp, parentCount);
    for (auto i = parents.begin(); i != parents.end(); i++) {
        fwriteSingle(fp, *i);
    }
}

bool GroupEntry::read(FILE* fp) {
    if (fread(&id, sizeof(id), 1, fp) == 0) {
        return false;
    }
    freadSingle(fp, com);
    freadSingle(fp, vel);
    freadSingle(fp, lang);
    freadSingle(fp, mass);
    freadSingle(fp, ekin);
    freadSingle(fp, epot);
    freadSingle(fp, r25);
    freadSingle(fp, r50);
    freadSingle(fp, r75);
    freadSingle(fp, r90);
    freadSingle(fp, rmax);
    freadSingle(fp, ravg);
    freadSingle(fp, vxdisp);
    freadSingle(fp, vydisp);
    freadSingle(fp, vzdisp);
    freadSingle(fp, ixx);
    freadSingle(fp, ixy);
    freadSingle(fp, ixz);
    freadSingle(fp, iyy);
    freadSingle(fp, iyz);
    freadSingle(fp, izz);
    int parentCount;
    freadSingle(fp, parentCount);
    for (int i = 0; i < parentCount; i++) {
        std::pair<GroupInteger, int> entry;
        freadSingle(fp, entry);
        parents[entry.first] = entry.second;
    }
    return true;
}
