/*
CosmicTiger - A cosmological N-Body code
Copyright (C) 2021  Dominic C. Marcello

Distributed under the Boost Software License, Version 1.0.
See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
*/

#include <cosmictiger/analysis/group_entry.hpp>
#include <cosmictiger/core/safe_io.hpp>

/// Write one binary value to the group catalogue stream.
template <class T>
void fwriteSingle(FILE* fp, T const& data) {
    fwrite(&data, sizeof(T), 1, fp);
}

/// Read one binary value from the group catalogue stream with short-read checking.
template <class T>
void freadSingle(FILE* fp, T& data) {
    FREAD(&data, sizeof(T), 1, fp);
}

/// Write group properties and parent membership counts in the catalogue's fixed field
/// order.
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
    fwriteSingle(fp, mGas);
    fwriteSingle(fp, mStar);
    int parentCount = parents.size();
    fwriteSingle(fp, parentCount);
    for (auto i = parents.begin(); i != parents.end(); i++) {
        fwriteSingle(fp, *i);
    }
}

/// Read one group record and its parent map, returning false when no next record ID is
/// available.
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
    freadSingle(fp, mGas);
    freadSingle(fp, mStar);
    int parentCount;
    freadSingle(fp, parentCount);
    for (int i = 0; i < parentCount; i++) {
        std::pair<GroupInteger, int> entry;
        freadSingle(fp, entry);
        parents[entry.first] = entry.second;
    }
    return true;
}
