#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <silo.h>
int main(int argc, char** argv) {
    if (argc != 2) { return 1; }
    DBfile* db = DBOpen(argv[1], DB_UNKNOWN, DB_READ);
    if (!db) { return 2; }
    std::puts("field,count,min,max,rms");
    for (char const* name : {"vx", "vy", "vz"}) {
        DBpointvar* field = DBGetPointvar(db, name);
        if (!field || field->nels != 512 || field->nvals != 1) { return 3; }
        double minimum = 1e300, maximum = -1e300, squares = 0.0;
        for (int i = 0; i < field->nels; ++i) {
            double const value = field->datatype == DB_FLOAT ? static_cast<float const*>(field->vals[0])[i] : static_cast<double const*>(field->vals[0])[i];
            if (!std::isfinite(value)) { return 4; }
            minimum = std::fmin(value, minimum);
            maximum = std::fmax(value, maximum);
            squares += value * value;
        }
        if (!(squares > 0.0) || !(minimum < 0.0) || !(maximum > 0.0)) { return 5; }
        std::printf("%s,%d,%.12g,%.12g,%.12g\n", name, field->nels, minimum, maximum, std::sqrt(squares / field->nels));
        DBFreePointvar(field);
    }
    DBClose(db);
}
