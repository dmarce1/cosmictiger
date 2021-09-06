#pragma once


using lc_real = double;

void lc_init(double);
int lc_add_particle(lc_real x0, lc_real y0, lc_real z0, lc_real x1, lc_real y1, lc_real z1, float vx, float vy, float vz, float t, float dt);
void lc_buffer2homes();
size_t lc_parts_waiting();
void lc_particle_boundaries();
void lc_form_trees(double tmax, double link_len);
size_t lc_find_groups();
void lc_groups2homes();
void lc_parts2groups();
