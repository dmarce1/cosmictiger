/* CosmicTiger - sampled gravity errors during evolution; BSL-1.0. */
#pragma once

// Root-locality calls. Begin chooses an evenly spaced sample of owned particles
// and saves an independent direct periodic reference before the production
// solve. Sample attributes follow particle sorting and migration. Finish must
// run after that solve's kick and before positions change.
void gravityErrorBegin();
void gravityErrorFinish(int coarseStep, double time, double scaleFactor);
