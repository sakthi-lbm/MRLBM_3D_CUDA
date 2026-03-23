#ifndef OUTPUTS_H
#define OUTPUTS_H

#include "constants.cuh"

#define PATH_FILES "OUTPUT/CYLINDER_NEW"
#ifndef ID_SIM
#define ID_SIM "000"
#endif

constexpr bool RESTART = 0;

constexpr int SCALE = D / U_MAX;
constexpr int MACR_SAVE = 5000;  // interval of output file saving
constexpr int TSTAR = 300;       // staionary state to start statistics
constexpr int STAT_PERIOD = 50; // period over which statistics are sampled
constexpr int MAX_ITER = (TSTAR + STAT_PERIOD) * SCALE;
// constexpr int MAX_ITER = 2000;
constexpr int STAT_END = MAX_ITER;
constexpr int STAT_START = TSTAR * SCALE;

constexpr int CHECKPOINT_SAVE = 10000; // interval to save restart file
constexpr bool POST_PROCESS = true;

#endif // OUTPUTS_H