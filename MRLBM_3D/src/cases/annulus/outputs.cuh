#ifndef OUTPUTS_H
#define OUTPUTS_H

#include "constants.cuh"

#define PATH_FILES "OUTPUT/ANNULUS"
#ifndef ID_SIM
#define ID_SIM "000"
#endif

constexpr bool RESTART = 0;

constexpr int MACR_SAVE = 1000;
constexpr int MAX_ITER = 10000;

constexpr int STAT_START = MAX_ITER - 1000;
constexpr int STAT_END = MAX_ITER;

// constexpr int SCALE = D / U_MAX;
// constexpr int MACR_SAVE = 5000;  // interval of output file saving
// constexpr int TSTAR = 300;       // staionary state to start statistics
// constexpr int STAT_PERIOD = 50; // period over which statistics are sampled
// constexpr int MAX_ITER = (TSTAR + STAT_PERIOD) * SCALE;
// constexpr int STAT_START = TSTAR * SCALE;
// constexpr int STAT_END = MAX_ITER;

constexpr int CHECKPOINT_SAVE = 10000; // interval to save restart file
constexpr bool POST_PROCESS = true;

#endif // OUTPUTS_H