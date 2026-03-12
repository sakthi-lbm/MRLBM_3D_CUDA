#ifndef OUTPUTS_H
#define OUTPUTS_H

#include "constants.h"

#define PATH_FILES "OUTPUT/CYLINDER_NEW"
#ifndef ID_SIM
#define ID_SIM "000"
#endif

#define RESTART 1

constexpr int SCALE = D / U_MAX;
// constexpr int MACR_SAVE = 100 * SCALE; // interval of output file saving
constexpr int MACR_SAVE = 100; // interval of output file saving
constexpr int TSTAR = 1000;     // staionary state to start statistics
constexpr int STAT_PERIOD = 10; // period over which statistics are sampled
// constexpr int MAX_ITER = (TSTAR + STAT_PERIOD) * SCALE;
constexpr int MAX_ITER = 50000;
constexpr int STAT_END = MAX_ITER;
constexpr int STAT_START = TSTAR * SCALE;

constexpr int CHECKPOINT_SAVE = 1000;

// constexpr int MACR_SAVE = 10000;
// constexpr int MAX_ITER = 20000;
// constexpr int STAT_END = MAX_ITER;
// constexpr int STAT_START = MAX_ITER - 5000;

constexpr bool POST_PROCESS = true;

#endif // OUTPUTS_H