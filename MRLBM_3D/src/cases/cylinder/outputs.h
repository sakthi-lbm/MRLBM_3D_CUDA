#ifndef OUTPUTS_H
#define OUTPUTS_H

#include "constants.h"

#define PATH_FILES "OUTPUT/CYLINDER_NEW"
#ifndef ID_SIM
#define ID_SIM "000"
#endif

#define RESTART 0

constexpr int SCALE = D / U_MAX;
constexpr int MACR_SAVE = 10 * SCALE; // interval of output file saving
constexpr int TSTAR = 100;             // staionary state to start statistics
constexpr int STAT_PERIOD = 100;       // period over which statistics are sampled
constexpr int MAX_ITER = (TSTAR + STAT_PERIOD) * SCALE;
constexpr int STAT_END = MAX_ITER;
constexpr int STAT_START = TSTAR * SCALE;

constexpr int CHECKPOINT_SAVE = 10*MACR_SAVE; // interval to save restart file
constexpr bool POST_PROCESS = true;

#endif // OUTPUTS_H