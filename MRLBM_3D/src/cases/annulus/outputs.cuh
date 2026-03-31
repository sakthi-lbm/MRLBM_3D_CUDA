#ifndef OUTPUTS_H
#define OUTPUTS_H

#include "constants.cuh"

#define PATH_FILES "OUTPUT/ANNULUS"
#ifndef ID_SIM
#define ID_SIM "000"
#endif

constexpr bool RESTART = 1;

// constexpr int MACR_SAVE = 1;
// constexpr int MAX_ITER = 10;

constexpr int num_cycle = 5;
constexpr real lattice_gap = R_OUT - R_IN;
constexpr int MAX_ITER = num_cycle * toInt(lattice_gap * lattice_gap / VISC); // by viscous timescale
constexpr int MACR_SAVE = MAX_ITER / 50;                                     // interval of output file saving
constexpr int STAT_PERIOD = 10000;                                            // period over which statistics are sampled

constexpr int STAT_END = MAX_ITER;
constexpr int STAT_START = MAX_ITER - STAT_PERIOD;

constexpr int CHECKPOINT_SAVE = MACR_SAVE; // interval to save restart file
constexpr bool POST_PROCESS = true;

#endif // OUTPUTS_H