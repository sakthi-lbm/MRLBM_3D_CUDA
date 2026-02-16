#ifndef MLBM_H
#define MLBM_H

#include "solver/initializeLBM.cuh"
#include HALO_INTERFACE
#include STREAMING
#include EVAL_MOMENTS

__global__ void streaming_and_evaluate_Mom(const cylinderVar cylinder, nodeVar fMom,
                                           haloData fHalo, haloData gHalo, const int iter);
__global__ void collision_halo_update(const cylinderVar cylinder, nodeVar fMom,
                                      haloData fHalo, haloData gHalo, const int iter);

#endif // MLBM_H