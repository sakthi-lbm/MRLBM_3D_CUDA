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

__global__ void apply_bc_cylinder(const int NB, const int NODE_TYPE, const cylinderVar &cylinder,
                                  nodeVar &fMom, const real UX_PRIME, const real UY_PRIME, const real UZ_PRIME,
                                  const real D_WALL, const int iter);

#endif // MLBM_H