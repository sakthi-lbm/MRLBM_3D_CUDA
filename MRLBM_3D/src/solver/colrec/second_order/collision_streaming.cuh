#ifndef COLLISION_STREAMING_H
#define COLLISION_STREAMING_H

#include "../../../config.h"
#include "../../../globalStructs.h"
#include "../../../halo_interface/halo_interface.cuh"
#include "../../cylinderLBM.cuh"
#include "../../../postprocess.cuh"

__device__ void mom_collision(real ux, real uy, real &mxx, real &myy, real &mxy);
__global__ void streaming_and_evaluate_Mom(const cylinderVar cylinder, nodeVar fMom,
                                           haloData fHalo, haloData gHalo, const int iter);
__global__ void apply_bc_cylinder(const int NB, const int NODE_TYPE, const cylinderVar &cylinder, nodeVar &fMom,
                                  const real UX_PRIME, const real UY_PRIME, const real D_WALL, const int iter);
__global__ void collision_halo_update(const cylinderVar cylinder_inner, nodeVar fMom, haloData fHalo,
                                      haloData gHalo, const int iter);

#endif