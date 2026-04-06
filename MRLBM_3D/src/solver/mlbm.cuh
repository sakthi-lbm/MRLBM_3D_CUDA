#ifndef MLBM_H
#define MLBM_H

#include "solver/initializeLBM.cuh"

__global__ void streaming_and_evaluate_Mom(void *caseData, nodeVar dMom, haloData fHalo, haloData gHalo,
                                           const int *d_active_blocks, int iter);

__device__ void evaluate_bounday_moments(void *caseData, int x, int y, int z, nodeType_t nodeType_packed,
                                         nodeVar dMom, real *pop, real *s_pop, real &rho, real &ux, real &uy, real &uz,
                                         real &mxx, real &myy, real &mzz, real &mxy, real &mxz, real &myz);

__global__ void collision_halo_update(nodeVar dMom, haloData fHalo, haloData gHalo,
                                      const int *d_active_blocks, const int iter);

__global__ void outlet_avg_ux(const real *__restrict__ ux);

void compute_convective_outlet_velocity(const real *d_ux);

#endif // MLBM_H