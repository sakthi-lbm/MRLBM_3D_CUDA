#ifndef EVAL_MOMENTS_CUH
#define EVAL_MOMENTS_CUH

#include "config.h"
#include LATTICE_PROPERTIES

__device__ inline void evaluate_moments(real rho, real ux, real uy, real uz,
                                        real mxx, real mxy, real mxz,
                                        real myy, real myz, real mzz,
                                        const real *pop)
{
    rho = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    const real invRho = 1 / (rho);

    ux = (pop[1] - pop[2] + pop[7] - pop[8] + pop[9] - pop[10] + pop[13] - pop[14] + pop[15] - pop[16]) * invRho;
    uy = (pop[3] - pop[4] + pop[7] - pop[8] + pop[11] - pop[12] + pop[14] - pop[13] + pop[17] - pop[18]) * invRho;
    uz = (pop[5] - pop[6] + pop[9] - pop[10] + pop[11] - pop[12] + pop[16] - pop[15] + pop[18] - pop[17]) * invRho;

    mxx = (pop[1] + pop[2] + pop[7] + pop[8] + pop[9] + pop[10] + pop[13] + pop[14] + pop[15] + pop[16]) * invRho - cs2;
    mxy = (pop[7] - pop[13] + pop[8] - pop[14]) * invRho;
    mxz = (pop[9] - pop[15] + pop[10] - pop[16]) * invRho;
    myy = (pop[3] + pop[4] + pop[7] + pop[8] + pop[11] + pop[12] + pop[13] + pop[14] + pop[17] + pop[18]) * invRho - cs2;
    myz = (pop[11] - pop[17] + pop[12] - pop[18]) * invRho;
    mzz = (pop[5] + pop[6] + pop[9] + pop[10] + pop[11] + pop[12] + pop[15] + pop[16] + pop[17] + pop[18]) * invRho - cs2;
}

#endif