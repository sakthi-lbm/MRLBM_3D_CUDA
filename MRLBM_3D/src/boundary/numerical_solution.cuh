#ifndef CUDA_HERLPERS_H
#define CUDA_HERLPERS_H

#include "extrapolation_utils.cuh"

__device__ void numerical_solution_rhoeq_rotated(real unit_nx, real unit_ny,
                                                 const uint32_t incomingMask, const uint32_t outgoingMask,
                                                 const real ux_prime, const real uy_prime, const real uz_prime,
                                                 const real mxx_prime, const real myy_prime, const real mzz_prime,
                                                 const real myz_prime, real &rhoVar, real &ux, real &uy, real &uz,
                                                 real &mxx, real &myy, real &mzz, real &mxy, real &mxz, real &myz,
                                                 const int iter);

__device__ void numerical_solution_strong_rotated(real unit_nx, real unit_ny,
                                                  const uint32_t incomingMask, const uint32_t outgoingMask,
                                                  const real ux_prime, const real uy_prime, const real uz_prime,
                                                  const real mxx_prime, const real myy_prime, const real mzz_prime,
                                                  const real myz_prime, real &rhoVar, real &ux, real &uy, real &uz,
                                                  real &mxx, real &myy, real &mzz, real &mxy, real &mxz, real &myz,
                                                  const int iter);

#endif