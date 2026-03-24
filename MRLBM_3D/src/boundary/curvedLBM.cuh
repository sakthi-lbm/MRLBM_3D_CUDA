#ifndef CYLINDERLBM_H
#define CYLINDERLBM_H

#include "numerical_solution.cuh"

__device__ void evaluate_incoming_moments_rotated(const real unit_nx, const real unit_ny,
                                                  const uint32_t incomingMask,
                                                  const real *pop, real &rho,
                                                  real &mxx, real &myy, real &mzz,
                                                  real &mxy, real &mxz, real &myz);

__device__ void curved_boundary_condition_rotated(const real unit_nx, const real unit_ny, const real delta,
                                                  const uint32_t incomingMask, const uint32_t outgoingMask,
                                                  unsigned int xb, unsigned int yb, unsigned int zb,
                                                  const real xw, const real yw, const real zw,
                                                  const nodeVar &dMom,
                                                  real &rho, real &ux, real &uy, real &uz,
                                                  real &mxx, real &myy, real &mzz,
                                                  real &mxy, real &mxz, real &myz,
                                                  const real UX_PRIME, const real UY_PRIME, const real UZ_PRIME,
                                                  const int iter);

#endif // CYLINDERLBM_H
