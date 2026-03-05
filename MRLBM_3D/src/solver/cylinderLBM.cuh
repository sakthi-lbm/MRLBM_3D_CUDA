#ifndef CYLINDERLBM_H
#define CYLINDERLBM_H

#include "initializeLBM_inline.cuh"
#include "extrapolation_utils.cuh"
#include "cylinder_helpers.cuh"

#ifdef CYLINDER
__device__ void evaluate_incoming_moments_rotated(const unsigned int x, const unsigned int y, const unsigned int z,
                                                  const nodeType_t nodeTag, const cylinderVar &cylinder,
                                                  const real *pop, real &rhoVar,
                                                  real &mxx, real &myy, real &mzz,
                                                  real &mxy, real &mxz, real &myz);

#endif

#endif // CYLINDERLBM_H
