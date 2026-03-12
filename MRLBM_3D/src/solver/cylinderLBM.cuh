#ifndef CYLINDERLBM_H
#define CYLINDERLBM_H

#include "../statistics/stat_header.cuh"
#include "extrapolation_utils.cuh"
#include "cylinder_helpers.cuh"

#ifdef CYLINDER
__device__ void evaluate_incoming_moments_rotated(const unsigned int x, const unsigned int y, const unsigned int z,
                                                  const nodeType_t nodeTag, const cylinderVar &cylinder,
                                                  const real *pop, real &rho,
                                                  real &mxx, real &myy, real &mzz,
                                                  real &mxy, real &mxz, real &myz);
                                                  
__device__ void cylinder_boundary_condition_rotated(const unsigned int x, const unsigned int y, const unsigned int z,
                                                    const cylinderVar &cylinder,
                                                    const nodeType_t nodeType, const nodeVar &dMom,
                                                    real &rho, real &ux, real &uy, real &uz,
                                                    real &mxx, real &myy, real &mzz,
                                                    real &mxy, real &mxz, real &myz,
                                                    const real UX_PRIME, const real UY_PRIME, const real UZ_PRIME,
                                                    const int NODE_TYPE, const real D_WALL, const int iter);

#endif

#endif // CYLINDERLBM_H
