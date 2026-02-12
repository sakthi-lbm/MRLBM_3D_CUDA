#ifndef CYLINDERLBM_H
#define CYLINDERLBM_H

#include "../all_headers.h"
#include "extrapolation_utils.cuh"
#include "cylinder_helpers.cuh"

#ifdef CYLINDER

__device__ void evaluate_incoming_moments(const nodeType_t nodeTag, const cylinderVar &cylinder,
                                          const real *pop, real &rhoVar, real &mxx, real &myy, real &mxy);
__device__ void evaluate_incoming_moments_rotated(const unsigned int x, const unsigned int y,
                                                  const nodeType_t nodeTag, const cylinderVar &cylinder,
                                                  const real *pop, real &rhoVar, real &mxx, real &myy, real &mxy);

__device__ void cylinder_boundary_condition(const unsigned int x, const unsigned int y, const cylinderVar &cylinder,
                                            const nodeType_t nodeType, const nodeVar &fMom,
                                            real &rhoVar, real &ux, real &uy, real &mxx, real &myy, real &mxy,
                                            const real UX_PRIME, const real UY_PRIME,
                                            const int NODE_TYPE, const real D_WALL);
__device__ void cylinder_boundary_condition_rotated(const unsigned int x, const unsigned int y, const cylinderVar &cylinder,
                                                    const nodeType_t nodeType, const nodeVar &fMom,
                                                    real &rhoVar, real &ux, real &uy, real &mxx, real &myy, real &mxy,
                                                    const real UX_PRIME, const real UY_PRIME,
                                                    const int NODE_TYPE, const real D_WALL, const int iter);

#endif

#endif // CYLINDERLBM_H
