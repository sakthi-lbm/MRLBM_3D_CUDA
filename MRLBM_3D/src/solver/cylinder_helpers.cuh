#ifndef CUDA_HERLPERS_H
#define CUDA_HERLPERS_H

#include "../all_headers.h"
#include "../globalStructs.h"

__device__ void numerical_solution_strong_rotated(const unsigned int x, const unsigned int y, const cylinderVar &cylinder,
                                                  const nodeType_t nodeType, const real ux_prime, const real uy_prime,
                                                  const real mxx_prime, const real myy_prime,
                                                  real &rhoVar, real &ux, real &uy, real &mxx, real &myy, real &mxy,
                                                  const nodeType_t NODE_TYPE);

#endif