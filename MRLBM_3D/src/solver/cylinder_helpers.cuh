#ifndef CUDA_HERLPERS_H
#define CUDA_HERLPERS_H

#include "../all_headers.h"
#include "../globalStructs.h"

__device__ void numerical_solution_rhoeq(const cylinderVar &cylinder, const nodeType_t nodeType,
                                         real &rhoVar, real &ux, real &uy, real &mxx, real &myy, real &mxy,
                                         const nodeType_t NODE_TYPE);

__device__ void numerical_solution_strong(const cylinderVar &cylinder, const nodeType_t nodeType,
                                          real &rhoVar, real &ux, real &uy, real &mxx, real &myy, real &mxy,
                                          const int NODE_TYPE);

__device__ void numerical_solution_weak(const cylinderVar &cylinder, const nodeType_t nodeType,
                                        real &rhoVar, real &ux, real &uy, real &mxx, real &myy, real &mxy,
                                        const int NODE_TYPE);

__device__ void numerical_solution_rhoeq_rotated(const unsigned int x, const unsigned int y, const cylinderVar &cylinder,
                                                 const nodeType_t nodeType, const real ux_prime, const real uy_prime,
                                                 const real mxx_prime, const real myy_prime,
                                                 real &rhoVar, real &ux, real &uy, real &mxx, real &myy, real &mxy,
                                                 const nodeType_t NODE_TYPE);

__device__ void numerical_solution_strong_rotated(const unsigned int x, const unsigned int y, const cylinderVar &cylinder,
                                                  const nodeType_t nodeType, const real ux_prime, const real uy_prime,
                                                  const real mxx_prime, const real myy_prime,
                                                  real &rhoVar, real &ux, real &uy, real &mxx, real &myy, real &mxy,
                                                  const nodeType_t NODE_TYPE);

__device__ void numerical_solution_weak_rotated(const unsigned int x, const unsigned int y, const cylinderVar &cylinder,
                                                const nodeType_t nodeType, const real ux_prime, const real uy_prime,
                                                const real mxx_prime, const real myy_prime,
                                                real &rhoVar, real &ux, real &uy, real &mxx, real &myy, real &mxy,
                                                const nodeType_t NODE_TYPE);

__device__ inline void solve_3x3_gauss(real A[3][3], real D[3], real M[3])
{
    const real factor_21 = A[1][0] / A[0][0];
    A[1][1] -= factor_21 * A[0][1];
    A[1][2] -= factor_21 * A[0][2];
    D[1] -= factor_21 * D[0];

    // Eliminate A[2][0] (Third row, first column)
    const real factor_31 = A[2][0] / A[0][0];
    A[2][1] -= factor_31 * A[0][1];
    A[2][2] -= factor_31 * A[0][2];
    D[2] -= factor_31 * D[0];

    // 2. Pivot Row 2 to eliminate A[2][1]
    // Eliminate A[2][1] (Third row, second column)
    const real factor_32 = A[2][1] / A[1][1];
    A[2][2] -= factor_32 * A[1][2];
    D[2] -= factor_32 * D[1];

    // Back substitution
    M[2] = D[2] / A[2][2];
    M[1] = (D[1] - A[1][2] * M[2]) / A[1][1];
    M[0] = (D[0] - A[0][1] * M[1] - A[0][2] * M[2]) / A[0][0];
}

#endif