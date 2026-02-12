#include <iostream>
#include "cylinder_helpers.cuh"

__device__ void numerical_solution_rhoeq_rotated(const unsigned int x, const unsigned int y, const cylinderVar &cylinder,
                                                 const nodeType_t nodeType, const real ux_prime, const real uy_prime,
                                                 const real mxx_prime, const real myy_prime,
                                                 real &rhoVar, real &ux, real &uy, real &mxx, real &myy, real &mxy,
                                                 const nodeType_t NODE_TYPE)
{
    const nodeType_t nodeTag = nodeType - toNodeTypeT(NODE_TYPE);

    // Defining coordinate trasform variables:sint, cost, sin2t, cos2t
    const real xb = toReal(x);
    const real yb = toReal(y);
    const real xc = toReal(XC);
    const real yc = toReal(YC);
    const real x_diff = xb - xc;
    const real y_diff = yb - yc;

    const real radius2 = x_diff * x_diff + y_diff * y_diff;
    const real inv_radius = rsqrt(radius2);
    const real cos_theta = x_diff * inv_radius;
    const real sin_theta = y_diff * inv_radius;
    const real sin_two_theta = toReal(2.0) * sin_theta * cos_theta;
    const real cos_two_theta = cos_theta * cos_theta - sin_theta * sin_theta;

    real E_prime = toReal(0.0);
    real D_xy_prime = toReal(0.0);

    real F11_xy_prime = toReal(0.0);
    real F22_xy_prime = toReal(0.0);
    real F12_xy_prime = toReal(0.0);

    constexpr real common_base_factor = toReal(0.5) * as2 * as2;

    for (size_t q = 0; q < Q; q++)
    {
        const real cx = toReal(d_cx[q]);
        const real cy = toReal(d_cy[q]);
        const real cx_prime = cx * cos_theta + cy * sin_theta;
        const real cy_prime = -cx * sin_theta + cy * cos_theta;

        const real Hxx_prime = cx_prime * cx_prime - cs2;
        const real Hyy_prime = cy_prime * cy_prime - cs2;
        const real Hxy_prime = cx_prime * cy_prime;

        const real wq = w[q];
        const real common_factor = common_base_factor * wq;

        if (cylinder.incomings[idxBoundPop(nodeTag, q)])
        {
            const real A_i = wq * (toReal(1.0) + as2 * (ux_prime * cx_prime + uy_prime * cy_prime));
            const real E_i = A_i + common_factor * (ux_prime * ux_prime * Hxx_prime +
                                                    uy_prime * uy_prime * Hyy_prime +
                                                    toReal(2.0) * ux_prime * uy_prime * Hxy_prime);

            E_prime += E_i;
            D_xy_prime += A_i * Hxy_prime;

            F11_xy_prime += (common_factor * Hxx_prime) * Hxy_prime;
            F22_xy_prime += (common_factor * Hyy_prime) * Hxy_prime;
            F12_xy_prime += (common_factor * Hxy_prime) * Hxy_prime;
        }
    }
    const real rhoI_prime = rhoVar;
    const real mxyI_prime = mxy;

    const real Rxy = E_prime * mxyI_prime - D_xy_prime;
    const real mxy_prime = (Rxy - mxx_prime * F11_xy_prime - myy_prime * F22_xy_prime) / (toReal(2.0) * F12_xy_prime);

    ux = ux_prime * cos_theta - uy_prime * sin_theta;
    uy = ux_prime * sin_theta + uy_prime * cos_theta;
    mxx = mxx_prime * cos_theta * cos_theta + myy_prime * sin_theta * sin_theta - mxy_prime * sin_two_theta;
    myy = mxx_prime * sin_theta * sin_theta + myy_prime * cos_theta * cos_theta + mxy_prime * sin_two_theta;
    mxy = (mxx_prime - myy_prime) * toReal(0.5) * sin_two_theta + mxy_prime * cos_two_theta;

    rhoVar = rhoI_prime / E_prime;
}

__device__ void numerical_solution_strong_rotated(const unsigned int x, const unsigned int y, const cylinderVar &cylinder,
                                                  const nodeType_t nodeType, const real ux_prime, const real uy_prime,
                                                  const real mxx_prime, const real myy_prime,
                                                  real &rhoVar, real &ux, real &uy, real &mxx, real &myy, real &mxy,
                                                  const nodeType_t NODE_TYPE)
{
    const nodeType_t nodeTag = nodeType - toNodeTypeT(NODE_TYPE);

    // Defining coordinate trasform variables:sint, cost, sin2t, cos2t
    const real xb = toReal(x);
    const real yb = toReal(y);
    const real xc = toReal(XC);
    const real yc = toReal(YC);
    const real x_diff = xb - xc;
    const real y_diff = yb - yc;

    const real radius2 = x_diff * x_diff + y_diff * y_diff;
    const real inv_radius = rsqrt(radius2);
    const real cos_theta = x_diff * inv_radius;
    const real sin_theta = y_diff * inv_radius;
    const real sin_two_theta = toReal(2.0) * sin_theta * cos_theta;
    const real cos_two_theta = cos_theta * cos_theta - sin_theta * sin_theta;

    real A_prime = toReal(0.0);
    real E_prime = toReal(0.0);
    real B11_prime = toReal(0.0);
    real B22_prime = toReal(0.0);
    real B12_prime = toReal(0.0);

    real D_xy_prime = toReal(0.0);

    real F11_xy_prime = toReal(0.0);
    real F22_xy_prime = toReal(0.0);
    real F12_xy_prime = toReal(0.0);

    constexpr real common_base_factor = toReal(0.5) * as2 * as2;

    for (size_t q = 0; q < Q; q++)
    {
        const real cx = toReal(d_cx[q]);
        const real cy = toReal(d_cy[q]);
        const real cx_prime = cx * cos_theta + cy * sin_theta;
        const real cy_prime = -cx * sin_theta + cy * cos_theta;

        const real Hxx_prime = cx_prime * cx_prime - cs2;
        const real Hyy_prime = cy_prime * cy_prime - cs2;
        const real Hxy_prime = cx_prime * cy_prime;

        const real wq = w[q];
        const real common_factor = common_base_factor * wq;

        const real A_i = wq * (toReal(1.0) + as2 * (ux_prime * cx_prime + uy_prime * cy_prime));

        if (cylinder.outgoings[idxBoundPop(nodeTag, q)])
        {
            const real E_i = A_i + common_factor * (ux_prime * ux_prime * Hxx_prime +
                                                    uy_prime * uy_prime * Hyy_prime +
                                                    toReal(2.0) * ux_prime * uy_prime * Hxy_prime);
            A_prime += A_i;
            E_prime += E_i;
            B11_prime += common_factor * Hxx_prime;
            B22_prime += common_factor * Hyy_prime;
            B12_prime += common_factor * Hxy_prime;
        }

        if (cylinder.incomings[idxBoundPop(nodeTag, q)])
        {
            D_xy_prime += A_i * Hxy_prime;

            F11_xy_prime += (common_factor * Hxx_prime) * Hxy_prime;
            F22_xy_prime += (common_factor * Hyy_prime) * Hxy_prime;
            F12_xy_prime += (common_factor * Hxy_prime) * Hxy_prime;
        }
    }
    const real rhoI_prime = rhoVar;
    const real mxyI_prime = mxy;

    const real omega_term = toReal(1.0) - OMEGA;
    const real G_prime = omega_term * A_prime + OMEGA * E_prime;

    const real L11_xy_prime = (omega_term * B11_prime * mxyI_prime) - F11_xy_prime;
    const real L22_xy_prime = (omega_term * B22_prime * mxyI_prime) - F22_xy_prime;
    const real L12_xy_prime = (omega_term * B12_prime * mxyI_prime) - F12_xy_prime;

    const real Rxy = D_xy_prime - G_prime * mxyI_prime;
    const real mxy_prime = (Rxy - mxx_prime * L11_xy_prime - myy_prime * L22_xy_prime) / (toReal(2.0) * L12_xy_prime);

    real denominator = G_prime + omega_term * (mxx_prime * B11_prime + myy_prime * B22_prime +
                                               toReal(2.0) * mxy_prime * B12_prime);
    real inv_denominator = toReal(1.0) / denominator;
    rhoVar = rhoI_prime * inv_denominator;

    ux = ux_prime * cos_theta - uy_prime * sin_theta;
    uy = ux_prime * sin_theta + uy_prime * cos_theta;
    mxx = mxx_prime * cos_theta * cos_theta + myy_prime * sin_theta * sin_theta - mxy_prime * sin_two_theta;
    myy = mxx_prime * sin_theta * sin_theta + myy_prime * cos_theta * cos_theta + mxy_prime * sin_two_theta;
    mxy = (mxx_prime - myy_prime) * toReal(0.5) * sin_two_theta + mxy_prime * cos_two_theta;
}

__device__ void numerical_solution_weak_rotated(const unsigned int x, const unsigned int y, const cylinderVar &cylinder,
                                                const nodeType_t nodeType, const real ux_prime, const real uy_prime,
                                                const real mxx_prime, const real myy_prime,
                                                real &rhoVar, real &ux, real &uy, real &mxx, real &myy, real &mxy,
                                                const nodeType_t NODE_TYPE)
{
    const nodeType_t nodeTag = nodeType - toNodeTypeT(NODE_TYPE);

    // Defining coordinate trasform variables:sint, cost, sin2t, cos2t
    const real xb = toReal(x);
    const real yb = toReal(y);
    const real xc = toReal(XC);
    const real yc = toReal(YC);
    const real x_diff = xb - xc;
    const real y_diff = yb - yc;

    const real radius2 = x_diff * x_diff + y_diff * y_diff;
    const real inv_radius = rsqrt(radius2);
    const real cos_theta = x_diff * inv_radius;
    const real sin_theta = y_diff * inv_radius;
    const real sin_two_theta = toReal(2.0) * sin_theta * cos_theta;
    const real cos_two_theta = cos_theta * cos_theta - sin_theta * sin_theta;

    real A_prime = toReal(0.0);
    real E_prime = toReal(0.0);
    real B11_prime = toReal(0.0);
    real B22_prime = toReal(0.0);
    real B12_prime = toReal(0.0);

    real D_xy_prime = toReal(0.0);

    real F11_xy_prime = toReal(0.0);
    real F22_xy_prime = toReal(0.0);
    real F12_xy_prime = toReal(0.0);

    constexpr real common_base_factor = toReal(0.5) * as2 * as2;

    for (size_t q = 0; q < Q; q++)
    {
        const real cx = toReal(d_cx[q]);
        const real cy = toReal(d_cy[q]);
        const real cx_prime = cx * cos_theta + cy * sin_theta;
        const real cy_prime = -cx * sin_theta + cy * cos_theta;

        const real Hxx_prime = cx_prime * cx_prime - cs2;
        const real Hyy_prime = cy_prime * cy_prime - cs2;
        const real Hxy_prime = cx_prime * cy_prime;

        const real wq = w[q];
        const real common_factor = common_base_factor * wq;

        if (cylinder.incomings[idxBoundPop(nodeTag, q)])
        {
            const real A_i = wq * (toReal(1.0) + as2 * (ux_prime * cx_prime + uy_prime * cy_prime));
            const real E_i = A_i + common_factor * (ux_prime * ux_prime * Hxx_prime +
                                                    uy_prime * uy_prime * Hyy_prime +
                                                    toReal(2.0) * ux_prime * uy_prime * Hxy_prime);
            A_prime += A_i;
            E_prime += E_i;

            B11_prime += common_factor * Hxx_prime;
            B22_prime += common_factor * Hyy_prime;
            B12_prime += common_factor * Hxy_prime;
            D_xy_prime += A_i * Hxy_prime;

            F11_xy_prime += (common_factor * Hxx_prime) * Hxy_prime;
            F22_xy_prime += (common_factor * Hyy_prime) * Hxy_prime;
            F12_xy_prime += (common_factor * Hxy_prime) * Hxy_prime;
        }
    }
    const real rhoI_prime = rhoVar;
    const real mxyI_prime = mxy;

    const real L11_xy_prime = (B11_prime * mxyI_prime) - F11_xy_prime;
    const real L22_xy_prime = (B22_prime * mxyI_prime) - F22_xy_prime;
    const real L12_xy_prime = (B12_prime * mxyI_prime) - F12_xy_prime;

    const real Rxy = D_xy_prime - A_prime * mxyI_prime;
    const real mxy_prime = (Rxy - mxx_prime * L11_xy_prime - myy_prime * L22_xy_prime) / (toReal(2.0) * L12_xy_prime);

    real denominator = A_prime + (mxx_prime * B11_prime +
                                  myy_prime * B22_prime +
                                  toReal(2.0) * mxy_prime * B12_prime);
    real inv_denominator = toReal(1.0) / denominator;
    rhoVar = rhoI_prime * inv_denominator;

    ux = ux_prime * cos_theta - uy_prime * sin_theta;
    uy = ux_prime * sin_theta + uy_prime * cos_theta;  
    mxx = mxx_prime * cos_theta * cos_theta + myy_prime * sin_theta * sin_theta - mxy_prime * sin_two_theta;
    myy = mxx_prime * sin_theta * sin_theta + myy_prime * cos_theta * cos_theta + mxy_prime * sin_two_theta;
    mxy = (mxx_prime - myy_prime) * toReal(0.5) * sin_two_theta + mxy_prime * cos_two_theta;
}

__device__ void numerical_solution_strong(const cylinderVar &cylinder, const nodeType_t nodeType,
                                          real &rhoVar, real &ux, real &uy, real &mxx, real &myy, real &mxy,
                                          const int NODE_TYPE)
{
    const nodeType_t nodeTag = nodeType - toNodeTypeT(NODE_TYPE);

    real A_prime = toReal(0.0);
    real E_prime = toReal(0.0);
    real B11_prime = toReal(0.0);
    real B22_prime = toReal(0.0);
    real B12_prime = toReal(0.0);

    real D_xx_prime = toReal(0.0);
    real D_yy_prime = toReal(0.0);
    real D_xy_prime = toReal(0.0);

    real F11_xx_prime = toReal(0.0);
    real F22_xx_prime = toReal(0.0);
    real F12_xx_prime = toReal(0.0);

    real F11_yy_prime = toReal(0.0);
    real F22_yy_prime = toReal(0.0);
    real F12_yy_prime = toReal(0.0);

    real F11_xy_prime = toReal(0.0);
    real F22_xy_prime = toReal(0.0);
    real F12_xy_prime = toReal(0.0);

    constexpr real common_base_factor = toReal(0.5) * as2 * as2;

    for (size_t q = 0; q < Q; q++)
    {
        const real Hxx = d_Hxx[q];
        const real Hyy = d_Hyy[q];
        const real Hxy = d_Hxy[q];

        const real wq = w[q];
        const real common_factor = common_base_factor * wq;

        const real A_i = wq * (toReal(1.0) + as2 * (ux * d_cx[q] + uy * d_cy[q]));

        if (cylinder.outgoings[idxBoundPop(nodeTag, q)])
        {
            const real E_i = A_i + common_factor * (ux * ux * Hxx + uy * uy * Hyy + toReal(2.0) * ux * uy * Hxy);
            A_prime += A_i;
            E_prime += E_i;
            B11_prime += common_factor * Hxx;
            B22_prime += common_factor * Hyy;
            B12_prime += common_factor * Hxy;
        }

        if (cylinder.incomings[idxBoundPop(nodeTag, q)])
        {
            D_xx_prime += A_i * Hxx;
            D_yy_prime += A_i * Hyy;
            D_xy_prime += A_i * Hxy;

            F11_xx_prime += (common_factor * Hxx) * Hxx;
            F22_xx_prime += (common_factor * Hyy) * Hxx;
            F12_xx_prime += (common_factor * Hxy) * Hxx;

            F11_yy_prime += (common_factor * Hxx) * Hyy;
            F22_yy_prime += (common_factor * Hyy) * Hyy;
            F12_yy_prime += (common_factor * Hxy) * Hyy;

            F11_xy_prime += (common_factor * Hxx) * Hxy;
            F22_xy_prime += (common_factor * Hyy) * Hxy;
            F12_xy_prime += (common_factor * Hxy) * Hxy;
        }
    }
    const real rhoI = rhoVar;
    const real mxxI = mxx;
    const real myyI = myy;
    const real mxyI = mxy;

    const real omega_term = toReal(1.0) - OMEGA;
    const real G_prime = omega_term * A_prime + OMEGA * E_prime;

    real Matrix_A[3][3];
    real Vector_D[3];

    Matrix_A[0][0] = (omega_term * B11_prime * mxxI) - F11_xx_prime; // a1
    Matrix_A[0][1] = (omega_term * B22_prime * mxxI) - F22_xx_prime; // b1
    Matrix_A[0][2] = toReal(2.0) * ((omega_term * B12_prime * mxxI) - F12_xx_prime);
    Vector_D[0] = -(D_xx_prime - G_prime * mxxI);

    // Row 2 (Corresponds to myy)
    Matrix_A[1][0] = (omega_term * B11_prime * myyI) - F11_yy_prime;                 // a2
    Matrix_A[1][1] = (omega_term * B22_prime * myyI) - F22_yy_prime;                 // b2
    Matrix_A[1][2] = toReal(2.0) * ((omega_term * B12_prime * myyI) - F12_yy_prime); // c2
    Vector_D[1] = -(D_yy_prime - G_prime * myyI);                                    // -d2

    // Row 3 (Corresponds to mxy)
    Matrix_A[2][0] = (omega_term * B11_prime * mxyI) - F11_xy_prime;                 // a3
    Matrix_A[2][1] = (omega_term * B22_prime * mxyI) - F22_xy_prime;                 // b3
    Matrix_A[2][2] = toReal(2.0) * ((omega_term * B12_prime * mxyI) - F12_xy_prime); // c3
    Vector_D[2] = -(D_xy_prime - G_prime * mxyI);                                    // -d3

    real Moments_M[3]; // Output vector {mxx, myy, mxy}
    solve_3x3_gauss(Matrix_A, Vector_D, Moments_M);

    mxx = Moments_M[0];
    myy = Moments_M[1];
    mxy = Moments_M[2];

    real denominator = omega_term * (A_prime + mxx * B11_prime + myy * B22_prime + toReal(2.0) * mxy * B12_prime) + OMEGA * E_prime;
    real inv_denominator = toReal(1.0) / denominator;
    rhoVar = rhoI * inv_denominator;
}

__device__ void numerical_solution_rhoeq(const cylinderVar &cylinder, const nodeType_t nodeType,
                                         real &rhoVar, real &ux, real &uy, real &mxx, real &myy, real &mxy,
                                         const nodeType_t NODE_TYPE)
{
    const nodeType_t nodeTag = nodeType - toNodeTypeT(NODE_TYPE);

    real E_prime = toReal(0.0);
    real D_xx_prime = toReal(0.0);
    real D_yy_prime = toReal(0.0);
    real D_xy_prime = toReal(0.0);

    real F11_xx_prime = toReal(0.0);
    real F22_xx_prime = toReal(0.0);
    real F12_xx_prime = toReal(0.0);

    real F11_yy_prime = toReal(0.0);
    real F22_yy_prime = toReal(0.0);
    real F12_yy_prime = toReal(0.0);

    real F11_xy_prime = toReal(0.0);
    real F22_xy_prime = toReal(0.0);
    real F12_xy_prime = toReal(0.0);

    constexpr real common_base_factor = toReal(0.5) * as2 * as2;

    for (size_t q = 0; q < Q; q++)
    {
        const real cx = d_cx[q];
        const real cy = d_cy[q];
        const real Hxx = d_Hxx[q];
        const real Hyy = d_Hyy[q];
        const real Hxy = d_Hxy[q];

        const real wq = w[q];
        const real common_factor = common_base_factor * wq;

        if (cylinder.incomings[idxBoundPop(nodeTag, q)])
        {
            const real A_i = wq * (toReal(1.0) + as2 * (ux * cx + uy * cy));
            const real E_i = A_i + common_factor * (ux * ux * Hxx + uy * uy * Hyy + toReal(2.0) * ux * uy * Hxy);

            E_prime += E_i;
            D_xx_prime += A_i * Hxx;
            D_yy_prime += A_i * Hyy;
            D_xy_prime += A_i * Hxy;

            F11_xx_prime += (common_factor * Hxx) * Hxx;
            F22_xx_prime += (common_factor * Hyy) * Hxx;
            F12_xx_prime += (common_factor * Hxy) * Hxx;

            F11_yy_prime += (common_factor * Hxx) * Hyy;
            F22_yy_prime += (common_factor * Hyy) * Hyy;
            F12_yy_prime += (common_factor * Hxy) * Hyy;

            F11_xy_prime += (common_factor * Hxx) * Hxy;
            F22_xy_prime += (common_factor * Hyy) * Hxy;
            F12_xy_prime += (common_factor * Hxy) * Hxy;
        }
    }
    const real rhoI = rhoVar;
    const real mxxI = mxx;
    const real myyI = myy;
    const real mxyI = mxy;

    real Matrix_A[3][3];
    real Vector_D[3];

    Matrix_A[0][0] = F11_xx_prime;                // a1
    Matrix_A[0][1] = F22_xx_prime;                // b1
    Matrix_A[0][2] = toReal(2.0) * F12_xx_prime;  // c1
    Vector_D[0] = -(E_prime * mxxI - D_xx_prime); //-d1

    // Row 2 (Corresponds to myy)
    Matrix_A[1][0] = F11_yy_prime;                // a2
    Matrix_A[1][1] = F22_yy_prime;                // b2
    Matrix_A[1][2] = toReal(2.0) * F12_yy_prime;  // c2
    Vector_D[1] = -(E_prime * myyI - D_yy_prime); // -d2

    // Row 3 (Corresponds to mxy)
    Matrix_A[2][0] = F11_xy_prime;                // a3
    Matrix_A[2][1] = F22_xy_prime;                // b3
    Matrix_A[2][2] = toReal(2.0) * F12_xy_prime;  // c3
    Vector_D[2] = -(E_prime * mxyI - D_xy_prime); // -d3

    real Moments_M[3]; // Output vector {mxx, myy, mxy}
    solve_3x3_gauss(Matrix_A, Vector_D, Moments_M);
    // solve_3x3_robust(Matrix_A, Vector_D, Moments_M);

    mxx = Moments_M[0];
    myy = Moments_M[1];
    mxy = Moments_M[2];

    rhoVar = rhoI / E_prime;
}
