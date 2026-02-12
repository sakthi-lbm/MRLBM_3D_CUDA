#ifndef EXTRAPOLATION_UTILS_H
#define EXTRAPOLATION_UTILS_H

#include "../all_headers.h"

__device__ __forceinline__ real bilinear_interpolation(const real x, const real y,
                                                       const real *__restrict__ variable_array)
{

    const int x0 = floor(x);
    const int y0 = floor(y);

    const int x1 = x0 + 1;
    const int y1 = y0 + 1;

    // The dimensions and indices remain the same
    const real xd = x - toReal(x0);
    const real yd = y - toReal(y0);

    // Calculate block and thread indices for the four corners
    const size_t tx0 = x0 % BLOCK_THREAD_X;
    const size_t tx1 = x1 % BLOCK_THREAD_X;
    const size_t ty0 = y0 % BLOCK_THREAD_Y;
    const size_t ty1 = y1 % BLOCK_THREAD_Y;

    const size_t bx0 = x0 / BLOCK_THREAD_X;
    const size_t bx1 = x1 / BLOCK_THREAD_X;
    const size_t by0 = y0 / BLOCK_THREAD_Y;
    const size_t by1 = y1 / BLOCK_THREAD_Y;

    // Retrieve the values from the passed array at the four corners
    const real q00 = variable_array[IDX_BLOCK(tx0, ty0, bx0, by0)];
    const real q10 = variable_array[IDX_BLOCK(tx1, ty0, bx1, by0)];
    const real q01 = variable_array[IDX_BLOCK(tx0, ty1, bx0, by1)];
    const real q11 = variable_array[IDX_BLOCK(tx1, ty1, bx1, by1)];

    // Perform the bilinear interpolation
    const real q0 = (toReal(1.0) - xd) * q00 + xd * q10;
    const real q1 = (toReal(1.0) - xd) * q01 + xd * q11;

    const real interp_val = (toReal(1.0) - yd) * q0 + yd * q1;

    return interp_val;
}

__device__ __forceinline__ real velocity_bilinear_interpolation_rotated(const real x, const real y,
                                                                        const unsigned int x0, const unsigned int y0,
                                                                        const unsigned int x1, const unsigned int y1,
                                                                        const real *__restrict__ variable_array)
{

    // The dimensions and indices remain the same
    const real xd = x - toReal(x0);
    const real yd = y - toReal(y0);

    // Calculate block and thread indices for the four corners
    const size_t tx0 = x0 % BLOCK_THREAD_X;
    const size_t tx1 = x1 % BLOCK_THREAD_X;
    const size_t ty0 = y0 % BLOCK_THREAD_Y;
    const size_t ty1 = y1 % BLOCK_THREAD_Y;

    const size_t bx0 = x0 / BLOCK_THREAD_X;
    const size_t bx1 = x1 / BLOCK_THREAD_X;
    const size_t by0 = y0 / BLOCK_THREAD_Y;
    const size_t by1 = y1 / BLOCK_THREAD_Y;

    // Retrieve the values from the passed array at the four corners
    const real q00 = variable_array[IDX_BLOCK(tx0, ty0, bx0, by0)];
    const real q10 = variable_array[IDX_BLOCK(tx1, ty0, bx1, by0)];
    const real q01 = variable_array[IDX_BLOCK(tx0, ty1, bx0, by1)];
    const real q11 = variable_array[IDX_BLOCK(tx1, ty1, bx1, by1)];

    // Perform the bilinear interpolation
    const real q0 = (toReal(1.0) - xd) * q00 + xd * q10;
    const real q1 = (toReal(1.0) - xd) * q01 + xd * q11;

    const real interp_val = (toReal(1.0) - yd) * q0 + yd * q1;

    return interp_val;
}

__device__ inline void rotate_velocity_moments(const real x, const real y,
                                               const real ux, const real uy, const real mxx, const real myy, const real mxy,
                                               real &ux_prime, real &uy_prime, real &mxx_prime, real &myy_prime, real &mxy_prime)
{
    const real xc = toReal(XC);
    const real yc = toReal(YC);
    const real x_diff = x - xc;
    const real y_diff = y - yc;

    const real radius2 = x_diff * x_diff + y_diff * y_diff;
    const real inv_radius = rsqrt(radius2);

    const real cos_theta = x_diff * inv_radius;
    const real sin_theta = y_diff * inv_radius;
    const real sin_two_theta = toReal(2.0) * sin_theta * cos_theta;
    const real cos_two_theta = cos_theta * cos_theta - sin_theta * sin_theta;

    ux_prime = ux * cos_theta + uy * sin_theta;
    uy_prime = -ux * sin_theta + uy * cos_theta;

    mxx_prime = mxx * cos_theta * cos_theta + myy * sin_theta * sin_theta + mxy * sin_two_theta;
    myy_prime = mxx * sin_theta * sin_theta + myy * cos_theta * cos_theta - mxy * sin_two_theta;
    mxy_prime = toReal(0.5) * (myy - mxx) * sin_two_theta + mxy * cos_two_theta;
}

__device__ inline void moment_bilinear_interpolation(real x, real y, size_t x0, size_t y0, size_t x1, size_t y1,
                                                     nodeVar fMom, real &mxx_prime_out, real &myy_prime_out)
{
    // The dimensions and indices remain the same
    const real xd = x - toReal(x0);
    const real yd = y - toReal(y0);

    // Defining coordinate trasform variables:sint, cost, sin2t, cos2t
    const real xc = toReal(XC);
    const real yc = toReal(YC);
    const real x0_diff = toReal(x0) - xc;
    const real y0_diff = toReal(y0) - yc;
    const real x1_diff = toReal(x1) - xc;
    const real y1_diff = toReal(y1) - yc;

    const real radius_1 = sqrt(x0_diff * x0_diff + y0_diff * y0_diff);
    const real radius_2 = sqrt(x1_diff * x1_diff + y0_diff * y0_diff);
    const real radius_3 = sqrt(x0_diff * x0_diff + y1_diff * y1_diff);
    const real radius_4 = sqrt(x1_diff * x1_diff + y1_diff * y1_diff);

    const real cos_theta_1 = x0_diff / radius_1;
    const real cos_theta_2 = x1_diff / radius_2;
    const real cos_theta_3 = x0_diff / radius_3;
    const real cos_theta_4 = x1_diff / radius_4;

    const real sin_theta_1 = y0_diff / radius_1;
    const real sin_theta_2 = y0_diff / radius_2;
    const real sin_theta_3 = y1_diff / radius_3;
    const real sin_theta_4 = y1_diff / radius_4;

    const real sin_two_theta_1 = toReal(2.0) * sin_theta_1 * cos_theta_1;
    const real sin_two_theta_2 = toReal(2.0) * sin_theta_2 * cos_theta_2;
    const real sin_two_theta_3 = toReal(2.0) * sin_theta_3 * cos_theta_3;
    const real sin_two_theta_4 = toReal(2.0) * sin_theta_4 * cos_theta_4;

    // Calculate block and thread indices for the four corners
    const size_t tx0 = x0 % BLOCK_THREAD_X;
    const size_t tx1 = x1 % BLOCK_THREAD_X;
    const size_t ty0 = y0 % BLOCK_THREAD_Y;
    const size_t ty1 = y1 % BLOCK_THREAD_Y;

    const size_t bx0 = x0 / BLOCK_THREAD_X;
    const size_t bx1 = x1 / BLOCK_THREAD_X;
    const size_t by0 = y0 / BLOCK_THREAD_Y;
    const size_t by1 = y1 / BLOCK_THREAD_Y;

    // Retrieve the moments at the four corners
    const real mxx_1 = fMom.mxx[IDX_BLOCK(tx0, ty0, bx0, by0)];
    const real mxx_2 = fMom.mxx[IDX_BLOCK(tx1, ty0, bx1, by0)];
    const real mxx_3 = fMom.mxx[IDX_BLOCK(tx0, ty1, bx0, by1)];
    const real mxx_4 = fMom.mxx[IDX_BLOCK(tx1, ty1, bx1, by1)];

    const real myy_1 = fMom.myy[IDX_BLOCK(tx0, ty0, bx0, by0)];
    const real myy_2 = fMom.myy[IDX_BLOCK(tx1, ty0, bx1, by0)];
    const real myy_3 = fMom.myy[IDX_BLOCK(tx0, ty1, bx0, by1)];
    const real myy_4 = fMom.myy[IDX_BLOCK(tx1, ty1, bx1, by1)];

    const real mxy_1 = fMom.mxy[IDX_BLOCK(tx0, ty0, bx0, by0)];
    const real mxy_2 = fMom.mxy[IDX_BLOCK(tx1, ty0, bx1, by0)];
    const real mxy_3 = fMom.mxy[IDX_BLOCK(tx0, ty1, bx0, by1)];
    const real mxy_4 = fMom.mxy[IDX_BLOCK(tx1, ty1, bx1, by1)];

    // calculate the prime moments at the four corners
    const real mxx_prime_1 = mxx_1 * cos_theta_1 * cos_theta_1 + myy_1 * sin_theta_1 * sin_theta_1 + mxy_1 * sin_two_theta_1;
    const real mxx_prime_2 = mxx_2 * cos_theta_2 * cos_theta_2 + myy_2 * sin_theta_2 * sin_theta_2 + mxy_2 * sin_two_theta_2;
    const real mxx_prime_3 = mxx_3 * cos_theta_3 * cos_theta_3 + myy_3 * sin_theta_3 * sin_theta_3 + mxy_3 * sin_two_theta_3;
    const real mxx_prime_4 = mxx_4 * cos_theta_4 * cos_theta_4 + myy_4 * sin_theta_4 * sin_theta_4 + mxy_4 * sin_two_theta_4;

    const real myy_prime_1 = mxx_1 * sin_theta_1 * sin_theta_1 + myy_1 * cos_theta_1 * cos_theta_1 - mxy_1 * sin_two_theta_1;
    const real myy_prime_2 = mxx_2 * sin_theta_2 * sin_theta_2 + myy_2 * cos_theta_2 * cos_theta_2 - mxy_2 * sin_two_theta_2;
    const real myy_prime_3 = mxx_3 * sin_theta_3 * sin_theta_3 + myy_3 * cos_theta_3 * cos_theta_3 - mxy_3 * sin_two_theta_3;
    const real myy_prime_4 = mxx_4 * sin_theta_4 * sin_theta_4 + myy_4 * cos_theta_4 * cos_theta_4 - mxy_4 * sin_two_theta_4;

    // Perform the bilinear interpolation
    const real mxx0 = (toReal(1.0) - xd) * mxx_prime_1 + xd * mxx_prime_2;
    const real mxx1 = (toReal(1.0) - xd) * mxx_prime_3 + xd * mxx_prime_4;
    const real mxx_interp = (toReal(1.0) - yd) * mxx0 + yd * mxx1;

    const real myy0 = (toReal(1.0) - xd) * myy_prime_1 + xd * myy_prime_2;
    const real myy1 = (toReal(1.0) - xd) * myy_prime_3 + xd * myy_prime_4;
    const real myy_interp = (toReal(1.0) - yd) * myy0 + yd * myy1;

    mxx_prime_out = mxx_interp;
    myy_prime_out = myy_interp;
}

__device__ inline real extrapolation_quadratic(const real delta,
                                               const real val_w, const real val_f1, const real val_f2)
{
    const real delta_over_delx = delta / delx;
    const real delta_over_delx2 = delta_over_delx * delta_over_delx;
    const real coeff_p1 = toReal(1.0) - toReal(0.5) * delta_over_delx2 + toReal(1.5) * delta_over_delx;
    const real coeff_p2 = delta_over_delx2 - toReal(2.0) * delta_over_delx;
    const real coeff_p3 = -toReal(0.5) * (delta_over_delx2 - delta_over_delx);

    const real extrapolated_value = coeff_p1 * val_w + coeff_p2 * val_f1 + coeff_p3 * val_f2;

    return extrapolated_value;
}

__device__ inline real extrapolation_hybrid(const real delta, const real valW, const real val1, const real val2)
{
    const real ratio = delta / delx;
    const real threshold = 0.1f; // Switch to linear if ghost node is within 10% of delx

    if (ratio < threshold)
    {
        // --- Linear Extrapolation ---
        // P(x) passes through valW at x=delta and val1 at x=delta+delx
        // We want P(0).
        // Gradient m = (val1 - valW) / delx
        // P(0) = valW - m * delta
        return valW - (val1 - valW) * ratio;
    }
    else
    {
        // --- Corrected Quadratic (Lagrange) ---
        const real xW = delta;
        const real x1 = delta + delx;
        const real x2 = delta + 2.0f * delx;

        const real coeff_W = (x1 * x2) / ((xW - x1) * (xW - x2));
        const real coeff_1 = (xW * x2) / ((x1 - xW) * (x1 - x2));
        const real coeff_2 = (xW * x1) / ((x2 - xW) * (x2 - x1));

        return (coeff_W * valW) + (coeff_1 * val1) + (coeff_2 * val2);
    }
}

__device__ inline real extrapolation_quadratic_corrected(const real delta, const real delx,
                                                         const real valW, const real val1, const real val2)
{
    // Distances from the Ghost Node (x=0) to the points
    const real xW = delta;
    const real x1 = delta + delx;
    const real x2 = delta + 2.0f * delx;

    // Lagrange basis coefficients calculated for the Ghost Node (x=0)
    // L_W(0) = (0 - x1)(0 - x2) / [(xW - x1)(xW - x2)]
    const real coeff_W = (x1 * x2) / ((xW - x1) * (xW - x2));

    // L_1(0) = (0 - xW)(0 - x2) / [(x1 - xW)(x1 - x2)]
    const real coeff_1 = (xW * x2) / ((x1 - xW) * (x1 - x2));

    // L_2(0) = (0 - xW)(0 - x1) / [(x2 - xW)(x2 - x1)]
    const real coeff_2 = (xW * x1) / ((x2 - xW) * (x2 - x1));

    return (coeff_W * valW) + (coeff_1 * val1) + (coeff_2 * val2);
}

__device__ inline real extrapolation_linear(const real delta,
                                            const real uw,
                                            const real u1)
{
    real alpha = delta / delx;
    alpha = fmin(toReal(1.0), fmax(toReal(0.0), alpha));
    return uw + alpha * (u1 - uw);
}

#endif // EXTRAPOLATION_UTILS_H
