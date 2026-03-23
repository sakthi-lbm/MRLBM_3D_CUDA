#pragma once

#include "../statistics/stat_header.cuh"

__device__ __forceinline__ real bilinear_interpolation(const real x, const real y, const real z,
                                                       const real *__restrict__ variable_array)
{

    const int x0 = floor(x);
    const int y0 = floor(y);

    const int x1 = x0 + 1;
    const int y1 = y0 + 1;

    const int zg = toInt(z);

    // The dimensions and indices remain the same
    const real xd = x - toReal(x0);
    const real yd = y - toReal(y0);

    // Calculate block and thread indices for the four corners
    const size_t tx0 = x0 % BLOCK_THREAD_X;
    const size_t tx1 = x1 % BLOCK_THREAD_X;
    const size_t ty0 = y0 % BLOCK_THREAD_Y;
    const size_t ty1 = y1 % BLOCK_THREAD_Y;
    const size_t tz = zg % BLOCK_THREAD_Z;

    const size_t bx0 = x0 / BLOCK_THREAD_X;
    const size_t bx1 = x1 / BLOCK_THREAD_X;
    const size_t by0 = y0 / BLOCK_THREAD_Y;
    const size_t by1 = y1 / BLOCK_THREAD_Y;
    const size_t bz = zg / BLOCK_THREAD_Z;

    // Retrieve the values from the passed array at the four corners
    const real q00 = variable_array[IDX_BLOCK(tx0, ty0, tz, bx0, by0, bz)];
    const real q10 = variable_array[IDX_BLOCK(tx1, ty0, tz, bx1, by0, bz)];
    const real q01 = variable_array[IDX_BLOCK(tx0, ty1, tz, bx0, by1, bz)];
    const real q11 = variable_array[IDX_BLOCK(tx1, ty1, tz, bx1, by1, bz)];

    // Perform the bilinear interpolation
    const real q0 = (toReal(1.0) - xd) * q00 + xd * q10;
    const real q1 = (toReal(1.0) - xd) * q01 + xd * q11;

    const real interp_val = (toReal(1.0) - yd) * q0 + yd * q1;

    return interp_val;
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

__device__ inline void rotate_velocity_moments(const real unit_nx, const real unit_ny,
                                               const real ux, const real uy, const real uz,
                                               const real mxx, const real myy, const real mzz,
                                               const real mxy, const real mxz, const real myz,
                                               real &ux_prime, real &uy_prime, real &uz_prime,
                                               real &mxx_prime, real &myy_prime, real &mzz_prime,
                                               real &mxy_prime, real &mxz_prime, real &myz_prime)
{
    const real cos_theta = unit_nx;
    const real sin_theta = unit_ny;
    const real sin_two_theta = toReal(2.0) * sin_theta * cos_theta;
    const real cos_two_theta = cos_theta * cos_theta - sin_theta * sin_theta;

    ux_prime = ux * cos_theta + uy * sin_theta;
    uy_prime = -ux * sin_theta + uy * cos_theta;
    uz_prime = uz;

    mxx_prime = mxx * cos_theta * cos_theta + myy * sin_theta * sin_theta + mxy * sin_two_theta;
    myy_prime = mxx * sin_theta * sin_theta + myy * cos_theta * cos_theta - mxy * sin_two_theta;
    mzz_prime = mzz;
    mxy_prime = mxy * cos_two_theta + toReal(0.5) * (myy - mxx) * sin_two_theta;
    mxz_prime = mxz * cos_theta + myz * sin_theta;
    myz_prime = myz * cos_theta - mxz * sin_theta;
}

__device__ __forceinline__ VelocityMoments initerpolate_and_rotate(const real unit_nx, const real unit_ny,
                                                                   const real x, const real y, const real z,
                                                                   const nodeVar &dMom)
{
    VelocityMoments vm;

    const real ux = bilinear_interpolation(x, y, z, dMom.ux);
    const real uy = bilinear_interpolation(x, y, z, dMom.uy);
    const real uz = bilinear_interpolation(x, y, z, dMom.uz);

    const real mxx = bilinear_interpolation(x, y, z, dMom.mxx);
    const real myy = bilinear_interpolation(x, y, z, dMom.myy);
    const real mzz = bilinear_interpolation(x, y, z, dMom.mzz);
    const real mxy = bilinear_interpolation(x, y, z, dMom.mxy);
    const real mxz = bilinear_interpolation(x, y, z, dMom.mxz);
    const real myz = bilinear_interpolation(x, y, z, dMom.myz);

    // Rotate velocity & moments
    rotate_velocity_moments(unit_nx, unit_ny, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz,
                            vm.ux, vm.uy, vm.uz, vm.mxx, vm.myy, vm.mzz, vm.mxy, vm.mxz, vm.myz);

    return vm;
}

__device__ __forceinline__ real surface_pressure_extrapolation(const real xw, const real yw,
                                                               const real x1, const real y1,
                                                               const real x2, const real y2,
                                                               const real x3, const real y3,
                                                               const real rho1, const real rho2, const real rho3)
{
    // pressure interpolation
    const real xc = toReal(XC);
    const real yc = toReal(YC);

    const real xw_diff = xw - xc;
    const real yw_diff = yw - yc;

    const real x1_diff = x1 - xc;
    const real y1_diff = y1 - yc;

    const real x2_diff = x2 - xc;
    const real y2_diff = y2 - yc;

    const real x3_diff = x3 - xc;
    const real y3_diff = y3 - yc;

    const real rw2 = xw_diff * xw_diff + yw_diff * yw_diff;
    const real r12 = x1_diff * x1_diff + y1_diff * y1_diff;
    const real r22 = x2_diff * x2_diff + y2_diff * y2_diff;
    const real r32 = x3_diff * x3_diff + y3_diff * y3_diff;

    const real rw = sqrt(rw2);
    const real r1 = sqrt(r12);
    const real r2 = sqrt(r22);
    const real r3 = sqrt(r32);

    const real denom = (r1 - r2) * (r1 - r3) * (r2 - r3);

    const real p1 = rho1 * cs2;
    const real p2 = rho2 * cs2;
    const real p3 = rho3 * cs2;

    const real a0 = (r1 * r3 * p2 * (r3 - r1) + (r2 * r2) * (r3 * p1 - r1 * p3) + r2 * ((r1 * r1) * p3 - (r3 * r3) * p1)) / denom;
    const real a1 = ((r3 * r3) * (p1 - p2) + (r1 * r1) * (p2 - p3) + (r2 * r2) * (p3 - p1)) / denom;
    const real a2 = (r3 * (p2 - p1) + r2 * (p1 - p3) + r1 * (p3 - p2)) / denom;

    const real pressure = a0 + a1 * rw + a2 * (rw * rw);

    return pressure;
}
