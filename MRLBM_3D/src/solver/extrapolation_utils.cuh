#ifndef EXTRAPOLATION_UTILS_H
#define EXTRAPOLATION_UTILS_H

#include "../all_headers.h"

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

#endif // EXTRAPOLATION_UTILS_H
