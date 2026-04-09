#pragma once

#include "../statistics/stat_header.cuh"



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
                                               VelocityMoments &vm)
{
    const real cos_theta = unit_nx;
    const real sin_theta = unit_ny;
    const real sin_two_theta = toReal(2.0) * sin_theta * cos_theta;
    const real cos_two_theta = cos_theta * cos_theta - sin_theta * sin_theta;

    vm.ux = ux * cos_theta + uy * sin_theta;
    vm.uy = -ux * sin_theta + uy * cos_theta;
    vm.uz = uz;

    vm.mxx = mxx * cos_theta * cos_theta + myy * sin_theta * sin_theta + mxy * sin_two_theta;
    vm.myy = mxx * sin_theta * sin_theta + myy * cos_theta * cos_theta - mxy * sin_two_theta;
    vm.mzz = mzz;
    vm.mxy = mxy * cos_two_theta + toReal(0.5) * (myy - mxx) * sin_two_theta;
    vm.mxz = mxz * cos_theta + myz * sin_theta;
    vm.myz = myz * cos_theta - mxz * sin_theta;
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
    rotate_velocity_moments(unit_nx, unit_ny, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz, vm);

    return vm;
}

