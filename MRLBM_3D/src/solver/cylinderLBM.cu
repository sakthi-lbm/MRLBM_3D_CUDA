#include <iostream>

#include "cylinderLBM.cuh"

#ifdef CYLINDER

__device__ void evaluate_incoming_moments_rotated(const unsigned int x, const unsigned int y, const unsigned int z,
                                                  const nodeType_t nodeTag, const cylinderVar &cylinder,
                                                  const real *pop, real &rhoVar,
                                                  real &mxx, real &myy, real &mzz,
                                                  real &mxy, real &mxz, real &myz)
{
    // Defining coordinate trasform variables:sint, cost, sin2t, cos2t
    const real xb = toReal(x);
    const real yb = toReal(y);
    const real xc = toReal(XC);
    const real yc = toReal(YC);
    const real x_diff = xb - xc;
    const real y_diff = yb - yc;

    const real radius2 = x_diff * x_diff + y_diff * y_diff;
    if (radius2 <= 0.0)
        return;

    const real inv_radius = rsqrt(radius2);
    const real cos_theta = x_diff * inv_radius;
    const real sin_theta = y_diff * inv_radius;

    real rhoI_prime = toReal(0.0);
    real mxxI_prime = toReal(0.0);
    real myyI_prime = toReal(0.0);
    real mzzI_prime = toReal(0.0);
    real mxyI_prime = toReal(0.0);
    real mxzI_prime = toReal(0.0);
    real myzI_prime = toReal(0.0);

    for (int q = 0; q < Q; q++)
    {
        if (cylinder.incomings[idxBoundPop(nodeTag, q)])
        {
            const real cx = toReal(d_cx[q]);
            const real cy = toReal(d_cy[q]);
            const real cz = toReal(d_cz[q]);

            const real cx_prime = cx * cos_theta + cy * sin_theta;
            const real cy_prime = -cx * sin_theta + cy * cos_theta;
            const real cz_prime = cz;

            const real Hxx_prime = cx_prime * cx_prime - cs2;
            const real Hyy_prime = cy_prime * cy_prime - cs2;
            const real Hzz_prime = cz_prime * cz_prime - cs2;
            const real Hxy_prime = cx_prime * cy_prime;
            const real Hxz_prime = cx_prime * cz_prime;
            const real Hyz_prime = cy_prime * cz_prime;

            rhoI_prime += pop[q];
            mxxI_prime += pop[q] * Hxx_prime;
            myyI_prime += pop[q] * Hyy_prime;
            mzzI_prime += pop[q] * Hzz_prime;
            mxyI_prime += pop[q] * Hxy_prime;
            mxzI_prime += pop[q] * Hxz_prime;
            myzI_prime += pop[q] * Hyz_prime;
        }
    }
    if (rhoI_prime <= 0.0)
        return;
    const real inv_rho = toReal(1.0) / rhoI_prime;

    rhoVar = rhoI_prime;
    mxx = mxxI_prime * inv_rho;
    myy = myyI_prime * inv_rho;
    mzz = myyI_prime * inv_rho;
    mxy = mxyI_prime * inv_rho;
    mxz = mxzI_prime * inv_rho;
    myz = myzI_prime * inv_rho;
}

__device__ void cylinder_boundary_condition_rotated(const unsigned int x, const unsigned int y, const unsigned int z,
                                                    const cylinderVar &cylinder,
                                                    const nodeType_t nodeType, const nodeVar &dMom,
                                                    real &rhoVar, real &ux, real &uy, real &uz,
                                                    real &mxx, real &myy, real &mzz,
                                                    real &mxy, real &mxz, real &myz,
                                                    const real UX_PRIME, const real UY_PRIME, const real UZ_PRIME,
                                                    const int NODE_TYPE, const real D_WALL, const int iter)
{
    real fluid_dir = 1.0;

    const real xc = toReal(XC);
    const real yc = toReal(YC);

    // Boundary node location
    const real xb = toReal(x);
    const real yb = toReal(y);
    const real zb = toReal(z);

    // unit normal calculation
    const real x_diff = xb - xc;
    const real y_diff = yb - yc;
    const real radius2 = x_diff * x_diff + y_diff * y_diff;
    const real inv_radius = rsqrt(radius2);

    const real unit_nx = x_diff * inv_radius;
    const real unit_ny = y_diff * inv_radius;

    // wall point location (cylinder)
    const real r_wall = toReal(0.5) * D_WALL;
    const real xw = xc + r_wall * unit_nx;
    const real yw = yc + r_wall * unit_ny;

    const real delta = rabs((xw - xb) * unit_nx + (yw - yb) * unit_ny);

    // First reference fluid point
    const real x1 = xw + fluid_dir * delx * unit_nx;
    const real y1 = yw + fluid_dir * delx * unit_ny;

    // second reference fluid point
    const real x2 = xw + fluid_dir * toReal(2.0) * delx * unit_nx;
    const real y2 = yw + fluid_dir * toReal(2.0) * delx * unit_ny;

    // Velocity and moments bilinear interpolation for reference fluid points
    const real ux1 = bilinear_interpolation(x1, y1, zb, dMom.ux);
    const real uy1 = bilinear_interpolation(x1, y1, zb, dMom.uy);
    const real uz1 = bilinear_interpolation(x1, y1, zb, dMom.uz);

    const real mxx1 = bilinear_interpolation(x1, y1, zb, dMom.mxx);
    const real myy1 = bilinear_interpolation(x1, y1, zb, dMom.myy);
    const real mzz1 = bilinear_interpolation(x1, y1, zb, dMom.mzz);
    const real mxy1 = bilinear_interpolation(x1, y1, zb, dMom.mxy);
    const real mxz1 = bilinear_interpolation(x1, y1, zb, dMom.mxz);
    const real myz1 = bilinear_interpolation(x1, y1, zb, dMom.myz);

    const real ux2 = bilinear_interpolation(x2, y2, zb, dMom.ux);
    const real uy2 = bilinear_interpolation(x2, y2, zb, dMom.uy);
    const real uz2 = bilinear_interpolation(x2, y2, zb, dMom.uz);

    const real mxx2 = bilinear_interpolation(x2, y2, zb, dMom.mxx);
    const real myy2 = bilinear_interpolation(x2, y2, zb, dMom.myy);
    const real mzz2 = bilinear_interpolation(x2, y2, zb, dMom.mzz);
    const real mxy2 = bilinear_interpolation(x2, y2, zb, dMom.mxy);
    const real mxz2 = bilinear_interpolation(x2, y2, zb, dMom.mxz);
    const real myz2 = bilinear_interpolation(x2, y2, zb, dMom.myz);

    // Converting Cartesian moments to Rotated moments
    real ux1_prime, uy1_prime, uz1_prime, mxx1_prime, myy1_prime, mzz1_prime, myz1_prime;
    real ux2_prime, uy2_prime, uz2_prime, mxx2_prime, myy2_prime, mzz2_prime, myz2_prime;

    rotate_velocity_moments(x1, y1, ux1, uy1, mxx1, myy1, mxy1, mxz1, myz1,
                            ux1_prime, uy1_prime, mxx1_prime, myy1_prime, myz1_prime);
    rotate_velocity_moments(x2, y2, ux2, uy2, mxx2, myy2, mzz2, mxy2, mxz2,
                            ux2_prime, uy2_prime, mxx2_prime, myy2_prime, myz2_prime);

    // no rotation for z component
    uz1_prime = uz1;
    uz2_prime = uz2;

    mzz1_prime = mzz1;
    mzz2_prime = mzz2;

    const real uxw_prime = UX_PRIME; //
    const real uyw_prime = UY_PRIME; //
    const real uzw_prime = UZ_PRIME; //
    const real mxxw_prime = UX_PRIME * UX_PRIME;
    const real myyw_prime = UY_PRIME * UY_PRIME;
    const real mzzw_prime = UZ_PRIME * UZ_PRIME;
    const real myzw_prime = UY_PRIME * UZ_PRIME;

    // if (iter > MAX_ITER / 2)
    //     printf("node:%d,mxxw:%.12f, myyw:%.12f\n", toInt(NODE_TYPE), mxxw_prime, myyw_prime);

    const real ux_prime = extrapolation_quadratic(delta, uxw_prime, ux1_prime, ux2_prime);
    const real uy_prime = extrapolation_quadratic(delta, uyw_prime, uy1_prime, uy2_prime);
    const real uz_prime = extrapolation_quadratic(delta, uzw_prime, uz1_prime, uz2_prime);
    const real mxx_prime = extrapolation_quadratic(delta, mxxw_prime, mxx1_prime, mxx2_prime);
    const real myy_prime = extrapolation_quadratic(delta, myyw_prime, myy1_prime, myy2_prime);
    const real mzz_prime = extrapolation_quadratic(delta, mzzw_prime, mzz1_prime, mzz2_prime);
    const real myz_prime = extrapolation_quadratic(delta, myzw_prime, myz1_prime, myz2_prime);

    // ux = extrapolation_hybrid(delta, uxw_prime, ux1_prime, ux2_prime);
    // uy = extrapolation_hybrid(delta, uyw_prime, uy1_prime, uy2_prime);

    // ux = uxw_prime;
    // uy = uyw_prime;

    if constexpr (MASS_CONSERV == MassBC ::Equilibrium)
    {
        // numerical_solution_rhoeq_rotated(x, y, cylinder, nodeType, ux_prime, uy_prime, mxx_prime, myy_prime, rhoVar, ux, uy, mxx, myy, mxy, NODE_TYPE);
    }
    else if constexpr (MASS_CONSERV == MassBC ::Strong)
    {
        numerical_solution_strong_rotated(x, y, cylinder, nodeType, ux_prime, uy_prime, mxx_prime, myy_prime, rhoVar, ux, uy, mxx, myy, mxy, NODE_TYPE);
    }
    else if constexpr (MASS_CONSERV == MassBC ::Weak)
    {
        // numerical_solution_weak_rotated(x, y, cylinder, nodeType, ux_prime, uy_prime, mxx_prime, myy_prime, rhoVar, ux, uy, mxx, myy, mxy, NODE_TYPE);
    }
}

#endif